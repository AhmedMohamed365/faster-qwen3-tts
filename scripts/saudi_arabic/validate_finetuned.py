#!/usr/bin/env python3
"""
Validate that the fine-tuned Qwen3-TTS model is actually learning.

Two checks are performed:
  1. Loss curve  — reads loss_history.json from the checkpoint and confirms
                   the loss at the last logged step is lower than the first.
  2. Audio output — generates speech from a held-out val sample with:
                    (a) the base model unchanged
                    (b) the fine-tuned adapter loaded on top of the base
                   Both .wav files are saved for manual listening comparison.

Usage:
    python scripts/saudi_arabic/validate_finetuned.py \
        --base-model   Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --finetuned-model outputs/qwen3tts-sada22-lora \
        --val-jsonl    data/sada22_small_qwen3tts/val_manifest.jsonl \
        --output-dir   outputs/validate
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 24_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def audio_stats(wav: np.ndarray, sr: int) -> dict:
    duration = len(wav) / sr
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wav)))
    return {"duration_sec": round(duration, 3), "rms": round(rms, 6), "peak": round(peak, 6)}


def pick_val_sample(val_jsonl: Path) -> dict | None:
    """Return first valid record from the val manifest."""
    if not val_jsonl.exists():
        return None
    base = val_jsonl.parent
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio_path = base / rec["audio"]
            if audio_path.exists():
                rec["_audio_abs"] = str(audio_path)
                return rec
    return None


# ---------------------------------------------------------------------------
# Check 1: Loss curve
# ---------------------------------------------------------------------------

def check_loss_curve(finetuned_dir: Path) -> bool:
    loss_file = finetuned_dir / "loss_history.json"
    if not loss_file.exists():
        # Also try trainer_state.json produced by HF Trainer
        trainer_state = finetuned_dir / "trainer_state.json"
        if trainer_state.exists():
            state = json.loads(trainer_state.read_text())
            history = [
                {"step": e["step"], "loss": e.get("loss", e.get("train_loss"))}
                for e in state.get("log_history", [])
                if "loss" in e or "train_loss" in e
            ]
            loss_file_data = history
        else:
            log.warning("No loss_history.json or trainer_state.json found in %s", finetuned_dir)
            return False
    else:
        loss_file_data = json.loads(loss_file.read_text())

    losses = [(e["step"], e["loss"]) for e in loss_file_data if e.get("loss") is not None]
    if not losses:
        log.warning("No loss entries found.")
        return False

    print("\n=== Loss Curve ===")
    for step, loss in losses:
        bar = "█" * min(40, int(loss * 10))
        print(f"  step {step:>5}: {loss:.4f}  {bar}")

    first_loss = losses[0][1]
    last_loss = losses[-1][1]
    improved = last_loss < first_loss

    print(f"\n  First step loss : {first_loss:.4f}")
    print(f"  Last step loss  : {last_loss:.4f}")
    print(f"  Learning signal : {'✓ IMPROVED' if improved else '✗ NOT IMPROVED (may need more steps)'}")
    return improved


# ---------------------------------------------------------------------------
# Check 2: Audio generation
# ---------------------------------------------------------------------------

def _load_base_and_generate(model_path: str, text: str, ref_audio_path: str, output_wav: Path) -> dict | None:
    """
    Generate audio from model_path for given text.
    Returns audio stats dict or None on failure.
    """
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        log.error("qwen_tts not importable; skipping audio generation")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    log.info("Loading model from %s …", model_path)
    _hf_token = os.environ.get("HF_TOKEN") or None
    with torch.no_grad():
        model = Qwen3TTSModel.from_pretrained(model_path, torch_dtype=dtype, token=_hf_token).to(device)
        model.eval()

        # Build voice clone prompt from reference audio (xvec only = fast)
        ref_wav, ref_sr = sf.read(ref_audio_path, dtype="float32")
        if ref_wav.ndim == 2:
            ref_wav = ref_wav.mean(axis=1)

        try:
            prompt_items = model.create_voice_clone_prompt(
                audio=ref_wav,
                sr=ref_sr,
                text=text,
                x_vector_only_mode=True,
            )
        except Exception as exc:
            log.warning("create_voice_clone_prompt failed (%s), trying plain generate…", exc)
            prompt_items = None

        # Generate
        try:
            if prompt_items is not None:
                # Use talker generate directly
                wav_out = model.generate(text=text, prompt=prompt_items)
            else:
                wav_out = model.generate(text=text)
        except Exception as exc:
            log.error("model.generate failed: %s", exc)
            return None

    if isinstance(wav_out, torch.Tensor):
        wav_out = wav_out.float().cpu().numpy().squeeze()
    elif isinstance(wav_out, np.ndarray):
        wav_out = wav_out.squeeze()
    else:
        wav_out = np.array(wav_out, dtype=np.float32).squeeze()

    if wav_out.ndim == 0 or wav_out.size == 0:
        log.error("Empty audio output from %s", model_path)
        return None

    # Normalize
    peak = np.max(np.abs(wav_out))
    if peak > 0:
        wav_out = wav_out / peak * 0.9

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wav_out, TARGET_SR)
    log.info("Saved %s (%.2f s)", output_wav, len(wav_out) / TARGET_SR)
    return audio_stats(wav_out, TARGET_SR)


def _load_finetuned_and_generate(
    base_model: str,
    finetuned_dir: Path,
    text: str,
    ref_audio_path: str,
    output_wav: Path,
) -> dict | None:
    """
    Load base + LoRA adapter from finetuned_dir, generate audio.
    """
    try:
        from qwen_tts import Qwen3TTSModel
        from peft import PeftModel
    except ImportError as exc:
        log.error("Missing dependency: %s", exc)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    log.info("Loading base model + LoRA adapter from %s …", finetuned_dir)
    _hf_token = os.environ.get("HF_TOKEN") or None
    model = Qwen3TTSModel.from_pretrained(base_model, torch_dtype=dtype, token=_hf_token).to(device)
    model.eval()

    # Apply LoRA adapter to the talker
    try:
        model.talker = PeftModel.from_pretrained(model.talker, str(finetuned_dir))
        log.info("LoRA adapter loaded successfully.")
    except Exception as exc:
        log.warning("PeftModel.from_pretrained failed (%s); trying as full fine-tuned weights …", exc)
        # Fallback: maybe it was saved as a full model
        try:
            from transformers import AutoModelForCausalLM
            model.talker = AutoModelForCausalLM.from_pretrained(
                str(finetuned_dir), torch_dtype=dtype
            ).to(device)
        except Exception as exc2:
            log.error("Could not load fine-tuned weights: %s", exc2)
            return None

    with torch.no_grad():
        ref_wav, ref_sr = sf.read(ref_audio_path, dtype="float32")
        if ref_wav.ndim == 2:
            ref_wav = ref_wav.mean(axis=1)

        try:
            prompt_items = model.create_voice_clone_prompt(
                audio=ref_wav, sr=ref_sr, text=text, x_vector_only_mode=True
            )
        except Exception:
            prompt_items = None

        try:
            wav_out = model.generate(text=text, prompt=prompt_items) if prompt_items else model.generate(text=text)
        except Exception as exc:
            log.error("Fine-tuned model.generate failed: %s", exc)
            return None

    if isinstance(wav_out, torch.Tensor):
        wav_out = wav_out.float().cpu().numpy().squeeze()
    elif isinstance(wav_out, np.ndarray):
        wav_out = wav_out.squeeze()
    else:
        wav_out = np.array(wav_out, dtype=np.float32).squeeze()

    if wav_out.ndim == 0 or wav_out.size == 0:
        log.error("Empty audio from fine-tuned model")
        return None

    peak = np.max(np.abs(wav_out))
    if peak > 0:
        wav_out = wav_out / peak * 0.9

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wav_out, TARGET_SR)
    log.info("Saved %s (%.2f s)", output_wav, len(wav_out) / TARGET_SR)
    return audio_stats(wav_out, TARGET_SR)


def check_audio(
    base_model: str,
    finetuned_dir: Path,
    val_sample: dict,
    output_dir: Path,
) -> bool:
    text = val_sample["text"]
    ref_audio = val_sample["_audio_abs"]
    log.info("Validation text: %s", text)

    print("\n=== Audio Generation Test ===")
    print(f"  Text : {text}")
    print(f"  Ref  : {ref_audio}")

    base_wav = output_dir / "base_output.wav"
    ft_wav = output_dir / "finetuned_output.wav"

    base_stats = _load_base_and_generate(base_model, text, ref_audio, base_wav)
    ft_stats = _load_finetuned_and_generate(base_model, finetuned_dir, text, ref_audio, ft_wav)

    print("\n  Results:")
    if base_stats:
        print(f"    Base model      → {base_wav.name}  {base_stats}")
    else:
        print("    Base model      → FAILED (check logs)")
    if ft_stats:
        print(f"    Fine-tuned model→ {ft_wav.name}  {ft_stats}")
    else:
        print("    Fine-tuned model→ FAILED (check logs)")

    both_ok = base_stats is not None and ft_stats is not None
    if both_ok:
        print("\n  Listen and compare:")
        print(f"    Base     : {base_wav.resolve()}")
        print(f"    Finetune : {ft_wav.resolve()}")
    return both_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--finetuned-model", type=Path, default=Path("outputs/qwen3tts-sada22-lora"))
    p.add_argument("--val-jsonl", type=Path, default=Path("data/sada22_small_qwen3tts/val_manifest.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/validate"))
    p.add_argument("--skip-audio", action="store_true", help="Only check loss curve, skip audio generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  VALIDATION: Qwen3-TTS Saudi Arabic fine-tune")
    print("=" * 60)

    # --- Check 1: loss ---
    loss_ok = check_loss_curve(args.finetuned_model)

    # --- Check 2: audio ---
    audio_ok = True
    if not args.skip_audio:
        val_sample = pick_val_sample(args.val_jsonl)
        if val_sample is None:
            log.warning("No valid val sample found in %s; skipping audio check.", args.val_jsonl)
            audio_ok = False
        else:
            audio_ok = check_audio(args.base_model, args.finetuned_model, val_sample, args.output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print(f"  Loss decreasing : {'PASS' if loss_ok else 'FAIL'}")
    print(f"  Audio generated : {'PASS' if audio_ok else 'FAIL (run with --skip-audio to ignore)'}")
    print("=" * 60 + "\n")

    if not loss_ok:
        print("TIP: If loss didn't improve, try increasing --max-steps (e.g. 200) or check data quality.")

    sys.exit(0 if loss_ok else 1)


if __name__ == "__main__":
    main()
