#!/usr/bin/env python3
"""
Validate that the fine-tuned Qwen3-TTS model is actually learning.

Two checks:
  1. Loss curve  — reads loss_history.json from checkpoint; confirms loss decreased.
  2. Audio output — generates speech from a val sample with base model and
                     fine-tuned model; saves both WAVs for listening comparison.

Usage:
    .venv/bin/python scripts/saudi_arabic/validate_finetuned.py \
        --base-model   Qwen/Qwen3-TTS-0.6B-Base \
        --finetuned-model outputs/qwen3tts-sada22-lora \
        --val-jsonl    data/sada22_small_qwen3tts/val_manifest.jsonl \
        --output-dir   outputs/validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 24_000


def audio_stats(wav: np.ndarray, sr: int) -> dict:
    duration = len(wav) / sr
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wav)))
    return {"duration_sec": round(duration, 3), "rms": round(rms, 6), "peak": round(peak, 6)}


def pick_val_sample(val_jsonl: Path) -> dict | None:
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
    # Try loss_history.json written by finetune_trainer.py
    history: list[dict] = []
    loss_file = finetuned_dir / "loss_history.json"
    if loss_file.exists():
        history = json.loads(loss_file.read_text())
    else:
        # Fall back to HF Trainer's trainer_state.json
        state_file = finetuned_dir / "trainer_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            history = [
                {"step": e["step"], "loss": e.get("loss", e.get("train_loss"))}
                for e in state.get("log_history", [])
                if "loss" in e or "train_loss" in e
            ]
        else:
            log.warning("No loss_history.json or trainer_state.json in %s", finetuned_dir)
            return False

    losses = [(e["step"], e["loss"]) for e in history if e.get("loss") is not None]
    if not losses:
        log.warning("No numeric loss entries found.")
        return False

    print("\n=== Loss Curve ===")
    for step, loss in losses:
        bar_len = min(40, max(1, int(loss * 5)))
        bar = "█" * bar_len
        print(f"  step {step:>5}: {loss:.4f}  {bar}")

    first_loss = losses[0][1]
    last_loss = losses[-1][1]
    improved = last_loss < first_loss

    print(f"\n  First step loss : {first_loss:.4f}")
    print(f"  Last step loss  : {last_loss:.4f}")
    print(f"  Learning signal : {'✓ IMPROVED (model is learning!)' if improved else '✗ NOT IMPROVED (try more steps)'}")
    return improved


# ---------------------------------------------------------------------------
# Check 2: Audio generation — correct qwen_tts API
# ---------------------------------------------------------------------------

def _generate_with_wrapper(wrapper, text: str, ref_audio_path: str, output_wav: Path) -> dict | None:
    """Use Qwen3TTSModel.generate_voice_clone() — the standard API."""
    import soundfile as sf

    ref_wav, ref_sr = sf.read(ref_audio_path, dtype="float32")
    if ref_wav.ndim == 2:
        ref_wav = ref_wav.mean(axis=1)

    try:
        # Pass ref audio as (ndarray, sr) tuple — supported natively
        wavs, sr = wrapper.generate_voice_clone(
            text=text,
            ref_audio=(ref_wav, ref_sr),
            x_vector_only_mode=True,   # speaker embedding only (no ref text needed)
            language="Arabic",
        )
    except Exception as exc:
        log.error("generate_voice_clone failed: %s", exc)
        return None

    wav_out = np.array(wavs[0], dtype=np.float32).squeeze()
    if wav_out.size == 0:
        log.error("Empty audio output.")
        return None

    peak = np.max(np.abs(wav_out))
    if peak > 0:
        wav_out = wav_out / peak * 0.9

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wav_out, sr)
    log.info("Saved %s  (%.2f s, sr=%d)", output_wav.name, len(wav_out) / sr, sr)
    return audio_stats(wav_out, sr)


def check_audio(
    base_model: str,
    finetuned_dir: Path,
    val_sample: dict,
    output_dir: Path,
) -> bool:
    try:
        from qwen_tts import Qwen3TTSModel
        from peft import PeftModel
    except ImportError as exc:
        log.error("Missing dependency: %s  — skipping audio check.", exc)
        return False

    text = val_sample["text"]
    ref_audio = val_sample["_audio_abs"]
    print(f"\n=== Audio Generation Test ===")
    print(f"  Text : {text}")
    print(f"  Ref  : {ref_audio}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # --- Base model ---
    print("\n  Loading base model …")
    base_wrapper = Qwen3TTSModel.from_pretrained(base_model, torch_dtype=dtype, device_map=device)
    base_stats = _generate_with_wrapper(base_wrapper, text, ref_audio, output_dir / "base_output.wav")
    del base_wrapper
    torch.cuda.empty_cache() if device == "cuda" else None

    # --- Fine-tuned model ---
    print("\n  Loading fine-tuned model (base + LoRA adapter) …")
    ft_wrapper = Qwen3TTSModel.from_pretrained(base_model, torch_dtype=dtype, device_map=device)
    try:
        ft_wrapper.model.talker = PeftModel.from_pretrained(ft_wrapper.model.talker, str(finetuned_dir))
        log.info("LoRA adapter loaded.")
    except Exception as exc:
        log.warning("PeftModel.from_pretrained failed (%s); treating as full fine-tune …", exc)
        try:
            from transformers import AutoModelForCausalLM
            ft_wrapper.model.talker = AutoModelForCausalLM.from_pretrained(
                str(finetuned_dir), torch_dtype=dtype
            ).to(device)
        except Exception as exc2:
            log.error("Could not load fine-tuned talker: %s", exc2)
            return False

    ft_stats = _generate_with_wrapper(ft_wrapper, text, ref_audio, output_dir / "finetuned_output.wav")
    del ft_wrapper
    torch.cuda.empty_cache() if device == "cuda" else None

    # Report
    print("\n  Results:")
    if base_stats:
        print(f"    Base model      → base_output.wav          {base_stats}")
    else:
        print("    Base model      → FAILED (check logs above)")
    if ft_stats:
        print(f"    Fine-tuned model→ finetuned_output.wav      {ft_stats}")
    else:
        print("    Fine-tuned model→ FAILED (check logs above)")

    if base_stats and ft_stats:
        print(f"\n  Files saved to: {output_dir.resolve()}/")
    return base_stats is not None and ft_stats is not None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-TTS-0.6B-Base")
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

    loss_ok = check_loss_curve(args.finetuned_model)

    audio_ok = True
    if not args.skip_audio:
        val_sample = pick_val_sample(args.val_jsonl)
        if val_sample is None:
            log.warning("No valid val sample found in %s — skipping audio check.", args.val_jsonl)
            audio_ok = False
        else:
            audio_ok = check_audio(args.base_model, args.finetuned_model, val_sample, args.output_dir)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print(f"  Loss decreasing : {'✓ PASS' if loss_ok else '✗ FAIL'}")
    print(f"  Audio generated : {'✓ PASS' if audio_ok else '✗ FAIL'}")
    print("=" * 60 + "\n")

    if not loss_ok:
        print("TIP: Loss didn't improve — try MAX_STEPS=200 or check data quality.")

    sys.exit(0 if loss_ok else 1)


if __name__ == "__main__":
    main()
