#!/usr/bin/env python3
"""
Convert normalized manifest into an SFT JSONL for Qwen3-TTS training.

Each output row contains:
  - id, text, audio (absolute path)
  - audio_codes: List[T × 16 ints]  — pre-encoded by Qwen3TTSTokenizer

Pre-encoding audio here (once) rather than inside the training collator:
  - mirrors the official Qwen3-TTS prepare_data.py recipe
  - eliminates all audio I/O overhead during training
  - removes the need for any encode() API-version fallback logic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 24_000   # Qwen3-TTS speech tokenizer expects 24 kHz


def load_audio_24k(path: str) -> np.ndarray:
    import math
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        src_t = np.linspace(0, 1, len(wav), endpoint=False)
        dst_n = int(math.ceil(len(wav) * TARGET_SR / sr))
        dst_t = np.linspace(0, 1, dst_n, endpoint=False)
        wav = np.interp(dst_t, src_t, wav).astype(np.float32)
    return wav


def load_speech_tokenizer(base_model: str):
    """
    Load only the speech tokenizer from the full Qwen3-TTS model (on CPU).
    Using CPU keeps GPU free for the subsequent training run.
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    hf_token = os.environ.get("HF_TOKEN") or None

    # Resolve local snapshot path (and apply preprocessor_config.json fix)
    load_path = base_model
    if not Path(base_model).exists():
        try:
            from huggingface_hub import snapshot_download
            load_path = snapshot_download(
                base_model,
                token=hf_token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            # Fix: copy preprocessor_config.json into speech_tokenizer/ if missing
            import shutil
            root_cfg = Path(load_path) / "preprocessor_config.json"
            sub_cfg  = Path(load_path) / "speech_tokenizer" / "preprocessor_config.json"
            if root_cfg.exists() and not sub_cfg.exists() and sub_cfg.parent.is_dir():
                shutil.copy2(str(root_cfg), str(sub_cfg))
                log.info("Copied preprocessor_config.json → speech_tokenizer/")
        except Exception as e:
            log.warning("snapshot_download failed (%s); using hub id directly", e)
            load_path = base_model

    log.info("Loading model to extract speech tokenizer (CPU) …")
    wrapper = Qwen3TTSModel.from_pretrained(
        load_path,
        dtype=torch.float32,
        device_map="cpu",
        token=hf_token,
    )
    speech_tok = wrapper.model.speech_tokenizer
    if speech_tok is None:
        raise RuntimeError("speech_tokenizer is None after from_pretrained")
    # speech_tokenizer may be a tokenizer object (not nn.Module), so only call
    # eval() if it actually has that method (i.e. it's a torch module).
    if hasattr(speech_tok, "eval"):
        speech_tok.eval()
    return speech_tok


def encode_audio(speech_tok, wav: np.ndarray) -> list[list[int]]:
    """
    Encode a 24 kHz waveform → List[T × 16 ints].

    speech_tok.encode(wav, sr=TARGET_SR) returns Qwen3TTSTokenizerV2EncoderOutput
    whose .audio_codes is a List[Tensor(T, 16)].  We take the first (and only)
    batch element and convert to a plain Python list — identical to the official
    prepare_data.py:  line['audio_codes'] = code.cpu().tolist()
    """
    result = speech_tok.encode(wav, sr=TARGET_SR)
    codes_tensor = result.audio_codes[0]   # (T, 16)
    return codes_tensor.cpu().tolist()     # List[T × [16 ints]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/sada22_qwen3tts/train_manifest.jsonl"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/sada22_qwen3tts/train_qwen3tts_sft.jsonl"),
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HF model ID or local path — used to load speech tokenizer for pre-encoding",
    )
    p.add_argument(
        "--system-prompt",
        type=str,
        default="Speak naturally in Saudi Arabic dialect.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.manifest.parent

    # ── Collect records from manifest ─────────────────────────────────────────
    raw_records: list[dict] = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio_path = Path(rec["audio"])
            if not audio_path.is_absolute():
                audio_path = (base / audio_path).resolve()
            rec["_audio_abs"] = str(audio_path)
            raw_records.append(rec)

    log.info("Loaded %d records from %s", len(raw_records), args.manifest)

    # ── Load speech tokenizer once on CPU ─────────────────────────────────────
    speech_tok = load_speech_tokenizer(args.base_model)

    # ── Encode audio + write output JSONL ─────────────────────────────────────
    from tqdm import tqdm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_skipped = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for rec in tqdm(raw_records, desc="Encoding audio", unit="utt"):
            audio_path = rec["_audio_abs"]
            if not Path(audio_path).exists():
                log.warning("Audio not found, skipping: %s", audio_path)
                rows_skipped += 1
                continue

            try:
                wav = load_audio_24k(audio_path)
                audio_codes = encode_audio(speech_tok, wav)
            except Exception as exc:
                log.warning("Encode failed for %s: %s — skipping", rec["id"], exc)
                rows_skipped += 1
                continue

            row = {
                "id":          rec["id"],
                "audio":       audio_path,
                "text":        rec["text"],
                "audio_codes": audio_codes,   # List[T × [16 ints]]
                "metadata": {
                    "language": rec.get("language", "Arabic"),
                    "locale":   rec.get("locale", "ar-SA"),
                    "speaker":  rec.get("speaker", "unknown"),
                },
                "messages": [
                    {"role": "system",    "content": args.system_prompt},
                    {"role": "user",      "content": f"Generate speech in Saudi Arabic for: {rec['text']}"},
                    {"role": "assistant", "content": "<speech>"},
                ],
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_skipped:
        log.warning("Skipped %d rows (missing audio or encode error)", rows_skipped)
    print(f"Wrote {rows_written} rows to {args.output}")


if __name__ == "__main__":
    main()
