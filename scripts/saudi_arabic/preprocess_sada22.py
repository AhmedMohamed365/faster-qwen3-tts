#!/usr/bin/env python3
"""Normalize SADA22 into a Qwen3-TTS-friendly audio/text manifest."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TARGET_SR = 24_000
ARABIC_LETTERS = re.compile(r"[\u0600-\u06FF]")
WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = WHITESPACE.sub(" ", text)
    # Keep only Arabic-script-centric samples to stay Saudi dialect focused.
    if not ARABIC_LETTERS.search(text):
        return ""
    return text


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        return audio.astype(np.float32)

    src_t = np.linspace(0, 1, num=len(audio), endpoint=False)
    dst_n = int(math.ceil(len(audio) * dst_sr / src_sr))
    dst_t = np.linspace(0, 1, num=dst_n, endpoint=False)
    out = np.interp(dst_t, src_t, audio).astype(np.float32)
    return out


def extract_audio_text(example: dict[str, Any], audio_key: str, text_key: str) -> tuple[np.ndarray, int, str]:
    audio_obj = example[audio_key]
    text = str(example[text_key])

    if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
        return np.asarray(audio_obj["array"], dtype=np.float32), int(audio_obj["sampling_rate"]), text

    if isinstance(audio_obj, dict) and "bytes" in audio_obj and audio_obj["bytes"]:
        wav, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
        return np.asarray(wav, dtype=np.float32), int(sr), text

    if isinstance(audio_obj, dict) and "path" in audio_obj and audio_obj["path"] and Path(audio_obj["path"]).exists():
        wav, sr = sf.read(audio_obj["path"])
        return np.asarray(wav, dtype=np.float32), int(sr), text

    if isinstance(audio_obj, str):
        wav, sr = sf.read(audio_obj)
        return np.asarray(wav, dtype=np.float32), int(sr), text

    raise ValueError(f"Unsupported audio field format: {type(audio_obj)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("data/sada22"))
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output-dir", type=Path, default=Path("data/sada22_qwen3tts"))
    p.add_argument("--audio-key", type=str, default="audio")
    p.add_argument("--text-key", type=str, default="text")
    p.add_argument("--speaker-key", type=str, default="speaker_id")
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("`datasets` is required. Install with: pip install datasets") from exc

    ds_root = args.input_dir
    if not ds_root.exists():
        raise SystemExit(f"Input directory not found: {ds_root}")

    ds = load_from_disk(str(ds_root))
    split = ds[args.split] if hasattr(ds, "keys") else ds

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(split))
    rng.shuffle(indices)
    val_size = int(len(indices) * args.val_ratio)
    val_idx = set(indices[:val_size].tolist())

    wav_dir = args.output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    kept = 0
    for i, ex in enumerate(split):
        try:
            wav, sr, text = extract_audio_text(ex, args.audio_key, args.text_key)
        except Exception:
            continue

        text = normalize_text(text)
        if not text:
            continue

        wav = resample_audio(wav, sr, TARGET_SR)
        peak = np.max(np.abs(wav)) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak

        if wav.size < TARGET_SR // 2:
            continue

        utt_id = f"sada22_{i:07d}"
        rel_path = Path("wav") / f"{utt_id}.wav"
        out_wav = args.output_dir / rel_path
        sf.write(out_wav, wav, TARGET_SR)

        speaker = str(ex.get(args.speaker_key, "unknown"))
        rec = {
            "id": utt_id,
            "audio": str(rel_path),
            "text": text,
            "language": "Arabic",
            "locale": "ar-SA",
            "speaker": speaker,
            "duration_sec": round(len(wav) / TARGET_SR, 3),
        }

        if i in val_idx:
            val_records.append(rec)
        else:
            train_records.append(rec)
        kept += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_records) + "\n", encoding="utf-8"
    )
    (args.output_dir / "val_manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_records) + "\n", encoding="utf-8"
    )

    stats = {
        "source": str(ds_root),
        "split": args.split,
        "kept": kept,
        "train": len(train_records),
        "val": len(val_records),
        "sample_rate": TARGET_SR,
    }
    (args.output_dir / "preprocess_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
