#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for Saudi Arabic SADA22 -> Qwen3-TTS fine-tune assets

python scripts/saudi_arabic/download_sada22.py --source hf --output-dir data/sada22
python scripts/saudi_arabic/preprocess_sada22.py --input-dir data/sada22 --output-dir data/sada22_qwen3tts
python scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
  --manifest data/sada22_qwen3tts/train_manifest.jsonl \
  --output data/sada22_qwen3tts/train_qwen3tts_sft.jsonl
python scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
  --manifest data/sada22_qwen3tts/val_manifest.jsonl \
  --output data/sada22_qwen3tts/val_qwen3tts_sft.jsonl

# Dry-run by default to show the accelerate command/config.
python scripts/saudi_arabic/train_qwen3tts.py --mode lora --dry-run
