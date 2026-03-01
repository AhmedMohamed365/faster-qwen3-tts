#!/usr/bin/env bash
set -euo pipefail

# Quick CPU smoke test for CI/local verification.
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/saudi_arabic/download_sada22.py --source hf --max-samples 12 --output-dir data/sada22_smoke
python scripts/saudi_arabic/preprocess_sada22.py --input-dir data/sada22_smoke --output-dir data/sada22_smoke_qwen3tts --val-ratio 0.2
python scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
  --manifest data/sada22_smoke_qwen3tts/train_manifest.jsonl \
  --output data/sada22_smoke_qwen3tts/train_qwen3tts_sft.jsonl
python scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
  --manifest data/sada22_smoke_qwen3tts/val_manifest.jsonl \
  --output data/sada22_smoke_qwen3tts/val_qwen3tts_sft.jsonl

python scripts/saudi_arabic/train_qwen3tts.py \
  --mode lora \
  --train-jsonl data/sada22_smoke_qwen3tts/train_qwen3tts_sft.jsonl \
  --val-jsonl data/sada22_smoke_qwen3tts/val_qwen3tts_sft.jsonl \
  --base-model Qwen/Qwen3-TTS-0.6B-Base \
  --trainer-entrypoint scripts.saudi_arabic.mock_trainer \
  --output-dir outputs/qwen3tts-sada22-smoke
