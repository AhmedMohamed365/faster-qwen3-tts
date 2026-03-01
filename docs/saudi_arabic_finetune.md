# Saudi Arabic (ar-SA) fine-tuning guide for Qwen3-TTS

This guide adds a reproducible path to adapt Qwen3-TTS to **Saudi dialect Arabic only** using **SADA22**.

## 1) Data sources

Primary source:
- Hugging Face: `MohamedRashad/SADA22`

Fallback mirror:
- Kaggle: `sdaiancai/sada2022`

## 2) Install extra dependencies

```bash
pip install datasets soundfile accelerate peft
```

If you use Kaggle as fallback:

```bash
pip install kaggle
```

## 3) Fast CPU smoke test (recommended first)

This validates the full data + training wiring quickly on CPU:

- downloads a tiny subset (`--max-samples 12`) from HF,
- preprocesses and formats JSONL,
- runs one mock optimizer step via `accelerate launch`.

```bash
scripts/saudi_arabic/smoke_test_pipeline.sh
```

Success signal:

```text
FIRST_TRAIN_STEP_OK ...
```

## 4) GPU smoke test (CUDA required)

To validate that the training plumbing works on GPU, run:

```bash
scripts/saudi_arabic/gpu_smoke_test.sh
```

Expected success includes:
- `FIRST_TRAIN_STEP_OK device=cuda ...`

## 5) Download SADA22

Hugging Face (recommended):

```bash
python scripts/saudi_arabic/download_sada22.py --source hf --output-dir data/sada22
```

For fast checks, limit rows:

```bash
python scripts/saudi_arabic/download_sada22.py --source hf --max-samples 200 --output-dir data/sada22_small
```

Kaggle fallback:

```bash
python scripts/saudi_arabic/download_sada22.py --source kaggle --output-dir data/sada22
```

## 6) Preprocess and normalize for Qwen3-TTS

This script:
- keeps only samples containing Arabic-script text,
- resamples audio to 24kHz mono,
- writes `train_manifest.jsonl` and `val_manifest.jsonl`,
- tags each row with `language=Arabic` and `locale=ar-SA`.

```bash
python scripts/saudi_arabic/preprocess_sada22.py \
  --input-dir data/sada22 \
  --output-dir data/sada22_qwen3tts
```

## 7) Build SFT JSONL files

```bash
python scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
  --manifest data/sada22_qwen3tts/train_manifest.jsonl \
  --output data/sada22_qwen3tts/train_qwen3tts_sft.jsonl

python scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
  --manifest data/sada22_qwen3tts/val_manifest.jsonl \
  --output data/sada22_qwen3tts/val_qwen3tts_sft.jsonl
```

## 8) Train (LoRA or full fine-tuning)

The launcher emits a JSON config and starts `accelerate launch`.

### LoRA (recommended first)

```bash
python scripts/saudi_arabic/train_qwen3tts.py \
  --mode lora \
  --train-jsonl data/sada22_qwen3tts/train_qwen3tts_sft.jsonl \
  --val-jsonl data/sada22_qwen3tts/val_qwen3tts_sft.jsonl \
  --base-model Qwen/Qwen3-TTS-0.6B-Base \
  --output-dir outputs/qwen3tts-sada22-lora
```

### Full fine-tuning

```bash
python scripts/saudi_arabic/train_qwen3tts.py \
  --mode full \
  --train-jsonl data/sada22_qwen3tts/train_qwen3tts_sft.jsonl \
  --val-jsonl data/sada22_qwen3tts/val_qwen3tts_sft.jsonl \
  --base-model Qwen/Qwen3-TTS-0.6B-Base \
  --output-dir outputs/qwen3tts-sada22-full
```

> Note: the launcher defaults to module `qwen_tts.finetune.train_sft`. If your upstream training entrypoint differs, pass `--trainer-entrypoint <module.path>`.

## 9) One-command data prep pipeline

```bash
scripts/saudi_arabic/run_end_to_end.sh
```

The final command in that script is a dry run, so it prints the exact training command without starting a long job.

## 10) Quality checklist for Saudi-only dialect

- Verify transcript normalization removes non-Arabic rows.
- Track per-speaker balance (avoid overfitting to one speaker).
- Hold out a Saudi-only validation set (`val_manifest.jsonl`).
- Run subjective MOS + dialectity checks with native Saudi listeners.

## 11) How to add another language/dialect

1. Replace the dataset source in `download_sada22.py` (or add a sibling downloader script).
2. In `preprocess_sada22.py`, update:
   - language filter regex/cleaning,
   - `language` and `locale` fields,
   - optional duration/speaker filters.
3. Rebuild SFT JSONL using `build_qwen3tts_sft_jsonl.py`.
4. Tune prompts in `--system-prompt` to enforce target dialect style.
5. Start with LoRA first; move to full fine-tuning only if quality plateaus.

## 12) Practical defaults

- Start from `Qwen/Qwen3-TTS-0.6B-Base` for faster iteration.
- Use LoRA (`r=64`) first; upgrade rank if underfitting.
- Keep `max_audio_seconds` around 12–16 for stability and memory.
- Use bf16 on Ampere/Hopper GPUs.
