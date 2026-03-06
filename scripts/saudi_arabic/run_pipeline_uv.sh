#!/usr/bin/env bash
# =============================================================================
# run_pipeline_uv.sh
#
# End-to-end Saudi Arabic Qwen3-TTS fine-tune pipeline using uv for fast
# dependency installation.
#
# Checklist:
#   [1] Install uv + create venv
#   [2] Install all Python dependencies
#   [3] Download 200 samples (Speaker1متحدث / Najdi dialect only)
#   [4] Preprocess → 24kHz WAV + manifests
#   [5] Build SFT JSONL
#   [6] Fine-tune (LoRA, 50 steps) — uses official trainer or our fallback
#   [7] Validate: loss curve + audio comparison
#
# Usage:
#   bash scripts/saudi_arabic/run_pipeline_uv.sh
#
# Optional env vars:
#   SPEAKER   - SADA22 speaker value (default: "Speaker1متحدث")
#   DIALECT   - SADA22 dialect value (default: "Najdi")
#   MAX_SAMPLES - number of samples to download (default: 200)
#   MAX_STEPS   - LoRA training steps (default: 50)
#   BASE_MODEL  - HF model id (default: Qwen/Qwen3-TTS-0.6B-Base)
#   DATA_DIR    - root data directory (default: data/sada22_small)
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SPEAKER="${SPEAKER:-Speaker1متحدث}"
DIALECT="${DIALECT:-Najdi}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_STEPS="${MAX_STEPS:-50}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-TTS-0.6B-Base}"

DATA_RAW="${DATA_DIR:-data/sada22_small}"
DATA_PROCESSED="${DATA_RAW}_qwen3tts"
OUTPUT_DIR="outputs/qwen3tts-sada22-lora"
VALIDATE_DIR="outputs/validate"

# Navigate to repo root regardless of where the script is called from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Use .venv/bin/python directly (symlink → ~/venvs/qwen3tts-finetune)
PY="$REPO_ROOT/.venv/bin/python"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}▶ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║      Qwen3-TTS Saudi Arabic LoRA Fine-tune Pipeline      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Speaker    : $SPEAKER"
echo "  Dialect    : $DIALECT"
echo "  Samples    : $MAX_SAMPLES"
echo "  Max steps  : $MAX_STEPS"
echo "  Base model : $BASE_MODEL"
echo "  Python     : $($PY --version 2>&1)  [using $PY]"
echo "  CUDA       : $($PY -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'torch not ready')"
echo ""

if [[ ! -x "$PY" ]]; then
    err "$PY not found. Run setup:\n  mkdir -p ~/venvs && uv venv ~/venvs/qwen3tts-finetune --python 3.11\n  ln -sfn ~/venvs/qwen3tts-finetune .venv\n  uv pip install -e '.[train]' --python .venv/bin/python"
fi

# =============================================================================
# [1/6] Verify packages
# =============================================================================
info "[1/6] Verifying required packages …"
MISSING=""
for pkg in datasets accelerate peft soundfile transformers torch; do
    "$PY" -c "import $pkg" 2>/dev/null || MISSING="$MISSING $pkg"
done
if [[ -n "$MISSING" ]]; then
    info "Installing missing:$MISSING"
    uv pip install $MISSING --python "$PY"
fi
ok "All required packages present."

# Check HuggingFace authentication
HF_AUTH_OK=$("$PY" -c "
from huggingface_hub import HfFolder
import os
t = HfFolder.get_token() or os.environ.get('HF_TOKEN','')
print('yes' if t else 'no')
" 2>/dev/null)
if [[ "$HF_AUTH_OK" != "yes" ]]; then
    echo ""
    echo -e "${RED}✗ Not logged in to HuggingFace.${NC}"
    echo "  Qwen3-TTS and SADA22 require a HuggingFace access token."
    echo "  Fix: .venv/bin/huggingface-cli login"
    echo "  Or:  export HF_TOKEN=hf_xxxx && bash scripts/saudi_arabic/run_pipeline_uv.sh"
    echo "  Accept license at: https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base"
    exit 1
fi
ok "HuggingFace authenticated."

# =============================================================================
# [2/6] Download data (speaker + dialect filter)
# =============================================================================
info "[2/6] Downloading SADA22 (${MAX_SAMPLES} samples, speaker='${SPEAKER}', dialect='${DIALECT}') …"

if [[ -f "$DATA_RAW/download_manifest.json" ]]; then
    EXISTING_ROWS=$("$PY" -c "import json; d=json.load(open('$DATA_RAW/download_manifest.json')); print(d.get('rows',0))" 2>/dev/null || echo 0)
    if [[ "$EXISTING_ROWS" -ge "$MAX_SAMPLES" ]]; then
        ok "Data already downloaded ($EXISTING_ROWS rows). Skipping."
    else
        info "Existing data has only $EXISTING_ROWS rows; re-downloading …"
        "$PY" scripts/saudi_arabic/download_sada22.py \
            --source hf \
            --max-samples "$MAX_SAMPLES" \
            --speaker "$SPEAKER" \
            --dialect "$DIALECT" \
            --output-dir "$DATA_RAW"
    fi
else
    "$PY" scripts/saudi_arabic/download_sada22.py \
        --source hf \
        --max-samples "$MAX_SAMPLES" \
        --speaker "$SPEAKER" \
        --dialect "$DIALECT" \
        --output-dir "$DATA_RAW"
fi

DOWNLOADED_ROWS=$("$PY" -c "import json; d=json.load(open('$DATA_RAW/download_manifest.json')); print(d['rows'])")
ok "[2/6] Downloaded: $DOWNLOADED_ROWS rows → $DATA_RAW"

# =============================================================================
# [3/6] Preprocess: resample to 24kHz, filter Arabic text, split train/val
# =============================================================================
info "[3/6] Preprocessing audio + building manifests …"

if [[ -f "$DATA_PROCESSED/train_manifest.jsonl" ]]; then
    TRAIN_LINES=$(wc -l < "$DATA_PROCESSED/train_manifest.jsonl")
    ok "Preprocessed data already exists ($TRAIN_LINES train rows). Skipping."
else
    # SADA22-specific column names:
    #   ProcessedText  = text column
    #   Speaker        = speaker id column
    "$PY" scripts/saudi_arabic/preprocess_sada22.py \
        --input-dir  "$DATA_RAW" \
        --output-dir "$DATA_PROCESSED" \
        --text-key   "ProcessedText" \
        --speaker-key "Speaker" \
        --val-ratio  0.1 \
        --seed       42
    TRAIN_LINES=$(wc -l < "$DATA_PROCESSED/train_manifest.jsonl")
    ok "[3/6] Preprocessed: $TRAIN_LINES train rows → $DATA_PROCESSED"
fi

# =============================================================================
# [4/6] Build SFT JSONL
# =============================================================================
info "[4/6] Building SFT JSONL files …"

TRAIN_JSONL="$DATA_PROCESSED/train_qwen3tts_sft.jsonl"
VAL_JSONL="$DATA_PROCESSED/val_qwen3tts_sft.jsonl"

if [[ -f "$TRAIN_JSONL" && -f "$VAL_JSONL" ]]; then
    ok "SFT JSONLs already exist. Skipping."
else
    "$PY" scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
        --manifest "$DATA_PROCESSED/train_manifest.jsonl" \
        --output   "$TRAIN_JSONL"

    "$PY" scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
        --manifest "$DATA_PROCESSED/val_manifest.jsonl" \
        --output   "$VAL_JSONL"
fi

ok "[4/6] SFT JSONL ready: $(wc -l < "$TRAIN_JSONL") train / $(wc -l < "$VAL_JSONL") val rows"

# =============================================================================
# [5/6] Fine-tune (LoRA via HF Trainer; works on CPU and CUDA)
# =============================================================================
info "[5/6] LoRA fine-tuning (max_steps=$MAX_STEPS, CUDA=$("$PY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '?')) …"

# BF16 only when CUDA supports it
BF16_FLAG=""
if "$PY" -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()" 2>/dev/null; then
    BF16_FLAG="--bf16"
fi

if "$PY" -c "import qwen_tts.finetune.train_sft" 2>/dev/null; then
    info "Using official qwen_tts.finetune.train_sft …"
    "$PY" scripts/saudi_arabic/train_qwen3tts.py \
        --mode lora --train-jsonl "$TRAIN_JSONL" --val-jsonl "$VAL_JSONL" \
        --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR" \
        --max-steps "$MAX_STEPS" $BF16_FLAG
else
    info "Using finetune_trainer.py (HF Trainer fallback) …"
    "$PY" scripts/saudi_arabic/finetune_trainer.py \
        --train-jsonl "$TRAIN_JSONL" --val-jsonl "$VAL_JSONL" \
        --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR" \
        --mode lora --max-steps "$MAX_STEPS" \
        --batch-size 1 --grad-accum 4 --lr 2e-5 --logging-steps 5 $BF16_FLAG
fi

# Verify checkpoint exists
[[ -d "$OUTPUT_DIR" ]] || err "No checkpoint at $OUTPUT_DIR after training."
ok "[5/6] Training done → $OUTPUT_DIR"

# =============================================================================
# [6/6] Validate: loss curve + audio comparison
# =============================================================================
info "[6/6] Validation — loss curve + audio comparison …"

"$PY" scripts/saudi_arabic/validate_finetuned.py \
    --base-model      "$BASE_MODEL" \
    --finetuned-model "$OUTPUT_DIR" \
    --val-jsonl       "$DATA_PROCESSED/val_manifest.jsonl" \
    --output-dir      "$VALIDATE_DIR"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Pipeline complete!                                       ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo "║  Checkpoint  : $OUTPUT_DIR"
echo "║  Audio WAVs  : $VALIDATE_DIR/"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Listen: $VALIDATE_DIR/base_output.wav  vs  finetuned_output.wav"
echo "  Scale:  MAX_STEPS=200 bash scripts/saudi_arabic/run_pipeline_uv.sh"
