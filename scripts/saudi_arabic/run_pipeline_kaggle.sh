#!/usr/bin/env bash
# =============================================================================
# run_pipeline_kaggle.sh
#
# End-to-end Saudi Arabic Qwen3-TTS fine-tune pipeline for Kaggle.
#
# Prerequisites (run once first):
#   !bash scripts/intialize_kaggle.sh
#
# Usage (from a Kaggle notebook cell):
#   !bash scripts/saudi_arabic/run_pipeline_kaggle.sh
#
# Optional env vars:
#   SPEAKER     - SADA22 speaker id     (default: "Speaker1متحدث")
#   DIALECT     - SADA22 dialect        (default: "Najdi")
#   MAX_SAMPLES - rows to download      (default: 200)
#   MAX_STEPS   - LoRA training steps   (default: 50)
#   BASE_MODEL  - HF model id           (default: Qwen/Qwen3-TTS-0.6B-Base)
#   DATA_DIR    - data root             (default: data/sada22_small)
#   ROOT_DIR    - miniconda root        (default: .)
#   HF_TOKEN    - HuggingFace token     (or set via Kaggle Secrets → Add-ons)
#
# Disk budget (Kaggle = 20 GB):
#   ~2 GB  Qwen3-TTS-0.6B-Base weights
#   ~0.1GB 200 SADA22 audio samples
#   ~0.2GB LoRA checkpoint
#   ~3 GB  conda env + packages
#   ──────────────────────────────
#   ~5-6 GB total  ✓  safe on 20 GB
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SPEAKER="${SPEAKER:-Speaker1متحدث}"
DIALECT="${DIALECT:-Najdi}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_STEPS="${MAX_STEPS:-50}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-TTS-0.6B-Base}"
ROOT_DIR="${ROOT_DIR:-.}"

# Kaggle-input dataset path — populated when the dataset is attached in Kaggle UI
# (Add Data → search "sada2022" by sdaiancai → attach)
KAGGLE_INPUT_DIR="${KAGGLE_INPUT_DIR:-/kaggle/input/datasets/sdaiancai/sada2022}"

DATA_RAW="${DATA_DIR:-data/sada22_small}"
DATA_PROCESSED="${DATA_RAW}_qwen3tts"
OUTPUT_DIR="outputs/qwen3tts-sada22-lora"
VALIDATE_DIR="outputs/validate"

# Point HF cache into /kaggle/working so it counts against the 20GB quota
# but remains accessible (not /root/.cache which is ephemeral on some runtimes)
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/hf}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Navigate to repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Conda env helpers ─────────────────────────────────────────────────────────
CONDA_ROOT="$ROOT_DIR/miniconda3"
ENV_NAME="my_env"
PYTHON_BIN="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

# run_py: run a Python script inside the conda env
# Usage: run_py script.py --arg1 val1 ...
run_py() {
    (source "$CONDA_ROOT/bin/activate" "$ENV_NAME" && python "$@")
}

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}▶ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }
disk_check() {
    local avail_gb
    avail_gb=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
    if [[ "$avail_gb" -lt 3 ]]; then
        echo -e "${RED}WARNING: Only ${avail_gb}GB free — proceeding but may run out of space!${NC}" >&2
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Qwen3-TTS Saudi Arabic LoRA Pipeline  [Kaggle Edition] ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Speaker    : $SPEAKER"
echo "  Dialect    : $DIALECT"
echo "  Samples    : $MAX_SAMPLES"
echo "  Max steps  : $MAX_STEPS"
echo "  Base model : $BASE_MODEL"
echo "  Python     : $("$PYTHON_BIN" --version 2>&1)  [$PYTHON_BIN]"
echo "  CUDA       : $("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'torch not ready')"
echo "  HF cache   : $HF_HOME"
echo ""
disk_check

# Guard: conda env must exist
if [[ ! -x "$PYTHON_BIN" ]]; then
    err "Conda env '$ENV_NAME' not found at $PYTHON_BIN\nRun first:\n  !bash scripts/intialize_kaggle.sh"
fi

# =============================================================================
# [1/6] Verify packages + HuggingFace auth
# =============================================================================
info "[1/6] Verifying packages and HF auth …"

MISSING=""
for pkg in datasets accelerate peft soundfile transformers torch; do
    "$PYTHON_BIN" -c "import $pkg" 2>/dev/null || MISSING="$MISSING $pkg"
done
if [[ -n "$MISSING" ]]; then
    info "Installing missing packages:$MISSING"
    (source "$CONDA_ROOT/bin/activate" "$ENV_NAME" && pip install --quiet $MISSING)
fi
ok "All required packages present."

# Decide download source — prefer attached Kaggle dataset (faster, no HF needed)
if [[ -f "$KAGGLE_INPUT_DIR/train.csv" ]]; then
    DOWNLOAD_SOURCE="kaggle-input"
    ok "Kaggle input dataset found at $KAGGLE_INPUT_DIR — will use local CSV (no HF download needed)."
    HF_NEEDED_FOR_DATA=0
else
    DOWNLOAD_SOURCE="hf"
    HF_NEEDED_FOR_DATA=1
    info "No local dataset at $KAGGLE_INPUT_DIR — will stream from HuggingFace."
    echo "  (Attach your Sada22 dataset in Kaggle: Add Data → search Sada22 → attach)"
fi

# HF authentication — only strictly needed if streaming from HF.
# Model weights always come from HF so check auth regardless.
HF_AUTH_OK=$("$PYTHON_BIN" -c "
from huggingface_hub import HfFolder
import os
t = HfFolder.get_token() or os.environ.get('HF_TOKEN','')
print('yes' if t else 'no')
" 2>/dev/null)

if [[ "$HF_AUTH_OK" != "yes" ]]; then
    echo ""
    echo -e "${RED}✗ Not logged in to HuggingFace.${NC}"
    echo "  Model weights ($BASE_MODEL) are always downloaded from HuggingFace."
    echo "  In Kaggle, run this in a notebook cell BEFORE the pipeline:"
    echo ""
    echo "    from kaggle_secrets import UserSecretsClient"
    echo "    from huggingface_hub import login"
    echo "    login(token=UserSecretsClient().get_secret('HF_TOKEN'))"
    echo ""
    echo "  Add your token: Notebook → Add-ons → Secrets → HF_TOKEN"
    echo "  Accept model license: https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base"
    exit 1
fi
ok "HuggingFace authenticated."

# =============================================================================
# [2/6] Prepare SADA22 data (local Kaggle input or HF streaming)
# =============================================================================
if [[ "$DOWNLOAD_SOURCE" == "kaggle-input" ]]; then
    info "[2/6] Filtering SADA22 from local dataset (speaker='${SPEAKER}', dialect='${DIALECT}', max=${MAX_SAMPLES}) …"
    echo "  Source: $KAGGLE_INPUT_DIR/train.csv  [no internet needed]"
else
    info "[2/6] Streaming SADA22 from HuggingFace (${MAX_SAMPLES} rows, speaker='${SPEAKER}', dialect='${DIALECT}') …"
    echo "  Tip: streaming stops as soon as ${MAX_SAMPLES} matching rows are found."
fi
echo ""
disk_check

if [[ -f "$DATA_RAW/download_manifest.json" ]]; then
    EXISTING_ROWS=$("$PYTHON_BIN" -c "import json; d=json.load(open('$DATA_RAW/download_manifest.json')); print(d.get('rows',0))" 2>/dev/null || echo 0)
    if [[ "$EXISTING_ROWS" -ge "$MAX_SAMPLES" ]]; then
        ok "Data already prepared ($EXISTING_ROWS rows). Skipping."
    else
        info "Existing data has only $EXISTING_ROWS rows; re-preparing …"
        run_py scripts/saudi_arabic/download_sada22.py \
            --source       "$DOWNLOAD_SOURCE" \
            --kaggle-input-dir "$KAGGLE_INPUT_DIR" \
            --max-samples  "$MAX_SAMPLES" \
            --speaker      "$SPEAKER" \
            --dialect      "$DIALECT" \
            --output-dir   "$DATA_RAW"
    fi
else
    run_py scripts/saudi_arabic/download_sada22.py \
        --source       "$DOWNLOAD_SOURCE" \
        --kaggle-input-dir "$KAGGLE_INPUT_DIR" \
        --max-samples  "$MAX_SAMPLES" \
        --speaker      "$SPEAKER" \
        --dialect      "$DIALECT" \
        --output-dir   "$DATA_RAW"
fi

DOWNLOADED_ROWS=$("$PYTHON_BIN" -c "import json; d=json.load(open('$DATA_RAW/download_manifest.json')); print(d['rows'])")
ok "[2/6] Downloaded: $DOWNLOADED_ROWS rows → $DATA_RAW"
disk_check

# =============================================================================
# [3/6] Preprocess: resample to 24kHz, filter Arabic text, split train/val
# =============================================================================
info "[3/6] Preprocessing audio + building manifests …"

NEED_PRE=1
if [[ -f "$DATA_PROCESSED/train_manifest.jsonl" ]]; then
    TRAIN_LINES=$(wc -l < "$DATA_PROCESSED/train_manifest.jsonl")
    if [[ "$TRAIN_LINES" -gt 5 ]]; then
        ok "Preprocessed data exists ($TRAIN_LINES train rows). Skipping."
        NEED_PRE=0
    fi
fi

if [[ $NEED_PRE -eq 1 ]]; then
    run_py scripts/saudi_arabic/preprocess_sada22.py \
        --input-dir   "$DATA_RAW" \
        --output-dir  "$DATA_PROCESSED" \
        --text-key    "ProcessedText" \
        --speaker-key "Speaker" \
        --val-ratio   0.1 \
        --seed        42
    TRAIN_LINES=$(wc -l < "$DATA_PROCESSED/train_manifest.jsonl")
    ok "[3/6] Preprocessed: $TRAIN_LINES train rows → $DATA_PROCESSED"
fi

# =============================================================================
# [4/6] Build SFT JSONL
# =============================================================================
info "[4/6] Building SFT JSONL files …"

TRAIN_JSONL="$DATA_PROCESSED/train_qwen3tts_sft.jsonl"
VAL_JSONL="$DATA_PROCESSED/val_qwen3tts_sft.jsonl"

[[ -f "$TRAIN_JSONL" ]] || run_py scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
    --manifest "$DATA_PROCESSED/train_manifest.jsonl" --output "$TRAIN_JSONL"
[[ -f "$VAL_JSONL" ]]   || run_py scripts/saudi_arabic/build_qwen3tts_sft_jsonl.py \
    --manifest "$DATA_PROCESSED/val_manifest.jsonl"   --output "$VAL_JSONL"

ok "[4/6] SFT JSONL: $(wc -l < "$TRAIN_JSONL") train / $(wc -l < "$VAL_JSONL") val rows"
disk_check

# =============================================================================
# [5/6] Fine-tune (LoRA via HF Trainer)
# =============================================================================
info "[5/6] LoRA fine-tuning (max_steps=$MAX_STEPS, CUDA=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '?')) …"

# BF16 only when CUDA supports it
BF16_FLAG=""
if "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()" 2>/dev/null; then
    BF16_FLAG="--bf16"
fi

if "$PYTHON_BIN" -c "import qwen_tts.finetune.train_sft" 2>/dev/null; then
    info "Using official qwen_tts.finetune.train_sft …"
    run_py scripts/saudi_arabic/train_qwen3tts.py \
        --mode lora --train-jsonl "$TRAIN_JSONL" --val-jsonl "$VAL_JSONL" \
        --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR" \
        --max-steps "$MAX_STEPS" $BF16_FLAG
else
    info "Using finetune_trainer.py (HF Trainer fallback) …"
    run_py scripts/saudi_arabic/finetune_trainer.py \
        --train-jsonl "$TRAIN_JSONL" --val-jsonl "$VAL_JSONL" \
        --base-model  "$BASE_MODEL"  --output-dir "$OUTPUT_DIR" \
        --mode lora   --max-steps "$MAX_STEPS" \
        --batch-size 1 --grad-accum 4 --lr 2e-5 --logging-steps 5 $BF16_FLAG
fi

[[ -d "$OUTPUT_DIR" ]] || err "No checkpoint at $OUTPUT_DIR after training."
ok "[5/6] Training done → $OUTPUT_DIR"
disk_check

# =============================================================================
# [6/6] Validate: loss curve + audio comparison
# =============================================================================
info "[6/6] Validation — loss curve + audio comparison …"

run_py scripts/saudi_arabic/validate_finetuned.py \
    --base-model      "$BASE_MODEL" \
    --finetuned-model "$OUTPUT_DIR" \
    --val-jsonl       "$DATA_PROCESSED/val_manifest.jsonl" \
    --output-dir      "$VALIDATE_DIR"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Pipeline complete!  [Kaggle Edition]                    ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
echo "║  Checkpoint  : $OUTPUT_DIR"
echo "║  Audio WAVs  : $VALIDATE_DIR/"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Listen: $VALIDATE_DIR/base_output.wav  vs  finetuned_output.wav"
echo "  Scale:  MAX_STEPS=200 bash scripts/saudi_arabic/run_pipeline_kaggle.sh"
echo ""
# Show final disk usage
echo "  Disk usage:"
du -sh "$HF_HOME" "$DATA_RAW" "$DATA_PROCESSED" "$OUTPUT_DIR" "$VALIDATE_DIR" 2>/dev/null | sort -h || true
df -h "$REPO_ROOT" | awk 'NR==2 {print "  Total disk: used=" $3 " / " $2 " (" $5 " full)"}'
