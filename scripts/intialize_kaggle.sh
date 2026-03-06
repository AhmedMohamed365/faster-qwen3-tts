#!/usr/bin/env bash
# =============================================================================
# intialize_kaggle.sh
#
# Run once at the start of a Kaggle session to set up the environment.
# Installs Miniconda, creates a conda env, and installs all training deps.
#
# Usage (from a Kaggle notebook cell):
#   !bash scripts/intialize_kaggle.sh
#
# Subsequent commands must source the env:
#   !source ./miniconda3/bin/activate my_env && python ...
# Or use run_pipeline_kaggle.sh which handles this automatically.
# =============================================================================
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-.}"
ENV_NAME="my_env"
CONDA="$ROOT_DIR/miniconda3/bin/conda"
PIP="$ROOT_DIR/miniconda3/envs/$ENV_NAME/bin/pip"
PYTHON="$ROOT_DIR/miniconda3/envs/$ENV_NAME/bin/python"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}▶ $*${NC}"; }

mkdir -p "$ROOT_DIR"

# ── 1. Install Miniconda (skip if already present) ────────────────────────────
if [[ ! -f "$ROOT_DIR/miniconda3/bin/conda" ]]; then
    info "Downloading Miniconda …"
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$ROOT_DIR/miniconda3" -f
    rm /tmp/miniconda.sh
    ok "Miniconda installed at $ROOT_DIR/miniconda3"
else
    ok "Miniconda already present — skipping install"
fi

# ── 2. Accept conda ToS (required for newer conda versions) ───────────────────
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true

# ── 3. Create env (skip if exists) ───────────────────────────────────────────
if [[ ! -d "$ROOT_DIR/miniconda3/envs/$ENV_NAME" ]]; then
    info "Creating conda env '$ENV_NAME' (Python 3.11) …"
    "$CONDA" create --name "$ENV_NAME" python=3.11 -y
    ok "Env '$ENV_NAME' created"
else
    ok "Conda env '$ENV_NAME' already exists — skipping"
fi

# ── 4. Install PyTorch (CUDA 12.x) ───────────────────────────────────────────
info "Installing system deps (sox, ffmpeg) …"
apt-get install -y -q sox libsox-fmt-all ffmpeg 2>/dev/null || true
ok "System deps ready"

info "Installing PyTorch (cu121) …"
"$PIP" install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
ok "PyTorch installed: $("$PYTHON" -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# ── 5. Install training dependencies ─────────────────────────────────────────
info "Installing training deps (transformers, peft, datasets, accelerate …) …"
"$PIP" install --quiet \
    "transformers>=4.45" \
    "peft>=0.10" \
    "datasets>=2.14" \
    "accelerate>=0.26" \
    "soundfile" \
    "torchaudio" \
    "tqdm" \
    "huggingface_hub"
# torchcodec: needed by newer datasets for Audio decoding — non-fatal if it fails
"$PIP" install --quiet torchcodec 2>/dev/null || true
ok "Training deps installed"

# ── 6. Install the faster-qwen3-tts package ───────────────────────────────────
info "Installing faster-qwen3-tts (editable) …"
"$PIP" install --quiet -e "$ROOT_DIR"
ok "faster-qwen3-tts installed"

# ── 7. Install qwen-tts ───────────────────────────────────────────────────────
info "Installing qwen-tts …"
"$PIP" install --quiet qwen-tts
ok "qwen-tts installed"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Kaggle environment ready!${NC}"
echo -e "${GREEN}  Python: $("$PYTHON" --version)${NC}"
echo -e "${GREEN}  Run the pipeline with:${NC}"
echo "    !bash scripts/saudi_arabic/run_pipeline_kaggle.sh"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"

# !source $root_dir/miniconda3/bin/activate my_env; uv add package_name
# !source $root_dir/miniconda3/bin/activate my_env; pip install torch
# !source $root_dir/miniconda3/bin/activate my_env; python test-benchmarked.py