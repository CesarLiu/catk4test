#!/usr/bin/env bash
# Sets up a Python 3.11 environment for CAT-K / SMART and the counterfactual-
# responsibility code on a Linux GPU server (tested target: Ubuntu 24.04, H200).
#
# Ubuntu 24.04 ships Python 3.12, but tensorflow 2.12 (needed to read WOMD and
# by the Waymo tools) has no 3.12 wheels, so the script brings its own 3.11:
#   uv (default)   installs a standalone 3.11 under ~/.local/share/uv, no root
#   conda          uses an existing conda/miniforge installation
#
# Usage:
#   bash install/setup_server.sh                 # venv at ~/venvs/catk via uv
#   ENV_DIR=/scratch/venvs/catk bash install/setup_server.sh
#   BACKEND=conda ENV_NAME=catk bash install/setup_server.sh
#   CUDA_TAG=cu124 bash install/setup_server.sh  # torch wheel flavour (cu121 default)
#
# Afterwards: source ~/venvs/catk/bin/activate  (or conda activate catk) and
# run  bash install/check_env.sh  from the repository root.
set -euo pipefail

BACKEND=${BACKEND:-uv}
ENV_DIR=${ENV_DIR:-$HOME/venvs/catk}
ENV_NAME=${ENV_NAME:-catk}
CUDA_TAG=${CUDA_TAG:-cu121}
TORCH_VERSION=2.4.1
TORCHVISION_VERSION=0.19.1
PYG_VERSION=2.6.1  # 2.8 made BaseTransform.forward abstract, which breaks the SMART transforms
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

log() { printf '\n=== %s\n' "$*"; }

if ! command -v nvidia-smi >/dev/null; then
  echo "WARNING: nvidia-smi not found; the environment will install but cannot use a GPU." >&2
else
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

if command -v apt-get >/dev/null && ! command -v ffmpeg >/dev/null; then
  # optional: ffmpeg is only used to re-encode demo videos; skip without root
  if sudo -n true 2>/dev/null; then
    log "installing ffmpeg"
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
  else
    echo "note: no passwordless sudo, skipping ffmpeg (demos still write .gif and .mp4 via OpenCV)"
  fi
fi

case "$BACKEND" in
  uv)
    if ! command -v uv >/dev/null; then
      log "installing uv"
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    log "creating venv $ENV_DIR (Python 3.11)"
    uv python install 3.11
    uv venv --python 3.11 --seed "$ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    PIP="uv pip"
    ;;
  conda)
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    log "creating conda env $ENV_NAME (Python 3.11)"
    conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
    python -m pip install --upgrade pip wheel
    PIP="python -m pip"
    ;;
  *)
    echo "unknown BACKEND=$BACKEND (use uv or conda)" >&2
    exit 1
    ;;
esac

log "torch $TORCH_VERSION ($CUDA_TAG)"
$PIP install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

log "requirements"
# torch==2.4.1 in there is already satisfied by the CUDA build above
$PIP install -r "$REPO_ROOT/install/requirements.txt"

log "torch_geometric $PYG_VERSION + torch_scatter / torch_cluster"
$PIP install "torch_geometric==$PYG_VERSION"
$PIP install torch_scatter torch_cluster -f "https://data.pyg.org/whl/torch-2.4.0+$CUDA_TAG.html"

log "waymo-open-dataset (no deps: its pins conflict with the ones above)"
$PIP install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4

log "test tooling"
$PIP install pytest scipy

log "done"
if [ "$BACKEND" = uv ]; then
  echo "activate with:  source $ENV_DIR/bin/activate"
else
  echo "activate with:  conda activate $ENV_NAME"
fi
echo "then check with: bash install/check_env.sh"
