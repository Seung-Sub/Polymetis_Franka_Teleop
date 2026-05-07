#!/usr/bin/env bash
# install_pro4000.sh -- bootstrap pro4000 (the inference / data-collection PC).
#
# Prerequisites you must have done by hand before running this:
#   * Ubuntu 22.04
#   * NVIDIA driver + CUDA (matching your GPU; we run RTX PRO 4000 Blackwell SFF
#     with driver 580+ / CUDA 13)
#   * miniconda or anaconda installed at $HOME/anaconda3 (or $HOME/miniconda3)
#   * Hyundai_motors_Gripper repo cloned + ART daemon installed (see
#     https://github.com/Seung-Sub/Hyundai_motors_Gripper -- one-shot
#     `sudo bash scripts/install_etherlab.sh && sudo bash scripts/install_daemon.sh --system`)
#   * ZED SDK 5.3 installed at /usr/local/zed (see docs/install_from_scratch.md
#     Phase F -- can't be silently scripted; vendor `.run` requires CUDA detection)
#   * Diffusion Policy repo cloned to $HOME/diffusion_policy (only used for
#     ReplayBuffer + checkpoint loading; not for training)
#
# What this script does:
#   1. Creates / updates the `groot-client` conda env (Python 3.8) with all
#      pro4000-side runtime deps for this repo.
#   2. `pip install -e .` of this repo + the Hyundai gripper Python client.
#   3. Drops a pre-flight env summary so you know what's missing.
#
# The conda env name `groot-client` matches the one used by Isaac-GR00T;
# both repos share it intentionally.
set -euo pipefail

ENV_NAME="${ENV_NAME:-groot-client}"
PY_VER="${PY_VER:-3.8}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
HYUNDAI_REPO="${HYUNDAI_REPO:-${HOME}/Hyundai_motors_Gripper}"
DIFFUSION_POLICY="${DIFFUSION_POLICY:-${HOME}/diffusion_policy}"

if [[ ! -d "$CONDA_BASE" ]]; then
    if [[ -d "${HOME}/miniconda3" ]]; then CONDA_BASE="${HOME}/miniconda3"
    else echo "[install_pro4000] no conda at $CONDA_BASE or ~/miniconda3" >&2; exit 1; fi
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[install_pro4000] conda env '$ENV_NAME' exists -- reusing"
else
    echo "[install_pro4000] creating conda env '$ENV_NAME' (python $PY_VER)..."
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"

# Polymetis client + DROID-style pose libraries -- `groot-client` already has
# these for Isaac-GR00T users; running pip again is a no-op.
echo "[install_pro4000] installing/updating Python deps..."
pip install --upgrade pip
pip install \
    'torch==1.13.1' 'torchvision==0.14.1' \
    grpcio==1.46.0 'protobuf<3.20.2' \
    hydra-core==1.1.1 omegaconf scipy 'numpy<2' \
    pyzmq msgpack pandas tyro moviepy==1.0.3 \
    pillow imageio opencv-python av zarr'<3' atomics pyarrow dill \
    pynput tqdm zerorpc threadpoolctl click

# This repo
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[install_pro4000] pip install -e $ROOT"
pip install -e "$ROOT"

# Hyundai gripper python client (sister repo)
if [[ -d "$HYUNDAI_REPO/python" ]]; then
    echo "[install_pro4000] pip install -e $HYUNDAI_REPO/python"
    pip install -e "$HYUNDAI_REPO/python"
else
    echo "[install_pro4000] WARN: $HYUNDAI_REPO/python not found -- skipping ART client install"
    echo "[install_pro4000]       (clone Seung-Sub/Hyundai_motors_Gripper if you need ART gripper)"
fi

# diffusion_policy reference (importable via PYTHONPATH; not pip installed
# because its pyproject pins are noisy)
if [[ -d "$DIFFUSION_POLICY" ]]; then
    echo "[install_pro4000] diffusion_policy detected at $DIFFUSION_POLICY -- importable via DIFFUSION_POLICY_PATH=$DIFFUSION_POLICY"
else
    echo "[install_pro4000] WARN: $DIFFUSION_POLICY not found -- needed for diffusion-policy eval (not for teleop)"
fi

echo
echo "[install_pro4000] done. Now run the env check:"
echo "  bash install/check_environment.sh"
