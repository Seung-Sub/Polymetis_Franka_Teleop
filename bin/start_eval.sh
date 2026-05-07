#!/usr/bin/env bash
# Wrapper: run a trained diffusion-policy checkpoint with KIST stack.
set -euo pipefail

CKPT="${1:?usage: start_eval.sh <checkpoint.ckpt> <output_dir> [extra args...]}"
OUTPUT="${2:?usage: start_eval.sh <checkpoint.ckpt> <output_dir> [extra args...]}"
shift 2 || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate groot-client

export DIFFUSION_POLICY_PATH="${DIFFUSION_POLICY_PATH:-${HOME}/diffusion_policy}"
export ART_GRIPPER_PYPATH="${ART_GRIPPER_PYPATH:-${HOME}/Hyundai_motors_Gripper/python}"

exec python scripts_real/eval_franka_policy.py \
    --input "$CKPT" --output "$OUTPUT" \
    --robot_ip 192.168.1.12 \
    --camera_backend zed --gripper_backend art \
    --camera_serials 33538770 --camera_serials 11667817 \
    --camera_resolution 1280x720 --camera_fps 60 \
    -v "$@"
