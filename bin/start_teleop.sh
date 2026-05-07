#!/usr/bin/env bash
# Wrapper: launch demo_franka_vive.py on pro4000 with the KIST default stack
#   ART gripper + ZED 2i exterior + ZED Mini wrist + Vive teleop.
# Override anything via CLI flags after $@.
set -euo pipefail

OUTPUT="${1:?usage: start_teleop.sh <output_dir> [extra args...]}"
shift || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate groot-client

export DIFFUSION_POLICY_PATH="${DIFFUSION_POLICY_PATH:-${HOME}/diffusion_policy}"
export ART_GRIPPER_PYPATH="${ART_GRIPPER_PYPATH:-${HOME}/Hyundai_motors_Gripper/python}"

exec python scripts_real/demo_franka_vive.py \
    --output "$OUTPUT" \
    --robot_ip 192.168.1.12 \
    --camera_backend zed --gripper_backend art \
    --camera_serials 33538770 --camera_serials 11667817 \
    --camera_resolution 1280x720 --camera_fps 60 \
    --frequency 10 --teleop_frequency 100 \
    -v "$@"
