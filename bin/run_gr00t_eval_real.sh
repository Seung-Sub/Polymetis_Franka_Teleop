#!/usr/bin/env bash
# Real-robot eval for fine-tuned GR00T-N1.7-DROID on KIST hardware.
#
# Architecture:
#   - GR00T inference server runs on kist_a6000_ss (where the checkpoint is)
#     on port 5555 (ZMQ).  Server-side handles RELATIVE->ABSOLUTE conversion.
#   - This script runs on pro4000 and talks to the remote server via
#     ssh -L 5555:127.0.0.1:5555 kist_a6000_ss   (you set this up separately).
#   - examples/DROID/main_gr00t.py with --env-mode kist_minimal uses our
#     franka_env_kist.py to drive the Franka arm (NUC polymetis :50051)
#     + ART gripper (:50053) + ZED 2i (exterior, 33538770) + ZED Mini
#     (wrist, 11667817).
#
# Pre-requisites on pro4000:
#   1. cleanup_pipeline.sh OK
#   2. Franka Desk unlocked + FCI Activate
#   3. NUC arm started manually (sudo bash /usr/local/sbin/start_franka_arm.sh)
#   4. preflight_full.sh PASS  (Vive stack not required for eval but harmless)
#   5. SSH tunnel from this machine to a6000_ss policy server:
#        ssh -N -L 5555:127.0.0.1:5555 kist_a6000_ss -p 2240 &
#
# Usage:
#   bash bin/run_gr00t_eval_real.sh [POLICY_HOST]
#
# Default POLICY_HOST=127.0.0.1 (i.e. you set up SSH tunnel).
set -euo pipefail

POLICY_HOST="${1:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"

ROOT="${HOME}/Isaac-GR00T"  # repo path on pro4000
cd "$ROOT"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate groot-client

# ART gripper -- our convention; main_gr00t.py + franka_env_kist.py handle the
# rest correctly because gr00t-droid training data is 1=close, 0=open and
# franka_env_kist already does `1 - width/max_width` for obs and
# `max * (1-cmd)` for action.
export KIST_GRIPPER=art_gripper
export ART_GRIPPER_PYPATH="${HOME}/Hyundai_motors_Gripper/python"

# Camera serials -- ZED 2i = exterior (33538770, larger baseline),
# ZED Mini = wrist (11667817, smaller baseline).  --right-camera-id is
# unused because our training is single-exterior; main_gr00t.py only
# needs --external-camera left.
LEFT_CAM=33538770
WRIST_CAM=11667817

# Results dir
RESULTS_DIR="${HOME}/Polymetis_Franka_Teleop/data/eval_$(date +%Y%m%d_%H%M%S)_gr00t_real"
mkdir -p "$RESULTS_DIR"

echo "==========================================="
echo "  Policy server : $POLICY_HOST:$POLICY_PORT"
echo "  External cam  : $LEFT_CAM (ZED 2i)"
echo "  Wrist cam     : $WRIST_CAM (ZED Mini)"
echo "  Results       : $RESULTS_DIR"
echo "==========================================="

exec python examples/DROID/main_gr00t.py \
    --env-mode kist_minimal \
    --polymetis-ip 192.168.1.12 \
    --polymetis-arm-port 50051 \
    --polymetis-gripper-port 50052 \
    --left-camera-id "$LEFT_CAM" \
    --right-camera-id "$LEFT_CAM" \
    --wrist-camera-id "$WRIST_CAM" \
    --external-camera left \
    --render-camera left \
    --policy-host "$POLICY_HOST" \
    --policy-port "$POLICY_PORT" \
    --results-dir "$RESULTS_DIR" \
    --max-timesteps 600 \
    --open-loop-horizon 15 \
    --delay-seconds 5
