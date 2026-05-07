#!/bin/bash
# start_franka_gripper.sh -- Polymetis franka_hand gRPC server on port 50052.
#
# Only needed for the Franka Hand workflow. Skip entirely when --gripper_backend art
# (KIST default) is used; the ART daemon runs on pro4000, not on the NUC.
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
export HOME="$TARGET_HOME"
LOG_DIR="$TARGET_HOME/.franka_logs"
mkdir -p "$LOG_DIR"
PIN_LOG="$LOG_DIR/franka_pin_gripper.log"

source "$TARGET_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate polymetis-local
cd "$TARGET_HOME/fairo/polymetis/polymetis/python/scripts"

(
    sleep 5
    {
        echo "=== $(date) ==="
        sudo -n /usr/local/sbin/franka_pin_helper.sh
    } > "$PIN_LOG" 2>&1
    echo "[gripper pinner] cores 6,7 pin applied (details: tail $PIN_LOG)"
) &

exec python launch_gripper.py gripper=franka_hand
