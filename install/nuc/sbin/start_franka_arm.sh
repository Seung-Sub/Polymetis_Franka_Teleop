#!/bin/bash
# start_franka_arm.sh -- bring up the Polymetis arm gRPC server on port 50051.
#
# Wraps `launch_robot.py robot_client=franka_hardware ip=0.0.0.0 port=50051`
# so the user can just `sudo bash /usr/local/sbin/start_franka_arm.sh` and
# get:
#   * the polymetis-local conda env activated for the launched python
#   * franka_pin_helper scheduled 5 s after launch (so the RT threads get
#     pinned to cores 6,7 once the C++ children have spawned)
#   * pin output sent to a per-user log file, not the terminal
#
# Stop with Ctrl-C in the terminal that ran this. The pin task is detached
# and exits on its own; no orphan cleanup needed.
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
export HOME="$TARGET_HOME"
LOG_DIR="$TARGET_HOME/.franka_logs"
mkdir -p "$LOG_DIR"
PIN_LOG="$LOG_DIR/franka_pin_arm.log"

source "$TARGET_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate polymetis-local
cd "$TARGET_HOME/fairo/polymetis/polymetis/python/scripts"

(
    sleep 5
    {
        echo "=== $(date) ==="
        sudo -n /usr/local/sbin/franka_pin_helper.sh
    } > "$PIN_LOG" 2>&1
    echo "[arm pinner] cores 6,7 pin applied (details: tail $PIN_LOG)"
) &

exec python launch_robot.py robot_client=franka_hardware ip=0.0.0.0 port=50051
