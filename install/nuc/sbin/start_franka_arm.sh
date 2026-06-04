#!/bin/bash
# This script must run as root: polymetis needs RT scheduling + mlockall + the
# franka_pin_helper.sh NOPASSWD sudoers entry. Auto-promote if invoked plainly
# (so ``bash start_franka_arm.sh`` and ``sudo bash start_franka_arm.sh`` both
# work). The interactive sudo prompt only fires when the user genuinely
# forgot it.
if [ "$EUID" -ne 0 ]; then
    exec sudo -H bash "$0" "$@"
fi

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
export HOME="$TARGET_HOME"
LOG_DIR="$TARGET_HOME/.franka_logs"
mkdir -p "$LOG_DIR"
chown "$TARGET_USER:$TARGET_USER" "$LOG_DIR" 2>/dev/null || true
PIN_LOG="$LOG_DIR/franka_pin_arm.log"

# Make sure hydra's outputs/ dir stays writable by the kist user — past root-
# launched runs left subdirs owned by root, which broke the next non-sudo
# attempt with PermissionError. Re-asserting ownership at every launch is
# cheap and idempotent.
SCRIPTS_DIR="$TARGET_HOME/fairo/polymetis/polymetis/python/scripts"
if [ -d "$SCRIPTS_DIR/outputs" ]; then
    chown -R "$TARGET_USER:$TARGET_USER" "$SCRIPTS_DIR/outputs" 2>/dev/null || true
fi

source "$TARGET_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate polymetis-local
cd "$TARGET_HOME/fairo/polymetis/polymetis/python/scripts"

# ---- Preflight: refuse to start if :50051 is already taken -----------------
# (2026-06-04) The shared NUC's Franka FCI is single-holder. If a polymetis
# server (yours, another user's, or the unified-interface stack) already owns
# :50051, launch_robot.py dies mid-startup with a cryptic
# "AssertionError: Port unavailable; possibly another server found...".
# Detect it up front and say exactly what to do instead of failing obscurely.
if ss -tlnp 2>/dev/null | grep -q ':50051 '; then
    PANDA=$(pgrep -f 'fairo/polymetis/polymetis/build/franka_panda_client' 2>/dev/null | tr '\n' ' ')
    if [ -n "$(echo "$PANDA" | tr -d ' ')" ]; then
        echo "[start_franka_arm] :50051 already bound and a franka_panda_client is LIVE"
        echo "    (PIDs: $PANDA) -- the arm is already running / in use."
        echo "    * Another user's session? Coordinate; do NOT start a second."
        echo "    * Your own stuck stack?  sudo /usr/local/sbin/cleanup_polymetis.sh --force"
        exit 1
    fi
    echo "[start_franka_arm] :50051 is bound but NO franka_panda_client is connected"
    echo "    -- a zombie socket from a prior Ctrl+C. Clean it first:"
    echo "        sudo /usr/local/sbin/cleanup_polymetis.sh"
    exit 1
fi
echo "[start_franka_arm] preflight OK -- :50051 free, starting polymetis arm server."

# pin은 백그라운드, 출력은 로그로. 완료 시 짧은 메시지만 터미널에 표시.
(
    sleep 5
    {
        echo "=== $(date) ==="
        sudo -n /usr/local/sbin/franka_pin_helper.sh
    } > "$PIN_LOG" 2>&1
    echo "[arm pinner] cores 6,7 핀 적용 완료 (상세: tail $PIN_LOG)"
) &

exec python launch_robot.py robot_client=franka_hardware ip=0.0.0.0 port=50051
