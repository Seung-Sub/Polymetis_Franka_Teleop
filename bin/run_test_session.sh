#!/usr/bin/env bash
# One-shot test-session launcher.
#
# Why setsid: this demo is sensitive to job-control signals because the main
# Python loop drives the cv2.imshow window. If the controlling shell
# accidentally sends SIGTSTP (Ctrl-Z) the whole foreground process group is
# stopped — cv2 window never appears, trackpad-HOME polling halts, but the
# child Vive/controller processes keep running, leaving you with a half-
# working teleop. setsid puts the demo in its own session/process group,
# detached from the terminal's job-control signals.
#
# You still see all output via `tail -f` of the log; press Ctrl-C on the tail
# to detach yourself, the demo keeps running. To quit cleanly: press 'q' in
# the cv2 window. To force-stop from another shell:
#   pkill -INT -f demo_franka_vive
#
# Usage:
#   bash ~/Polymetis_Franka_Teleop/bin/run_test_session.sh [extra start_teleop args...]

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export POLYMETIS_SUDO_PASSWORD="${POLYMETIS_SUDO_PASSWORD:- }"
export PYTHONUNBUFFERED=1

TS="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${ROOT}/data/test_groot_${TS}"
LOG="/tmp/teleop_session.log"
PIDFILE="/tmp/teleop_session.pid"

echo "=========================================="
echo "[run_test_session] data : $DATA_DIR"
echo "[run_test_session] log  : $LOG"
echo "[run_test_session] PID will be written to $PIDFILE"
echo "[run_test_session] To quit: 'q' in cv2 window OR 'pkill -INT -f demo_franka_vive'"
echo "=========================================="

# truncate previous log
: > "$LOG"

# setsid + redirect — demo runs in its own session, immune to terminal SIGTSTP/SIGINT.
# stdout/stderr → log file (single redirect, no tee pipe to break).
setsid bash bin/start_teleop.sh "$DATA_DIR" "$@" >>"$LOG" 2>&1 < /dev/null &
DEMO_PID=$!
echo "$DEMO_PID" > "$PIDFILE"
echo "[run_test_session] demo started (PID $DEMO_PID)"
echo

# tail -f gives the user live output; user can Ctrl-C this safely without
# affecting the demo process. We use --pid=$DEMO_PID so tail exits as soon
# as the demo dies (e.g. user pressed q in the cv2 window) — otherwise the
# launcher would hang showing a stale log forever and force the user to
# Ctrl-C just to recover their shell.
echo "[run_test_session] tailing log (auto-exit when demo dies):"
echo "------------------------------------------"
tail -n +1 --pid="$DEMO_PID" -f "$LOG"
TAIL_RC=$?

# Demo may still be in cleanup at this point (multiprocessing teardown). Wait
# briefly, then surface any orphans so the user is never left with a half-
# alive session.
sleep 2
LEFT=$(pgrep -fc -- "demo_franka_vive|spawn_main|FrankaPositional|ViveTele|SingleZed|MultiZed|ArtGripperController|cv2_viewer" || true)
if [[ "$LEFT" -gt 0 ]]; then
    echo
    echo "[run_test_session] WARNING: $LEFT demo subprocess(es) still alive after demo PID exit."
    echo "[run_test_session] Run 'pkill -KILL -f demo_franka_vive' to force-clean if they linger."
fi
echo "[run_test_session] session ended."
exit "$TAIL_RC"
