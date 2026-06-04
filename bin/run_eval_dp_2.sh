#!/usr/bin/env bash
# Real-robot DP eval — minimal version.
# No `set` options. Every step echoes its result so failures are visible.

trap 'echo "[wrapper] got SIGINT, exiting" >&2; exit 130' INT
trap 'echo "[wrapper] got SIGTERM, exiting" >&2; exit 143' TERM
# Don't die on ssh disconnect or broken pipe — eval keeps running until done.
trap 'echo "[wrapper] got SIGHUP (ssh disconnected?), continuing" >&2' HUP
trap 'echo "[wrapper] got SIGPIPE, continuing" >&2' PIPE

echo "[wrapper] === START ==="
echo "[wrapper] PID=$$  args=$*"

CKPT="$1"
if [ -z "$CKPT" ]; then echo "Usage: $0 <ckpt> [eval args...]"; exit 1; fi
shift
EXTRA=("$@")
DAEMON_PORT="${DAEMON_PORT:-5555}"

cd "$(dirname "$0")/.." || exit 2
ROOT="$(pwd)"
echo "[wrapper] cwd=$ROOT  ckpt=$CKPT  daemon_port=$DAEMON_PORT"

if [ ! -f "$CKPT" ]; then echo "[wrapper] ckpt missing"; exit 3; fi

UMI_PY=/home/kist/anaconda3/envs/umi/bin/python
GROOT_PY=/home/kist/anaconda3/envs/groot-client/bin/python
[ -x "$UMI_PY" ] || { echo "[wrapper] umi env python missing"; exit 4; }
[ -x "$GROOT_PY" ] || { echo "[wrapper] groot-client env python missing"; exit 5; }

export POLYMETIS_SUDO_PASSWORD=" "
export DIFFUSION_POLICY_PATH="${DIFFUSION_POLICY_PATH:-${HOME}/diffusion_policy}"
export ART_GRIPPER_PYPATH="${ART_GRIPPER_PYPATH:-${HOME}/Hyundai_motors_Gripper/python}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# --- Stale process cleanup ---
echo "[wrapper] --- step 1: kill stale eval processes ---"
fuser -k -n tcp "$DAEMON_PORT" 2>/dev/null
pkill -9 -f "eval_dp_inference_daemon" 2>/dev/null
pkill -9 -f "eval_dp_real" 2>/dev/null
pkill -9 -f "cv2_viewer.py" 2>/dev/null
echo "[wrapper] step 1 done"

echo "[wrapper] step 2 skipped (cleanup_pipeline must be run manually before this wrapper)"
echo "[wrapper] step 3 skipped (ART daemon is already systemd-managed; run restart_gripper.sh manually if needed)"

# --- daemon launch (umi env via explicit python path; no conda activate needed) ---
echo "[wrapper] --- step 4: inference daemon (umi env, GPU) ---"
DAEMON_LOG=/tmp/dp_eval_daemon.log
DAEMON_PYPATH="/home/kist/fairo-polymetis/polymetis/python:${DIFFUSION_POLICY_PATH}:${ROOT}"
PYTHONPATH="$DAEMON_PYPATH" CONDA_PREFIX="/home/kist/anaconda3/envs/umi" PYTHONUNBUFFERED=1 \
    nohup "$UMI_PY" -u scripts_real/eval_dp_inference_daemon.py \
        --ckpt "$CKPT" --port "$DAEMON_PORT" --device cuda:0 --collect_calib \
        > "$DAEMON_LOG" 2>&1 &
disown
DAEMON_PID=$!
echo "[wrapper] step 4 launched (PID=$DAEMON_PID log=$DAEMON_LOG)"

trap "echo '[wrapper] EXIT trap: kill daemon $DAEMON_PID'; kill -TERM $DAEMON_PID 2>/dev/null; sleep 0.5; kill -KILL $DAEMON_PID 2>/dev/null" EXIT

# --- wait for daemon to bind ---
echo "[wrapper] --- step 5: wait for daemon socket bind ---"
READY=0
for i in $(seq 1 60); do
    if grep -q "listening on tcp://127.0.0.1:${DAEMON_PORT}" "$DAEMON_LOG" 2>/dev/null; then
        READY=1
        echo "[wrapper] step 5 done (daemon ready after ${i}s)"
        break
    fi
    sleep 1
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "[wrapper] daemon died early. log:"
        tail -20 "$DAEMON_LOG"
        exit 6
    fi
done
if [ $READY -eq 0 ]; then
    echo "[wrapper] daemon did not bind within 60s. log:"
    tail -20 "$DAEMON_LOG"
    exit 7
fi

# --- main eval (groot-client env via FULL conda activate, same as start_eval.sh / start_teleop_groot_droid_ft.sh)
# Polymetis needs CONDA_PREFIX *and* LD_LIBRARY_PATH set together so libtorchscript_pinocchio.so can find libpinocchio_wrapper.so.
echo "[wrapper] --- step 6: main eval (groot-client env) ---"
cat <<EOF
======================================================================
  ckpt    : $CKPT
  daemon  : PID $DAEMON_PID  port $DAEMON_PORT
  cameras : 35766817 ext + 11667817 wrist
  robot   : 192.168.1.14
======================================================================
EOF

# shellcheck disable=SC1091
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate groot-client
echo "[wrapper] conda env active: ${CONDA_DEFAULT_ENV:-unknown}  prefix=${CONDA_PREFIX:-unknown}"
echo "[wrapper] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

export PYTHONUNBUFFERED=1

# (2026-05-21) Isolate eval main from the controlling terminal.
# 2026-05-21 00:48 + 01:10 runs both died ~1 minute into the first inference,
# silently (no finally/atexit/signal print). Root cause: USB 3-3.4 (Logitech
# Wireless Unifying dongle) momentarily disconnected -> X server keyboard/mouse
# remove -> NVIDIA Xorg session reconfigure -> gnome-terminal X connection
# broken -> SIGHUP/SIGPIPE cascade to bash wrapper + tee + python eval.
# Because every fd in the eval main was inherited from the terminal, BrokenPipe
# silently killed every print() including signal handler stack dumps.
#
# Fix: setsid + redirect stdout/stderr to a stable file (not the pipe), and
# disown so a terminal SIGHUP cannot reach eval. The wrapper's caller can
# still `tee` if it wants the live stream -- our log file is the source of
# truth.
EVAL_LOG="${EVAL_LOG:-/tmp/eval_dp_main_$(date +%Y%m%d_%H%M%S).log}"
echo "[wrapper] launching eval (setsid, log=$EVAL_LOG)"
# CRITICAL: use `setsid` WITHOUT `-f`. `setsid -f` (--fork) makes setsid
# fork() and the parent (setsid itself) exits IMMEDIATELY -- bash's $!
# captures setsid's PID which is already dead, so `wait $EVAL_PID`
# returns instantly. The wrapper then hits its EXIT trap and SIGTERMs
# the DP daemon while the real eval child is still alive, causing the
# eval's first client.predict() to get zmq EAGAIN ten seconds later.
#
# Without -f, `setsid` execs the command IN PLACE -- $! is the actual
# python PID, and `wait` blocks correctly until eval finishes.
# (Verified failure pattern: 2026-05-21 01:30 wrapper exited in <1s and
# trap'd daemon kill before eval ping->predict cycle could complete.)
setsid python -u scripts_real/collect_calib_data.py \
    --output_dir "/home/kist/Polymetis_Franka_Teleop/calib_data" \
    --daemon_port "$DAEMON_PORT" \
    --robot_ip 192.168.1.14 \
    --auto_start_after 10 \
    "${EXTRA[@]}" \
    >"$EVAL_LOG" 2>&1 < /dev/null &
EVAL_PID=$!
echo "[wrapper] eval PID=$EVAL_PID (detached). Live tail: tail -f $EVAL_LOG"

# Stream eval log to wrapper stdout in parallel; tail dies naturally when
# eval PID exits.
tail --pid="$EVAL_PID" -F "$EVAL_LOG" 2>/dev/null &
TAIL_PID=$!
wait "$EVAL_PID" 2>/dev/null
RC=$?
kill "$TAIL_PID" 2>/dev/null || true
conda deactivate
echo "[wrapper] step 6 done (main eval rc=$RC, full log: $EVAL_LOG)"
echo "[wrapper] === END ==="
exit $RC
