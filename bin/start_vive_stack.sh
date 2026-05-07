#!/usr/bin/env bash
# Bring up the **headless** Vive stack on pro4000.
# Modeled on the user's existing ~/Isaac-GR00T/vive_aliases.sh — uses
# `vrserver --keepalive` directly (no Steam GUI / Qt dependency), so it
# works fine over SSH.
#
# Order:
#   1. /dev/hidraw* permissions  (sudo chmod +rw)
#   2. SteamVR vrserver --keepalive  (background)
#   3. vive_input binary  (background, TCP :12345 + UDP :12346)
#   4. verify port 12345 listening
#
# Usage:
#   bash bin/start_vive_stack.sh           # full stack
#   bash bin/start_vive_stack.sh --stop    # tear everything down
#   bash bin/start_vive_stack.sh --status  # show current state
#
# Override binary locations via env vars (defaults track KIST setup):
#   STEAMVR=$HOME/.steam/debian-installation/steamapps/common/SteamVR
#   VIVE_INPUT_BIN=$HOME/Isaac-GR00T/vive_input/build/vive_input
#   VIVE_INPUT_FREQ=50

set -uo pipefail

STEAMVR="${STEAMVR:-${HOME}/.steam/debian-installation/steamapps/common/SteamVR}"
VIVE_INPUT_BIN_DEFAULT="${HOME}/Isaac-GR00T/vive_input/build/vive_input"
VIVE_INPUT_BIN_FALLBACK="${HOME}/vive_ws/build/vive_ros2/vive_input"
VIVE_INPUT_BIN="${VIVE_INPUT_BIN:-${VIVE_INPUT_BIN_DEFAULT}}"
[[ -x "$VIVE_INPUT_BIN" ]] || VIVE_INPUT_BIN="$VIVE_INPUT_BIN_FALLBACK"
VIVE_INPUT_FREQ="${VIVE_INPUT_FREQ:-50}"

LOG_DIR="/tmp"
LOG_VRSERVER="$LOG_DIR/vrserver.log"
LOG_VIVE_INPUT="$LOG_DIR/vive_input.log"

cmd_status() {
    echo "==== Vive stack status ===="
    if pgrep -x vrserver >/dev/null 2>&1; then
        echo "  vrserver       : RUNNING (pid $(pgrep -x vrserver))"
    else
        echo "  vrserver       : not running"
    fi
    if pgrep -f "$VIVE_INPUT_BIN" >/dev/null 2>&1; then
        echo "  vive_input     : RUNNING ($VIVE_INPUT_BIN)"
    else
        echo "  vive_input     : not running"
    fi
    if ss -tln 2>/dev/null | grep -q ':12345 '; then
        echo "  port 12345 TCP : LISTENING"
    else
        echo "  port 12345 TCP : not listening"
    fi
    if ss -uln 2>/dev/null | grep -q ':12346 '; then
        echo "  port 12346 UDP : bound"
    fi
    local htc=$(lsusb 2>/dev/null | grep -ci 'htc\|valve')
    echo "  HTC/Valve USB  : $htc device(s)"
}

cmd_stop() {
    echo "==== stopping Vive stack ===="
    pkill -f "$VIVE_INPUT_BIN" 2>/dev/null || true
    pkill -x vrserver 2>/dev/null || true
    pkill -x vrcompositor 2>/dev/null || true
    pkill -x vrwebhelper 2>/dev/null || true
    sleep 1
    cmd_status
}

cmd_start() {
    echo "==== starting Vive stack ===="

    # 1. HID permissions  (vive_aliases.sh:setup_vive step 1)
    echo "[1/4] /dev/hidraw* permissions ..."
    if ls /dev/hidraw* >/dev/null 2>&1; then
        echo ' ' | sudo -S -p '' chmod +rw /dev/hidraw* 2>/dev/null \
            && echo "      OK" \
            || echo "      WARN: chmod failed (may already be world-rw)"
    else
        echo "      WARN: no /dev/hidraw* devices — Vive Link Box not connected?"
    fi

    # 2. vrserver --keepalive (matches run_steamvr in vive_aliases.sh)
    echo "[2/4] vrserver --keepalive ..."
    if pgrep -x vrserver >/dev/null 2>&1; then
        echo "      already running"
    else
        if [[ ! -x "$STEAMVR/bin/linux64/vrserver" ]]; then
            echo "      FAIL: vrserver not found at $STEAMVR/bin/linux64/vrserver" >&2
            return 1
        fi
        export STEAMVR LD_LIBRARY_PATH="$STEAMVR/bin/linux64:${LD_LIBRARY_PATH:-}"
        ( cd "$STEAMVR/bin/linux64" && nohup ./vrserver --keepalive >"$LOG_VRSERVER" 2>&1 & )
        for _ in $(seq 1 8); do
            sleep 1; pgrep -x vrserver >/dev/null && break
        done
        if pgrep -x vrserver >/dev/null; then
            echo "      OK (log: $LOG_VRSERVER)"
        else
            echo "      FAIL: vrserver did not start. tail of log:"
            tail -10 "$LOG_VRSERVER" 2>/dev/null
            return 1
        fi
    fi

    # 3. vive_input (matches run_vive)
    echo "[3/4] vive_input :12345 ..."
    if ss -tln 2>/dev/null | grep -q ':12345 '; then
        echo "      already listening"
    else
        if [[ ! -x "$VIVE_INPUT_BIN" ]]; then
            echo "      FAIL: vive_input binary not found." >&2
            echo "         tried: $VIVE_INPUT_BIN_DEFAULT" >&2
            echo "         and:   $VIVE_INPUT_BIN_FALLBACK" >&2
            return 1
        fi
        export LD_LIBRARY_PATH="$STEAMVR/bin/linux64:$(dirname "$VIVE_INPUT_BIN"):${LD_LIBRARY_PATH:-}"
        ( cd "$(dirname "$VIVE_INPUT_BIN")" && \
          nohup "$VIVE_INPUT_BIN" "$VIVE_INPUT_FREQ" >"$LOG_VIVE_INPUT" 2>&1 & )
        for _ in $(seq 1 8); do
            sleep 1; ss -tln 2>/dev/null | grep -q ':12345 ' && break
        done
        if ss -tln 2>/dev/null | grep -q ':12345 '; then
            echo "      OK (log: $LOG_VIVE_INPUT)"
        else
            echo "      FAIL: 12345 did not come up. tail of log:"
            tail -10 "$LOG_VIVE_INPUT" 2>/dev/null
            return 1
        fi
    fi

    # 4. summary
    echo "[4/4] verify"
    cmd_status
}

case "${1:-start}" in
    start|"")  cmd_start ;;
    stop)      cmd_stop ;;
    status)    cmd_status ;;
    restart)   cmd_stop; sleep 1; cmd_start ;;
    *) echo "usage: $0 [start|stop|status|restart]" >&2; exit 2 ;;
esac
