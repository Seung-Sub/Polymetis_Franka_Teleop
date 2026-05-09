#!/usr/bin/env bash
# monitor_session.sh -- live multi-channel health monitor for an in-flight
# Polymetis_Franka_Teleop session. Run from a separate SSH terminal while
# the demo is going via run_test_session_groot_ft.sh (or similar).
#
# What it surveils every 5 s:
#   * ZED grab fps (each camera, last 5 s window)
#   * FrankaInterpolationController teleop frequency + overrun count
#   * ArtGripperController loop frequency + overrun count
#   * IK fail streak from FrankaInterface
#   * Recovery count + most recent recovery cause (NUC libfranka log)
#   * NUC control_command_success_rate from libfranka
#   * Active episode index + main loop iter
#
# Stops on Ctrl-C. Read-only — does not interfere with the demo.
#
# Usage:
#   bash bin/monitor_session.sh                              # default log path
#   bash bin/monitor_session.sh /tmp/teleop_groot_ft.log     # explicit
LOG="${1:-/tmp/teleop_groot_ft.log}"
NUC_USER="${NUC_USER:-kist}"
NUC_HOST="${NUC_HOST:-192.168.1.12}"
NUC_PWD="${NUC_PWD:-kist}"
NUC_LOG="${NUC_LOG:-/tmp/franka_arm.log}"
INTERVAL="${INTERVAL:-5}"

if [[ ! -f "$LOG" ]]; then
    echo "[monitor] log not found: $LOG" >&2
    exit 1
fi

echo "[monitor] watching $LOG every ${INTERVAL}s (Ctrl-C to stop)"
echo "[monitor] NUC log via ssh ${NUC_USER}@${NUC_HOST}:${NUC_LOG}"

while true; do
    ts=$(date +%H:%M:%S)
    last_iter=$(grep '\[main\] iter' "$LOG" | tail -1 | sed 's/.*iter=\([0-9]*\) ep=\([0-9]*\).*/iter=\1 ep=\2/')
    cam_a=$(grep 'SingleZed 33538770.*FPS' "$LOG" | tail -1 | sed 's/.*FPS \([0-9.]*\).*/\1/')
    cam_b=$(grep 'SingleZed 11667817.*FPS' "$LOG" | tail -1 | sed 's/.*FPS \([0-9.]*\).*/\1/')
    arm_freq=$(grep 'FrankaPositionalController.*Actual frequency' "$LOG" | tail -1 \
        | sed 's/.*Actual frequency: \([0-9.]*\)Hz  overruns=\([0-9]*\).*/\1Hz over=\2\/100/')
    grip_status=$(grep 'ArtGripperController.*overruns' "$LOG" | tail -1 \
        | sed 's/.*target=\([0-9]*\)Hz overruns=\([0-9]*\)\/.*/\1Hz over=\2\/300/')
    rec_count=$(grep -c 'Recovery #' "$LOG" 2>/dev/null)
    ik_fail=$(grep -c 'IK STUCK' "$LOG" 2>/dev/null)
    last_recovery=$(grep 'Recovery #' "$LOG" | tail -1 | sed 's/.*Recovery #\([0-9]*\):.*/\1/')

    # NUC side: success rate + reflex count (1 s SSH timeout to avoid blocking)
    nuc_info=$(timeout 2 sshpass -p "$NUC_PWD" ssh -o StrictHostKeyChecking=no \
        -o BatchMode=yes -o ConnectTimeout=1 "${NUC_USER}@${NUC_HOST}" \
        "grep -c 'motion aborted by reflex' $NUC_LOG; grep 'control_command_success_rate' $NUC_LOG | tail -1 | awk '{print \$NF}'" 2>/dev/null \
        | tr '\n' ' ')

    printf "[%s] iter=%-15s cam33538770=%4s cam11667817=%4s | arm=%s | grip=%s | rec=%s ik_stuck=%s | NUC reflex=%s last_succ=%s\n" \
        "$ts" "${last_iter:-?}" "${cam_a:-?}" "${cam_b:-?}" "${arm_freq:-?}" "${grip_status:-?}" \
        "${rec_count:-0}" "${ik_fail:-0}" "${nuc_info:-?}"
    sleep "$INTERVAL"
done
