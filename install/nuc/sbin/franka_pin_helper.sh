#!/bin/bash
# franka_pin_helper.sh -- pin the Polymetis RT threads to the isolated cores.
#
# isolcpus by itself only *prevents* the scheduler from migrating onto cores
# 6,7 by default — it does not pull existing threads in. Without an explicit
# taskset, the Polymetis run_server / franka_panda_client / franka_hand_client
# threads stay on whatever core they spawned on (usually a P-core under load),
# which defeats the whole isolation strategy.
#
# This helper polls for the targets for up to 15 s after launch_robot.py /
# launch_gripper.py is invoked, then pins every thread (TID) of every match
# to PIN_CORES via taskset -a. Re-runnable safely.
#
# Usage:
#   sudo /usr/local/sbin/franka_pin_helper.sh
# (or via the start_franka_arm.sh / start_franka_gripper.sh wrappers, which
#  schedule it 5 s after launch_robot.py / launch_gripper.py spawn).
PIN_CORES="6,7"
TARGETS="run_server franka_panda_client franka_hand_client"

pin_all_threads() {
    local pid="$1"
    [ -d /proc/$pid ] || return 0
    taskset -cpa "$PIN_CORES" "$pid" >/dev/null 2>&1
    for tid_dir in /proc/$pid/task/*; do
        taskset -cpa "$PIN_CORES" "$(basename "$tid_dir")" >/dev/null 2>&1
    done
}

# Wait up to 15 s for the targets to appear
FOUND_ANY=0
for i in $(seq 1 30); do
    PIDS=""
    for t in $TARGETS; do
        PIDS="$PIDS $(pgrep -f "fairo/polymetis/polymetis/build/$t" 2>/dev/null)"
    done
    PIDS=$(echo "$PIDS" | tr ' ' '\n' | sort -u | grep -v '^$')
    if [ -n "$PIDS" ]; then
        FOUND_ANY=1
        for pid in $PIDS; do pin_all_threads "$pid"; done
    fi
    sleep 0.5
done

if [ "$FOUND_ANY" -eq 0 ]; then
    echo "[franka_pin_helper] no target processes found -- launch_robot.py / launch_gripper.py started?" >&2
    exit 1
fi

# Final report
echo "[franka_pin_helper] pin result:"
for t in $TARGETS; do
    for pid in $(pgrep -f "fairo/polymetis/polymetis/build/$t" 2>/dev/null); do
        for tid_dir in /proc/$pid/task/*; do
            tid=$(basename "$tid_dir")
            comm=$(cat "$tid_dir/comm" 2>/dev/null)
            aff=$(taskset -cp "$tid" 2>/dev/null | awk '{print $NF}')
            printf "    %-22s tid=%-7s aff=%s\n" "$comm" "$tid" "$aff"
        done
    done
done
