#!/bin/bash
# /usr/local/sbin/cleanup_polymetis.sh
# Run this on the NUC after Ctrl+C-ing start_franka_arm.sh, before relaunching.
#
# Problem this solves:
#   Ctrl+C from the shell sends SIGINT to the foreground process group, but
#   the polymetis stack (run_server + launch_robot.py + franka_panda_client)
#   sometimes leaves one or more children alive (the C++ binaries don't
#   always honor SIGINT cleanly).  Their open socket on :50051 then blocks
#   the next start_franka_arm.sh with
#
#       AssertionError: Port unavailable; possibly another server found on
#       designated address.
#
# Idempotent.  Run as: sudo bash /usr/local/sbin/cleanup_polymetis.sh
set +e

echo "=== Polymetis cleanup on $(hostname) ==="

# Patterns ordered child -> parent so child cleanup doesn't trigger respawn.
PATTERNS=(
    "fairo/polymetis/polymetis/build/franka_panda_client"
    "fairo/polymetis/polymetis/build/franka_hand_client"
    "fairo/polymetis/polymetis/build/run_server"
    "launch_robot.py"
    "launch_gripper.py"
    "/usr/local/sbin/start_franka_arm.sh"
)

echo
echo "[step 1] SIGTERM round..."
killed=0
for pat in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ')
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null && killed=$((killed + 1))
        echo "  TERM $pat -> PIDs: $pids"
    fi
done
[ "$killed" = "0" ] && echo "  (nothing to TERM)"

sleep 2

echo
echo "[step 2] SIGKILL survivors..."
survived=0
for pat in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ')
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null && survived=$((survived + 1))
        echo "  KILL $pat -> PIDs: $pids"
    fi
done
[ "$survived" = "0" ] && echo "  (all gone on SIGTERM)"

sleep 1

echo
echo "[step 3] Verify port 50051 released..."
if ss -tlnp 2>/dev/null | grep -q ':50051 '; then
    echo "  WARN: :50051 still bound:"
    ss -tlnp 2>/dev/null | grep ':50051 '
    echo
    echo "  Force-killing whoever owns :50051..."
    fuser -k -9 50051/tcp 2>&1 | head -3
    sleep 1
    if ss -tlnp 2>/dev/null | grep -q ':50051 '; then
        echo "  STILL bound -- inspect manually: 'sudo ss -tlnp | grep 50051'"
        exit 1
    fi
fi
echo "  :50051 released."

echo
echo "[step 4] Final check..."
remaining=$(pgrep -af 'run_server|launch_robot|franka_panda_client|franka_hand_client|start_franka_arm' 2>/dev/null)
if [ -n "$remaining" ]; then
    echo "  WARN: still alive:"
    echo "$remaining" | sed 's/^/    /'
    exit 1
fi
echo "  all clear"

echo
echo "Done.  Now you can run:  sudo bash /usr/local/sbin/start_franka_arm.sh"
