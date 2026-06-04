#!/bin/bash
# /usr/local/sbin/cleanup_polymetis.sh
# Run on the NUC after Ctrl+C-ing start_franka_arm.sh, before relaunching.
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
# Multi-user safety (2026-06-04):
#   This NUC is SHARED and the Franka FCI is a single-holder lock. polymetis
#   (franka_panda_client), the unified-interface server, and ROS2 franka
#   (franka_ros2_ws) all contend for it. The old version killed EVERY
#   launch_robot.py / run_server by path -- which would also tear down ANOTHER
#   user's live session. New behaviour:
#     * If a HEALTHY polymetis stack is connected to the robot (a live
#       franka_panda_client), the arm is IN USE. We REPORT who holds it and
#       EXIT WITHOUT killing anything. Pass --force only when you KNOW it is
#       your own stuck stack and you intend to kill it.
#     * Otherwise we clean leftover / zombie polymetis processes + the :50051
#       socket -- safe, because no client is connected.
#     * Competing arm drivers from other projects (launch_unified_interface_
#       server.py, polymetis_zerorpc_server.py, polymetis_gripper_server.py,
#       ROS2 franka nodes) are NEVER auto-killed -- only reported so you can
#       coordinate before taking the FCI.
#
# Idempotent.  Run as: sudo bash /usr/local/sbin/cleanup_polymetis.sh [--force]
set +e

FORCE=0
for a in "$@"; do
    case "$a" in
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    esac
done

echo "=== Polymetis cleanup on $(hostname) ==="

# ---- Detect live FCI / arm-control state -----------------------------------
PANDA_PIDS=$(pgrep -f 'fairo/polymetis/polymetis/build/franka_panda_client' 2>/dev/null | tr '\n' ' ')
RUNSRV_PIDS=$(pgrep -f 'fairo/polymetis/polymetis/build/run_server' 2>/dev/null | tr '\n' ' ')
PORT_BOUND=0; ss -tlnp 2>/dev/null | grep -q ':50051 ' && PORT_BOUND=1

# Competing drivers (report only, never auto-killed).
UNIFIED_PIDS=$(pgrep -f 'launch_unified_interface_server' 2>/dev/null | tr '\n' ' ')
ZRPC_PIDS=$(pgrep -f 'polymetis_zerorpc_server|polymetis_gripper_server' 2>/dev/null | tr '\n' ' ')
# Match ACTUAL ROS2 node executables only. Do NOT match bare 'franka_ros2' /
# 'franka_hardware' -- those substrings appear inside the long LD_LIBRARY_PATH
# of the polymetis/unified sudo wrappers (…/franka_ros2_ws/install/franka_hardware/lib)
# and would false-positive them as ROS nodes (verified 2026-06-04).
ROS_PIDS=$(pgrep -f 'ros2_control_node|robot_state_publisher|franka_hardware_node|franka_control_node|franka_gripper_node' 2>/dev/null | tr '\n' ' ')

report_proc() {  # $1=label  $2=pids
    local label="$1" pids="$2"
    [ -z "$(echo "$pids" | tr -d ' ')" ] && return 0
    echo "  [$label]"
    for p in $pids; do
        local u tty st
        u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
        tty=$(ps -o tty= -p "$p" 2>/dev/null | tr -d ' ')
        st=$(ps -o lstart= -p "$p" 2>/dev/null)
        echo "      pid=$p user=${u:-?} tty=${tty:-?} since='${st}'"
    done
}

echo
echo "[detect] current arm-control / FCI state:"
report_proc "polymetis run_server (:50051)"            "$RUNSRV_PIDS"
report_proc "polymetis franka_panda_client (HOLDS FCI)" "$PANDA_PIDS"
report_proc "unified-interface server"                 "$UNIFIED_PIDS"
report_proc "polymetis zerorpc/gripper server"         "$ZRPC_PIDS"
report_proc "ROS2 franka nodes (HOLD FCI)"             "$ROS_PIDS"
[ "$PORT_BOUND" = "1" ] && echo "  :50051 is BOUND" || echo "  :50051 is free"

# ---- Active-stack guard -----------------------------------------------------
if [ -n "$(echo "$PANDA_PIDS" | tr -d ' ')" ] && [ "$FORCE" = "0" ]; then
    echo
    echo "  ⚠  A LIVE polymetis stack is connected to the Franka"
    echo "     (franka_panda_client PIDs: $PANDA_PIDS). The arm is IN USE."
    echo "     NOT killing anything (multi-user safety)."
    echo "       * Someone else running it? Coordinate before taking the FCI."
    echo "       * Your own stuck stack? Re-run with: sudo $0 --force"
    exit 3
fi

if [ -n "$(echo "$ROS_PIDS$UNIFIED_PIDS" | tr -d ' ')" ]; then
    echo
    echo "  ⚠  Competing arm driver(s) detected above (unified / ROS2). They hold or"
    echo "     can take the Franka FCI and are NOT auto-killed (different owner/project)."
    echo "     If the arm is unexpectedly busy after this cleanup, coordinate with them."
fi

# ---- Kill leftover/zombie polymetis processes (safe: no live FCI client,
#      or --force) -----------------------------------------------------------
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
