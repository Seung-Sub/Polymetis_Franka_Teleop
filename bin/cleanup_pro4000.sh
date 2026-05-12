#!/usr/bin/env bash
# Polymetis_Franka_Teleop / Isaac-GR00T 데이터 수집 전 pro4000 cleanup.
#
# Run before every new ``bash bin/start_teleop_*.sh ...`` invocation.  Idempotent.
# Sanitizes stale processes, leaked SHM, ZED SDK locks, and cv2_viewer temp
# files that intermittently cause "camera not detected" / "device busy" /
# "address already in use" on the next launch.
#
# Safe: never touches systemd-managed daemons (art-gripper-daemon, ethercat),
# the active SSH session, or SteamVR / Vive vrserver (their cooldown is the
# user's call; killing vrserver mid-session loses Vive tracking).
#
# Usage:
#   bash ~/Polymetis_Franka_Teleop/bin/cleanup_pro4000.sh
#   # then re-run preflight + start_teleop_*.sh as normal.

set +e   # never abort on a transient kill / chmod failure

section() { printf "\n=== %s ===\n" "$*"; }

# Self-PID so kill_pattern never SIGKILLs the very script that's running.
SELF_PID=$$

# ──────────────────────────────────────────────────────────────────────
section "1. Pipeline processes (Polymetis_Franka_Teleop / Isaac-GR00T client side)"

# Patterns chosen to be specific enough to avoid hitting unrelated python
# scripts. SIGTERM first, sleep 2, then SIGKILL the survivors.
PATTERNS=(
    "scripts_real/demo_franka_vive"
    "scripts_real/eval_franka_policy"
    "bin/cv2_viewer"
    "polymetis_franka_teleop/real_world/single_zed"
    "polymetis_franka_teleop/real_world/multi_zed"
    "polymetis_franka_teleop/real_world/single_realsense"
    "polymetis_franka_teleop/real_world/multi_realsense"
    "polymetis_franka_teleop/real_world/vive_teleop_process"
    "polymetis_franka_teleop/real_world/franka_interpolation_controller"
    "polymetis_franka_teleop/real_world/franka_policy_env"
    "polymetis_franka_teleop/real_world/video_recorder"
    "polymetis_franka_teleop/real_world/franka_gripper_controller"
    "scripts_real/bench_fk_consistency"
    # Isaac-GR00T side (3090 abandoned, but keep for completeness if anyone runs here)
    "gr00t/eval/run_gr00t_server"
    "gr00t/eval/open_loop_eval"
    "examples/DROID/main_gr00t"
)

kill_proc_pattern() {
    local pat="$1" sig="$2" pids
    pids=$(pgrep -f "$pat" 2>/dev/null | grep -v "^${SELF_PID}\$" | tr '\n' ' ')
    if [ -n "$pids" ]; then
        kill -$sig $pids 2>/dev/null
        printf "  %-65s [%s] PIDs: %s\n" "$pat" "$sig" "$pids"
    fi
}

# SIGTERM round
for pat in "${PATTERNS[@]}"; do kill_proc_pattern "$pat" TERM; done
sleep 2
# SIGKILL survivors
killed_any=0
for pat in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null | grep -v "^${SELF_PID}\$")
    if [ -n "$pids" ]; then
        kill_proc_pattern "$pat" KILL
        killed_any=1
    fi
done
[ "$killed_any" = "0" ] && echo "  (all stopped on SIGTERM, no SIGKILL needed)"

# ──────────────────────────────────────────────────────────────────────
section "2. Python multiprocessing SHM leftovers"

# Patterns left behind by killed UMI SharedMemoryRingBuffer / Python's
# resource_tracker when its parent dies SIGKILL.
shm_removed=0
for shm in /dev/shm/wnsm_* /dev/shm/psm_* /dev/shm/sem.mp-* /dev/shm/u${UID}-Shm_*; do
    [ -e "$shm" ] || continue
    # SAFETY: do NOT touch SteamVR's per-user named segments
    case "$(basename "$shm")" in
        *SteamVR*|*ValveIPC*) continue ;;
    esac
    rm -f "$shm" 2>/dev/null && shm_removed=$((shm_removed + 1))
done
printf "  removed %d Python-mp SHM objects (SteamVR / Valve segments preserved)\n" "$shm_removed"

# ──────────────────────────────────────────────────────────────────────
section "3. ZED SDK ipc / lock files"

# ZED SDK leaves /tmp/.zed*, /tmp/zed_*, /dev/shm/zed_* on hard-kill.
zed_removed=0
for pattern in '/tmp/.zed*' '/tmp/zed_*' '/dev/shm/zed_*' '/var/lib/zed/.cam_lock_*'; do
    # shellcheck disable=SC2086  # we want glob expansion
    for f in $pattern; do
        [ -e "$f" ] || continue
        rm -rf "$f" 2>/dev/null && zed_removed=$((zed_removed + 1))
    done
done
printf "  removed %d ZED ipc / lock artifacts\n" "$zed_removed"

# ──────────────────────────────────────────────────────────────────────
section "4. cv2_viewer JPEG temp + demo viz state"

rm -f /tmp/teleop_vis*.jpg /tmp/franka_vive_*.jpg 2>/dev/null
echo "  removed cv2_viewer JPEG temp (if any)"

# ──────────────────────────────────────────────────────────────────────
section "5. Hardware visibility check"

echo "  USB devices (ZED + RealSense):"
lsusb 2>/dev/null | grep -iE "stereolabs|zed|intel corp.*realsense|8086:0b[0-9a-f]" | sed 's/^/    /' || echo "    (none detected -- check cable / USB power)"

# RealSense via librealsense if available
if command -v rs-enumerate-devices >/dev/null 2>&1; then
    echo "  rs-enumerate-devices:"
    rs-enumerate-devices --compact 2>&1 | grep -E "Device info|Name|Serial" | sed 's/^/    /' | head -10
fi

echo "  /dev/video* nodes:"
ls -la /dev/video* 2>/dev/null | awk '{printf "    %s %s %s\n", $1, $5, $NF}' | head -8

# Who is holding camera nodes? (helps diagnose "device busy" on the next launch)
echo "  /dev/video* holders (lsof):"
lsof /dev/video* 2>/dev/null | awk 'NR==1 || /\/dev\/video/' | sed 's/^/    /' | head -10

# ──────────────────────────────────────────────────────────────────────
section "6. NUC arm + ART gripper reachability"

if (echo > /dev/tcp/192.168.1.12/50051) 2>/dev/null; then
    echo "  polymetis :50051 (NUC) -- REACHABLE"
else
    echo "  polymetis :50051 (NUC) -- UNREACHABLE.  Restart on NUC: sudo bash /usr/local/sbin/start_franka_arm.sh"
fi

if (echo > /dev/tcp/127.0.0.1/50053) 2>/dev/null; then
    echo "  art_gripper :50053       -- REACHABLE"
else
    echo "  art_gripper :50053       -- UNREACHABLE.  systemctl status art-gripper-daemon"
fi

if (echo > /dev/tcp/127.0.0.1/12345) 2>/dev/null; then
    echo "  vive_input :12345        -- REACHABLE (vrserver alive)"
else
    echo "  vive_input :12345        -- UNREACHABLE.  Launch SteamVR on the desktop side (vrserver must be running)"
fi

# ──────────────────────────────────────────────────────────────────────
section "Done"
echo "Pre-flight ready.  Next:"
echo "  bash ~/Polymetis_Franka_Teleop/bin/preflight_full.sh"
echo "  bash ~/Polymetis_Franka_Teleop/bin/start_teleop_groot_droid_ft.sh \\"
echo "       ~/Polymetis_Franka_Teleop/data/session_\$(date +%Y%m%d_%H%M%S) \\"
echo "       --task \"<your task instruction>\""
