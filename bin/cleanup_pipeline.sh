#!/usr/bin/env bash
# Polymetis_Franka_Teleop / Isaac-GR00T full-pipeline sanitiser.
#
# Idempotent.  Run before every demo / fine-tune / eval session.  Tries to
# restore a healthy state across the **entire pipeline**:
#
#   pro4000  -- demo / cv2_viewer / single_zed / vive_teleop / franka_interp
#               + Python multiprocessing SHM + ZED SDK ipc + cv2 temp
#   NUC      -- polymetis arm server (:50051), franka_*_client orphans
#   Franka   -- (libfranka error recovery is left to demo's controller, but
#                we surface the symptom)
#   Gripper  -- art_gripper_daemon (:50053) + ethercat systemd
#   Cameras  -- ZED / RealSense USB visibility report + SDK lock cleanup
#   Vive     -- vive_input :12345 (vrserver liveness report)
#
# Policy
# ------
# * Kill orphan client-side processes aggressively (we control them all).
# * Restart pro4000-LOCAL systemd services (ethercat, art-gripper-daemon)
#   only if they are **down**.  These are local and idempotent to start.
# * NUC polymetis arm is **NEVER auto-started** by this script.  The operator
#   starts the arm manually on the NUC at session start, and uses Franka Desk
#   to unlock joints + FCI Activate.  This avoids surprise-running the arm
#   in the background, and keeps physical/cyber control of the arm explicit.
#   If :50051 is up while running cleanup, we leave NUC alone entirely;
#   if down, we just report the manual command the operator should run.
# * Hardware-level issues (USB unplugged, SteamVR not launched) cannot be
#   auto-fixed; report them precisely so the operator can act.
# * Refuse to run if a demo / eval is already running locally on pro4000
#   (the user almost certainly ran us by mistake -- prompt to quit the
#   existing session first).
#
# Flags
# -----
#   --no-nuc                 skip the NUC SSH section entirely
#   --no-gripper-restart     skip auto-restart of art_gripper_daemon
#   --force                  proceed even if a session is currently running
#                            (only when you really mean it)
#   --quiet                  suppress section banners (still prints findings)
#
# Deprecated flags (still accepted, no-op)
# ----------------------------------------
#   --no-arm-restart         NUC arm is no longer auto-restarted by default;
#                            this flag is now redundant.

set +e

NUC_HOST="${NUC_HOST:-kist@192.168.1.12}"
NUC_PASS="${NUC_PASS:-kist}"
SELF_PID=$$

# -------- arg parsing --------
SKIP_NUC=0
SKIP_GRIPPER_RESTART=0
SKIP_ARM_RESTART=0
FORCE=0
QUIET=0
for arg in "$@"; do
    case "$arg" in
        --no-nuc)              SKIP_NUC=1 ;;
        --no-gripper-restart)  SKIP_GRIPPER_RESTART=1 ;;
        --no-arm-restart)      : ;;   # deprecated no-op (NUC arm is never auto-restarted)
        --force)               FORCE=1 ;;
        --quiet)               QUIET=1 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
    esac
done

section() { [ "$QUIET" = "1" ] || printf "\n=== %s ===\n" "$*"; }
note()    { printf "  %s\n" "$*"; }
warn()    { printf "  ⚠  %s\n" "$*" >&2; }
ok()      { printf "  ✓  %s\n" "$*"; }

# Probe TCP port (no nc / netcat dependency).
port_alive() {
    local host="$1" port="$2" timeout="${3:-2}"
    timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null
}

# ──────────────────────────────────────────────────────────────────────
# Pre-flight: refuse if a session is already running locally
# ──────────────────────────────────────────────────────────────────────
section "0. Active-session sanity"
RUNNING=$(pgrep -f "scripts_real/demo_franka_vive|scripts_real/eval_franka_policy" 2>/dev/null | grep -v "^${SELF_PID}\$")
if [ -n "$RUNNING" ] && [ "$FORCE" = "0" ]; then
    warn "demo / eval is currently running (PIDs: $(echo $RUNNING | tr '\n' ' '))"
    warn "Quit the existing session first (cv2 viewer: q q), or re-run with --force"
    exit 2
fi
[ -z "$RUNNING" ] && ok "no active demo / eval -- safe to clean"

# ──────────────────────────────────────────────────────────────────────
# 1. pro4000 client-side processes
# ──────────────────────────────────────────────────────────────────────
section "1. pro4000 pipeline processes"

PATTERNS=(
    "scripts_real/demo_franka_vive"
    "scripts_real/eval_franka_policy"
    "scripts_real/bench_fk_consistency"
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
    # Isaac-GR00T side (3090 abandoned but kept for completeness)
    "gr00t/eval/run_gr00t_server"
    "gr00t/eval/open_loop_eval"
    "examples/DROID/main_gr00t"
)

kill_pattern() {
    local pat="$1" sig="$2" pids
    pids=$(pgrep -f "$pat" 2>/dev/null | grep -v "^${SELF_PID}\$" | tr '\n' ' ')
    [ -n "$pids" ] || return 0
    kill "-$sig" $pids 2>/dev/null
    note "[$sig] $pat (PIDs: $pids)"
}

for pat in "${PATTERNS[@]}"; do kill_pattern "$pat" TERM; done
sleep 2
SURVIVORS=0
for pat in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null | grep -v "^${SELF_PID}\$")
    [ -n "$pids" ] || continue
    kill_pattern "$pat" KILL
    SURVIVORS=$((SURVIVORS + 1))
done
[ "$SURVIVORS" = "0" ] && ok "all clean on SIGTERM"

# ──────────────────────────────────────────────────────────────────────
# 2. Python multiprocessing SHM leftovers
# ──────────────────────────────────────────────────────────────────────
section "2. /dev/shm cleanup (Python mp / SharedMemoryRingBuffer)"

shm_removed=0
for pattern in '/dev/shm/wnsm_*' '/dev/shm/psm_*' '/dev/shm/sem.mp-*' "/dev/shm/u${UID}-Shm_*"; do
    for f in $pattern; do
        [ -e "$f" ] || continue
        case "$(basename "$f")" in
            *SteamVR*|*ValveIPC*) continue ;;
        esac
        rm -f "$f" 2>/dev/null && shm_removed=$((shm_removed + 1))
    done
done
ok "removed $shm_removed mp/SHM objects (SteamVR / Valve preserved)"

# ──────────────────────────────────────────────────────────────────────
# 3. ZED SDK ipc / lock files
# ──────────────────────────────────────────────────────────────────────
section "3. ZED SDK ipc / lock files"

zed_removed=0
for pattern in '/tmp/.zed*' '/tmp/zed_*' '/dev/shm/zed_*' '/var/lib/zed/.cam_lock_*'; do
    for f in $pattern; do
        [ -e "$f" ] || continue
        rm -rf "$f" 2>/dev/null && zed_removed=$((zed_removed + 1))
    done
done
ok "removed $zed_removed ZED ipc / lock artifacts"

# ──────────────────────────────────────────────────────────────────────
# 4. cv2_viewer / demo viz temp
# ──────────────────────────────────────────────────────────────────────
section "4. cv2_viewer JPEG temp"
rm -f /tmp/teleop_vis*.jpg /tmp/franka_vive_*.jpg 2>/dev/null
ok "removed cv2_viewer JPEG temp"

# ──────────────────────────────────────────────────────────────────────
# 5. ART gripper daemon + ethercat (pro4000 systemd)
# ──────────────────────────────────────────────────────────────────────
section "5. ART gripper + ethercat (pro4000 systemd)"

# Check each unit's state; restart only if inactive/failed.
restart_if_dead() {
    local unit="$1"
    if ! systemctl is-active --quiet "$unit"; then
        local state
        state=$(systemctl is-failed --quiet "$unit" 2>/dev/null && echo "failed" || systemctl is-active "$unit" 2>/dev/null || echo "unknown")
        warn "$unit is $state -- restarting"
        if [ "$SKIP_GRIPPER_RESTART" = "1" ] && [ "$unit" = "art-gripper-daemon" ]; then
            warn "  (--no-gripper-restart was passed -- skipping)"
            return 1
        fi
        sudo systemctl restart "$unit" 2>&1 | sed 's/^/    /'
        sleep 2
        if systemctl is-active --quiet "$unit"; then
            ok "$unit restarted OK"
        else
            warn "$unit restart FAILED -- check 'systemctl status $unit'"
            return 1
        fi
    else
        ok "$unit is active"
    fi
}
restart_if_dead ethercat
restart_if_dead art-gripper-daemon

# Port-level health check on art_gripper :50053
if port_alive 127.0.0.1 50053 2; then
    ok "art_gripper :50053 reachable"
else
    warn "art_gripper :50053 NOT reachable despite systemd active -- daemon may be in init phase"
    warn "  retry once after 3 s..."
    sleep 3
    if port_alive 127.0.0.1 50053 2; then
        ok "art_gripper :50053 now reachable"
    else
        warn "art_gripper :50053 still down -- run 'systemctl status art-gripper-daemon' + 'journalctl -u art-gripper-daemon -n 50'"
    fi
fi

# ──────────────────────────────────────────────────────────────────────
# 6. NUC polymetis arm (Franka) -- DIAGNOSE ONLY
# ──────────────────────────────────────────────────────────────────────
# Policy: the NUC arm is owned by the operator.  This script never starts
# or restarts it -- doing so previously caused "phantom" arm processes the
# operator didn't ask for (sub bug: a brief :50051 outage during cleanup
# was misread as "arm down, please restart").  Instead we just report
# current state, and tell the operator exactly what to type on the NUC if
# they need to bring the arm up.
section "6. NUC polymetis arm (Franka) -- diagnostic only"

if [ "$SKIP_NUC" = "1" ]; then
    note "--no-nuc passed -- skipped"
elif ! port_alive 192.168.1.12 22 2; then
    warn "NUC SSH (port 22) NOT reachable -- check NUC power / LAN cable"
else
    # Choose SSH prefix
    if command -v sshpass >/dev/null 2>&1; then
        SSH_PFX="sshpass -p $NUC_PASS ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=no"
    else
        SSH_PFX="ssh -o ConnectTimeout=5 -o BatchMode=yes"
    fi

    port_open=0
    port_alive 192.168.1.12 50051 2 && port_open=1

    # Process inventory on NUC (always check, regardless of port state).
    proc_count=$($SSH_PFX "$NUC_HOST" \
        "pgrep -f 'fairo/polymetis/polymetis/build/run_server|launch_robot.py' 2>/dev/null | wc -l" \
        2>/dev/null | tr -d '[:space:]')
    proc_count=${proc_count:-0}

    if [ "$port_open" = "1" ] && [ "$proc_count" -gt 0 ]; then
        # Port + process both alive — healthy.
        ok "polymetis :50051 reachable + $proc_count NUC process(es) running"
        $SSH_PFX "$NUC_HOST" \
            "pgrep -af 'start_franka_arm|launch_robot.py|fairo/polymetis/polymetis/build/run_server' 2>/dev/null | head -5" \
            2>/dev/null | sed 's/^/    /'
    elif [ "$port_open" = "1" ] && [ "$proc_count" = "0" ]; then
        # Port bound but no process — ZOMBIE state (Ctrl+C left lingering socket).
        warn "polymetis :50051 PORT BOUND but no polymetis process found on NUC"
        warn "  -> zombie socket from a prior Ctrl+C.  Next start_franka_arm.sh will fail with:"
        warn "        AssertionError: Port unavailable; possibly another server found..."
        warn ""
        warn "  Fix: clean up NUC with the bundled helper, then relaunch:"
        warn "       ssh kist@192.168.1.12"
        warn "       sudo bash /usr/local/sbin/cleanup_polymetis.sh"
        warn "       sudo bash /usr/local/sbin/start_franka_arm.sh"
    elif [ "$port_open" = "0" ] && [ "$proc_count" -gt 0 ]; then
        # No port but processes alive — half-dead state.
        warn "polymetis :50051 NOT reachable but $proc_count NUC process(es) still alive"
        warn "  -> partially crashed.  Clean up first:"
        warn "       ssh kist@192.168.1.12 'sudo bash /usr/local/sbin/cleanup_polymetis.sh'"
        warn "       ssh kist@192.168.1.12 'sudo bash /usr/local/sbin/start_franka_arm.sh'"
    else
        # Both down — clean state, just needs operator to start.
        warn "polymetis :50051 NOT reachable on NUC (clean state)"
        warn "  Manual action on NUC:"
        warn "    ssh kist@192.168.1.12"
        warn "    sudo bash /usr/local/sbin/start_franka_arm.sh"
        warn "  Don't forget Franka Desk: unlock joints + FCI Activate."
    fi
fi

# ──────────────────────────────────────────────────────────────────────
# 7. Cameras (ZED + RealSense)
# ──────────────────────────────────────────────────────────────────────
section "7. Camera visibility (USB-level)"

zed_count=$(lsusb 2>/dev/null | grep -c -iE "stereolabs|2b03:f[0-9a-f]+")
rs_count=$(lsusb 2>/dev/null | grep -c -iE "intel corp.*realsense|8086:0b")

note "USB ZED cameras visible: $zed_count"
lsusb 2>/dev/null | grep -iE "stereolabs|2b03:f[0-9a-f]+" | sed 's/^/    /'
note "USB RealSense cameras visible: $rs_count"
lsusb 2>/dev/null | grep -iE "intel corp.*realsense|8086:0b" | sed 's/^/    /'

if [ "$zed_count" = "0" ] && [ "$rs_count" = "0" ]; then
    warn "NO cameras detected on USB.  Physical actions:"
    warn "  1. Re-seat USB cables (especially ZED 2i power-hungry side cam)"
    warn "  2. Try a different USB 3.0 port (some Mini-PC ports throttle power)"
    warn "  3. lsusb again -- if still missing, power-cycle the camera"
else
    ok "$zed_count ZED + $rs_count RealSense detected via lsusb"
fi

# rs-enumerate-devices for finer RealSense check
if command -v rs-enumerate-devices >/dev/null 2>&1 && [ "$rs_count" -gt 0 ]; then
    rs-enumerate-devices --compact 2>&1 | grep -E "Device info|Name|Serial" | head -10 | sed 's/^/    /'
fi

# Who is holding /dev/video* nodes (useful for diagnosis even if not blocking ZED SDK)
holders=$(lsof /dev/video* 2>/dev/null | awk 'NR>1 {print $1, $2}' | sort -u)
if [ -n "$holders" ]; then
    note "/dev/video* holders:"
    echo "$holders" | sed 's/^/    /'
fi

# ──────────────────────────────────────────────────────────────────────
# 8. Vive / SteamVR
# ──────────────────────────────────────────────────────────────────────
section "8. Vive input server (SteamVR + vive_input)"

if port_alive 127.0.0.1 12345 2; then
    ok "vive_input :12345 reachable (vrserver alive)"
else
    warn "vive_input :12345 NOT reachable"
    warn "  Manual action: launch SteamVR on the desktop side"
    warn "  (Vive lighthouse + controllers must be powered on AND vrserver running)"
fi

# ──────────────────────────────────────────────────────────────────────
# 9. Summary
# ──────────────────────────────────────────────────────────────────────
section "Summary"

probes=(
    "polymetis_arm  192.168.1.12 50051"
    "art_gripper    127.0.0.1    50053"
    "vive_input     127.0.0.1    12345"
)
n_ok=0; n_fail=0
for line in "${probes[@]}"; do
    read name host port <<< "$line"
    if port_alive "$host" "$port" 3; then
        ok "$name $host:$port"
        n_ok=$((n_ok + 1))
    else
        warn "$name $host:$port DOWN"
        n_fail=$((n_fail + 1))
    fi
done

echo
if [ "$n_fail" = "0" ] && [ "$zed_count" -gt 0 ]; then
    echo "Pipeline READY.  Next:"
    echo "  bash ~/Polymetis_Franka_Teleop/bin/preflight_full.sh"
    echo "  bash ~/Polymetis_Franka_Teleop/bin/start_teleop_groot_droid_ft.sh \\"
    echo "       ~/Polymetis_Franka_Teleop/data/session_\$(date +%Y%m%d_%H%M%S) \\"
    echo "       --task \"<your task instruction>\""
    exit 0
else
    echo "Pipeline NOT fully ready ($n_fail service(s) down, $zed_count ZED visible)."
    echo "Address the warnings above, then re-run this script."
    exit 1
fi
