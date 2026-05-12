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
# * Restart a service only if it is **down** (systemd inactive / port
#   unreachable).  Don't touch healthy services -- the goal is to fix what
#   is broken, not to disrupt what is already working.
# * Hardware-level issues (USB unplugged, SteamVR not launched) cannot be
#   auto-fixed; report them precisely so the operator can act.
# * Refuse to run if a demo / eval is already running locally on pro4000
#   (the user almost certainly ran us by mistake -- prompt to quit the
#   existing session first).
#
# Flags
# -----
#   --no-nuc                 skip the NUC SSH section (offline / no LAN to NUC)
#   --no-gripper-restart     skip auto-restart of art_gripper_daemon
#   --no-arm-restart         skip auto-restart of NUC polymetis arm
#   --force                  proceed even if a session is currently running
#                            (only when you really mean it)
#   --quiet                  suppress section banners (still prints findings)

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
        --no-arm-restart)      SKIP_ARM_RESTART=1 ;;
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
# 6. NUC (Franka arm via polymetis :50051)
# ──────────────────────────────────────────────────────────────────────
section "6. NUC polymetis arm (Franka)"

if [ "$SKIP_NUC" = "1" ]; then
    note "--no-nuc passed -- skipped"
elif ! port_alive 192.168.1.12 22 2; then
    warn "NUC SSH (port 22) NOT reachable -- check NUC power / LAN cable"
else
    # Choose SSH prefix once (used by both orphan-kill and restart paths).
    if command -v sshpass >/dev/null 2>&1; then
        SSH_PFX="sshpass -p $NUC_PASS ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5"
    else
        SSH_PFX="ssh -o ConnectTimeout=5"
    fi

    # First check :50051 directly from pro4000.
    if port_alive 192.168.1.12 50051 2; then
        ok "polymetis :50051 reachable -- NOT touching NUC processes (would tear down a healthy server)"
    else
        warn "polymetis :50051 NOT reachable -- cleaning NUC orphans then restarting"

        # Step 1: kill any orphaned polymetis processes left over from a crashed
        # previous run.  Only safe to do this when :50051 is already down -- the
        # patterns include launch_robot.py and run_server which ARE the live
        # arm server when it's healthy.
        $SSH_PFX "$NUC_HOST" "bash -s" <<'NUC_EOF' 2>&1 | sed 's/^/    /'
PASS="kist"
killed=0
for pat in fairo/polymetis/polymetis/build/run_server franka_panda_client franka_hand_client launch_robot.py launch_gripper.py ; do
    pids=$(pgrep -f "$pat" 2>/dev/null)
    [ -n "$pids" ] || continue
    echo "$PASS" | sudo -S kill $pids 2>/dev/null
    killed=$((killed + 1))
done
sleep 1
for pat in fairo/polymetis/polymetis/build/run_server franka_panda_client franka_hand_client launch_robot.py launch_gripper.py ; do
    pids=$(pgrep -f "$pat" 2>/dev/null)
    [ -n "$pids" ] || continue
    echo "$PASS" | sudo -S kill -9 $pids 2>/dev/null
done
echo "  NUC orphans cleared: $killed pattern(s)"
NUC_EOF

        # Step 2: restart the arm server (unless caller suppressed it).
        # start_franka_arm.sh runs the arm server in the foreground forever,
        # so we MUST detach it on the NUC (nohup + & + redirect all fds) and
        # let SSH return immediately.  Without this the cleanup script hangs.
        if [ "$SKIP_ARM_RESTART" = "1" ]; then
            warn "  (--no-arm-restart was passed -- skipping restart)"
        else
            note "launching start_franka_arm.sh on NUC (detached, log -> /tmp/franka_arm.log)"
            $SSH_PFX "$NUC_HOST" \
                "echo $NUC_PASS | sudo -S nohup bash /usr/local/sbin/start_franka_arm.sh </dev/null >/tmp/franka_arm.log 2>&1 &" \
                2>&1 | sed 's/^/    /'
            # Poll up to ~30 s for :50051 to come up (arm init + libfranka handshake takes ~5-8 s).
            for i in 1 2 3 4 5 6; do
                sleep 5
                if port_alive 192.168.1.12 50051 2; then
                    ok "polymetis :50051 now reachable -- arm server up (after ${i}x5 s)"
                    break
                fi
                [ "$i" = "6" ] && warn "polymetis :50051 still down after 30 s -- check NUC: 'tail /tmp/franka_arm.log; sudo bash /usr/local/sbin/start_franka_arm.sh'"
            done
        fi
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
