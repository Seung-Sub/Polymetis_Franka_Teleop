#!/usr/bin/env bash
# preflight_full.sh -- comprehensive pre-launch health check + auto-recovery.
#
# Run this BEFORE the demo. Every issue we hit during 2026-05-09 testing has
# an auto-recovery path here, so a fresh boot or a botched previous session
# can be fixed with a single command instead of manual SSH-and-debug.
#
# Issues this script auto-recovers from:
#   * Stale Python multiprocessing children (orphaned by previous demo crash)
#     holding ZED cameras / ART TCP slot / SHM resources.
#   * Stale ART daemon TCP connections (CLOSE-WAIT / FIN-WAIT-2 / ESTAB) from
#     a killed previous client process. ART daemon's single-client lock
#     refuses new connects. Auto-restart via restart_gripper.sh.
#   * ZED Mini half-enumerated (HID visible, UVC missing) -- typical after
#     reboot. Cannot auto-reset USB at OS level without root + udev tricks,
#     so this just FAILS LOUDLY with the right physical-action instruction.
#   * Vive controller not detected (vrserver lost device handles after
#     reboot or USB hot-plug). Auto-restart vrserver + vive_input stack.
#   * NUC polymetis arm :50051 down. Auto-start via SSH + sudo + setsid.
#
# Returns 0 if everything is ready, non-zero with stderr describing the
# specific fault that requires human attention.
#
# Usage (from any pro4000 shell):
#   bash ~/Polymetis_Franka_Teleop/bin/preflight_full.sh
# OR (typical) it is auto-invoked by run_test_session_groot_ft.sh.
#
# Tunables via env:
#   NUC_USER (default: kist)
#   NUC_HOST (default: 192.168.1.12)
#   NUC_PWD  (default: kist)
#   AUTO_FIX (default: yes; set to "no" for read-only diagnostics)

set -uo pipefail

NUC_USER="${NUC_USER:-kist}"
NUC_HOST="${NUC_HOST:-192.168.1.12}"
NUC_PWD="${NUC_PWD:-kist}"
AUTO_FIX="${AUTO_FIX:-yes}"

# Console colors -- tty only
if [[ -t 1 ]]; then
    R=$(tput setaf 1); G=$(tput setaf 2); Y=$(tput setaf 3); B=$(tput setaf 4); N=$(tput sgr0)
else
    R=""; G=""; Y=""; B=""; N=""
fi

ok()  { echo -e "  ${G}OK${N}    $*"; }
warn(){ echo -e "  ${Y}WARN${N}  $*"; }
fail(){ echo -e "  ${R}FAIL${N}  $*"; }
fix() { echo -e "  ${B}FIX${N}   $*"; }

FAULTS=0

echo "============================================================"
echo "  Pre-flight: full health check + auto-recovery"
echo "============================================================"

# ---------- 1. Stale Python demo processes ----------
echo
echo "[1/6] Stale process cleanup..."
STALE_PIDS=$(pgrep -f "demo_franka_vive|spawn_main|cv2_viewer|FrankaPositional|ArtGripper|SingleZed|MultiZed|ViveTeleop" 2>/dev/null || true)
if [[ -n "$STALE_PIDS" ]]; then
    if [[ "$AUTO_FIX" == "yes" ]]; then
        fix "killing stale processes: $(echo $STALE_PIDS | tr '\n' ' ')"
        # SIGINT first (clean shutdown), then SIGKILL
        echo "$STALE_PIDS" | xargs -r kill -INT 2>/dev/null || true
        sleep 2
        echo "$STALE_PIDS" | xargs -r kill -KILL 2>/dev/null || true
        sleep 1
        REMAINING=$(pgrep -f "demo_franka_vive|spawn_main|cv2_viewer" 2>/dev/null | grep -v $$ || true)
        if [[ -n "$REMAINING" ]]; then
            fail "stale processes survived KILL: $REMAINING"
            FAULTS=$((FAULTS+1))
        else
            ok "stale processes cleared"
        fi
    else
        warn "stale processes detected (AUTO_FIX=no, leaving alone): $STALE_PIDS"
    fi
else
    ok "no stale demo processes"
fi

# ---------- 2. ART gripper daemon ----------
echo
echo "[2/6] ART gripper daemon (:50053)..."
if ! systemctl is-active --quiet art-gripper-daemon; then
    if [[ "$AUTO_FIX" == "yes" ]]; then
        fix "art-gripper-daemon inactive -- starting"
        sudo systemctl start art-gripper-daemon
        sleep 2
    else
        fail "art-gripper-daemon inactive"
        FAULTS=$((FAULTS+1))
    fi
fi

# Test by actual TCP ping (catches the stale-connection-holding-mutex bug
# that systemctl is-active does not detect). Try once with short timeout.
ART_OK=0
# OP_PING in art_gripper_client protocol: 1-byte opcode 0x01, 4-byte LE length=0
if timeout 3 python3 - <<'PYEOF' 2>/dev/null
import socket, struct
s = socket.create_connection(('127.0.0.1', 50053), timeout=2)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.sendall(bytes([0x01]) + struct.pack('<I', 0))   # OP_PING
s.settimeout(2)
status = s.recv(1)
plen_bytes = s.recv(4)
plen = struct.unpack('<I', plen_bytes)[0]
pl = s.recv(plen) if plen else b''
s.close()
ok = (status == bytes([0x00]) and pl == b'pong')   # STATUS_OK = 0x00
exit(0 if ok else 1)
PYEOF
then
    ART_OK=1
fi

if [[ "$ART_OK" == "1" ]]; then
    ok "ART daemon responsive on :50053"
else
    if [[ "$AUTO_FIX" == "yes" ]]; then
        fix "ART daemon hung -- running restart_gripper.sh"
        sudo bash "$HOME/Hyundai_motors_Gripper/scripts/restart_gripper.sh" >/tmp/preflight_art.log 2>&1 || true
        sleep 2
        # Re-test using same OP_PING protocol
        if timeout 3 python3 - <<'PYEOF' 2>/dev/null
import socket, struct
s = socket.create_connection(('127.0.0.1', 50053), timeout=2)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.sendall(bytes([0x01]) + struct.pack('<I', 0))
s.settimeout(2)
status = s.recv(1)
plen = struct.unpack('<I', s.recv(4))[0]
pl = s.recv(plen) if plen else b''
s.close()
exit(0 if (status == bytes([0x00]) and pl == b'pong') else 1)
PYEOF
        then
            ok "ART daemon recovered (see /tmp/preflight_art.log)"
        else
            fail "ART daemon still unreachable after restart_gripper.sh"
            fail "  -> tail /tmp/preflight_art.log for details, or 24V power-cycle the gripper."
            FAULTS=$((FAULTS+1))
        fi
    else
        fail "ART daemon TCP unresponsive (AUTO_FIX=no)"
        FAULTS=$((FAULTS+1))
    fi
fi

# ---------- 3. ZED enumeration ----------
echo
echo "[3/6] ZED cameras (USB enumeration + pyzed)..."
ZED_USB_COUNT=$(lsusb 2>/dev/null | grep -c -i stereolabs || echo 0)
# Expected: 4 entries -- ZED-2i UVC (f880), ZED-2i HID (f881), ZED-M UVC (f682), ZED-M HID (f681)
if [[ "$ZED_USB_COUNT" -lt 4 ]]; then
    fail "lsusb shows only $ZED_USB_COUNT/4 ZED USB devices"
    fail "  -> at least one ZED is missing or half-enumerated."
    if ! lsusb 2>/dev/null | grep -q '2b03:f880'; then
        fail "  -> ZED 2i UVC (2b03:f880) missing -- replug the EXTERIOR camera USB-3 cable"
    fi
    if ! lsusb 2>/dev/null | grep -q '2b03:f682'; then
        fail "  -> ZED Mini UVC (2b03:f682) missing -- replug the WRIST camera USB-3 cable"
    fi
    if ! lsusb 2>/dev/null | grep -q '2b03:f881'; then
        fail "  -> ZED 2i HID (2b03:f881) missing -- check exterior cam USB-2 side"
    fi
    if ! lsusb 2>/dev/null | grep -q '2b03:f681'; then
        fail "  -> ZED Mini HID (2b03:f681) missing -- check wrist cam USB-2 side"
    fi
    FAULTS=$((FAULTS+1))
else
    # Both UVC + HID present in lsusb -- check pyzed actually enumerates 2.
    # pyzed only available in groot-client conda env -- activate before testing.
    PYZED_COUNT=$(bash -c "
        source $HOME/anaconda3/etc/profile.d/conda.sh
        conda activate groot-client 2>/dev/null
        python -c 'import pyzed.sl as sl; print(len(sl.Camera.get_device_list()))' 2>/dev/null
    " || echo 0)
    PYZED_COUNT="${PYZED_COUNT:-0}"
    if [[ "$PYZED_COUNT" == "2" ]]; then
        ok "ZED USB OK (4/4) and pyzed sees both cameras"
    else
        warn "ZED USB OK but pyzed sees only ${PYZED_COUNT}/2 cameras"
        warn "  -> usually means a stale Python child still holds /dev/video* -- step 1 should have cleared it."
        warn "  -> try: lsof /dev/video0 /dev/video1 to find the holder"
        FAULTS=$((FAULTS+1))
    fi
fi

# ---------- 4. Vive stack + controllers ----------
echo
echo "[4/6] Vive stack (vrserver + vive_input + controller)..."
if ! pgrep -x vrserver >/dev/null 2>&1; then
    if [[ "$AUTO_FIX" == "yes" ]]; then
        fix "vrserver not running -- starting Vive stack"
        bash "$HOME/Polymetis_Franka_Teleop/bin/start_vive_stack.sh" start >/tmp/preflight_vive.log 2>&1
        sleep 5
    else
        fail "vrserver not running"
        FAULTS=$((FAULTS+1))
    fi
fi

if ! ss -tln 2>/dev/null | grep -q ':12345 '; then
    fail "vive_input :12345 not listening"
    FAULTS=$((FAULTS+1))
else
    # Check vive_input log for actual controller detection.
    if [[ -f /tmp/vive_input.log ]]; then
        # The log line "Summary: HMD=N Controllers=M Trackers=K" tells us.
        SUMMARY=$(tac /tmp/vive_input.log 2>/dev/null | grep -m 1 "Summary:" || true)
        if [[ -z "$SUMMARY" ]]; then
            warn "vive_input log has no Summary line yet -- waiting 3 s..."
            sleep 3
            SUMMARY=$(tac /tmp/vive_input.log 2>/dev/null | grep -m 1 "Summary:" || true)
        fi
        if echo "$SUMMARY" | grep -q "Controllers=0"; then
            if [[ "$AUTO_FIX" == "yes" ]]; then
                fix "vive_input sees 0 controllers -- restarting Vive stack"
                pkill -KILL -f vrserver 2>/dev/null || true
                pkill -KILL -f vive_input 2>/dev/null || true
                sleep 3
                bash "$HOME/Polymetis_Franka_Teleop/bin/start_vive_stack.sh" start >>/tmp/preflight_vive.log 2>&1
                sleep 6
                # Re-check
                SUMMARY=$(tac /tmp/vive_input.log 2>/dev/null | grep -m 1 "Summary:" || true)
                if echo "$SUMMARY" | grep -q "Controllers=0"; then
                    fail "still 0 controllers after restart -- physical action required:"
                    fail "  -> Press the system button on Vive controller (long-press for power)"
                    fail "  -> Verify base stations are powered + green LED"
                    fail "  -> Hold controller in line-of-sight of the base stations"
                    FAULTS=$((FAULTS+1))
                else
                    ok "Vive stack recovered: $SUMMARY"
                fi
            else
                fail "Vive stack reports 0 controllers (AUTO_FIX=no)"
                FAULTS=$((FAULTS+1))
            fi
        else
            ok "Vive stack: $SUMMARY"
        fi
    else
        warn "no /tmp/vive_input.log yet (just-started? try again in 5 s)"
    fi
fi

# ---------- 5. NUC polymetis arm ----------
# Policy: the NUC arm is owned by the operator -- preflight NEVER auto-
# starts it.  An earlier version did, which caused phantom arm processes
# the operator didn't request and obscured Franka Desk e-stop / FCI state.
# Now we diagnose only and print the exact command to run on the NUC.
echo
echo "[5/6] NUC polymetis arm (192.168.1.12:50051)..."
if ! ping -c 1 -W 2 "$NUC_HOST" >/dev/null 2>&1; then
    fail "NUC $NUC_HOST not pingable -- check network / NUC power / LAN cable"
    FAULTS=$((FAULTS+1))
elif ! nc -z -w 2 "$NUC_HOST" 50051 2>/dev/null; then
    fail "NUC :50051 down -- start the arm manually on the NUC:"
    fail "    ssh kist@$NUC_HOST"
    fail "    sudo bash /usr/local/sbin/start_franka_arm.sh"
    fail "  Don't forget Franka Desk (https://172.16.0.2/desk/):"
    fail "    unlock joints + FCI Activate + verify external e-stop released."
    FAULTS=$((FAULTS+1))
else
    ok "NUC polymetis :50051 reachable"
fi

# ---------- 6. RT settings + ulimit -r ----------
echo
echo "[6/6] RT environment..."
RT_LIMIT=$(ulimit -r 2>/dev/null || echo 0)
if [[ "$RT_LIMIT" -ge 50 ]]; then
    ok "ulimit -r = $RT_LIMIT  (SCHED_RR up to 50 available)"
else
    fail "ulimit -r = $RT_LIMIT  -- SCHED_RR will fall back to nice"
    fail "  -> log out from desktop and back in, or ssh kist@localhost (forces fresh PAM session)"
    FAULTS=$((FAULTS+1))
fi
if systemctl is-active --quiet franka-client-rt-tune; then
    ok "franka-client-rt-tune service active (NIC IRQ pinned to cores 0,1)"
else
    warn "franka-client-rt-tune service inactive -- run: sudo systemctl start franka-client-rt-tune"
fi

# ---------- summary ----------
echo
echo "============================================================"
if [[ "$FAULTS" -eq 0 ]]; then
    echo -e "  ${G}All checks passed -- ready to launch demo.${N}"
    echo "============================================================"
    exit 0
else
    echo -e "  ${R}$FAULTS fault(s) remaining -- resolve above before launching.${N}"
    echo "============================================================"
    exit 1
fi
