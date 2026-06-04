#!/usr/bin/env bash
# recover_gripper.sh -- one-button ART gripper recovery, run FROM pro4000.
#
# Why this exists
# ---------------
# The ART gripper has TWO independent health layers and the old preflight only
# checked the first:
#   1. daemon liveness  -- TCP :50053 answers 'pong'         (art_gripper_daemon)
#   2. bus liveness     -- EtherCAT master has >=1 slave + Link UP (the actual
#                          gripper drive is powered and on the wire)
# A daemon can answer 'pong' while the gripper is unplugged / unpowered
# (observed 2026-06-04: ping=pong but `ethercat master` => Slaves: 0, Link: DOWN).
# Commands then silently do nothing. This script checks BOTH layers and walks
# through escalating recovery so a session can re-init cleanly from pro4000:
#
#   Tier 0  diagnose (ping + slaves/link)
#   Tier 1  software recovery: scripts/restart_gripper.sh
#           (stop daemon -> reload EtherCAT master -> start daemon)
#   Tier 2  if the bus is STILL empty, the drive itself latched a fault that
#           only a POWER CYCLE clears. Guide the operator to power-cycle the
#           ART gripper, wait for confirmation, then auto re-run Tier 1 + verify.
#
# Usage:
#   bash ~/Polymetis_Franka_Teleop/bin/recover_gripper.sh           # interactive
#   bash ~/Polymetis_Franka_Teleop/bin/recover_gripper.sh --yes     # assume gripper
#                                                                    # was already
#                                                                    # power-cycled
# Exit 0 = gripper healthy (daemon pong + >=1 slave + Link UP).
set +e

ECAT_BIN="${ECAT_BIN:-/opt/etherlab/bin/ethercat}"
GRIPPER_REPO="${GRIPPER_REPO:-$HOME/Hyundai_motors_Gripper}"
RESTART_SH="$GRIPPER_REPO/scripts/restart_gripper.sh"
ASSUME_YES=0
for a in "$@"; do [ "$a" = "--yes" ] && ASSUME_YES=1; done

note() { printf "  %s\n" "$*"; }
hdr()  { printf "\n=== %s ===\n" "$*"; }

# ---- health probes ---------------------------------------------------------
ping_daemon() {  # 0 = pong
    python3 - <<'PY' 2>/dev/null
import socket, struct, sys
try:
    s = socket.create_connection(("127.0.0.1", 50053), timeout=3)
    s.sendall(bytes([0x01]) + struct.pack("<I", 0))
    st = s.recv(1)[0]; plen = struct.unpack("<I", s.recv(4))[0]
    reply = s.recv(plen) if plen else b""
    s.close()
    sys.exit(0 if (reply == b"pong" and st == 0) else 1)
except Exception:
    sys.exit(1)
PY
}
slave_count() { "$ECAT_BIN" master 2>/dev/null | awk -F: '/Slaves:/{gsub(/ /,"",$2); print $2; exit}'; }
link_up()     { "$ECAT_BIN" master 2>/dev/null | grep -qi 'Link: *UP'; }

diagnose() {   # prints status, returns 0 if fully healthy
    local pong=1 slaves link
    ping_daemon && pong=0
    slaves=$(slave_count); slaves=${slaves:-0}
    if link_up; then link="UP"; else link="DOWN"; fi
    note "daemon :50053  : $([ $pong -eq 0 ] && echo 'pong (alive)' || echo 'NO RESPONSE')"
    note "EtherCAT slaves: ${slaves}"
    note "EtherCAT link  : ${link}"
    [ $pong -eq 0 ] && [ "${slaves:-0}" -ge 1 ] && [ "$link" = "UP" ]
}

hdr "Tier 0: diagnose"
if diagnose; then
    note "gripper HEALTHY (daemon + bus). Nothing to do."
    exit 0
fi

hdr "Tier 1: software recovery (restart_gripper.sh)"
if [ ! -x "$RESTART_SH" ]; then
    note "restart helper not found/executable: $RESTART_SH"
    note "set GRIPPER_REPO=... or fix the path, then re-run."
    exit 2
fi
sudo bash "$RESTART_SH"
sleep 1
hdr "Re-diagnose after Tier 1"
if diagnose; then
    note "gripper RECOVERED via software restart."
    exit 0
fi

hdr "Tier 2: gripper drive needs a POWER CYCLE"
note "Daemon/EtherCAT were restarted but the bus is still empty (Slaves:0 / Link:DOWN)."
note "This means the ART gripper drive itself is off or latched a fault that only"
note "a power cycle clears. Software alone cannot fix this."
note ""
note "ACTION:"
note "  1. Power OFF the ART gripper, wait ~5s, power it back ON."
note "  2. Wait until its drive LED indicates ready."
if [ $ASSUME_YES -eq 0 ]; then
    note ""
    read -r -p "  Done power-cycling the gripper? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) note "Aborted. Re-run after power-cycling."; exit 3 ;; esac
fi
hdr "Tier 2: re-init after power cycle"
sudo bash "$RESTART_SH"
sleep 1
hdr "Final diagnose"
if diagnose; then
    note "gripper RECOVERED after power cycle + re-init."
    exit 0
fi
note "STILL unhealthy. Check physically:"
note "  * gripper power + drive LED"
note "  * EtherCAT cable into pro4000 NIC (ip link show => is the gripper NIC UP?)"
note "  * journalctl -u ethercat -n 40 ; journalctl -u art-gripper-daemon -n 40"
note "If the pro4000 EtherCAT master itself is wedged, a pro4000 reboot is the last resort"
note "(systemd brings ethercat + art-gripper-daemon back automatically on boot)."
exit 1
