#!/usr/bin/env bash
# install_nuc.sh -- one-shot NUC RT-tuning installer.
#
# Run on the NUC (NOT pro4000):
#   sudo bash install/install_nuc.sh
#
# What it does:
#   1. Copies install/nuc/sbin/*    -> /usr/local/sbin/   (chmod 755)
#   2. Copies install/nuc/systemd/* -> /etc/systemd/system/
#   3. Copies install/nuc/sudoers.d/franka_rt -> /etc/sudoers.d/  (chmod 440)
#   4. systemctl enable --now franka-rt-tune franka-dma-latency
#
# What it does NOT do (manual prerequisites, see docs/install_from_scratch.md
# Phase B and Phase C):
#   * PREEMPT_RT kernel install
#   * GRUB cmdline change (isolcpus=domain,managed_irq,6-7 nohz_full=6,7
#     rcu_nocbs=6,7 irqaffinity=0-5 intel_idle.max_cstate=0
#     processor.max_cstate=0)
#   * libfranka build
#   * Polymetis (fairo) build + polymetis-local conda env
#
# After this script completes, point the user at:
#   sudo bash /usr/local/sbin/start_franka_arm.sh
#
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[install_nuc] need root -- re-run with sudo" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/install/nuc"

echo "[install_nuc] copying sbin scripts..."
install -m 755 -t /usr/local/sbin/ \
    "$SRC/sbin/franka_rt_apply.sh" \
    "$SRC/sbin/franka_dma_latency.py" \
    "$SRC/sbin/franka_pin_helper.sh" \
    "$SRC/sbin/start_franka_arm.sh" \
    "$SRC/sbin/start_franka_gripper.sh"

echo "[install_nuc] copying systemd units..."
install -m 644 -t /etc/systemd/system/ \
    "$SRC/systemd/franka-rt-tune.service" \
    "$SRC/systemd/franka-dma-latency.service" \
    "$SRC/systemd/franka-realtime-setup.service"

echo "[install_nuc] copying sudoers drop-in..."
install -m 440 "$SRC/sudoers.d/franka_rt" /etc/sudoers.d/franka_rt
# visudo style sanity check
if command -v visudo >/dev/null 2>&1; then
    visudo -c -f /etc/sudoers.d/franka_rt
fi

echo "[install_nuc] reloading + enabling systemd units..."
systemctl daemon-reload
systemctl enable --now franka-rt-tune.service
systemctl enable --now franka-dma-latency.service
# franka-realtime-setup is optional sysctl tuning -- comment out the next line
# if you do not want kernel.nmi_watchdog=0 etc.
systemctl enable --now franka-realtime-setup.service || true

echo
echo "[install_nuc] OK -- installed scripts + units"
echo
echo "Verify:"
echo "  systemctl status franka-rt-tune franka-dma-latency"
echo "  cat /sys/class/net/enp86s0/queues/rx-0/rps_cpus       # expect: f000"
echo "  cat /sys/devices/system/cpu/intel_pstate/no_turbo     # expect: 1"
echo "  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # expect: performance"
echo
echo "Then bring up Polymetis:"
echo "  sudo bash /usr/local/sbin/start_franka_arm.sh"
