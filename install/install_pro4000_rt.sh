#!/usr/bin/env bash
# install_pro4000_rt.sh -- one-shot pro4000 RT-tuning installer.
#
# Run on pro4000:
#   sudo bash install/install_pro4000_rt.sh
#
# What it does:
#   1. Copies install/pro4000/sbin/franka_client_rt_apply.sh -> /usr/local/sbin/
#   2. Copies install/pro4000/systemd/franka-client-rt-tune.service -> /etc/systemd/system/
#   3. systemctl enable --now franka-client-rt-tune
#   4. Drops /etc/security/limits.d/franka_client_rt.conf with rtprio + nice
#      so apply_realtime() can negative-nice without root.
#
# Manual prereq the script can't do:
#   * /etc/sudoers.d/franka_client_rt (granting setcap) — only install if you
#     want the demo to be able to elevate to SCHED_RR mid-process. CPU pinning
#     alone (the default) already eliminates the reflex storm seen at KIST
#     2026-05-09.
#
# What does NOT need RT tuning on pro4000 (different from NUC):
#   * No PREEMPT_RT kernel needed. PREEMPT_DYNAMIC + CPU pinning + governor
#     performance is sufficient because the timing budget here is ~10 ms
#     (10 Hz / 15 Hz / 60 Hz / 100 Hz loops), not the 1 ms libfranka inner
#     loop. NUC handles the tight timing, pro4000 just needs to deliver
#     fresh commands within 10 ms slack.
#   * No GRUB isolcpus needed. CPU affinity at the Python multiprocessing
#     level (PRO4000_CORE_MAP in common/realtime_util.py) is enough.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[install_pro4000_rt] need root -- re-run with sudo" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/install/pro4000"

echo "[install_pro4000_rt] copying sbin script..."
install -m 755 "$SRC/sbin/franka_client_rt_apply.sh" /usr/local/sbin/

echo "[install_pro4000_rt] copying systemd unit..."
install -m 644 "$SRC/systemd/franka-client-rt-tune.service" /etc/systemd/system/

echo "[install_pro4000_rt] writing /etc/security/limits.d/franka_client_rt.conf..."
TARGET_USER="${SUDO_USER:-kist}"
cat > /etc/security/limits.d/franka_client_rt.conf <<EOF
# Lets ${TARGET_USER} use rtprio (SCHED_RR/FIFO up to 50) and negative nice
# (down to -15). Required by polymetis_franka_teleop.common.realtime_util.
${TARGET_USER}    -    rtprio    50
${TARGET_USER}    -    nice      -15
${TARGET_USER}    -    memlock   unlimited
EOF
chmod 644 /etc/security/limits.d/franka_client_rt.conf

# systemd drop-in override — without this, processes inside user@uid.service
# slice (i.e. anything launched from a desktop session terminal) inherit
# systemd's compile-time DefaultLimitRTPRIO=0 *despite* PAM limits, because
# systemd applies its slice limits AFTER pam_limits.so. Symptom: ssh-to-pro4000
# gives ulimit -r=50 but a gnome-terminal on the desktop shows ulimit -r=0.
# Set the system-wide and user-level defaults to 50/-15/infinity so the
# discrepancy goes away. Takes effect on the next user@.service startup
# (i.e. next desktop login or reboot).
echo "[install_pro4000_rt] writing /etc/systemd/system.conf.d/franka_rt.conf + user.conf.d..."
mkdir -p /etc/systemd/system.conf.d /etc/systemd/user.conf.d
cat > /etc/systemd/system.conf.d/franka_rt.conf <<'EOF'
[Manager]
DefaultLimitRTPRIO=50
DefaultLimitNICE=-15
DefaultLimitMEMLOCK=infinity
EOF
cp /etc/systemd/system.conf.d/franka_rt.conf /etc/systemd/user.conf.d/franka_rt.conf
chmod 644 /etc/systemd/system.conf.d/franka_rt.conf /etc/systemd/user.conf.d/franka_rt.conf
systemctl daemon-reexec

echo "[install_pro4000_rt] reloading + enabling systemd unit..."
systemctl daemon-reload
systemctl enable --now franka-client-rt-tune.service

echo
echo "[install_pro4000_rt] OK"
echo
echo "Verify:"
echo "  systemctl status franka-client-rt-tune"
echo "  cat /sys/class/net/enp130s0/queues/rx-0/rps_cpus  # expect 0003"
echo "  for i in \$(grep enp130s0 /proc/interrupts|awk '{print\$1}'|tr -d ':');do echo \"IRQ \$i: \$(cat /proc/irq/\$i/smp_affinity_list)\"; done"
echo "  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # expect performance"
echo
echo "Then re-login (so limits.conf takes effect for the user) and re-launch"
echo "the demo. The Python controllers will now pin themselves via"
echo "polymetis_franka_teleop.common.realtime_util.apply_realtime()."
