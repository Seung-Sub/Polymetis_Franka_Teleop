#!/bin/bash
# franka_rt_apply.sh — boot-time RT tuning for the Franka FCI link.
#
# Idempotent. Called once at boot via the franka-rt-tune.service unit.
# Pins the NIC IRQs to E-cores, enables RPS/XPS, forces the CPU governor
# to performance, disables Turbo (Intel pstate) + PCIe ASPM, raises the
# NIC ring buffer, and herds the GUI processes off the realtime cores.
#
# Tunables — adjust if your NIC name or RT cores differ:
#   NIC          : Ethernet interface attached to the Franka box (172.16.0.x)
#   IRQ_E_CORES  : list of E-cores to receive NIC IRQs
#   RPS_E_MASK   : same E-cores expressed as a CPU bitmask (hex)
#   GUI_RANGE    : P-cores reserved for GUI (mass-taskset target)
#
# Defaults below are KIST i7-1360P (4 P-core HT + 8 E-core; isolcpus=6,7).
set -u
NIC="enp86s0"
declare -a IRQ_E_CORES=(12 13 14 15)
RPS_E_MASK="f000"
GUI_RANGE="0-3"

# [A] NIC IRQ -> E-cores (round-robin across IRQ_E_CORES)
i=0
for irq in $(grep "$NIC" /proc/interrupts | awk '{print $1}' | tr -d ':'); do
    [ -f /proc/irq/$irq/smp_affinity_list ] && \
        echo "${IRQ_E_CORES[$((i % ${#IRQ_E_CORES[@]}))]}" > /proc/irq/$irq/smp_affinity_list && \
        i=$((i+1))
done

# [B] RPS / XPS / per-queue flow_cnt
for q in /sys/class/net/$NIC/queues/rx-*/rps_cpus;       do echo "$RPS_E_MASK" > "$q" 2>/dev/null; done
for q in /sys/class/net/$NIC/queues/tx-*/xps_cpus;       do echo "$RPS_E_MASK" > "$q" 2>/dev/null; done
for q in /sys/class/net/$NIC/queues/rx-*/rps_flow_cnt;   do echo 4096          > "$q" 2>/dev/null; done
echo 32768 > /proc/sys/net/core/rps_sock_flow_entries 2>/dev/null

# [C] CPU governor = performance (every core)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$c" 2>/dev/null || true; done

# [D] PCIe ASPM = performance
echo performance > /sys/module/pcie_aspm/parameters/policy 2>/dev/null || true

# [E] Turbo Boost OFF (intel_pstate)
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

# [F] NIC ring buffer to max
ethtool -G $NIC rx 4096 tx 4096 2>/dev/null || true

# [G] GUI processes -> P-core 0-3 (best effort — they may not all be running yet)
for proc in firefox firefox-bin Isolated gnome-shell Xorg gdm pulseaudio; do
    for pid in $(pgrep -u kist $proc 2>/dev/null) $(pgrep -u root $proc 2>/dev/null); do
        taskset -cpa "$GUI_RANGE" "$pid" >/dev/null 2>&1 || true
    done
done

# [H] irqbalance OFF (otherwise it overrides our manual affinity)
systemctl is-active --quiet irqbalance && systemctl stop irqbalance 2>/dev/null || true

logger "franka-rt-tune: applied (IRQ->E-cores, RPS/XPS, governor, ASPM perf, no_turbo, ring max, GUI->0-3)"
echo "OK -- NIC IRQ ${IRQ_E_CORES[*]}, RPS/XPS ${RPS_E_MASK}, no_turbo=1, ASPM=perf, ring=4096"
