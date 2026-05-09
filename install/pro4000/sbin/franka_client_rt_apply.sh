#!/bin/bash
# franka_client_rt_apply.sh -- pro4000 (client side) RT tuning.
#
# pro4000 runs the Polymetis client (FrankaInterpolationController),
# the ART gripper controller, ZED camera workers, the Vive teleop
# process, the main demo loop, and the cv2 viewer subprocess. Default
# Linux scheduling can put any of these on cores that also handle the
# NIC IRQ for the NUC-subnet link, causing the gRPC update_desired_*
# calls to be delayed by softirq processing -> NUC libfranka misses
# the 1 ms FCI window -> communication_constraints_violation reflex.
#
# This script:
#   1. Pins enp130s0 (NUC subnet, 192.168.1.20) NIC IRQs to cores 0,1
#      so softirq RX/TX handling never lands on the cores our Python
#      controllers run on.
#   2. Configures RPS / XPS to match (steers softirq onto IRQ cores).
#   3. Forces all CPUs into 'performance' governor (no DVFS jitter).
#   4. Stops irqbalance so our manual pin sticks.
#   5. (Hybrid CPU note) Intel Core Ultra 5 245K is 14 P-cores (no HT) +
#      no E-cores. We have plenty of headroom; we just need to keep the
#      two NIC handler cores (0,1) away from Python compute.
#
# What this script does NOT do (handled at process level via
# os.sched_setaffinity inside the Python multiprocessing children):
#   * Pinning FrankaInterpolationController to cores 6,7
#   * Pinning ArtGripperController to cores 8,9
#   * Pinning ZED camera workers to cores 10,11
#
# Idempotent. Called by franka-client-rt-tune.service at boot.
set -u
NIC="enp130s0"                # NUC subnet NIC (find with `ip -br addr | grep 192.168.1`)
declare -a IRQ_CORES=(0 1)    # dedicate cores 0,1 to NIC handling
RPS_MASK="0003"               # cores 0,1 bitmask

# [A] NIC IRQ -> dedicated cores
i=0
for irq in $(grep "$NIC" /proc/interrupts | awk '{print $1}' | tr -d ':'); do
    [ -f /proc/irq/$irq/smp_affinity_list ] && \
        echo "${IRQ_CORES[$((i % ${#IRQ_CORES[@]}))]}" > /proc/irq/$irq/smp_affinity_list && \
        i=$((i+1))
done

# [B] RPS / XPS / per-queue flow_cnt
for q in /sys/class/net/$NIC/queues/rx-*/rps_cpus;     do echo "$RPS_MASK" > "$q" 2>/dev/null; done
for q in /sys/class/net/$NIC/queues/tx-*/xps_cpus;     do echo "$RPS_MASK" > "$q" 2>/dev/null; done
for q in /sys/class/net/$NIC/queues/rx-*/rps_flow_cnt; do echo 4096        > "$q" 2>/dev/null; done
echo 32768 > /proc/sys/net/core/rps_sock_flow_entries 2>/dev/null

# [C] CPU governor performance everywhere
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$c" 2>/dev/null || true
done

# [D] NIC ring buffer to max
ethtool -G $NIC rx 4096 tx 4096 2>/dev/null || true

# [E] irqbalance OFF (otherwise it overrides our manual affinity on the next rebalance)
systemctl is-active --quiet irqbalance && systemctl stop irqbalance 2>/dev/null || true

logger "franka-client-rt-tune: applied (NIC ${NIC} IRQ -> cores ${IRQ_CORES[*]}, governor=performance, irqbalance stopped)"
echo "OK -- NIC ${NIC} IRQ ${IRQ_CORES[*]}, RPS/XPS ${RPS_MASK}, governor=performance"
