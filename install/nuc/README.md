# NUC install artifacts

These files live on the **NUC** (the realtime PC connected directly to the
Franka Controller Box). They are versioned here so that a fresh NUC can be
brought up with one `sudo bash install/install_nuc.sh` from a clone of this
repo.

| File | Where it goes | What it does |
|---|---|---|
| `sbin/franka_rt_apply.sh`        | `/usr/local/sbin/`  | Boot-time NIC IRQ pinning, RPS/XPS, governor, ASPM, Turbo OFF, ring buffer max, GUI eviction |
| `sbin/franka_dma_latency.py`     | `/usr/local/sbin/`  | Long-lived process holding `/dev/cpu_dma_latency` at 0us (forbids deep C-states) |
| `sbin/franka_pin_helper.sh`      | `/usr/local/sbin/`  | Post-launch `taskset` for `run_server` / `franka_panda_client` / `franka_hand_client` to cores 6,7 |
| `sbin/start_franka_arm.sh`       | `/usr/local/sbin/`  | Wrapper: activate `polymetis-local` conda env, schedule pin helper, exec `launch_robot.py` |
| `sbin/start_franka_gripper.sh`   | `/usr/local/sbin/`  | Same pattern for `launch_gripper.py` (Franka Hand only) |
| `systemd/franka-rt-tune.service` | `/etc/systemd/system/` | `oneshot` running `franka_rt_apply.sh` after `network-online.target` |
| `systemd/franka-dma-latency.service` | `/etc/systemd/system/` | `simple` running the DMA-latency holder, `Restart=always` |
| `systemd/franka-realtime-setup.service` | `/etc/systemd/system/` | Optional: `sysctl` knobs (RT throttling off, NMI watchdog off, TCP low-latency) |
| `sudoers.d/franka_rt`            | `/etc/sudoers.d/`   | Lets the regular user run `franka_pin_helper.sh` without a password (so the wrappers work) |

## Manual prerequisites that the installer cannot do

| Step | Why it can't be scripted | How to do it |
|---|---|---|
| PREEMPT_RT kernel | Kernel build / package install -- one-time | See [Phase B in `docs/install_from_scratch.md`](../../docs/install_from_scratch.md#phase-b-nuc-os-base--preempt_rt-kernel) |
| GRUB `isolcpus=domain,managed_irq,6-7 nohz_full=6,7 rcu_nocbs=6,7 irqaffinity=0-5 intel_idle.max_cstate=0 processor.max_cstate=0` | Boot parameter; reboot required | Edit `/etc/default/grub`, `sudo update-grub`, reboot |
| libfranka                                                              | Apt-pinned to a libfranka version that matches the Franka Controller Box | `sudo apt install libfranka` (or build from `frankaemika/libfranka`) |
| Polymetis (`fairo`) build + `polymetis-local` conda env                | Heavy build, drives `polymetis-local` env layout | [Phase C in `docs/install_from_scratch.md`](../../docs/install_from_scratch.md#phase-c-nuc-polymetis-build) |

## Tunables -- if your NIC name or RT cores differ

Edit `sbin/franka_rt_apply.sh` and `sbin/franka_pin_helper.sh` before installing:

```bash
# franka_rt_apply.sh
NIC="enp86s0"                  # ip link show | look for the Franka 172.16.0.x NIC
declare -a IRQ_E_CORES=(12 13 14 15)   # E-cores that receive NIC IRQs
RPS_E_MASK="f000"              # same E-cores expressed as a CPU bitmask
GUI_RANGE="0-3"                # P-cores reserved for GUI (away from RT)

# franka_pin_helper.sh
PIN_CORES="6,7"                # the cores you isolcpus'd
```

The KIST baseline assumes `isolcpus=6,7` (P-core 3 of the i7-1360P) +
`IRQ_E_CORES=12-15` (the 4 E-cores). If your CPU has a different P/E layout,
look at `lscpu --extended` and pick analogous cores.

## Verify

After `sudo bash install/install_nuc.sh` finishes:

```bash
systemctl status franka-rt-tune franka-dma-latency
cat /sys/class/net/enp86s0/queues/rx-0/rps_cpus            # expect: f000
cat /sys/devices/system/cpu/intel_pstate/no_turbo          # expect: 1
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # expect: performance
for irq in $(grep enp86s0 /proc/interrupts | awk '{print $1}' | tr -d ':'); do
    echo "IRQ $irq: $(cat /proc/irq/$irq/smp_affinity_list)"
done
```

Then bring up Polymetis:

```bash
sudo bash /usr/local/sbin/start_franka_arm.sh
# In another terminal (only if Franka Hand workflow):
sudo bash /usr/local/sbin/start_franka_gripper.sh
```

When `start_franka_arm.sh` prints
`[arm pinner] cores 6,7 pin applied (details: tail .../franka_pin_arm.log)`
the RT thread is on the isolated cores and Polymetis is ready for clients
on `pro4000` to connect to `192.168.1.12:50051`.
