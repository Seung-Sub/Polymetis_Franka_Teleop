# Polymetis_Franka_Teleop — Install from scratch

Hardware-only → fully working Vive teleop + data collection in one pass.
Modeled on the same phase structure as the sister Isaac-GR00T install
guide; the two share the NUC realtime setup so a NUC tuned for one is
ready for the other.

## Table of contents

- [§0 How to use this document](#0-how-to-use-this-document)
- [§1 Hardware bill of materials](#1-hardware-bill-of-materials)
- [§2 System topology + networking](#2-system-topology--networking)
- [§A Franka Desk pre-config](#phase-a--franka-desk-pre-config)
- [§B NUC OS base + PREEMPT_RT kernel](#phase-b-nuc-os-base--preempt_rt-kernel)
- [§C NUC Polymetis build](#phase-c-nuc-polymetis-build)
- [§D NUC RT permanent tuning (KIST scripts, included in this repo)](#phase-d-nuc-rt-permanent-tuning)
- [§E pro4000 OS base + GPU](#phase-e-pro4000-os-base--gpu)
- [§F pro4000 ZED SDK + cameras](#phase-f-pro4000-zed-sdk--cameras)
- [§G pro4000 conda env (`groot-client`) + Polymetis client](#phase-g-pro4000-groot-client-env)
- [§H pro4000 ART gripper daemon](#phase-h-pro4000-art-gripper-daemon)
- [§I pro4000 SteamVR + vive_input](#phase-i-pro4000-steamvr--vive_input)
- [§J Verification sequence (incremental)](#phase-j-verification-sequence)
- [§13 Troubleshooting catalog](#13-troubleshooting-catalog)

---

## 0. How to use this document

### Recommended order

1. Read §1, §2 — confirm you have the hardware.
2. §A (Franka Desk web UI, one-time, in a browser) — independent of any host.
3. §B → §C → §D — NUC, in front of its monitor (ssh comes later).
4. §E → §F → §G → §H → §I — pro4000, monitor or ssh.
5. §J — incremental verification, do **not** skip.
6. Once §J passes, daily use lives in [`USAGE.md`](USAGE.md).

### Conventions

- Each command block runs on **one host**. The host is named in the section heading: `(NUC)` or `(pro4000)`.
- `<TODO>` placeholders mean "fill in for your environment".
- Numbers in the "Verify" column of §J are KIST measurements; yours should be in the same ballpark.

---

## 1. Hardware bill of materials

| Role | KIST baseline (verified) | Minimum / acceptable substitute |
|---|---|---|
| Robot | Franka Panda + Franka Controller Box (172.16.0.2) + ART 2-finger gripper *or* Franka Hand | Any FCI-licensed Franka Panda |
| **Realtime PC (NUC)** | Intel i7-1360P (4 P-core HT + 8 E-core), 64 GB RAM, 2 NICs (Franka direct + Lab) | Any Intel hybrid CPU, PREEMPT_RT-capable, ≥2 NICs |
| **Inference / data PC (pro4000)** | RTX PRO 4000 Blackwell SFF 24 GB sm_120 (verified) — RTX 3090 also verified | RTX 3090 / 4090 / Pro 4000–6000; 64 GB RAM; Ubuntu 22.04 + recent CUDA |
| Exterior camera | ZED 2i (S/N `33538770`) | ZED 2 / 2i (DROID baseline) |
| Wrist camera | ZED Mini (S/N `11667817`) | ZED Mini |
| USB cables (cameras) | USB 3.0 SuperSpeed C-C | USB 3.0 only — USB 2.0 cables enumerate only the HID side, UVC will fail |
| Teleop input | HTC Vive controller + 2 Lighthouse base stations + (HMD or tracker dummy) | HTC Vive Pro / 1.0 controllers; Index works too if you re-bind buttons |
| EtherCAT NIC (ART gripper) | `enxb0386cf13036` direct to NETX 90-RE/ECS | Any IgH-EtherCAT-compatible NIC |
| Networking | 1 Gbps NIC NUC↔Franka (direct), 1 Gbps NIC NUC↔pro4000 (lab subnet) | FCI guideline: ≤1 ms RTT |

The Vive controller in particular is the dominant constraint: this repo
expects the C++ binary `vive_input` (built from the user's `vive_ws`) to
publish the controller pose over TCP `:12345`. SteamVR Linux can be flaky
on first boot — see §I for the headless `vrserver --keepalive` workaround.

---

## 2. System topology + networking

```
[Franka Panda + ART (or Franka Hand)]
    │ EtherCAT
    ▼
[Franka Controller Box  172.16.0.2]
    │ 1 Gbps  (FCI ≤1 ms RTT)
    ▼
[NUC]
  enp86s0  172.16.0.1/24    ← Franka FCI (direct, dedicated NIC)
  enp87s0  192.168.1.12/24  ← Lab subnet (talks to pro4000)
    │
    │ 1 Gbps  (≤0.2 ms RTT)
    ▼
[pro4000]                                   (kist-eval, 161.122.114.90)
  enp130s0          192.168.1.20/24  ← Lab subnet
  enxb0386cf13036   161.122.115.1    ← EtherCAT direct to ART (or empty for Franka Hand workflow)
  USB 3.0           ZED 2i (sn 33538770)
  USB 3.0           ZED Mini (sn 11667817)
  USB 2.0           HTC Vive controller (via Lighthouse)
```

Ports the stack listens on:

| Service | Where | Port | Notes |
|---|---|---|---|
| Polymetis arm gRPC | NUC | 50051 | `start_franka_arm.sh` |
| Polymetis franka_hand gRPC | NUC | 50052 | `start_franka_gripper.sh` (skip for ART) |
| ZeroRPC unified bridge (optional) | NUC | 4242 | `bin/start_unified_bridge_on_nuc.sh` |
| ART gripper daemon | pro4000 | 50053 | systemd, auto-start |
| `vive_input` | pro4000 | 12345 (TCP) / 12346 (UDP) | from `vive_ws` |

> **Why NUC↔Franka must be direct** (not through a switch): switches add
> jitter and an extra IRQ path that's hard to keep off the realtime cores.
> Already-verified at KIST: 172.16.0.2 ↔ 172.16.0.1 RTT 0.13 ms over a
> direct cable, ~0.6 ms with a small consumer switch.

---

## Phase A — Franka Desk pre-config

(Browser only, one-time per Franka Controller Box.)

1. From any PC on the lab subnet, open `https://172.16.0.2/desk/` and accept the self-signed certificate.
2. Log in with the Franka account printed on the controller box.
3. **Unlock joints** — click the grey indicator until it turns blue ("Joints brake released").
4. **End-Effector → Settings → Custom EE** (because we run ART, not Franka Hand). For ART + ZED Mini wrist, use these (re-measure after any swap with `~/Isaac-GR00T/scripts/kist/measure_ee_load.py`):

| Field | Value |
|---|---|
| Mass | 1.05 kg |
| Flange→CoM | (0, 0.010, 0.100) m |
| Flange→TCP | (0, 0, 0.216, 0, 0, 0) — finger tip |
| Inertia diag | (0.004, 0.004, 0.001) |

For Franka Hand workflow, just pick the built-in **Franka Hand** EE.

5. **Activate FCI** — top-right menu → Activate FCI. Status pill should turn green.
6. e-stop released, base mounted, no obstacles inside the workspace.

> If FCI Activate is greyed out: the EE selection is wrong, or the EE
> is not physically attached. Re-seat the connector and reboot the
> controller box. Catalog #11 in §13.

---

## Phase B — NUC OS base + PREEMPT_RT kernel

```bash
# (Ubuntu 22.04.5 LTS installed.)
# PREEMPT_RT 6.8.0-rt8 kernel — install per Franka's guide:
#   https://frankaemika.github.io/docs/installation_linux.html
# Or via Ubuntu Pro -> ubuntu-realtime metapackage.

uname -a   # must contain 'PREEMPT_RT'  (KIST: 6.8.0-rt8 SMP PREEMPT_RT)
```

### B-2. GRUB cmdline (`/etc/default/grub`)

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_idle.max_cstate=0 processor.max_cstate=0 isolcpus=domain,managed_irq,6-7 nohz_full=6,7 rcu_nocbs=6,7 irqaffinity=0-5"
```

`isolcpus=6-7` reserves the second P-core (both hyperthreads) of the
i7-1360P for realtime work. `managed_irq` keeps managed device IRQs
(NVMe, etc.) off those cores. `nohz_full` + `rcu_nocbs` stop the timer
tick + RCU callbacks on those cores, and `irqaffinity=0-5` is the
fallback for any IRQs we did not explicitly pin in §D.

```bash
sudo update-grub
sudo reboot
# After reboot:
cat /proc/cmdline   # verify the parameters above
```

### B-3. libfranka

```bash
sudo apt install -y libfranka                 # 0.15.0 verified on KIST
# or, if the apt pin is wrong for your controller box version:
git clone --recursive https://github.com/frankaemika/libfranka.git ~/libfranka
cd ~/libfranka && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j && sudo make install
```

### B-4. SSH key from pro4000 to NUC

The pro4000 helper scripts (`bin/start_unified_bridge_on_nuc.sh`,
`scripts/kist/cleanup_all.sh` in Isaac-GR00T) ssh into the NUC. Set up
key-based auth once:

```bash
# On pro4000:
ssh-copy-id kist@192.168.1.12
ssh kist@192.168.1.12 'echo OK'   # must print OK with no password prompt
```

---

## Phase C — NUC Polymetis build

NUC-side, ~30–60 min.

```bash
# C-1. miniconda3 (skip if already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh        # installs to ~/miniconda3

# C-1b. build deps
sudo apt install -y build-essential cmake git libpoco-dev libeigen3-dev \
    pkg-config python3-dev libhidapi-dev libusb-1.0-0-dev

# C-2. fairo (Polymetis source)
git clone --recursive https://github.com/facebookresearch/fairo.git ~/fairo
cd ~/fairo && git submodule update --init --recursive

# C-3. polymetis-local conda env
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/fairo/polymetis
conda env create -f polymetis/environment.yml -n polymetis-local
conda activate polymetis-local

# C-4. build
mkdir -p polymetis/build && cd polymetis/build
cmake .. -DBUILD_FRANKA=ON
make -j$(nproc)
ls polymetis/build/run_server polymetis/build/franka_panda_client polymetis/build/franka_hand_client
ls polymetis/build/torch_isolation/lib*.so

# C-5. polymetis Python package
cd ~/fairo/polymetis
pip install -e .
python -c "from polymetis import RobotInterface; print('OK')"

# C-6. ping the Franka box
ping -c 3 172.16.0.2     # expect <1 ms RTT
```

### C-7. Smoke test (Polymetis on its own, no motion)

```bash
# Terminal A: arm server
conda activate polymetis-local
cd ~/fairo/polymetis/polymetis/python/scripts
python launch_robot.py robot_client=franka_hardware ip=0.0.0.0 port=50051
# wait for "Connected" (one sudo prompt)

# Terminal B: gripper server (only Franka Hand workflow)
conda activate polymetis-local
python launch_gripper.py gripper=franka_hand
# Connecting to robot_ip 172.16.0.2 -> listening
```

Ctrl-C both terminals when done — we will run them via the wrappers from §D from now on.

---

## Phase D — NUC RT permanent tuning

> **This phase is what makes Polymetis stable.** Skip it and you will see
> reflex trips on contact (`communication_constraints_violation`,
> `success_rate ≈ 0.79`) within seconds of starting any client.

### D-1. Why each tunable matters (KIST measurements)

| Knob | Default | What goes wrong | Fixed by |
|---|---|---|---|
| `isolcpus=6,7` | none | RT thread shares cores with GUI / Chrome / Slack | GRUB §B-2 |
| `taskset -cpa 6,7` of `run_server` & `franka_panda_client` | not done | `isolcpus` only *prevents* default scheduling; it doesn't *pull* threads in. RT threads still spawn on whatever core | `franka_pin_helper.sh` (this phase) |
| NIC IRQ affinity | spread across P-cores 0-5 | NIC IRQ competes with GUI on P-core; success_rate drops to 0.79 → reflex | `franka_rt_apply.sh` pins to E-cores 12-15 |
| RPS / XPS | off | Softirq RX/TX bounces between cores; cache-cold packets | `franka_rt_apply.sh` sets RPS/XPS mask |
| Turbo Boost | on | P-core frequency oscillates → RT jitter | `franka_rt_apply.sh` writes `no_turbo=1` |
| PCIe ASPM | default | NIC wake-up adds tens of µs | `franka_rt_apply.sh` sets policy=performance |
| GUI cpuset | unbounded | Firefox / GNOME shell drift onto cores 6,7 (they're "isolated", not "forbidden") | `franka_rt_apply.sh` mass-tasksets GUI to 0-3 |
| `/dev/cpu_dma_latency` | open | Cores enter deep C-states between 1 kHz iterations | `franka_dma_latency.py` holds it at 0us |
| `irqbalance` | running | Periodically rebalances IRQs, undoing our pin | `franka_rt_apply.sh` stops the unit |

### D-2. One-shot install from this repo

The 9 scripts + 2 systemd units + 1 sudoers drop-in are in
[`install/nuc/`](../install/nuc/). The wrapper [`install/install_nuc.sh`](../install/install_nuc.sh)
copies them into place and enables both services:

```bash
# (NUC) clone this repo (just for the install dir; runtime lives elsewhere)
git clone https://github.com/Seung-Sub/Polymetis_Franka_Teleop.git ~/Polymetis_Franka_Teleop
cd ~/Polymetis_Franka_Teleop

# Edit the user name in install/nuc/sudoers.d/franka_rt if it isn't 'kist'
sudo bash install/install_nuc.sh
```

The script:
1. Copies `install/nuc/sbin/*` → `/usr/local/sbin/` (mode 755).
2. Copies `install/nuc/systemd/*.service` → `/etc/systemd/system/`.
3. Copies `install/nuc/sudoers.d/franka_rt` → `/etc/sudoers.d/` (mode 440), with `visudo -c` validation.
4. `systemctl enable --now franka-rt-tune franka-dma-latency` (and optionally `franka-realtime-setup`).

### D-3. Verify

```bash
# IRQ -> E-cores 12-15
for irq in $(grep enp86s0 /proc/interrupts | awk '{print $1}' | tr -d ':'); do
    echo "IRQ $irq: $(cat /proc/irq/$irq/smp_affinity_list)"
done

# RPS mask
cat /sys/class/net/enp86s0/queues/rx-0/rps_cpus            # f000

# governor / turbo / ASPM
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # performance
cat /sys/devices/system/cpu/intel_pstate/no_turbo          # 1
cat /sys/module/pcie_aspm/parameters/policy                # [performance]

# systemd
systemctl status franka-rt-tune franka-dma-latency
```

### D-4. Run Polymetis through the wrappers (from now on, every session)

```bash
# (NUC, terminal 1) arm
sudo bash /usr/local/sbin/start_franka_arm.sh
# 5 s after launch_robot.py prints "Connected", you should see:
#   [arm pinner] cores 6,7 pin applied (details: tail ~/.franka_logs/franka_pin_arm.log)

# (NUC, terminal 2)  ONLY for Franka Hand workflow:
sudo bash /usr/local/sbin/start_franka_gripper.sh
```

If the pinner line doesn't appear, the install was incomplete — re-run `install_nuc.sh` and reboot the NUC.

---

## Phase E — pro4000 OS base + GPU

```bash
# Ubuntu 22.04, NVIDIA driver + CUDA pre-installed (manufacturer image works).
nvidia-smi   # KIST: RTX PRO 4000 Blackwell SFF, driver 580.126.09, CUDA 13
lsb_release -a   # 22.04 jammy

# anaconda or miniconda at $HOME/anaconda3 (or $HOME/miniconda3)
ls $HOME/anaconda3/bin/conda || ls $HOME/miniconda3/bin/conda
```

This repo does **not** require the `groot` (training) env; only `groot-client`
(client-side Polymetis + ZED + Vive). If you've already done Isaac-GR00T's
INSTALL_FROM_SCRATCH §7, the training env is a bonus, not a prerequisite.

### E-2. pro4000 RT tuning (one-shot install)

Without this, the Franka client experiences NIC IRQ contention on
`enp130s0` (the NUC subnet), `FrankaInterpolationController` floats across
cores under the default scheduler, and `update_desired_joint_positions`
calls miss libfranka's 1 ms FCI window — leading to
`communication_constraints_violation` reflex storms (catalog #25).

```bash
cd ~/Polymetis_Franka_Teleop
sudo bash install/install_pro4000_rt.sh
```

This installs:
- `/usr/local/sbin/franka_client_rt_apply.sh` → systemd `franka-client-rt-tune.service`
  pins enp130s0 NIC IRQs to cores 0-1, sets governor=performance, stops irqbalance.
- `/etc/security/limits.d/franka_client_rt.conf` → `kist` user gets
  rtprio=50, nice=-15, memlock=unlimited.
- `/etc/systemd/system.conf.d/franka_rt.conf` + user.conf.d → systemd
  default RTPRIO/NICE/MEMLOCK match (otherwise systemd's user@*.service
  slice caps desktop-terminal RTPRIO at 0 — see catalog #33).

After install, log out from the desktop and back in (or reboot once) so
the new limits propagate to the desktop session. Verify:

```bash
ulimit -r            # expect 50 (was 0 before)
systemctl is-active franka-client-rt-tune     # active
cat /sys/class/net/enp130s0/queues/rx-0/rps_cpus    # expect 0003 (cores 0,1)
```

The Python multiprocessing children (FrankaInterpolationController,
ArtGripperController, etc.) self-pin to dedicated cores via
`polymetis_franka_teleop/common/realtime_util.py` at process start —
no per-process taskset commands needed.

---

## Phase F — pro4000 ZED SDK + cameras

### F-1. ZED SDK 5.3

```bash
mkdir -p ~/Downloads && cd ~/Downloads
curl -L -o ZED_SDK.run \
    'https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.3/ZED_SDK_Ubuntu22_cuda13.0_tensorrt10.13_v5.3.0.zstd.run'
chmod +x ZED_SDK.run
sudo bash ZED_SDK.run --silent --skip_python --skip_cuda
sudo chmod -R a+rX /usr/local/zed
```

If the silent installer hangs on `pam-auth-update` (some Ubuntu images do this):

```bash
sudo killall pam-auth-update
sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a
```

### F-2. Camera calibration files

Each ZED needs its serial-number-keyed `.conf` to be on disk locally
(otherwise pyzed tries to download it via its built-in libcurl, which
fights with the conda LD-library overrides — see catalog #22):

```bash
sudo mkdir -p /usr/local/zed/settings
for SN in 33538770 11667817; do
    sudo curl -fsSL -o /usr/local/zed/settings/SN${SN}.conf \
        "https://calib.stereolabs.com/?SN=${SN}"
done
ls /usr/local/zed/settings/
```

### F-3. USB enumeration check

```bash
lsusb | grep -iE 'stereolabs|zed'
# Expect:
#   2b03:f880 STEREOLABS ZED 2i           (UVC, USB 3.0)
#   2b03:f682 STEREOLABS ZED-M camera     (UVC, USB 3.0)
#   2b03:f881 STEREOLABS ZED-2i HID
#   2b03:f681 STEREOLABS ZED-M HID

lsusb -t | grep -B1 5000M    # both UVCs must show 5 Gbps
```

If only the HID side appears, you have a USB 2.0 cable masquerading as 3.0.
Replace it. (`dmesg | grep -i "USB cable is bad"` confirms.)

---

## Phase G — pro4000 `groot-client` env

Same env Isaac-GR00T uses, so the two repos can coexist with no env switching.

```bash
cd ~/Polymetis_Franka_Teleop
sudo apt install -y ffmpeg                # H264 mux for video_recorder.py
bash install/install_pro4000.sh           # creates / updates groot-client
```

`install_pro4000.sh`:
1. Creates conda env `groot-client` (Python 3.8) if absent.
2. `pip install` torch 1.13.1, grpcio, hydra, zarr<3, av, opencv, zerorpc, pynput, etc.
3. `pip install -e .` for this repo.
4. `pip install -e ~/Hyundai_motors_Gripper/python` for the ART client (skipped if not present).

If you don't have an existing Polymetis client artifact tree on pro4000,
also follow Isaac-GR00T's INSTALL_FROM_SCRATCH §9 (Phase G) — that step
rsyncs the NUC's `fairo-polymetis/build/torch_isolation/` and
`nuc_libs/` to pro4000 so `from polymetis import RobotInterface` works
on the client side. See `Polymetis_Franka_Teleop/docs/troubleshooting.md`
catalog #6 for the LD_LIBRARY_PATH pitfalls.

---

## Phase H — pro4000 ART gripper daemon

The Hyundai ART gripper has its own ROS-free EtherCAT daemon, shipped
in the [`Hyundai_motors_Gripper`](https://github.com/Seung-Sub/Hyundai_motors_Gripper)
sister repo.

```bash
git clone https://github.com/Seung-Sub/Hyundai_motors_Gripper.git ~/Hyundai_motors_Gripper
cd ~/Hyundai_motors_Gripper

# IgH EtherCAT master kernel module (one-shot)
sudo bash scripts/install_etherlab.sh

# Daemon binary + systemd unit
sudo bash scripts/install_daemon.sh --system
sudo systemctl enable --now art-gripper-daemon
systemctl is-active art-gripper-daemon       # active

# TCP port 50053 should be listening
ss -tln | grep 50053
```

`art-gripper-daemon.service` requires `ethercat.service` (also installed by
`install_etherlab.sh`) and pulls in the `LimitMEMLOCK=infinity` +
`LimitRTPRIO=99` knobs the daemon needs to enter realtime priority.

If the gripper latches a fault later (hot-unplug, motor stall, etc.):

```bash
sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh
# stops daemon -> reloads ethercat kmod -> starts daemon -> ping verify
```

---

## Phase I — pro4000 SteamVR + vive_input

The Vive controller pose enters this repo over TCP `:12345` (and an
optional UDP haptic feedback channel on `:12346`). The publisher is the
C++ binary `vive_input` from the user's `vive_ws` workspace. SteamVR
provides the raw OpenVR poses to it.

### I-1. SteamVR

Install Steam GUI, log in, install SteamVR. Verify the binary exists:

```bash
ls ~/.steam/debian-installation/steamapps/common/SteamVR/bin/linux64/vrserver
```

The first time you launch SteamVR, you also need to pair your Vive
controller through the Steam GUI. After pairing, daily use never needs
the GUI again — `vrserver --keepalive` is headless.

### I-2. `vive_input` binary

```bash
ls ~/vive_ws/build/vive_ros2/vive_input
# or the symlink:
ls ~/vive_ws/install/vive_ros2/lib/vive_ros2/vive_input
```

This binary is built outside this repo (it's part of `vive_ws`, a
ROS-free OpenVR + nlohmann-json publisher); the source isn't shipped
here. KIST keeps it under `~/vive_ws/`. If you don't have it, see the
Isaac-GR00T documentation — the same `vive_aliases.sh` setup applies.

### I-3. udev / hidraw

```bash
lsusb | grep -i htc          # HTC controller(s) appear
sudo chmod +rw /dev/hidraw*   # one-shot per boot or write a udev rule
```

### I-4. Bring it up

```bash
bash bin/start_vive_stack.sh start    # starts vrserver --keepalive + vive_input :12345
bash bin/start_vive_stack.sh status

# Expected:
#   vrserver       : RUNNING (pid …)
#   vive_input     : RUNNING (~/vive_ws/build/vive_ros2/vive_input)
#   port 12345 TCP : LISTENING
#   port 12346 UDP : bound
#   HTC/Valve USB  : ≥2 device(s)
```

`vrserver --keepalive` runs without a desktop, so the whole Vive stack
works fine over SSH after the one-time GUI pairing.

---

## Phase J — Verification sequence

> Each step must pass before moving on. If something fails, jump to §13 troubleshooting.

### J-1. Pre-flight (pro4000)

```bash
cd ~/Polymetis_Franka_Teleop
bash install/check_environment.sh
# All [OK]; [WARN] for things you intentionally don't use is fine; [FAIL] is not.

# Comprehensive preflight with auto-recovery (recommended every session
# before launching the demo; run_test_session_groot_ft.sh invokes it
# automatically). Auto-fixes: stale demo processes, hung ART daemon TCP
# slot, missing Vive controllers, NUC :50051 down. Returns 0 if ready.
bash bin/preflight_full.sh
```

### J-2. NUC reachability + Polymetis state read (no motion)

```bash
# (NUC, terminal A)
sudo bash /usr/local/sbin/start_franka_arm.sh
# wait for: "Connected." + "[arm pinner] cores 6,7 pin applied"

# (pro4000)
conda activate groot-client
python -c "
from polymetis import RobotInterface
import numpy as np
r = RobotInterface(ip_address='192.168.1.12', port=50051)
s = r.get_robot_state()
print('joint (deg):', np.degrees(np.asarray(s.joint_positions)))
"
# 7 numbers print. Stop with Ctrl-C in the NUC terminal.
```

### J-3. ZED capture (no robot motion)

```bash
python ~/Isaac-GR00T/examples/DROID/zed_live_preview.py \
    --left-sn 33538770 --wrist-sn 11667817
# Two live windows. q to quit. (Skip if you don't have Isaac-GR00T cloned —
# zed_live_preview is a small standalone tool, not required by this repo.)
```

### J-4. ART gripper test

```bash
# Quickest: poke the daemon directly via the python client
python -c "
from art_gripper_client import ArtGripperClient
c = ArtGripperClient(ip='127.0.0.1', port=50053)
c.connect()
print('width m:', c.get_width())
c.goto(width=0.0, speed=0.05, force=10)   # close
import time; time.sleep(1.5)
c.goto(width=0.10, speed=0.05, force=10)  # open
time.sleep(1.5)
c.disconnect()
"
```

### J-5. Vive pose read

```bash
# vive_stack already up from §I-4
python -c "
from polymetis_franka_teleop.real_world.vive_shared_memory import ViveSharedMemory
import multiprocessing as mp, time
m = mp.Manager(); shm = ViveSharedMemory(m, host='127.0.0.1', port=12345)
shm.start(); time.sleep(1.0)
print(shm.read_latest_pose())
shm.stop()
"
# Expect a pose dict with controller pos/quat + buttons.
```

### J-6. Smallest possible joint motion (e-stop in hand!)

```bash
python <<'PY'
import torch, numpy as np
from polymetis import RobotInterface
r = RobotInterface(ip_address="192.168.1.12", port=50051)
ready = torch.tensor([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
r.move_to_joint_positions(ready, time_to_go=4.0)
print('ready')
PY
```

### J-7. Full data collection (everything together)

```bash
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S)
# In the cv2 window: press 'c' to start an episode, 's' to stop, 'q' to quit.
# Vive: Grip = clutch, Trigger = gripper toggle, Trackpad press = HOME.
```

When you stop, validate the recording:

```bash
python examples/check_recording.py data/<your-run-dir>
# Expect: per-stream shape print, video FPS report, monotonic timestamps
```

---

## 13. Troubleshooting catalog

The full list lives in [`docs/troubleshooting.md`](troubleshooting.md). Highlights:

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 1 | `motion aborted by reflex! [communication_constraints_violation] success_rate: 0.79` | NIC IRQ on P-core competing with GUI | §D — `franka_rt_apply.sh` re-pins NIC IRQ to E-cores |
| 2 | `success_rate: 0.24` (worse) | RT threads not pinned to cores 6,7 | §D — `start_franka_arm.sh` schedules `franka_pin_helper.sh` |
| 3 | Recovery cascade (8–12 trips/min during teleop) | IK silent failure when Vive target is unreachable | This repo's `franka_interpolation_controller.py` falls back to `last_good_joint_target` (no fix needed beyond pulling current code) |
| 4 | `c` key doesn't start recording inside the cv2 window | Old launcher had no signal relay | `bin/cv2_viewer.py` (this repo) sends SIGUSR1/2 + SIGHUP to the demo PID |
| 5 | `??? IDLE` garbled banners in cv2 window | Em-dash (Unicode) in `cv2.putText` | This repo uses ASCII colons only |
| 6 | `libtorchscript_pinocchio.so: cannot open` on pro4000 | NUC build artifact not on pro4000 | rsync `~/fairo/polymetis/build/torch_isolation/` from NUC; see Isaac-GR00T INSTALL_FROM_SCRATCH §9-5 |
| 7 | `curl: libcrypto.so.1.1` after activating `groot-client` | nuc_libs has libcrypto removed but LD_LIBRARY_PATH still points there | `env -u LD_LIBRARY_PATH curl …` or `conda deactivate` first |
| 8 | ZED 2i UVC missing (HID only) | USB 2.0 cable | Replace with USB 3.0 SuperSpeed C-C |
| 9 | ZED open `CALIBRATION FILE NOT AVAILABLE` | Calibration not on disk + ZED's libcurl can't fetch with conda LD overrides | §F-2 — pre-download calibration |
| 10 | Vive trackpad HOME does nothing | `last_teleop_cmd` stale across HOME | This repo's `art_gripper_controller.py` has the explicit reset (no fix needed in current code) |
| 11 | Franka Desk shows EE as "(none)" / FCI greyed out | Franka Hand connector not seated | Re-seat the EE connector + power-cycle controller box |
| 12 | NUC `start_franka_arm.sh` runs but no pin message | sudoers drop-in not installed | `sudo bash install/install_nuc.sh` again |

---

## Sister repos

This repo intentionally stays small. Related KIST workspaces:

| Repo | Provides | Used by this repo |
|---|---|---|
| [`Hyundai_motors_Gripper`](https://github.com/Seung-Sub/Hyundai_motors_Gripper) | ART gripper EtherCAT daemon + Python client | systemd `art-gripper-daemon` on pro4000 + `art_gripper_client` import |
| [`Isaac-GR00T`](https://github.com/Seung-Sub/Isaac-GR00T) | GR00T model + DROID inference + `vive_input` C++ binary source | NUC RT setup is shared (literally the same scripts in §D) |
| [`diffusion_policy`](https://github.com/columbia-ai-robotics/diffusion_policy) | UMI ReplayBuffer + checkpoint loader | `eval_franka_policy.py` and the UMI Zarr converter |

---

Document version: 2026-05-07 (full self-contained Phase A→J; install scripts shipped under `install/nuc/`)
