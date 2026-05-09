# Polymetis_Franka_Teleop

ROS-free Franka Panda teleoperation + data collection workspace, optimised for KIST hardware (NUC + RTX 4000 SFF + ZED 2i/Mini + HTC Vive + ART gripper or Franka Hand).

The pipeline is forked from [Stanford UMI](https://github.com/real-stanford/universal_manipulation_interface)'s real-robot stack — every timing-critical algorithm (timestamp-aligned obs, 200 Hz interpolation controller, latency-compensated action scheduling, Zarr+H264 storage) is preserved verbatim. KIST-specific changes are confined to the hardware boundary (camera/gripper/teleop drivers).

## Architecture

```
HTC Vive controller
    │ TCP :12345 (vive_input, ROS-free)
    ▼
ViveSharedMemory (200 Hz)
    │
    ▼
ViveTeleopProcess (100 Hz, clutch + toggle + velocity clamp)
    │ robot_ring_buffer        gripper_ring_buffer       action_ring_buffer
    ▼                          ▼                         ▼
FrankaInterpolationController  Franka/ArtGripperController   Main loop (10 Hz)
    │ 100 Hz                    │ 30 Hz (discrete)             │
    ▼                          ▼                              ├── MultiZed   60 fps H264 MP4
NUC :50051 polymetis           pro4000 :50053 ART             └── Zarr replay_buffer (per-step state/action)
   1 kHz libfranka loop          (or NUC :50052 Franka Hand)
```

Hardware Hz / call-rate decisions are documented in `docs/pipeline.md`.

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/install_from_scratch.md`](docs/install_from_scratch.md) | Hardware → fully working teleop, Phase A→J, self-contained. NUC RT scripts shipped under [`install/nuc/`](install/nuc/), pro4000 install via [`install/install_pro4000.sh`](install/install_pro4000.sh) |
| [`docs/usage.md`](docs/usage.md) | Daily TL;DR — once installed, this is the manual |
| [`docs/pipeline.md`](docs/pipeline.md) | Hz / algorithm deep-dive (UMI controller, latency calibration, timestamp accumulator) |
| [`docs/hardware_setup.md`](docs/hardware_setup.md) | Networking + cabling cheat sheet |
| [`docs/teleop_tuning.md`](docs/teleop_tuning.md) | Vive ↔ Franka feel knobs (pos_scale, rot_scale, Kx, Kxd) |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Symptom → fix catalog |

## Hardware (KIST)

| Role | Where | Notes |
|---|---|---|
| Franka Panda + 2-finger ART (or Franka Hand) | bench | Franka Desk → FCI Activate before bring-up |
| NUC `192.168.1.12` (`kist@kist-NUC13ANHi7`) | wired direct | PREEMPT_RT kernel + RT IRQ pinning (Phase D — `install/nuc/`) |
| pro4000 (`kist@kist-eval`, `161.122.114.90`) | wired | runs this repo + GR00T workspace + ART daemon |
| ART gripper (Hyundai Motors) | EtherCAT NIC `enxb0386cf13036` | systemd `art-gripper-daemon` :50053 (auto-boot) |
| ZED 2i `33538770` (exterior) | USB | LEFT eye only, 60 fps native VGA (672×376) |
| ZED Mini `11667817` (wrist) | USB | LEFT eye only |
| HTC Vive controller | base stations × 2 | SteamVR + `vive_input` TCP :12345 |

## Output formats

| Variant | Use | Script |
|---|---|---|
| Native UMI Zarr + H264 MP4 | Diffusion Policy / UMI training | `scripts_real/convert_franka_vive_to_umi_format.py` |
| GR00T LeRobot v2 (DROID embodiment) | `nvidia/GR00T-N1.7-3B` / `nvidia/GR00T-N1.7-DROID` fine-tune | `scripts_real/convert_to_gr00t_lerobot.py` |

Training itself is out of scope here — fine-tune in `Isaac-GR00T` or your `diffusion_policy` repo and bring the checkpoint back to `eval_franka_policy.py`.

## Install (one-shot per host)

```bash
# (NUC) RT scripts + systemd units + sudoers drop-in
sudo bash install/install_nuc.sh

# (pro4000) groot-client conda env + this repo + ART client
bash install/install_pro4000.sh
```

Full install walk-through in [`docs/install_from_scratch.md`](docs/install_from_scratch.md) — go there if you have only hardware.

## Bring up the stack (after install)

Pre-flight: `bash install/check_environment.sh` (each line OK / WARN / FAIL with hints).

```bash
# Franka Desk web UI:  unlock joints + Activate FCI + e-stop released

# NUC — terminal 1
ssh kist@192.168.1.12
sudo bash /usr/local/sbin/start_franka_arm.sh         # polymetis arm :50051

# NUC — terminal 2 (Franka Hand workflow only — skip for ART)
sudo bash /usr/local/sbin/start_franka_gripper.sh

# pro4000 — Vive
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start    # vrserver --keepalive + vive_input

# pro4000 — ART daemon (auto-systemd, just verify)
systemctl is-active art-gripper-daemon
```

Optional **ZeroRPC bridge** mode (UMI/DROID-style — runs polymetis client locally on the NUC, exposes one ZeroRPC endpoint):

```bash
bash ~/Polymetis_Franka_Teleop/bin/start_unified_bridge_on_nuc.sh   # auto-deploys + launches on NUC
# then run demo_franka_vive.py with --polymetis_mode zerorpc --robot_port 4242
```

## Data collection

```bash
# ART + ZED (KIST default)
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S)

# or directly with full CLI control
python scripts_real/demo_franka_vive.py \
    --output ./data/pap \
    --robot_ip 192.168.1.12 \
    --camera_backend zed --gripper_backend art \
    --camera_serials 33538770 --camera_serials 11667817 \
    --camera_resolution 1280x720 --camera_fps 60 \
    -v
```

In-app keys: `c` start, `s` stop, `Backspace`+`y` drop, `h` home, `q` quit.
Vive: `Grip` clutch, `Trigger` gripper toggle, `Trackpad press` HOME.

Drop-in alternatives:
* `--gripper_backend franka`     — Franka Hand instead of ART (also start `start_franka_gripper.sh` on NUC)
* `--camera_backend realsense`   — RealSense instead of ZED (legacy UMI)
* `--polymetis_mode zerorpc --robot_port 4242` — UMI/DROID-style bridge
* `LIVE_DURATION=60 python examples/run_live_test.py` — 60 s headless live test

Teleop feel: pass `--tuning_preset {coarse|normal|precise|custom}` to switch
between Vive↔robot mappings and Cartesian impedance gains. See
[`docs/teleop_tuning.md`](docs/teleop_tuning.md) for the symptom→knob table
and safe ranges.

## Convert recorded data

```bash
# UMI / Diffusion Policy
python scripts_real/convert_franka_vive_to_umi_format.py \
    -i ./data/pap -o ./data/pap/dataset.zarr.zip -r 224,224

# GR00T LeRobot v2 (DROID embodiment, ready for fine-tune)
python scripts_real/convert_to_gr00t_lerobot.py \
    -i ./data/pap -o ./data/pap_gr00t \
    -t "Pick up the yellow cup" \
    --gripper_max_width 0.100   # ART: 0.100 ; Franka Hand: 0.080
```

The GR00T export is directly usable:

```bash
cd ~/Isaac-GR00T
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/Polymetis_Franka_Teleop/data/pap_gr00t \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --num-gpus 1 --output-dir /tmp/franka_finetune \
    --max-steps 5000 --global-batch-size 32
```

## Verify a recording

```bash
python examples/check_recording.py data/pap
# Zarr stream shapes + monotonic timestamps + video FPS/decode check + GR00T metadata if converted
```

## Latency calibration

All per-channel latency constants live in [`install/latency_calibration.json`](install/latency_calibration.json) and are loaded automatically by `FrankaViveEnv` / `FrankaPolicyEnv` at construction time (with hardcoded V3 fallbacks if the file is missing). Backend-aware: the ZED camera and the ART gripper carry their own keys.

Current defaults (V3 2026-01-25, RealSense + Franka Hand calibrated; ZED + ART use placeholder values until the new calibrators are run):

| Channel | Latency | Source |
|---|---|---|
| Camera obs (RealSense) | 15 ms | hardware timestamp |
| Camera obs (ZED) | 15 ms | placeholder — run `calibrate_zed_latency.py` |
| Robot obs | 1 ms | round-trip / 2 |
| Gripper obs (Franka) | 1 ms | ZeroRPC round-trip / 2 |
| Gripper obs (ART) | 1 ms | placeholder — run `calibrate_art_gripper_latency.py` |
| Robot action | 55 ms | schedule_waypoint → arrival |
| Gripper action (Franka) | 85 ms | direct ZeroRPC |
| Gripper action (ART) | 85 ms | placeholder — run `calibrate_art_gripper_latency.py` |

Re-measure on hardware change. Each calibrator prints stats and (with `--patch`, the default) writes the measured value back into `install/latency_calibration.json` so subsequent recordings pick it up automatically:

```bash
# Existing (Franka Hand + RealSense paths)
python scripts_real/calibrate_franka_robot_latency.py
python scripts_real/calibrate_franka_gripper_latency.py
python scripts_real/calibrate_realsense_latency.py
python scripts_real/calibrate_all_latencies.py        # orchestrator

# NEW for ZED + ART (KIST-specific paths)
python scripts_real/calibrate_zed_latency.py --serial 33538770 --serial 11667817
python scripts_real/calibrate_art_gripper_latency.py
```

## Policy eval

```bash
bash bin/start_eval.sh path/to/checkpoint.ckpt ~/Polymetis_Franka_Teleop/data/eval_$(date +%H%M%S)
```

Loads a Diffusion Policy checkpoint, runs inference at 10 Hz, executes via the same controller stack, and records into the same Zarr+MP4 format as data collection.

For GR00T evaluation, use the existing GR00T server-client setup in `Isaac-GR00T/examples/DROID/main_gr00t.py` (this repo's `eval_franka_policy.py` is targeted at local diffusion-policy checkpoints).

## Repository layout

```
Polymetis_Franka_Teleop/
├── README.md                             ← you are here
├── LICENSE                               ← MIT (UMI portions also MIT)
├── pyproject.toml                        ← pip install -e .
├── docs/
│   ├── install_from_scratch.md           ← Phase A→J full install (hardware → working teleop)
│   ├── usage.md                          ← daily TL;DR
│   ├── pipeline.md                       ← Hz/algorithm deep-dive
│   ├── hardware_setup.md                 ← network + cabling
│   ├── teleop_tuning.md                  ← Vive ↔ Franka feel knobs
│   └── troubleshooting.md                ← symptom → fix catalog
├── install/
│   ├── check_environment.sh              ← preflight dependency check
│   ├── install_nuc.sh                    ← (NUC) copies install/nuc/* into /usr/local/sbin etc.
│   ├── install_pro4000.sh                ← (pro4000) groot-client conda env + this repo
│   └── nuc/                              ← raw RT scripts + systemd units shipped with the repo
│       ├── README.md                     ← what each file does, manual prereqs
│       ├── sbin/franka_rt_apply.sh        # boot-time NIC IRQ pin / governor / ASPM / Turbo
│       ├── sbin/franka_dma_latency.py     # /dev/cpu_dma_latency 0us holder
│       ├── sbin/franka_pin_helper.sh      # post-launch taskset of polymetis RT threads to 6,7
│       ├── sbin/start_franka_arm.sh       # polymetis arm wrapper (auto pin)
│       ├── sbin/start_franka_gripper.sh   # polymetis franka_hand wrapper
│       ├── systemd/franka-rt-tune.service       # boot-time franka_rt_apply.sh
│       ├── systemd/franka-dma-latency.service   # holds /dev/cpu_dma_latency at 0us
│       ├── systemd/franka-realtime-setup.service  # optional sysctl tunings
│       └── sudoers.d/franka_rt           # passwordless franka_pin_helper
├── bin/
│   ├── start_teleop.sh                   ← demo_franka_vive wrapper (ART+ZED defaults)
│   ├── start_eval.sh                     ← eval_franka_policy wrapper
│   ├── start_vive_stack.sh               ← vrserver --keepalive + vive_input bring-up
│   ├── start_unified_bridge_on_nuc.sh    ← optional ZeroRPC bridge launcher (UMI/DROID compat)
│   ├── run_test_session.sh               ← setsid'd long-running session launcher
│   └── cv2_viewer.py                     ← cv2.imshow subprocess (signal relay to demo)
├── polymetis_franka_teleop/              ← Python package (pip install -e .)
│   ├── shared_memory/                    ← lock-free SHM primitives (UMI vendored)
│   ├── common/                           ← pose math, precise_sleep, latency, accumulators
│   └── real_world/
│       ├── franka_interpolation_controller.py   200 Hz polymetis arm controller (mode={direct,zerorpc})
│       ├── franka_gripper_controller.py         Franka Hand 30 Hz (ZeroRPC)
│       ├── art_gripper_controller.py     ★KIST  ART gripper 30 Hz (TCP daemon)
│       ├── single_zed.py / multi_zed.py  ★KIST  ZED camera workers
│       ├── single_realsense.py / multi_realsense.py  legacy UMI camera workers
│       ├── vive_shared_memory.py / vive_teleop_process.py
│       ├── video_recorder.py / image_transform.py / keystroke_counter.py
│       ├── multi_camera_visualizer.py
│       ├── franka_vive_env.py            data-collection env (backend-selectable)
│       ├── franka_policy_env.py          policy-eval env (backend-selectable)
│       └── real_inference_util.py        obs/action transforms shared with training
├── scripts_real/                         ← user-facing entry points
│   ├── demo_franka_vive.py               data collection
│   ├── eval_franka_policy.py             policy eval
│   ├── preflight_check.py                interactive preflight
│   ├── launch_franka_unified_server.py   NUC ZeroRPC bridge (deployed by bin/start_unified_bridge_on_nuc.sh)
│   ├── calibrate_*.py                    latency calibration
│   ├── convert_franka_vive_to_umi_format.py  → UMI / Diffusion Policy dataset
│   └── convert_to_gr00t_lerobot.py       ★KIST → GR00T DROID embodiment dataset
└── examples/
    ├── run_live_test.py                  headless 3-min teleop + auto convert
    └── check_recording.py                Zarr + video integrity + GR00T metadata check
```

## Sister repositories

This workspace depends on (and stays consistent with) two adjacent KIST repos — both are clone-and-go, no source modifications required here:

| Repo | What it provides | How we use it |
|---|---|---|
| [`Hyundai_motors_Gripper`](https://github.com/Seung-Sub/Hyundai_motors_Gripper) | ART gripper EtherCAT daemon + Python client | `art_gripper_client` library import, daemon runs as systemd on pro4000 |
| `Isaac-GR00T` (your fork) | GR00T model + DROID inference + Vive_input source | NUC RT setup is shared (see `docs/install_from_scratch.md` §1), Vive_input C++ binary reused from `~/Isaac-GR00T/vive_input/build/` |
