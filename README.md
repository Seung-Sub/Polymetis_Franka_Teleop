# Polymetis_Franka_Teleop

ROS-free Franka Panda teleoperation + data collection workspace, built around an HTC Vive controller, a NUC running PREEMPT_RT polymetis, and a workstation running this repo. Records data in formats consumed by GR00T-N1.7-DROID, ACT / generic LeRobot, and Diffusion-Policy / UMI — pick the converter that matches your target model.

The pipeline is forked from [Stanford UMI](https://github.com/real-stanford/universal_manipulation_interface)'s real-robot stack. The timing-critical algorithms (timestamp-aligned obs, 100 Hz interpolation controller, latency-compensated action scheduling, Zarr+H264 storage) are preserved verbatim. KIST-specific changes are confined to the hardware boundary (camera/gripper/teleop drivers) and the post-recording converters.

## Architecture

```
HTC Vive controller
    │ TCP :12345 (vive_input)
    ▼
ViveSharedMemory (200 Hz)
    │
    ▼
ViveTeleopProcess (100 Hz, clutch + toggle + velocity clamp)
    │ robot_ring_buffer        gripper_ring_buffer       action_ring_buffer
    ▼                          ▼                         ▼
FrankaInterpolationController  Art / FrankaGripperCtrl   Main loop (10 / 15 Hz)
    │ 100 Hz direct gRPC       │ 60 Hz (discrete)          │
    ▼                          ▼                            ├── MultiZed / MultiRealsense  60 fps H264 MP4
NUC :50051 polymetis           pro4000 :50053 ART          └── Zarr replay_buffer
   1 kHz libfranka loop          (or NUC :4242 Franka Hand)
```

Hz / call-rate decisions are documented in [`docs/pipeline.md`](docs/pipeline.md).

## What's modular

The same demo + eval scripts cover a 4-axis backend matrix; pick at the CLI:

| Axis | Choices | CLI flag |
|---|---|---|
| Gripper | `art` (Hyundai EtherCAT, default) · `franka` (Franka Hand) | `--gripper_backend` |
| Camera | `zed` (ZED 2i / Mini, default) · `realsense` (Intel RealSense) | `--camera_backend` |
| Output format | `groot` · `umi` · `diffusion` (selects ready-pose at startup) | `--data_format` |
| Frequency | `10` (DP/ACT default) · `15` (GR00T-DROID baseline) | `--frequency` |

The robot-arm transport is direct polymetis gRPC `:50051`. The Franka Hand gripper still uses fairo's standalone zerorpc service `:4242` (intrinsic to the polymetis Franka Hand interface).

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/install_from_scratch.md`](docs/install_from_scratch.md) | Hardware → working teleop, Phase A→J. NUC RT scripts ship under [`install/nuc/`](install/nuc/), pro4000 scripts under [`install/pro4000/`](install/pro4000/) |
| [`docs/usage.md`](docs/usage.md) | Per-host phase-by-phase commands, model-specific data-collection workflow |
| [`docs/pipeline.md`](docs/pipeline.md) | Hz / algorithm deep-dive (UMI controller, latency calibration, timestamp accumulator) |
| [`docs/hardware_setup.md`](docs/hardware_setup.md) | Networking + cabling cheat sheet |
| [`docs/teleop_tuning.md`](docs/teleop_tuning.md) | Vive ↔ Franka feel knobs (`pos_scale`, `rot_scale`, Kx, Kxd) |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Symptom → fix catalog (#25 onwards: stutter / recovery / RT) |

## Hardware (KIST default)

| Role | Where | Notes |
|---|---|---|
| Franka Panda + ART gripper (default) or Franka Hand | bench | Franka Desk → unlock joints + Activate FCI before bring-up |
| NUC `192.168.1.12` (`kist@kist-NUC13ANHi7`) | wired direct | PREEMPT_RT kernel + RT IRQ pinning (Phase D — `install/nuc/`) |
| pro4000 (`kist@kist-eval`, `161.122.114.90`) | wired | this repo + GR00T workspace + ART daemon |
| ART gripper (Hyundai Motors) | EtherCAT NIC `enxb0386cf13036` | systemd `art-gripper-daemon` :50053 (auto-boot) |
| ZED 2i `33538770` (exterior) | USB | LEFT eye only, 60 fps native VGA (672×376) |
| ZED Mini `11667817` (wrist) | USB | LEFT eye only |
| HTC Vive controller | base stations × 2 | SteamVR + `vive_input` TCP :12345 |

## Output formats and converters

You collect once and convert to whichever target format you need.

| Target model | Converter | Output |
|---|---|---|
| `nvidia/GR00T-N1.7-3B` / `nvidia/GR00T-N1.7-DROID` | `scripts_real/convert_to_gr00t_lerobot.py` | LeRobot v2.1 with DROID 17-D state/action (eef_9d + gripper + joint) |
| ACT / HuggingFace LeRobot / generic | `scripts_real/convert_to_lerobot.py` | LeRobot v2.1, raw state/action (`--state_format {joint,eef,full}`) |
| Diffusion Policy / UMI | `scripts_real/convert_franka_vive_to_umi_format.py` | UMI `dataset.zarr.zip` |

Training itself is out of scope — fine-tune in [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), HuggingFace LeRobot (ACT), or your `diffusion_policy` checkout, and bring the resulting checkpoint back to `eval_franka_policy.py`.

## Install (one-shot per host)

```bash
# (NUC) RT scripts + systemd units + sudoers drop-in for franka_pin_helper
sudo bash install/install_nuc.sh

# (pro4000) base deps + repo install
bash install/install_pro4000.sh
sudo bash install/install_pro4000_rt.sh   # RT-tune (rtprio, NIC IRQ, governor)
```

Full install walk-through in [`docs/install_from_scratch.md`](docs/install_from_scratch.md).

## Daily bring-up

`docs/usage.md` is the manual. Quick reference:

```bash
# (pro4000) preflight: stale-process kill + ART/ZED/Vive/NUC checks (auto-fix where safe)
bash bin/preflight_full.sh

# (NUC) terminal (only if not running as systemd)
ssh kist@192.168.1.12
sudo bash /usr/local/sbin/start_franka_arm.sh           # polymetis arm :50051
sudo bash /usr/local/sbin/start_franka_gripper.sh       # only for --gripper_backend franka

# (pro4000) Vive bring-up
bash bin/start_vive_stack.sh start

# (pro4000) collect data — pick the wrapper for your target model
bash bin/start_teleop_groot_droid_ft.sh ./data/$(date +%Y%m%d_%H%M%S)   # GR00T (15 Hz, DROID ready-pose)
bash bin/start_teleop.sh                ./data/$(date +%Y%m%d_%H%M%S)   # ACT/DP/UMI (10 Hz, Franka home)

# In the cv2 window:
#   c     start episode    s    stop episode
#   bksp  drop episode     y    confirm
#   h     home (alt: trackpad press)
#   q     quit (double-press within 1.5 s — single press is a no-op)
# On the controller:
#   Grip      clutch (hold to drive)
#   Trigger   gripper toggle
#   Trackpad  HOME
```

Mid-session monitoring (separate terminal): `bash bin/monitor_session.sh`.

## Convert recorded data

```bash
# GR00T-DROID (15 Hz dataset, DROID 17-D)
python scripts_real/convert_to_gr00t_lerobot.py \
    -i ./data/pap -o ./data/pap_gr00t \
    -t "Pick up the yellow cup"

# ACT / generic LeRobot (joint-space, 8-D)
python scripts_real/convert_to_lerobot.py \
    -i ./data/pap -o ./data/pap_act \
    -t "Pick up the yellow cup" \
    --state_format joint --gripper_repr normalized

# UMI / Diffusion-Policy
python scripts_real/convert_franka_vive_to_umi_format.py \
    -i ./data/pap -o ./data/pap/dataset.zarr.zip -r 224,224
```

`gripper_max_width`, `fps`, and per-episode tasks are read from `replay_buffer.zarr/meta/.attrs` if present (set during recording), so most flags are optional.

## Latency calibration (run once per hardware change)

All per-channel latency constants live in [`install/latency_calibration.json`](install/latency_calibration.json) and are loaded automatically by `FrankaViveEnv` / `FrankaPolicyEnv` (with V3 hardcoded fallback if the file is missing).

```bash
python scripts_real/calibrate_franka_arm_direct.py    # arm via :50051
python scripts_real/calibrate_zed_latency.py --serial 33538770 --serial 11667817
python scripts_real/calibrate_art_gripper_latency.py  # ART daemon :50053
```

Each calibrator prints stats and writes back to `install/latency_calibration.json`. If you wire up the Franka Hand backend instead of ART, copy `calibrate_art_gripper_latency.py` as a template — the protocol is different but the floor-offset measurement methodology is the same.

## Policy eval

Loads a trained checkpoint, runs inference at the recorded frequency, executes via the same controller stack, and records into the same Zarr+MP4 format as data collection. Latencies are compensated automatically from `install/latency_calibration.json`, so the action timestamps match the training distribution.

```bash
bash bin/start_eval.sh path/to/checkpoint.ckpt ./data/eval_$(date +%H%M%S)
```

For GR00T evaluation, the inference path is the GR00T server-client setup in `Isaac-GR00T/examples/DROID/main_gr00t.py` (this repo's `eval_franka_policy.py` targets local diffusion-policy and HF-LeRobot checkpoints).

## Repository layout

```
Polymetis_Franka_Teleop/
├── README.md                                 ← you are here
├── LICENSE                                   ← MIT (UMI portions also MIT)
├── pyproject.toml                            ← pip install -e .
├── docs/                                     (install / usage / troubleshooting / ...)
├── install/
│   ├── check_environment.sh                  preflight dependency check
│   ├── install_nuc.sh                        (NUC) deploy install/nuc/* into /usr/local/sbin
│   ├── install_pro4000.sh                    (pro4000) base install (conda env + repo)
│   ├── install_pro4000_rt.sh                 (pro4000) RT-tune (rtprio, NIC IRQ, governor)
│   ├── latency_calibration.json              live JSON consumed by env/policy classes
│   ├── nuc/                                  raw RT scripts + systemd units shipped to NUC
│   └── pro4000/                              pro4000 RT scripts + systemd unit
├── bin/
│   ├── preflight_full.sh                     auto-fix preflight (stale-proc / ART / ZED / Vive / NUC)
│   ├── start_teleop.sh                       data collection wrapper (DP/UMI defaults: 10 Hz)
│   ├── start_teleop_groot_droid_ft.sh        data collection wrapper (GR00T defaults: 15 Hz)
│   ├── run_test_session.sh                   setsid'd one-shot wrapper (DP/UMI)
│   ├── run_test_session_groot_ft.sh          setsid'd one-shot wrapper (GR00T)
│   ├── start_eval.sh                         policy-eval wrapper
│   ├── start_vive_stack.sh                   vrserver --keepalive + vive_input
│   ├── monitor_session.sh                    live 5-s dashboard (overruns, recoveries, fps)
│   └── cv2_viewer.py                         cv2.imshow subprocess (signal relay to demo)
├── polymetis_franka_teleop/                  Python package (pip install -e .)
│   ├── shared_memory/                        lock-free SHM primitives (UMI vendored)
│   ├── common/
│   │   ├── pose_*.py / interpolation_util.py / rotation_transformer.py / cv2_util.py
│   │   ├── precise_sleep.py
│   │   ├── timestamp_accumulator.py
│   │   ├── realtime_util.py                  apply_realtime() — affinity + SCHED_RR pinning
│   │   └── latency_config.py                 backend-aware latency lookup
│   └── real_world/
│       ├── franka_interpolation_controller.py    100 Hz polymetis arm controller (direct gRPC)
│       ├── art_gripper_controller.py             ART gripper 60 Hz (TCP daemon)
│       ├── franka_gripper_controller.py          Franka Hand 60 Hz (zerorpc :4242)
│       ├── single_zed.py / multi_zed.py          ZED camera workers
│       ├── single_realsense.py / multi_realsense.py   RealSense camera workers
│       ├── vive_shared_memory.py / vive_teleop_process.py
│       ├── video_recorder.py / image_transform.py / keystroke_counter.py
│       ├── multi_camera_visualizer.py
│       ├── franka_vive_env.py                data-collection env (backend-selectable)
│       ├── franka_policy_env.py              policy-eval env (backend-selectable)
│       └── real_inference_util.py            obs/action transforms shared with training
└── scripts_real/
    ├── demo_franka_vive.py                   data collection
    ├── eval_franka_policy.py                 policy eval (DP / HF-LeRobot)
    ├── preflight_check.py                    Python preflight (called by demo)
    ├── calibrate_franka_arm_direct.py        arm latency via direct :50051
    ├── calibrate_art_gripper_latency.py      ART gripper latency via :50053
    ├── calibrate_zed_latency.py              ZED HW timestamp latency
    ├── calibrate_realsense_latency.py        RealSense HW timestamp latency
    ├── convert_to_gr00t_lerobot.py           → GR00T DROID embodiment dataset
    ├── convert_to_lerobot.py                 → generic LeRobot v2 (ACT / HF-LeRobot)
    └── convert_franka_vive_to_umi_format.py  → UMI / Diffusion-Policy zarr.zip
```

## Sister repositories

| Repo | What it provides | How we use it |
|---|---|---|
| [`Hyundai_motors_Gripper`](https://github.com/Seung-Sub/Hyundai_motors_Gripper) | ART gripper EtherCAT daemon + Python client | `art_gripper_client` library import, daemon runs as systemd on pro4000 |
| [`Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T) (or your fork) | GR00T model + DROID inference + `vive_input` C++ source | NUC RT setup is shared (see `docs/install_from_scratch.md` §1); `vive_input` binary reused from `~/Isaac-GR00T/vive_input/build/` |
