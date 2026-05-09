# USAGE — daily operation

This is the everyday manual. First-time setup lives in
[`install_from_scratch.md`](install_from_scratch.md); once that document's
final phase passes, the steps below are all you need.

The pipeline is **modular**: pick a model target (GR00T / ACT / Diffusion-Policy),
a gripper (`art` or `franka`), and a camera (`zed` or `realsense`) at the CLI;
the same demo and eval scripts handle every combination.

## Table of contents

- [Hosts and processes](#hosts-and-processes)
- [Step 1 — Franka Desk (every session)](#step-1--franka-desk-every-session)
- [Step 2 — NUC: bring up polymetis](#step-2--nuc-bring-up-polymetis)
- [Step 3 — pro4000: bring up Vive + ART daemon](#step-3--pro4000-bring-up-vive--art-daemon)
- [Step 4 — Preflight (auto-fix)](#step-4--preflight-auto-fix)
- [Step 5 — Data collection (pick your target model)](#step-5--data-collection-pick-your-target-model)
- [Step 6 — Mid-session monitoring](#step-6--mid-session-monitoring)
- [Step 7 — Convert recordings](#step-7--convert-recordings)
- [Step 8 — Latency calibration (one-shot per hardware change)](#step-8--latency-calibration-one-shot-per-hardware-change)
- [Step 9 — Policy evaluation](#step-9--policy-evaluation)
- [Step 10 — Shutdown sequence](#step-10--shutdown-sequence)
- [Cheat sheets](#cheat-sheets)

---

## Hosts and processes

| Host | What runs there |
|------|-----------------|
| **NUC** (`192.168.1.12`, `kist@kist-NUC13ANHi7`) | polymetis arm `:50051` (always); polymetis Franka Hand `:4242` (only for `--gripper_backend franka`) |
| **pro4000** (`kist@kist-eval`, `161.122.114.90`) | this repo (`demo_franka_vive.py` / `eval_franka_policy.py`), Vive stack (`vrserver --keepalive` + `vive_input :12345`), ART gripper daemon `:50053` (systemd, only for `--gripper_backend art`) |

Run all NUC commands inside `ssh kist@192.168.1.12`. Run all pro4000 commands locally.

The arm transport is direct polymetis gRPC `:50051` end-to-end. The Franka Hand still uses fairo's standalone zerorpc service on `:4242` because the polymetis Franka Hand interface is implemented as zerorpc — that's intrinsic to the polymetis package, not an indirection added by this repo.

---

## Step 1 — Franka Desk (every session)

In Franka Desk web UI:

1. Unlock joints
2. **Activate FCI**
3. Confirm the e-stop is released (or you'll see "User stop" in the libfranka log)

Skipping any of these makes `start_franka_arm.sh` log "FCI not active" and exit.

---

## Step 2 — NUC: bring up polymetis

```bash
ssh kist@192.168.1.12

# ARM (always required)
sudo bash /usr/local/sbin/start_franka_arm.sh
# Healthy output (both must appear):
#   [INFO] Connected.
#   [arm pinner] cores 6,7 pin applied
# If only the first appears: tail ~/.franka_logs/franka_pin_arm.log

# FRANKA HAND (only if you'll pass --gripper_backend franka — skip for ART)
sudo bash /usr/local/sbin/start_franka_gripper.sh
```

Both units stay up across sessions. If you want them as systemd services that auto-restart, see [`docs/install_from_scratch.md`](install_from_scratch.md) §E.

---

## Step 3 — pro4000: bring up Vive + ART daemon

```bash
# Vive headless stack (vrserver --keepalive + vive_input TCP :12345)
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start

# ART gripper daemon — already running as systemd, just verify
systemctl is-active art-gripper-daemon         # → active
```

Skip the ART step entirely if you'll pass `--gripper_backend franka`.

To shut Vive down at the end of the day: `bash bin/start_vive_stack.sh stop`.

---

## Step 4 — Preflight (auto-fix)

Run this once before the first session of the day. It cleans stale processes, verifies ART/ZED/Vive, optionally bring polymetis up over SSH if it isn't running, and confirms `ulimit -r` is high enough:

```bash
bash ~/Polymetis_Franka_Teleop/bin/preflight_full.sh
# 6 phases. Each line is OK / WARN / FAIL. AUTO_FIX=1 (default) cleans
# what it can. Re-run until all green before launching the demo.
```

Lighter dependency-only check (Python imports, no hardware): `bash install/check_environment.sh`.

---

## Step 5 — Data collection (pick your target model)

Pick the wrapper that matches the model you're collecting for. **The wrapper sets the right ready-pose, frequency, and metadata so the converter has everything it needs.** All wrappers accept `<output_dir>` as `$1` and pass-through extra flags after that.

### 5.A — GR00T-N1.7-DROID fine-tune (default in this repo)

```bash
bash ~/Polymetis_Franka_Teleop/bin/start_teleop_groot_droid_ft.sh \
    ~/Polymetis_Franka_Teleop/data/groot_$(date +%Y%m%d_%H%M%S)

# Sets: --frequency 15 --data_format groot --camera_backend zed --gripper_backend art
# Ready pose: DROID-tilted (matches the GR00T-N1.7-DROID training distribution).
```

Or **one-shot mode** (setsid'd, auto-tail, exits when the demo exits):

```bash
bash ~/Polymetis_Franka_Teleop/bin/run_test_session_groot_ft.sh
```

### 5.B — ACT / HuggingFace LeRobot / Diffusion-Policy

```bash
bash ~/Polymetis_Franka_Teleop/bin/start_teleop.sh \
    ~/Polymetis_Franka_Teleop/data/dp_$(date +%Y%m%d_%H%M%S)

# Sets: --frequency 10 --data_format umi --camera_backend zed --gripper_backend art
# Ready pose: standard Franka home.
```

Or one-shot: `bash bin/run_test_session.sh`.

### 5.C — Custom backend combination

`start_teleop*.sh` are thin wrappers around `demo_franka_vive.py`. Pass any of these flags after the output dir to override:

| Flag | Choices | Effect |
|------|---------|--------|
| `--gripper_backend` | `art` (default) · `franka` | Switch to Franka Hand. Also requires `start_franka_gripper.sh` on NUC. |
| `--camera_backend` | `zed` (default) · `realsense` | Switch to RealSense. Requires `pyrealsense2` in `groot-client`: `pip install pyrealsense2`. |
| `--data_format` | `groot` · `umi` · `diffusion` | Picks the joint ready-pose. Saved to zarr meta so converters can read it. |
| `--frequency` | int | Data-loop Hz. 10 (DP/UMI), 15 (GR00T-DROID), 20-30 (ACT high-rate). |
| `--camera_resolution` | `WxH` | ZED HD720 native = `1280x720`; RealSense default = `640x480`. |
| `--camera_fps` | int | 60 for ZED HD720; 30/60 for RealSense. |
| `--tuning_preset` | `coarse` · `normal` · `precise` · `custom` | Vive ↔ Franka mapping + impedance gains. See [`teleop_tuning.md`](teleop_tuning.md). |
| `--tcp_offset` | float (m) | Auto: 0.216 (ART) · 0.1034 (Franka Hand). Override only for unusual setups. |
| `--task` | string | Task instruction for this session (saved to zarr meta, becomes default for converters). |

Example — Franka Hand + RealSense, 20 Hz collection for ACT high-rate:

```bash
bash bin/start_teleop.sh ~/data/act_hf_$(date +%H%M%S) \
    --gripper_backend franka \
    --camera_backend realsense \
    --frequency 20 \
    --data_format umi \
    --task "Stack the red block on the green block"
```

### 5.D — In-app keys and Vive controls

Inside the cv2 status window:

| Key | Action |
|-----|--------|
| `c` | Start episode (begin recording into the zarr) |
| `s` | Stop episode (commit the zarr / mp4 files) |
| `Backspace` then `y` | Drop the current episode (erase mid-recording) |
| `h` | HOME (move arm + gripper back to ready pose) |
| `q` (double-press within 1.5 s) | Quit cleanly |

On the Vive controller:

| Input | Action |
|-------|--------|
| `Grip` (hold) | Clutch — robot moves only while held |
| `Trigger` | Gripper toggle (open ↔ closed) |
| `Trackpad press` | HOME (full reset cycle) |
| `Trackpad touch + move Y` | Joint-7 yaw (when `--enable_trackpad_rotation` is on) |

### 5.E — How `--data_format` shapes the recording

The wrapper sets `--data_format`, which determines the ready-pose joints saved at the start of each episode and the metadata tag in `replay_buffer.zarr/meta/.attrs[data_format]`. Converters read this attribute to validate that the dataset matches the target embodiment.

| `--data_format` | Joints 1–6 | Joint 7 (ART / Franka Hand) | Used by |
|------|-------|----|----|
| `groot` | DROID-tilted | 0 / π/4 | `convert_to_gr00t_lerobot.py` |
| `umi` | Franka home | 0 / π/4 | `convert_to_lerobot.py`, `convert_franka_vive_to_umi_format.py` |
| `diffusion` | Franka home | 0 / π/4 | `convert_to_lerobot.py`, `convert_franka_vive_to_umi_format.py` |

The action representation written into the zarr is the **same** in all three modes (`[eef_pos, eef_aa, gripper_width]`); the difference is purely the joint ready-pose at episode start. Re-record (don't post-hoc convert) if you need to switch ready-pose.

---

## Step 6 — Mid-session monitoring

Open a separate pro4000 terminal:

```bash
bash ~/Polymetis_Franka_Teleop/bin/monitor_session.sh
# Live dashboard updated every 5 s:
#   ZED 60 fps drop count
#   FrankaInterp overruns (target 0/100)
#   ArtGripper overruns (target 0/300)
#   recovery / IK STUCK / auto-HOME counters
#   NUC libfranka reflex_stack + success_rate
```

Yellow numbers are warning-level. Red means the run is degraded (stop, drop episode, restart).

---

## Step 7 — Convert recordings

Pick the converter for your model target. All converters read `gripper_max_width`, `frequency`, and (if present) `episode_tasks` from `replay_buffer.zarr/meta/.attrs`, so most flags are optional.

### 7.A — GR00T-N1.7-DROID

```bash
python scripts_real/convert_to_gr00t_lerobot.py \
    -i ./data/groot_session \
    -o ./data/groot_session_gr00t \
    -t "Pick up the yellow cup and place it in the bowl"
# Output: LeRobot v2.1, 17-D state/action (eef_9d + gripper_position + joint_position)
# Embodiment tag: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

The output is directly fine-tunable in Isaac-GR00T:

```bash
cd ~/Isaac-GR00T
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/Polymetis_Franka_Teleop/data/groot_session_gr00t \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --num-gpus 1 --output-dir /tmp/franka_finetune \
    --max-steps 5000 --global-batch-size 32
```

### 7.B — ACT / HuggingFace LeRobot

```bash
# Joint-space (typical for ACT) — 8-D state/action [joint(7), gripper(1)]
python scripts_real/convert_to_lerobot.py \
    -i ./data/dp_session -o ./data/dp_session_act \
    -t "Pick up the yellow cup" \
    --state_format joint --gripper_repr normalized

# EEF-space — 7-D state/action [eef_pos(3), eef_aa(3), gripper(1)]
python scripts_real/convert_to_lerobot.py \
    -i ./data/dp_session -o ./data/dp_session_lerobot_eef \
    --state_format eef --gripper_repr width

# Combined — 14-D (joint + eef + gripper) for multi-modal policies
python scripts_real/convert_to_lerobot.py \
    -i ./data/dp_session -o ./data/dp_session_lerobot_full \
    --state_format full --gripper_repr normalized
```

| Flag | Choices | Effect |
|------|---------|--------|
| `--state_format` | `joint` (8 D) · `eef` (7 D) · `full` (14 D) | State/action layout |
| `--gripper_repr` | `normalized` (0=open, 1=closed) · `width` (raw m) | Gripper signal repr |
| `--video_keys` | repeated string | Per-camera key (default `cam_high cam_wrist`; for DP-via-LeRobot pass `camera0_rgb camera1_rgb`) |

### 7.C — Diffusion-Policy / UMI (zarr.zip)

```bash
python scripts_real/convert_franka_vive_to_umi_format.py \
    -i ./data/dp_session \
    -o ./data/dp_session/dataset.zarr.zip \
    -r 224,224
# Output: UMI-style zarr.zip, ready for the diffusion-policy training pipeline
```

---

## Step 8 — Latency calibration (one-shot per hardware change)

Latencies are stored in [`install/latency_calibration.json`](../install/latency_calibration.json). They're loaded automatically by `FrankaViveEnv` (recording) and `FrankaPolicyEnv` (eval), with V3 fallbacks hardcoded in `latency_config.py` if the JSON is absent.

Re-measure when you change a camera, swap grippers, or move the workstation. Each calibrator writes back to the JSON:

```bash
# Arm — direct gRPC :50051 (production path)
python scripts_real/calibrate_franka_arm_direct.py

# ART gripper — TCP :50053
python scripts_real/calibrate_art_gripper_latency.py

# ZED cameras — HW timestamp
python scripts_real/calibrate_zed_latency.py --serial 33538770 --serial 11667817

# RealSense — HW timestamp (only if --camera_backend realsense)
python scripts_real/calibrate_realsense_latency.py
```

Franka Hand latency calibration: copy `calibrate_art_gripper_latency.py` as a template and substitute the polymetis Franka Hand zerorpc protocol (`gripper.goto`, `gripper.get_state`). The floor-offset measurement methodology is identical.

---

## Step 9 — Policy evaluation

Run a trained checkpoint on real hardware. Latency compensation is automatic from `install/latency_calibration.json` — the action timestamps will match the training distribution as long as the JSON values reflect this hardware:

```bash
# Diffusion-Policy / HF-LeRobot checkpoint
bash bin/start_eval.sh path/to/checkpoint.ckpt \
    ~/Polymetis_Franka_Teleop/data/eval_$(date +%H%M%S)

# Or directly
python scripts_real/eval_franka_policy.py \
    -i path/to/checkpoint.ckpt \
    -o ./data/eval \
    --camera_backend zed --gripper_backend art \
    -c 33538770 -c 11667817 \
    --auto_start --record_episode
```

In-app keys (for the eval cv2 window):

| Key | Action |
|-----|--------|
| `c` | Start policy execution |
| `s` | Stop policy and return to waiting mode |
| `h` | HOME |
| `q` | Quit (stops policy first) |

**Always keep your hand on the e-stop** — the robot moves autonomously while the policy is running.

For **GR00T evaluation**, this repo's `eval_franka_policy.py` is not the right entry point. Use the GR00T server-client setup in `Isaac-GR00T/examples/DROID/main_gr00t.py` — it loads a GR00T checkpoint via the GR00T inference server and the Vive teleop env imports here as a library.

---

## Step 10 — Shutdown sequence

```bash
# pro4000
# - Inside the cv2 window: q (twice) to exit demo cleanly
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh stop

# NUC (only if you brought polymetis up manually for this session)
# - Ctrl-C the start_franka_arm.sh / start_franka_gripper.sh windows
```

The ART daemon is systemd-managed — leave it running. Same for ZED USB enumeration.

If anything is stuck, `bin/preflight_full.sh` will detect and (with `AUTO_FIX=1`, the default) clean stale processes, free the FCI port, and reset USB enumeration before the next session.

---

## Cheat sheets

### Per-host one-liners

| Where | Command | Purpose |
|-------|---------|---------|
| NUC | `sudo bash /usr/local/sbin/start_franka_arm.sh` | Bring up arm `:50051` |
| NUC | `sudo bash /usr/local/sbin/start_franka_gripper.sh` | Bring up Franka Hand `:4242` (only if `--gripper_backend franka`) |
| pro4000 | `bash bin/start_vive_stack.sh start` | Vive bring-up (vrserver + vive_input) |
| pro4000 | `systemctl is-active art-gripper-daemon` | Verify ART daemon (only if `--gripper_backend art`) |
| pro4000 | `bash bin/preflight_full.sh` | Full preflight w/ auto-fix |
| pro4000 | `bash bin/start_teleop_groot_droid_ft.sh <out>` | Collect (GR00T) |
| pro4000 | `bash bin/start_teleop.sh <out>` | Collect (DP/ACT/UMI) |
| pro4000 | `bash bin/monitor_session.sh` | Live dashboard |
| pro4000 | `bash bin/start_eval.sh <ckpt> <out>` | DP/HF-LeRobot eval |
| pro4000 | `bash bin/start_vive_stack.sh stop` | Vive shutdown |

### Convert quick reference

| Target | Command |
|--------|---------|
| GR00T-DROID | `python scripts_real/convert_to_gr00t_lerobot.py -i <in> -o <out> -t "<task>"` |
| ACT (joint) | `python scripts_real/convert_to_lerobot.py -i <in> -o <out> -t "<task>" --state_format joint` |
| LeRobot eef | `python scripts_real/convert_to_lerobot.py -i <in> -o <out> -t "<task>" --state_format eef` |
| UMI / DP | `python scripts_real/convert_franka_vive_to_umi_format.py -i <in> -o <out>/dataset.zarr.zip -r 224,224` |

### Calibrators

| Channel | Command |
|---------|---------|
| Arm (direct :50051) | `python scripts_real/calibrate_franka_arm_direct.py` |
| ART gripper (:50053) | `python scripts_real/calibrate_art_gripper_latency.py` |
| ZED | `python scripts_real/calibrate_zed_latency.py --serial 33538770 --serial 11667817` |
| RealSense | `python scripts_real/calibrate_realsense_latency.py` |

All calibrators write back to `install/latency_calibration.json` automatically.
