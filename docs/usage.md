# USAGE — daily operation

This is the *everyday* manual. First-time setup is in
[`install_from_scratch.md`](install_from_scratch.md). Once §J of that
document passes, the steps below are all you need.

## Table of contents

- [TL;DR — 5-step bring-up](#tldr--5-step-bring-up)
- [1. NUC: bring up Polymetis](#1-nuc-bring-up-polymetis)
- [2. pro4000: bring up Vive](#2-pro4000-bring-up-vive)
- [3. Pre-flight check](#3-pre-flight-check)
- [4. Data collection](#4-data-collection)
- [5. Verify the recording](#5-verify-the-recording)
- [6. Convert recordings](#6-convert-recordings)
- [7. Policy evaluation](#7-policy-evaluation)
- [8. Shutdown sequence](#8-shutdown-sequence)
- [9. Teleop tuning shortcuts](#9-teleop-tuning-shortcuts)
- [10. Cheat sheet — common commands](#10-cheat-sheet--common-commands)

---

## TL;DR — 5-step bring-up

```bash
# 1. Franka Desk web UI: unlock joints, FCI Activate, e-stop released

# 2. NUC: arm gRPC server
ssh kist@192.168.1.12
sudo bash /usr/local/sbin/start_franka_arm.sh
# wait for "Connected" + "[arm pinner] cores 6,7 pin applied"

# 3. pro4000: Vive
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start

# 4. pro4000: pre-flight
cd ~/Polymetis_Franka_Teleop && bash install/check_environment.sh

# 5. pro4000: data collection
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S)
```

In the cv2 window: `c` start episode, `s` stop, `Backspace` then `y` drop, `h` home, `q` quit.
On the Vive: `Grip` clutch, `Trigger` gripper toggle, `Trackpad press` HOME.

---

## 1. NUC: bring up Polymetis

```bash
ssh kist@192.168.1.12

# Arm — always required.
sudo bash /usr/local/sbin/start_franka_arm.sh

# (Optional, only if --gripper_backend franka)
# sudo bash /usr/local/sbin/start_franka_gripper.sh
```

`start_franka_arm.sh` activates the `polymetis-local` conda env, spawns
`launch_robot.py`, and 5 seconds later schedules `franka_pin_helper.sh`
(via `sudo -n`) to pin the RT threads to cores 6,7. Two messages tell you it's healthy:

```
[INFO] Connected.                                 ← libfranka talked to the controller box
[arm pinner] cores 6,7 pin applied (details: tail ~/.franka_logs/franka_pin_arm.log)
```

If only the first line appears: the pinner failed silently. `tail
~/.franka_logs/franka_pin_arm.log` to see why (almost always: sudoers
drop-in missing — re-run `sudo bash install/install_nuc.sh`).

### When to use the ZeroRPC bridge instead

The `direct` mode (above) makes the pro4000 client speak Polymetis gRPC
directly to NUC `:50051`. Each control tick is up to 4 RPCs (state
read, IK, joint command, status). With ART gripper at 100 Hz this
amounts to ~400 RPCs/s and stays well under the 1 s watchdog.

For UMI/DROID compatibility, you can launch a **ZeroRPC bridge** that
collapses everything into one RPC per tick:

```bash
# (pro4000) - the launcher ssh's into the NUC and starts the bridge there
bash bin/start_unified_bridge_on_nuc.sh
# Then run demos with: --polymetis_mode zerorpc --robot_port 4242
```

You only need this if you're cross-comparing against UMI/DROID baselines,
or you start hitting the watchdog at higher control rates. KIST default = `direct`.

---

## 2. pro4000: bring up Vive

```bash
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh status
```

Expected status output:

```
==== Vive stack status ====
  vrserver       : RUNNING (pid …)
  vive_input     : RUNNING (~/vive_ws/build/vive_ros2/vive_input)
  port 12345 TCP : LISTENING
  port 12346 UDP : bound
  HTC/Valve USB  : 2 device(s)
```

Stop with `bash bin/start_vive_stack.sh --stop`. The stack uses
`vrserver --keepalive` so it works fine over SSH — no Steam GUI required
after the first-time controller pairing.

---

## 3. Pre-flight check

```bash
cd ~/Polymetis_Franka_Teleop
bash install/check_environment.sh
```

The script checks the conda env, all imports, the pro4000 services, NUC
reachability, and NUC port `:50051`. Any `[FAIL]` blocks teleop; `[WARN]`
on a feature you aren't using is fine.

---

## 4. Data collection

The default wrapper runs `demo_franka_vive.py` with KIST defaults
(ART + ZED + Vive, 10 Hz main loop, 100 Hz Vive poll, 60 fps cameras
at native VGA 672×376):

```bash
# Basic — output dir is the only required arg
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S)

# With teleop-feel preset (see teleop_tuning.md)
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S) \
    --tuning_preset precise

# Full-throttle CLI (override anything)
python scripts_real/demo_franka_vive.py \
    --output ./data/pap \
    --robot_ip 192.168.1.12 \
    --camera_backend zed --gripper_backend art \
    --camera_serials 33538770 --camera_serials 11667817 \
    --camera_resolution 1280x720 --camera_fps 30 \
    -v
```

### Hands-off "test session" mode

For a session that survives accidental Ctrl-Z / SSH timeouts:

```bash
bash bin/run_test_session.sh
# - setsid'd demo so it owns its own session/group
# - logs to /tmp/teleop_session.log
# - tail follows the log; Ctrl-C the tail any time, demo keeps running
# - tail auto-exits when the demo dies (you don't have to hunt orphans)
```

Force-stop from another shell: `pkill -INT -f demo_franka_vive`.

### In-app keys

| Key | Action |
|---|---|
| `c` | start a new episode (record) |
| `s` | stop the current episode (commit Zarr + close mp4) |
| `Backspace`, then `y` | drop the in-progress episode (do not commit) |
| `h` | move robot to ready pose |
| `q` | quit cleanly (closes Zarr, releases cameras + gripper) |

`cv2_viewer.py` runs as a separate subprocess and forwards key presses
back to the demo via signals — that way the OpenCV window never deadlocks
the main loop's Qt5 / multiprocessing state.

### Vive controller bindings

| Vive button | Action |
|---|---|
| Grip (squeeze) | clutch (engage / disengage motion mapping) |
| Trigger | gripper toggle (open ↔ closed) |
| Trackpad press | HOME (move to ready pose) |
| Menu | unused |

### Drop-in alternatives

```bash
--gripper_backend franka                  # Franka Hand instead of ART
                                          # (also start_franka_gripper.sh on NUC)
--camera_backend realsense                # RealSense instead of ZED (legacy UMI)
--polymetis_mode zerorpc --robot_port 4242   # UMI/DROID-style bridge (see §1)
LIVE_DURATION=60 python examples/run_live_test.py   # 60 s headless live test
```

---

## 5. Verify the recording

```bash
python examples/check_recording.py data/<your-run-dir>
```

The script prints, per episode:
- Zarr stream shapes + dtypes
- monotonic timestamps + dt mean / std
- video frame count vs. zarr step count (must be a clean integer ratio)
- end-effector workspace bounds
- gripper open/closed distribution

A pass looks like (KIST 2-episode session, 1052 steps):

```
2 episodes (444 + 608 steps), 1052 total
in-episode dt = 100.00ms ± 0.000ms       (perfect 10 Hz)
camera frames per episode = exactly 6× zarr steps   (60fps / 10Hz)
two cameras frame-synchronized (within 1 frame)
0 NaNs anywhere
```

---

## 6. Convert recordings

### UMI / Diffusion Policy

```bash
python scripts_real/convert_franka_vive_to_umi_format.py \
    -i ./data/pap \
    -o ./data/pap/dataset.zarr.zip \
    -r 224,224
```

### GR00T LeRobot v2 (DROID embodiment)

```bash
python scripts_real/convert_to_gr00t_lerobot.py \
    -i ./data/pap \
    -o ./data/pap_gr00t \
    -t "Pick up the yellow cup" \
    --gripper_max_width 0.100        # ART: 0.100 ; Franka Hand: 0.080
```

Output is directly fine-tunable:

```bash
cd ~/Isaac-GR00T
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/Polymetis_Franka_Teleop/data/pap_gr00t \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --num-gpus 1 --output-dir /tmp/franka_finetune \
    --max-steps 5000 --global-batch-size 32
```

---

## 7. Policy evaluation

```bash
bash bin/start_eval.sh path/to/checkpoint.ckpt \
    ~/Polymetis_Franka_Teleop/data/eval_$(date +%H%M%S)
```

`start_eval.sh` loads a Diffusion Policy checkpoint, runs inference at
10 Hz through the same controller stack as data collection, and records
into the same Zarr+MP4 format — so eval rollouts are directly
inspectable with `examples/check_recording.py`.

For GR00T evaluation, use Isaac-GR00T's
`examples/DROID/main_gr00t.py --env-mode kist_minimal` — that's a
separate code path (server-client), and `eval_franka_policy.py` here is
focused on local diffusion-policy checkpoints.

---

## 8. Shutdown sequence

```bash
# 1. In the cv2 window: press 'q'  (or pkill -INT -f demo_franka_vive)

# 2. Vive stack
bash bin/start_vive_stack.sh --stop

# 3. NUC: Ctrl-C the start_franka_arm.sh terminal
ssh kist@192.168.1.12 'pkill -INT launch_robot' 2>/dev/null   # if you're remote

# 4. (optional) move robot back to ready pose
python -c "
import torch, numpy as np
from polymetis import RobotInterface
r = RobotInterface(ip_address='192.168.1.12', port=50051)
r.move_to_joint_positions(
    torch.tensor([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
    time_to_go=5.0)
"
```

`demo_franka_vive.py` does **not** automatically move the robot back to
ready pose on quit — the arm holds wherever it was. If you start the
next session immediately, that pose carries over (the next session's
ready-pose move handles it). If you're stopping for the day, run the
explicit move so the next person inherits a known pose.

---

## 9. Teleop tuning shortcuts

`--tuning_preset {coarse|normal|precise|custom}` switches Vive ↔ robot
mappings + Cartesian impedance gains in one go. Full table in
[`teleop_tuning.md`](teleop_tuning.md). Quick table:

| Preset | When |
|---|---|
| `coarse` | Layout shots, transit, large reach |
| `normal` | Default — UMI baseline; what the policy expects |
| `precise` | Fine inserts, alignment, contact tasks |
| `custom` | Override any subset of `--pos_scale`, `--rot_scale`, `--kx_scale`, `--kxd_scale`, `--velocity_clamp`, `--max_pos_velocity`, `--max_rot_velocity` |

For data collection that you'll fine-tune on, use `normal` or `precise`
(they keep the Vive↔robot mapping near identity, which the policy can
learn). `coarse`'s `pos_scale=1.5` distorts the mapping enough that
rollouts visibly under-shoot.

---

## 10. Cheat sheet — common commands

```bash
# Bring up everything (after first-time install)
ssh kist@192.168.1.12 'sudo bash /usr/local/sbin/start_franka_arm.sh' &
bash bin/start_vive_stack.sh start
bash install/check_environment.sh
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S)

# Force-stop teleop demo from elsewhere
pkill -INT -f demo_franka_vive

# Restart ART gripper after a latched fault
sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh

# Cycle SteamVR cleanly
bash bin/start_vive_stack.sh --stop
bash bin/start_vive_stack.sh start

# What's running on which port?
ss -tln 2>/dev/null | grep -E ':50051|:50052|:50053|:4242|:12345|:12346'

# Where are recordings on disk?
ls -lh ~/Polymetis_Franka_Teleop/data/

# Check NUC RT health
ssh kist@192.168.1.12 '
    cat /sys/class/net/enp86s0/queues/rx-0/rps_cpus
    cat /sys/devices/system/cpu/intel_pstate/no_turbo
    systemctl is-active franka-rt-tune franka-dma-latency
'
```
