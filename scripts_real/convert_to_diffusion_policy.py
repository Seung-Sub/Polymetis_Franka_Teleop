#!/usr/bin/env python
"""Convert a Polymetis_Franka_Teleop session into a robomimic-style HDF5
dataset for Diffusion Policy training.

Phase 2-4 final spec (cmd-based DROID-style action, DP-standard 7D action +
DP's RotationTransformer converts axis_angle -> rot6d at training time; obs
exposed at 10D rot6d with axis_angle also stored for future obs_keys
flexibility).

Layout produced
---------------
<output>/
  demos.hdf5                              robomimic-style:
    data/
      demo_<i>/
        obs/
          robot0_eef_pos              (T, 3)  float32
          robot0_eef_rot6d            (T, 6)  float32
          robot0_eef_rot_axis_angle   (T, 3)  float32
          robot0_gripper_position     (T, 1)  float32  (normalized [0, 1])
        actions                       (T, 7)  float32  (pos+aa+gripper)
        rewards                       (T,)    float32  (zeros placeholder)
        dones                         (T,)    int8     (last frame = 1)
        states                        (T, 10) float32  (obs concat: pos+rot6d+gripper)
        .attrs: num_samples, language_instruction, task_id, video_paths
      .attrs: total, env_args
  videos/
    demo_<i>/
      exterior_image_1_left.mp4   symlink to session videos/<i>/<cam_idx>.mp4
      wrist_image_left.mp4        same, different cam
  meta.json                       conversion metadata

Action / state representation
-----------------------------
At STORE TIME (HDF5):
  state[0:3]  = robot0_eef_pos[t]                          (TCP frame, m)
  state[3:9]  = rot6d(robot0_eef_orientation_quat[t])      (Zhou et al.)
  state[9:10] = robot0_gripper_width[t] / max_width_m      (normalized)

  action[0:3]  = action_ee_position_cmd[t]                 (commanded TCP pos)
  action[3:6]  = quat_to_aa(action_ee_orientation_quat_cmd[t])  (axis-angle)
  action[6:7]  = robot0_gripper_width[t+1] / max_width_m   (gripper, next-step
                                                            state proxy with
                                                            last-frame clamp)

At LOAD TIME (DP RobomimicReplayLowdimDataset with abs_action=True,
                rotation_rep='rotation_6d'):
  rotation_transformer applies axis_angle -> rot6d on action[3:6].
  Result: 10D model-side action (pos 3 + rot6d 6 + gripper 1), matching
  GR00T's representation. Storage stays DP-standard 7D so the upstream
  abs_action_only_normalizer applies as designed.

obs is concatenated by the DP loader from the obs_keys list the user
chooses at training time -- we just provide the four keys; the user
picks 3 (or 4) of them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from polymetis_franka_teleop.common.rotation_util import quat_to_rot6d

# ``_conversion_common`` lives next to this script; Python adds the
# script's directory to sys.path[0] when invoked as
# ``python scripts_real/<file>.py``, so a same-dir import works without
# making ``scripts_real`` a package.
from _conversion_common import (  # noqa: E402  (sibling-script import)
    classify_cams,
    load_session,
)


# ============================== Constants ==============================

ACTION_DIM = 7   # pos(3) + axis_angle(3) + gripper(1)
STATE_DIM  = 10  # pos(3) + rot6d(6) + gripper(1)  -- stored in /states for robomimic

# DP-style role names; symlinked under videos/demo_<i>/<role>.mp4. These
# are the values ``classify_cams(role_naming="plain")`` returns.
ROLE_WRIST    = "wrist_image_left"
ROLE_EXTERIOR = "exterior_image_1_left"


# ============================== Per-episode obs / state / action ==============================


def compute_obs_state_action(
    root: zarr.Group,
    max_width: float,
    ep_start: int,
    ep_end: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Slice one episode and assemble (obs_dict, states, actions).

    Returns
    -------
    obs : dict
        Four float32 arrays: robot0_eef_pos (T,3), robot0_eef_rot6d (T,6),
        robot0_eef_rot_axis_angle (T,3), robot0_gripper_position (T,1).
    states : (T, 10) float32
        Concatenation of pos + rot6d + gripper -- matches what the DP
        loader produces when obs_keys = ['robot0_eef_pos',
        'robot0_eef_rot6d', 'robot0_gripper_position']. Kept in /states
        for robomimic compatibility (some downstream tools read it directly
        for reset-state replay).
    actions : (T, 7) float32
        DP-standard format consumed by RobomimicReplayLowdimDataset's
        abs_action transform (pos 3 + axis_angle 3 + gripper 1).
    """
    s = slice(ep_start, ep_end)
    eef_pos    = np.asarray(root["data/robot0_eef_pos"][s],                  dtype=np.float64)
    eef_quat   = np.asarray(root["data/robot0_eef_orientation_quat"][s],     dtype=np.float64)
    eef_aa_raw = np.asarray(root["data/robot0_eef_rot_axis_angle"][s],       dtype=np.float64)
    g_width    = np.asarray(root["data/robot0_gripper_width"][s],            dtype=np.float64).reshape(-1, 1)
    cmd_pos    = np.asarray(root["data/action_ee_position_cmd"][s],          dtype=np.float64)
    cmd_quat   = np.asarray(root["data/action_ee_orientation_quat_cmd"][s],  dtype=np.float64)

    n = eef_pos.shape[0]
    if n == 0:
        raise ValueError(f"empty episode slice [{ep_start}:{ep_end}]")

    eef_rot6d = quat_to_rot6d(eef_quat)                              # (T, 6)
    cmd_aa    = Rotation.from_quat(cmd_quat).as_rotvec()              # (T, 3)
    g_norm    = g_width / max_width                                    # (T, 1)

    # action gripper: next-step state with last-frame clamp.
    g_next = g_norm.copy()
    if n > 1:
        g_next[:-1] = g_norm[1:]
    # g_next[-1] stays as g_norm[-1] (fallback)

    obs = {
        "robot0_eef_pos":            eef_pos.astype(np.float32),
        "robot0_eef_rot6d":          eef_rot6d.astype(np.float32),
        "robot0_eef_rot_axis_angle": eef_aa_raw.astype(np.float32),
        "robot0_gripper_position":   g_norm.astype(np.float32),
    }
    states = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_rot6d"],
        obs["robot0_gripper_position"],
    ], axis=-1).astype(np.float32)
    actions = np.concatenate([
        cmd_pos.astype(np.float32),
        cmd_aa.astype(np.float32),
        g_next.astype(np.float32),
    ], axis=-1)

    return obs, states, actions


# ============================== Cam classification (shared logic) ==============================


def symlink_videos(
    session_dir: Path,
    output_dir: Path,
    n_episodes: int,
    cam_video_map: dict[int, str],
) -> None:
    """videos/demo_<i>/<role>.mp4 -> session videos/<ep>/<cam_idx>.mp4."""
    for ep in range(n_episodes):
        demo_dir = output_dir / "videos" / f"demo_{ep}"
        demo_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx, role in cam_video_map.items():
            src = (session_dir / "videos" / str(ep) / f"{cam_idx}.mp4").resolve()
            if not src.exists():
                raise FileNotFoundError(f"missing mp4 for ep={ep} cam={cam_idx}: {src}")
            dst = demo_dir / f"{role}.mp4"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)


# ============================== HDF5 write ==============================


def write_hdf5(
    output_path: Path,
    demos_data: list[dict],
    total: int,
    fps: int,
) -> None:
    """robomimic-style HDF5: data/demo_<i>/{obs, actions, rewards, dones, states}."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["total"] = int(total)
        # env_args is required by some robomimic tools but not by DP's
        # RobomimicReplayLowdimDataset; keep a documentation-quality JSON
        # string so robomimic envrunner / playback tools don't choke.
        data_grp.attrs["env_args"] = json.dumps({
            "env_name": "franka_real_pick_place",
            "type": "real",
            "env_kwargs": {"fps": fps},
        })

        for i, d in enumerate(demos_data):
            demo_grp = data_grp.create_group(f"demo_{i}")

            # obs/<key>
            obs_grp = demo_grp.create_group("obs")
            for key, arr in d["obs"].items():
                obs_grp.create_dataset(key, data=arr, dtype="float32")

            T = int(d["actions"].shape[0])
            demo_grp.create_dataset("actions", data=d["actions"], dtype="float32")
            demo_grp.create_dataset("rewards", data=np.zeros(T, dtype=np.float32), dtype="float32")
            dones = np.zeros(T, dtype=np.int8); dones[-1] = 1
            demo_grp.create_dataset("dones", data=dones, dtype="int8")
            demo_grp.create_dataset("states", data=d["states"], dtype="float32")

            demo_grp.attrs["num_samples"]         = T
            demo_grp.attrs["language_instruction"] = d["instruction"]
            demo_grp.attrs["task_id"]              = d["task_id"]
            demo_grp.attrs["video_paths"]          = json.dumps(d["video_paths"])


# ============================== Main ==============================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-session",  required=True, type=Path,
                        help="Polymetis_Franka_Teleop session dir.")
    parser.add_argument("--output-dataset", required=True, type=Path,
                        help="DP HDF5 dataset output dir.")
    parser.add_argument("--max-episodes",   type=int, default=-1,
                        help="Convert only the first N episodes (debugging). -1 = all.")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    session_dir = args.input_session.resolve()
    output_dir  = args.output_dataset.resolve()

    session = load_session(session_dir)
    fps = int(session["ds_meta"]["main_loop_rate_hz"])

    n_eps = session["n_episodes"]
    if args.max_episodes > 0:
        n_eps = min(n_eps, args.max_episodes)
    if n_eps == 0:
        print("No episodes to convert.", file=sys.stderr)
        return 1

    # ``role_naming="plain"`` returns bare role strings (wrist_image_left,
    # exterior_image_1_left) suitable for the DP videos/demo_<i>/<role>.mp4 layout.
    cam_video_map = classify_cams(session_dir, episode_index=0, role_naming="plain")
    if args.verbose:
        print(f"  cam_video_map = {cam_video_map}", flush=True)

    demos_data: list[dict] = []
    total = 0
    for ep in range(n_eps):
        ep_start = 0 if ep == 0 else int(session["episode_ends"][ep - 1])
        ep_end   = int(session["episode_ends"][ep])

        obs, states, actions = compute_obs_state_action(
            session["root"], session["max_width"], ep_start, ep_end,
        )

        lj = session["languages"][ep]
        instruction = lj.get("instruction") or ""
        task_id     = lj.get("task_id") or ""
        video_paths = {role: f"videos/demo_{ep}/{role}.mp4" for role in cam_video_map.values()}

        demos_data.append({
            "obs":         obs,
            "states":      states,
            "actions":     actions,
            "instruction": instruction,
            "task_id":     task_id,
            "video_paths": video_paths,
        })
        total += int(actions.shape[0])
        if args.verbose:
            print(f"  demo {ep}: T={actions.shape[0]}  state.shape={states.shape}  "
                  f"action.shape={actions.shape}  task_id={task_id}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_hdf5(output_dir / "demos.hdf5", demos_data, total, fps)
    symlink_videos(session_dir, output_dir, n_eps, cam_video_map)

    # meta.json side-car (documentation, not consumed by DP loader)
    with open(output_dir / "meta.json", "w") as f:
        json.dump({
            "source_session":     str(session_dir),
            "n_demos":            n_eps,
            "total_frames":       int(total),
            "fps":                fps,
            "action_format":      f"{ACTION_DIM}D (pos+axis_angle+gripper), abs cmd, DROID convention",
            "action_loader_note": ("DP RobomimicReplayLowdimDataset with abs_action=True "
                                   "and rotation_rep='rotation_6d' will auto-convert "
                                   "axis_angle -> rot6d at training time, producing 10D "
                                   "action on the model side."),
            "state_format":       f"{STATE_DIM}D (pos+rot6d+gripper), stored as /states",
            "obs_keys": [
                "robot0_eef_pos",
                "robot0_eef_rot6d",
                "robot0_eef_rot_axis_angle",
                "robot0_gripper_position",
            ],
            "rot6d_convention_note": (
                "obs/robot0_eef_rot6d uses Zhou et al. COLUMN-major flatten "
                "(via polymetis_franka_teleop.common.rotation_util.quat_to_rot6d): "
                "[m00, m10, m20, m01, m11, m21] -- consistent with the GR00T converter. "
                "DP's RotationTransformer (pytorch3d) instead produces ROW-major rot6d "
                "[m00, m01, m02, m10, m11, m12] when it transforms the action's axis_angle "
                "at training time. Both are valid Zhou-style 6D representations -- they "
                "share the same first two basis vectors but in different element ordering. "
                "For strict obs<->action format consistency, use obs_keys without "
                "robot0_eef_rot6d (e.g. ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', "
                "'robot0_gripper_position']) and let the DP loader / training script "
                "transform axis_angle to rot6d uniformly on both sides."
            ),
            "cam_video_map":      {str(k): v for k, v in cam_video_map.items()},
        }, f, indent=2)

    print(f"\nConverted {n_eps} demo(s) -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
