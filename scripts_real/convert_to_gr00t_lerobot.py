"""Convert Polymetis_Franka_Teleop data → GR00T LeRobot v2.0 dataset.

Targets the OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT embodiment so that the
exported dataset can be used to fine-tune `nvidia/GR00T-N1.7-3B` (or
`nvidia/GR00T-N1.7-DROID`) directly.

Input  (produced by demo_franka_vive.py):
  <input>/replay_buffer.zarr                  Zarr replay buffer
    data/timestamp                       (N,)
    data/action                          (N, 7)   [x,y,z,rx,ry,rz, gripper_width_m]
    data/robot0_eef_pos                  (N, 3)
    data/robot0_eef_rot_axis_angle       (N, 3)
    data/robot0_joint_pos                (N, 7)
    data/robot0_gripper_width            (N, 1)
    meta/episode_ends                    (E,)
  <input>/videos/<episode_id>/<cam_idx>.mp4   per-camera H264 mp4

Output (LeRobot v2.0 + GR00T modality.json):
  <output>/data/chunk-000/episode_000000.parquet ...
  <output>/videos/chunk-000/<video_key>/episode_000000.mp4 ...
  <output>/meta/info.json
  <output>/meta/modality.json
  <output>/meta/tasks.jsonl
  <output>/meta/episodes.jsonl

Usage:
    python scripts_real/convert_to_gr00t_lerobot.py \
        --input ./data/pap \
        --output ./data/pap_gr00t \
        --task "Pick up the yellow cup" \
        --gripper_max_width 0.100   # ART; use 0.080 for Franka Hand
"""
from __future__ import annotations

import json
import shutil
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr
from scipy.spatial.transform import Rotation


# Same matrix as gr00t/examples/DROID/main_gr00t.py — converts robot EEF
# into the OXE-DROID egocentric convention before computing rot6d.
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64,
)


def _axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle → (N, 6) rot6d (first two rows of corrected R)."""
    R = Rotation.from_rotvec(axis_angle).as_matrix()        # (N, 3, 3)
    R = R @ DROID_EEF_ROTATION_CORRECT                      # apply DROID convention
    return R[:, :2, :].reshape(-1, 6)                       # first two rows flattened


def _build_eef_9d(pos: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, _axis_angle_to_rot6d(axis_angle)], axis=1).astype(np.float32)


def _build_state_action(replay, gripper_max_width: float):
    """Pack the per-step state and action arrays in DROID 17-D layout.

    State 17-D: [eef_9d(9), gripper_position(1), joint_position(7)]
    Action 17-D: same dimensions.
    """
    n = replay['data']['timestamp'].shape[0]
    pos = replay['data']['robot0_eef_pos'][:]                # (N, 3)
    aa = replay['data']['robot0_eef_rot_axis_angle'][:]      # (N, 3)
    joint = replay['data']['robot0_joint_pos'][:]            # (N, 7)
    gw = replay['data']['robot0_gripper_width'][:].reshape(-1)  # (N,)
    action = replay['data']['action'][:]                     # (N, 7)

    eef_state = _build_eef_9d(pos, aa)                       # (N, 9)
    # gripper_position normalized: 0=open, 1=closed (matches DROID convention)
    gripper_state = (1.0 - gw / gripper_max_width).clip(0.0, 1.0).astype(np.float32).reshape(-1, 1)
    state = np.concatenate([eef_state, gripper_state, joint.astype(np.float32)], axis=1)  # (N, 17)

    # Action — use the recorded teleop target_pose (action[:6] = pose, action[6] = gripper_width)
    a_pos = action[:, 0:3]
    a_aa = action[:, 3:6]
    a_eef = _build_eef_9d(a_pos, a_aa)                       # (N, 9)
    a_gripper = (1.0 - action[:, 6] / gripper_max_width).clip(0.0, 1.0).astype(np.float32).reshape(-1, 1)
    # For joint action: target joint of the next timestep — proxy with state[t+1]; tail is repeated
    a_joint = np.concatenate([joint[1:], joint[-1:]], axis=0).astype(np.float32)
    act = np.concatenate([a_eef, a_gripper, a_joint], axis=1)  # (N, 17)

    assert state.shape == (n, 17) and act.shape == (n, 17), (state.shape, act.shape)
    return state, act


def _episode_slices(episode_ends: np.ndarray):
    """Yield (episode_idx, start, end) for each episode using cumulative ends."""
    starts = np.concatenate([[0], episode_ends[:-1]])
    for i, (s, e) in enumerate(zip(starts, episode_ends)):
        yield i, int(s), int(e)


def _write_modality_json(out_meta_dir: Path, n_cameras: int):
    """Match gr00t/configs/data/embodiment_configs.py oxe_droid_relative_eef_relative_joint."""
    video_block = {}
    if n_cameras >= 1:
        video_block['exterior_image_1_left'] = {'original_key': 'observation.images.exterior_image_1_left'}
    if n_cameras >= 2:
        video_block['wrist_image_left'] = {'original_key': 'observation.images.wrist_image_left'}
    modality = {
        'state': {
            'eef_9d':           {'start': 0,  'end': 9},
            'gripper_position': {'start': 9,  'end': 10},
            'joint_position':   {'start': 10, 'end': 17},
        },
        'action': {
            'eef_9d':           {'start': 0,  'end': 9},
            'gripper_position': {'start': 9,  'end': 10},
            'joint_position':   {'start': 10, 'end': 17},
        },
        'video': video_block,
        'annotation': {
            'language.language_instruction': {'original_key': 'task_index'},
        },
    }
    (out_meta_dir / 'modality.json').write_text(json.dumps(modality, indent=2))


def _write_info_json(out_meta_dir: Path, total_episodes: int, total_frames: int,
                     fps: int, n_cameras: int, image_hw: tuple[int, int]):
    h, w = image_hw
    features = {
        'observation.state': {'dtype': 'float32', 'shape': [17]},
        'action': {'dtype': 'float32', 'shape': [17]},
        'task_index': {'dtype': 'int64', 'shape': [1]},
    }
    if n_cameras >= 1:
        features['observation.images.exterior_image_1_left'] = {
            'dtype': 'video', 'shape': [h, w, 3]}
    if n_cameras >= 2:
        features['observation.images.wrist_image_left'] = {
            'dtype': 'video', 'shape': [h, w, 3]}
    info = {
        'codebase_version': 'v2.1',
        'robot_type': 'franka_panda',
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'fps': fps,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'chunks_size': 1000,
        'splits': {'train': f'0:{total_episodes}'},
        'features': features,
    }
    (out_meta_dir / 'info.json').write_text(json.dumps(info, indent=2))


def _write_tasks_jsonl(out_meta_dir: Path, task_text: str):
    with (out_meta_dir / 'tasks.jsonl').open('w') as f:
        f.write(json.dumps({'task_index': 0, 'task': task_text}) + '\n')


def _write_episodes_jsonl(out_meta_dir: Path, episode_lengths: list[int], task_text: str):
    with (out_meta_dir / 'episodes.jsonl').open('w') as f:
        for i, n in enumerate(episode_lengths):
            f.write(json.dumps({
                'episode_index': i,
                'tasks': [task_text],
                'length': int(n),
            }) + '\n')


@click.command()
@click.option('--input', '-i', required=True, help='Input directory from demo_franka_vive.py')
@click.option('--output', '-o', required=True, help='Output GR00T-LeRobot v2 dataset directory')
@click.option('--task', '-t', required=True, help='Task instruction (one string for all episodes)')
@click.option('--gripper_max_width', default=0.100, type=float,
              help='Gripper full-open width in meters (0.080 Franka Hand, 0.100 ART)')
@click.option('--fps', default=10, type=int, help='Recording frequency (=demo --frequency)')
@click.option('--video_keys', multiple=True,
              default=['exterior_image_1_left', 'wrist_image_left'],
              help='GR00T video keys in camera-index order')
@click.option('--copy_videos/--symlink_videos', default=True,
              help='Copy mp4 (default) vs symlink')
def main(input, output, task, gripper_max_width, fps, video_keys, copy_videos):
    in_dir = Path(input).resolve()
    out_dir = Path(output).resolve()

    in_zarr = in_dir / 'replay_buffer.zarr'
    in_videos = in_dir / 'videos'
    if not in_zarr.is_dir():
        raise SystemExit(f'replay_buffer.zarr not found at {in_zarr}')
    if not in_videos.is_dir():
        raise SystemExit(f'videos/ not found at {in_videos}')

    print(f'[convert] in:  {in_dir}')
    print(f'[convert] out: {out_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
    (out_dir / 'meta').mkdir(parents=True, exist_ok=True)

    replay = zarr.open(str(in_zarr), mode='r')
    episode_ends = replay['meta']['episode_ends'][:]
    state_full, action_full = _build_state_action(replay, gripper_max_width)
    timestamps = replay['data']['timestamp'][:]

    # Detect actual camera count from the first episode's video dir
    ep0_video_dir = in_videos / '0' if (in_videos / '0').exists() else None
    n_cameras = 0
    if ep0_video_dir is not None:
        n_cameras = len(sorted(ep0_video_dir.glob('*.mp4')))
    n_cameras = max(n_cameras, len(video_keys))

    # Probe image resolution from first video (for info.json features.shape)
    image_hw = (0, 0)
    if ep0_video_dir is not None:
        import cv2
        first_mp4 = sorted(ep0_video_dir.glob('*.mp4'))[0]
        cap = cv2.VideoCapture(str(first_mp4))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_hw = (h, w)
        cap.release()

    episode_lengths = []
    for ep_idx, s, e in _episode_slices(episode_ends):
        n = e - s
        episode_lengths.append(n)

        df = pd.DataFrame({
            'observation.state': list(state_full[s:e].astype(np.float32)),
            'action': list(action_full[s:e].astype(np.float32)),
            'timestamp': timestamps[s:e].astype(np.float64),
            'task_index': np.zeros(n, dtype=np.int64),
            'frame_index': np.arange(n, dtype=np.int64),
            'episode_index': np.full(n, ep_idx, dtype=np.int64),
            'index': np.arange(s, e, dtype=np.int64),
        })
        out_pq = out_dir / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
        df.to_parquet(out_pq, index=False)

        # videos: input <input>/videos/<ep_idx>/<cam_idx>.mp4
        # output <output>/videos/chunk-000/<video_key>/episode_<ep>.mp4
        src_ep = in_videos / str(ep_idx)
        for cam_idx, key in enumerate(video_keys[:n_cameras]):
            src = src_ep / f'{cam_idx}.mp4'
            if not src.exists():
                print(f'  [warn] missing {src}')
                continue
            dst_dir = out_dir / 'videos' / 'chunk-000' / key
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f'episode_{ep_idx:06d}.mp4'
            if dst.exists():
                dst.unlink()
            if copy_videos:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)

        print(f'  [ep {ep_idx}] {n} steps → {out_pq.name}')

    total_frames = int(sum(episode_lengths))
    total_episodes = len(episode_lengths)

    out_meta = out_dir / 'meta'
    _write_info_json(out_meta, total_episodes, total_frames, fps, n_cameras, image_hw)
    _write_modality_json(out_meta, n_cameras)
    _write_tasks_jsonl(out_meta, task)
    _write_episodes_jsonl(out_meta, episode_lengths, task)

    print(f'[convert] OK — {total_episodes} episodes, {total_frames} frames')
    print(f'[convert] embodiment-tag suggestion: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT')


if __name__ == '__main__':
    main()
