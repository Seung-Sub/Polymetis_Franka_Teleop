"""Convert Polymetis_Franka_Teleop data → generic LeRobot v2.1 dataset.

Targets HuggingFace-LeRobot consumers (ACT, Diffusion Policy via LeRobot,
SmolVLA, ...). The state/action layout is configurable so the same
recording can be reshaped for whichever policy you're training.

This converter does NOT bake in the GR00T-DROID 17-D layout (use
``convert_to_gr00t_lerobot.py`` for that) or the UMI zarr.zip layout
(use ``convert_franka_vive_to_umi_format.py`` for that).

State / action layouts (selectable via ``--state_format``):

    joint  (default):   [joint_pos(7),  gripper(1)]                     = 8 D
    eef:                [eef_pos(3),    eef_aa(3),  gripper(1)]         = 7 D
    full:               [joint_pos(7),  eef_pos(3), eef_aa(3), grip(1)] = 14 D

Gripper repr (selectable via ``--gripper_repr``):

    normalized (default): 0.0 = open, 1.0 = closed  (matches GR00T / DROID)
    width:                raw meters (matches UMI / Diffusion-Policy convention)

Usage:
    # ACT — joint-space, normalized gripper, 15 fps
    python scripts_real/convert_to_lerobot.py \\
        --input  ./data/pap \\
        --output ./data/pap_act \\
        --task   "Pick up the yellow cup" \\
        --state_format joint \\
        --gripper_repr normalized

    # Diffusion-Policy via LeRobot — eef-space, raw gripper width
    python scripts_real/convert_to_lerobot.py \\
        --input  ./data/pap \\
        --output ./data/pap_dp \\
        --task   "Pick up the yellow cup" \\
        --state_format eef \\
        --gripper_repr width
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr


def _normalize_gripper(width_m: np.ndarray, gripper_max_width: float) -> np.ndarray:
    """0 = open, 1 = closed."""
    return np.clip(1.0 - width_m / gripper_max_width, 0.0, 1.0).astype(np.float32)


def _build_state(replay, state_format: str, gripper_repr: str,
                 gripper_max_width: float) -> np.ndarray:
    pos = replay['data']['robot0_eef_pos'][:].astype(np.float32)         # (N, 3)
    aa = replay['data']['robot0_eef_rot_axis_angle'][:].astype(np.float32)  # (N, 3)
    joint = replay['data']['robot0_joint_pos'][:].astype(np.float32)     # (N, 7)
    gw = replay['data']['robot0_gripper_width'][:].reshape(-1).astype(np.float32)

    g = _normalize_gripper(gw, gripper_max_width) if gripper_repr == 'normalized' else gw
    g = g.reshape(-1, 1)

    if state_format == 'joint':
        return np.concatenate([joint, g], axis=1)                # (N, 8)
    if state_format == 'eef':
        return np.concatenate([pos, aa, g], axis=1)              # (N, 7)
    if state_format == 'full':
        return np.concatenate([joint, pos, aa, g], axis=1)       # (N, 14)
    raise ValueError(f'Unknown state_format={state_format!r}')


def _build_action(replay, state_format: str, gripper_repr: str,
                  gripper_max_width: float, joint_state: np.ndarray) -> np.ndarray:
    """Action layout MUST match state layout — that's the LeRobot convention."""
    action = replay['data']['action'][:].astype(np.float32)              # (N, 7) — [pos(3), aa(3), gripper_width(1)]
    a_pos = action[:, 0:3]
    a_aa = action[:, 3:6]
    a_gw = action[:, 6]
    a_g = _normalize_gripper(a_gw, gripper_max_width) if gripper_repr == 'normalized' else a_gw
    a_g = a_g.reshape(-1, 1)

    if state_format == 'joint':
        # Joint-space target: shift state by 1 (target = next joint state),
        # tail repeated. Same proxy as GR00T-DROID converter.
        a_joint = np.concatenate([joint_state[1:], joint_state[-1:]], axis=0)
        return np.concatenate([a_joint, a_g], axis=1)            # (N, 8)
    if state_format == 'eef':
        return np.concatenate([a_pos, a_aa, a_g], axis=1)        # (N, 7)
    if state_format == 'full':
        a_joint = np.concatenate([joint_state[1:], joint_state[-1:]], axis=0)
        return np.concatenate([a_joint, a_pos, a_aa, a_g], axis=1)  # (N, 14)
    raise ValueError(f'Unknown state_format={state_format!r}')


def _episode_slices(episode_ends: np.ndarray):
    starts = np.concatenate([[0], episode_ends[:-1]])
    for i, (s, e) in enumerate(zip(starts, episode_ends)):
        yield i, int(s), int(e)


def _state_dim(state_format: str) -> int:
    return {'joint': 8, 'eef': 7, 'full': 14}[state_format]


def _write_info_json(out_meta_dir: Path, total_episodes: int, total_frames: int,
                     fps: int, video_keys: list[str], image_hw: tuple[int, int],
                     state_dim: int) -> None:
    h, w = image_hw
    features = {
        'observation.state': {'dtype': 'float32', 'shape': [state_dim]},
        'action':            {'dtype': 'float32', 'shape': [state_dim]},
        'task_index':        {'dtype': 'int64',   'shape': [1]},
    }
    _PREFIX = 'observation.images.'
    for key in video_keys:
        bare = key[len(_PREFIX):] if key.startswith(_PREFIX) else key
        features[f'{_PREFIX}{bare}'] = {
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


def _write_tasks_jsonl(out_meta_dir: Path, unique_tasks: list[str]) -> None:
    with (out_meta_dir / 'tasks.jsonl').open('w') as f:
        for i, t in enumerate(unique_tasks):
            f.write(json.dumps({'task_index': i, 'task': t}) + '\n')


def _write_episodes_jsonl(out_meta_dir: Path, episode_lengths: list[int],
                          per_episode_tasks: list[str]) -> None:
    with (out_meta_dir / 'episodes.jsonl').open('w') as f:
        for i, (n, t) in enumerate(zip(episode_lengths, per_episode_tasks)):
            f.write(json.dumps({
                'episode_index': i,
                'tasks': [t],
                'length': int(n),
            }) + '\n')


@click.command()
@click.option('--input', '-i', required=True,
              help='Input directory from demo_franka_vive.py')
@click.option('--output', '-o', required=True,
              help='Output LeRobot v2.1 dataset directory')
@click.option('--task', '-t', default=None,
              help='Task instruction. If omitted, falls back to per-episode '
                   'tasks recorded in zarr meta (set during demo_franka_vive '
                   'via env.start_episode(task=...)). One of (CLI --task, '
                   'zarr meta/episode_tasks) must be present.')
@click.option('--state_format', default='joint',
              type=click.Choice(['joint', 'eef', 'full']),
              help='Layout of observation.state and action. ACT typically '
                   'wants joint; eef-space DP wants eef; multi-modal wants full.')
@click.option('--gripper_repr', default='normalized',
              type=click.Choice(['normalized', 'width']),
              help='Gripper representation. normalized: 0=open 1=closed; '
                   'width: raw meters (UMI/DP convention).')
@click.option('--gripper_max_width', default=None, type=float,
              help='Gripper full-open width in meters. Default: read from '
                   'zarr meta/.attrs[gripper_max_width].')
@click.option('--fps', default=None, type=int,
              help='Dataset frequency. Default: read from zarr meta/'
                   '.attrs[frequency]. ACT/DP baseline = 10 Hz.')
@click.option('--video_keys', multiple=True,
              default=['observation.images.cam_high', 'observation.images.cam_wrist'],
              help='Per-camera video key (in camera-index order). Defaults '
                   'follow the LeRobot Aloha/ACT naming. For Diffusion-Policy '
                   'pass --video_keys observation.images.camera0_rgb '
                   '--video_keys observation.images.camera1_rgb.')
@click.option('--copy_videos/--symlink_videos', default=True,
              help='Copy mp4 (default) vs symlink')
def main(input, output, task, state_format, gripper_repr, gripper_max_width,
         fps, video_keys, copy_videos):
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
    print(f'[convert] state_format={state_format} ({_state_dim(state_format)}D), '
          f'gripper_repr={gripper_repr}')

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
    (out_dir / 'meta').mkdir(parents=True, exist_ok=True)

    replay = zarr.open(str(in_zarr), mode='r')
    meta_attrs = dict(replay['meta'].attrs)

    if gripper_max_width is None:
        gripper_max_width = float(meta_attrs.get('gripper_max_width', 0.100))
        print(f'[convert] gripper_max_width = {gripper_max_width:.4f} (from zarr meta)')
    else:
        print(f'[convert] gripper_max_width = {gripper_max_width:.4f} (CLI override)')
    if fps is None:
        fps = int(meta_attrs.get('frequency', 10))
        print(f'[convert] fps = {fps} (from zarr meta)')
    else:
        print(f'[convert] fps = {fps} (CLI override)')

    episode_tasks_meta = list(meta_attrs.get('episode_tasks', []))
    episode_ends = replay['meta']['episode_ends'][:]
    n_eps_input = len(episode_ends)
    if episode_tasks_meta and len(episode_tasks_meta) == n_eps_input:
        per_episode_tasks = list(episode_tasks_meta)
        print(f'[convert] using per-episode tasks from zarr meta '
              f'({n_eps_input} entries, {len(set(per_episode_tasks))} unique)')
    elif episode_tasks_meta and len(episode_tasks_meta) != n_eps_input:
        raise SystemExit(
            f'[convert] zarr meta has {len(episode_tasks_meta)} episode_tasks '
            f'but {n_eps_input} episodes recorded.')
    elif task is not None:
        per_episode_tasks = [task] * n_eps_input
        print(f'[convert] using single CLI --task for all {n_eps_input} episodes')
    else:
        raise SystemExit(
            '[convert] no task instruction available -- pass --task, or '
            'pre-record per-episode tasks via env.start_episode(task=...)')

    unique_tasks: list[str] = []
    task_to_idx: dict[str, int] = {}
    for t in per_episode_tasks:
        if t not in task_to_idx:
            task_to_idx[t] = len(unique_tasks)
            unique_tasks.append(t)
    ep_to_task_idx = [task_to_idx[t] for t in per_episode_tasks]

    # joint_state needed for action shift in joint/full modes
    joint_state = replay['data']['robot0_joint_pos'][:].astype(np.float32)
    state_full = _build_state(replay, state_format, gripper_repr, gripper_max_width)
    action_full = _build_action(replay, state_format, gripper_repr,
                                gripper_max_width, joint_state)
    timestamps = replay['data']['timestamp'][:]
    state_dim = state_full.shape[1]
    assert state_dim == _state_dim(state_format)

    # Detect cameras
    ep0_video_dir = in_videos / '0' if (in_videos / '0').exists() else None
    n_cameras = 0
    if ep0_video_dir is not None:
        n_cameras = len(sorted(ep0_video_dir.glob('*.mp4')))
    video_keys = list(video_keys)[:max(n_cameras, len(video_keys))]
    if n_cameras < len(video_keys):
        video_keys = video_keys[:n_cameras]

    image_hw = (0, 0)
    if ep0_video_dir is not None and n_cameras > 0:
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
            'task_index': np.full(n, ep_to_task_idx[ep_idx], dtype=np.int64),
            'frame_index': np.arange(n, dtype=np.int64),
            'episode_index': np.full(n, ep_idx, dtype=np.int64),
            'index': np.arange(s, e, dtype=np.int64),
        })
        out_pq = out_dir / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
        df.to_parquet(out_pq, index=False)

        src_ep = in_videos / str(ep_idx)
        for cam_idx, key in enumerate(video_keys):
            src = src_ep / f'{cam_idx}.mp4'
            if not src.exists():
                print(f'  [warn] missing {src}')
                continue
            # LeRobot v2.1 stores videos under <video_key> exactly as named
            # in features. Strip leading "observation.images." for the path
            # if present, so the on-disk dir matches the standard layout.
            dir_name = key.replace('observation.images.', '')
            dst_dir = out_dir / 'videos' / 'chunk-000' / dir_name
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
    _write_info_json(out_meta, total_episodes, total_frames, fps,
                     video_keys, image_hw, state_dim)
    _write_tasks_jsonl(out_meta, unique_tasks)
    _write_episodes_jsonl(out_meta, episode_lengths, per_episode_tasks)

    print(f'[convert] OK — {total_episodes} episodes, {total_frames} frames, '
          f'{state_dim}D state/action')


if __name__ == '__main__':
    main()
