"""Resolve per-channel latency constants from install/latency_calibration.json.

The hardcoded defaults below mirror what was in franka_vive_env.py /
franka_policy_env.py before this refactor, so an absent JSON file (or a
JSON file missing a key) gives the exact same numbers as the previous code.
That makes the migration 100% backwards-compatible.

Programmatic use:

    from polymetis_franka_teleop.common.latency_config import (
        get_camera_obs_latency, get_gripper_obs_latency,
        get_robot_obs_latency, get_robot_action_latency,
        get_gripper_action_latency, patch_calibration,
    )

    cam_lat = get_camera_obs_latency('zed')          # 0.015
    grp_lat = get_gripper_obs_latency('art')         # 0.001
    grp_act = get_gripper_action_latency('art')      # 0.085

    # Calibrators write back into the file:
    patch_calibration({
        'camera_obs_latency': {'zed': 0.0123},
        '_calibration_dates': {'camera_zed': '2026-05-09'},
    })
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

# These hardcoded fallbacks are the V3 (2026-01-25) values that the env
# code shipped before the JSON-config refactor. Keep them here so the env
# behaves identically when no JSON file is present.
_FALLBACK = {
    'camera_obs_latency': {
        'zed': 0.015,
        'realsense': 0.015,
    },
    'robot_obs_latency': 0.001,
    'gripper_obs_latency': {
        'franka': 0.001,
        'art': 0.001,
    },
    'robot_action_latency': 0.055,
    'gripper_action_latency': {
        'franka': 0.085,
        'art': 0.085,
    },
}


def _config_path() -> Path:
    # repo_root/install/latency_calibration.json
    return Path(__file__).resolve().parent.parent.parent / 'install' / 'latency_calibration.json'


def load_calibration() -> dict:
    """Return the merged calibration: JSON file overlaid on _FALLBACK."""
    cfg = deepcopy(_FALLBACK)
    p = _config_path()
    if p.is_file():
        try:
            disk = json.loads(p.read_text())
        except Exception as e:
            print(f'[latency_config] WARN: cannot parse {p}: {e} -- using fallback')
            return cfg
        for k, v in disk.items():
            if k.startswith('_'):
                continue  # comments / metadata
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def get_camera_obs_latency(camera_backend: str) -> float:
    cfg = load_calibration()
    return float(cfg['camera_obs_latency'].get(camera_backend,
                 _FALLBACK['camera_obs_latency'].get(camera_backend, 0.015)))


def get_robot_obs_latency() -> float:
    return float(load_calibration()['robot_obs_latency'])


def get_gripper_obs_latency(gripper_backend: str) -> float:
    cfg = load_calibration()
    return float(cfg['gripper_obs_latency'].get(gripper_backend,
                 _FALLBACK['gripper_obs_latency'].get(gripper_backend, 0.001)))


def get_robot_action_latency() -> float:
    return float(load_calibration()['robot_action_latency'])


def get_gripper_action_latency(gripper_backend: str) -> float:
    cfg = load_calibration()
    return float(cfg['gripper_action_latency'].get(gripper_backend,
                 _FALLBACK['gripper_action_latency'].get(gripper_backend, 0.085)))


def patch_calibration(updates: dict) -> Path:
    """Deep-merge ``updates`` into the on-disk JSON, then return its path.

    Used by scripts_real/calibrate_*.py to write measured numbers back so
    subsequent env constructions pick them up automatically.
    """
    p = _config_path()
    if p.is_file():
        try:
            existing = json.loads(p.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    def _merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(existing, updates)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(existing, indent=2) + '\n')
    return p
