"""Common conversion utilities shared by the GR00T-DROID and DP converters.

Purpose
-------
Single source of truth for the per-session bookkeeping that both
``convert_to_gr00t_droid.py`` and ``convert_to_diffusion_policy.py`` need:

  - ``load_session``      — open zarr, dataset_meta.json, per-episode
                            language.json once, resolve gripper max_width
                            from the gripper_convention block.
  - ``classify_cams``     — baseline-based wrist / exterior detection
                            with selectable role naming.
  - ``WRIST_BASELINE_MAX_M`` — single threshold for the classifier.

Anything that diverges between converters (action layout, output tree,
parquet vs HDF5, etc.) stays in the converter modules.  Per Phase 2-6
spec: ``compute_state_action`` / ``write_*`` / ``symlink_videos`` /
``resolve_instruction`` are **not** moved here — their semantics are
target-specific.

Regression invariant
--------------------
This module's outputs are byte-identical to the previous inline
implementations in each converter (verified by V-A 90-cell round-trip
post-refactor: GR00T 48 + DP 42 cells, all bitwise 0.0e+00).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr


# Stereo baseline threshold separating ZED Mini (~63 mm = wrist) from
# ZED 2i (~120 mm = exterior). Same value previously inlined in both
# ``convert_to_gr00t_droid.py`` and ``convert_to_diffusion_policy.py``.
WRIST_BASELINE_MAX_M: float = 0.08


# Role naming conventions for classify_cams() output. Different consumers
# of the cam-to-video mapping expect different forms of the role string:
#
#   "lerobot": LeRobot v2.1 video path expects
#              ``observation.images.<role>`` keys (used by the GR00T
#              converter when writing the info.json features block and
#              video tree directory names).
#   "plain":   robomimic / DP HDF5 stores the role as a bare string under
#              the demo's video_paths attr (used by the DP converter when
#              symlinking ``videos/demo_<i>/<role>.mp4``).
_VALID_ROLE_NAMINGS = ("lerobot", "plain")


# ============================== load_session ==============================


def load_session(session_dir: Path) -> dict:
    """Open the session zarr + dataset_meta.json + per-episode language.json.

    Returns
    -------
    dict with keys::
        session_dir   : Path (resolved input dir, useful for downstream
                        video-path resolution)
        root          : zarr.Group (read-mode handle to replay_buffer.zarr)
        ds_meta       : dict (parsed dataset_meta.json)
        max_width     : float (resolved gripper max width for the active
                        backend, e.g. 0.095 for ART, 0.075 for Franka Hand)
        episode_ends  : np.ndarray (int64) (UMI ReplayBuffer episode boundaries)
        n_episodes    : int
        languages     : list[dict] (per-episode language.json contents,
                        empty dict for episodes whose JSON is missing)

    Raises
    ------
    FileNotFoundError
        If ``session_dir`` does not exist as a directory.
    KeyError
        If ``dataset_meta.json`` lacks the gripper_convention.max_width_m
        entry for the active backend.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session dir not found: {session_dir}")
    root = zarr.open(str(session_dir / "replay_buffer.zarr"), mode="r")

    with open(session_dir / "dataset_meta.json") as f:
        ds_meta = json.load(f)

    # ``gripper_type`` in dataset_meta.json is the backend short name
    # ("art" / "franka"); the gripper_convention block keys are the full
    # backend names ("art_gripper" / "franka_hand"); translate.
    gripper_type = ds_meta["gripper_type"]
    max_width_key = (
        "art_gripper"  if gripper_type == "art"
        else "franka_hand" if gripper_type == "franka"
        else f"{gripper_type}_gripper"
    )
    max_width = float(ds_meta["gripper_convention"]["max_width_m"][max_width_key])

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    n_episodes = int(episode_ends.shape[0])

    languages: list[dict] = []
    for ep in range(n_episodes):
        lj = session_dir / "videos" / str(ep) / "language.json"
        if lj.exists():
            with open(lj) as f:
                languages.append(json.load(f))
        else:
            languages.append({})

    return {
        "session_dir":  session_dir,
        "root":         root,
        "ds_meta":      ds_meta,
        "max_width":    max_width,
        "episode_ends": episode_ends,
        "n_episodes":   n_episodes,
        "languages":    languages,
    }


# ============================== classify_cams ==============================


def classify_cams(
    session_dir: Path,
    episode_index: int = 0,
    role_naming: str = "lerobot",
) -> dict[int, str]:
    """Map each ``cam_<i>.json`` to its DROID video role by stereo baseline.

    Reads the per-camera intrinsics emitted by the session at end_episode
    time and splits on stereo baseline -- ZED Mini (~63 mm) is the wrist
    cam, ZED 2i (~120 mm) is the third / exterior cam. Robust to operator
    reordering ``--camera_serials`` at the demo wrapper since the
    classification is driven by a measured intrinsic, not the positional
    index.

    Parameters
    ----------
    session_dir : Path
        Raw Polymetis_Franka_Teleop session directory.
    episode_index : int
        Which episode's calibration JSON to inspect (default 0). The
        physical setup is constant across a session, so any episode
        works; we use 0 by convention.
    role_naming : {"lerobot", "plain"}
        - ``"lerobot"``: returns ``observation.images.<role>`` keys, used
          by the GR00T converter when building the LeRobot video tree
          and info.json features block.
        - ``"plain"``: returns bare ``<role>`` keys, used by the DP
          converter under ``videos/demo_<i>/<role>.mp4``.

    Returns
    -------
    dict[int, str]
        ``{cam_idx: role_name}`` with exactly two entries.

    Raises
    ------
    ValueError
        If ``role_naming`` is not in ``("lerobot", "plain")``.
    RuntimeError
        If the calibration directory doesn't contain exactly 2
        ``cam_<i>.json`` files, or if both cameras fall on the same
        side of ``WRIST_BASELINE_MAX_M`` (ambiguous setup -- two ZED 2i
        or two ZED Mini).
    """
    if role_naming not in _VALID_ROLE_NAMINGS:
        raise ValueError(
            f"role_naming must be one of {_VALID_ROLE_NAMINGS}, got {role_naming!r}"
        )

    session_dir = Path(session_dir)
    calib_dir = session_dir / "videos" / str(episode_index) / "calibration"
    cam_files = sorted(calib_dir.glob("cam_*.json"))
    if len(cam_files) != 2:
        raise RuntimeError(
            f"expected 2 cam_<i>.json in {calib_dir}, found {len(cam_files)}: "
            f"{[p.name for p in cam_files]}"
        )

    baselines: dict[int, float] = {}
    for p in cam_files:
        with open(p) as fp:
            calib = json.load(fp)
        cam_idx = int(p.stem.split("_")[1])
        baselines[cam_idx] = float(calib["intrinsics"]["baseline_m"])

    wrist_idx    = min(baselines, key=baselines.get)
    exterior_idx = max(baselines, key=baselines.get)
    if baselines[wrist_idx] >= WRIST_BASELINE_MAX_M:
        raise RuntimeError(
            f"Cannot classify wrist vs exterior cam: both baselines "
            f">= {WRIST_BASELINE_MAX_M} m ({baselines}). Looks like two "
            f"ZED 2i (or two of the same model)."
        )
    if baselines[exterior_idx] < WRIST_BASELINE_MAX_M:
        raise RuntimeError(
            f"Cannot classify wrist vs exterior cam: both baselines "
            f"< {WRIST_BASELINE_MAX_M} m ({baselines}). Looks like two "
            f"ZED Mini."
        )

    if role_naming == "lerobot":
        return {
            wrist_idx:    "observation.images.wrist_image_left",
            exterior_idx: "observation.images.exterior_image_1_left",
        }
    # role_naming == "plain"
    return {
        wrist_idx:    "wrist_image_left",
        exterior_idx: "exterior_image_1_left",
    }
