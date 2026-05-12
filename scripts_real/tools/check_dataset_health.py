#!/usr/bin/env python
"""Automated health check for a raw Polymetis_Franka_Teleop v2 session.

Automates the ``validation_checks_planned`` block written into every
session's ``dataset_meta.json`` (see ``franka_vive_env._write_recording_meta``).
Each check returns PASS / FAIL / SKIP plus a short message; the tool
exits 0 only when every selected check is PASS (or SKIP). Intended for:

  - Pre-conversion sanity (run before convert_to_gr00t_droid /
    convert_to_diffusion_policy to catch a corrupted session early).
  - Post-collection sanity (run after a recording session to verify
    the Batch 1.5 invariants held in production).
  - CI gate (--strict makes any FAIL exit 1).

Checks
------
1. obs_native truncation invariant
     obs_native arrays' shape[0] match obs_native_episode_ranges[-1, ?, 1]
     for both robot (8 keys) and gripper (2 keys) streams.
2. V-frame mp4 <-> npy 1:1
     ffprobe nb_read_frames == np.load(cam_<i>_frame_timestamps.npy).shape[0]
     for every (episode, cam) pair.
3. F2-path post-hoc resolve
     each videos/<ep>/calibration/cam_<i>.json frame_timestamps_path
     resolves to an existing npy file under the session root.
4. F4-legacy dict equality
     zarr meta.attrs['gripper_convention'] == dataset_meta.json
     ['gripper_convention'] (both dicts, key/value equal).
5. 6 attrs lists length == n_episodes
     episode_tasks / episode_ids / episode_task_ids / episode_scene_ids /
     episode_software_versions / episode_start_iso all match.
6. language.json normality
     each videos/<ep>/language.json exists and carries non-null
     instruction + task_id.
7. episode_ends consistency
     episode_ends is monotonic non-decreasing, last value matches the
     length of every data/<key> array.
8. (Optional) Controller log baseline -- requires --tee-log
     Recovery=0, Auto-HOME=0, j-limit warns under a soft threshold.

Usage
-----
    python scripts_real/tools/check_dataset_health.py \\
        --session data/session_20260511_234242

    # CI gate (any FAIL -> exit 1):
    python scripts_real/tools/check_dataset_health.py \\
        --session data/session_XYZ --strict

    # Subset of checks:
    python scripts_real/tools/check_dataset_health.py \\
        --session data/session_XYZ --checks 1,2,4

    # Include controller log analysis:
    python scripts_real/tools/check_dataset_health.py \\
        --session data/session_XYZ --tee-log /tmp/controller.log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr


# ============================== Result type ==============================


class _Result:
    """Tiny status carrier for the report formatter."""
    __slots__ = ("status", "message")

    def __init__(self, status: str, message: str = "") -> None:
        assert status in ("PASS", "FAIL", "SKIP"), status
        self.status = status
        self.message = message


# ============================== Check helpers ==============================


_NATIVE_ROBOT_KEYS = frozenset({
    "ts_robot", "joint_position", "joint_velocity", "joint_torque_external",
    "ee_pose_axis_angle", "ee_orientation_quat",
    "ee_linear_velocity", "ee_angular_velocity",
})
_NATIVE_GRIPPER_KEYS = frozenset({"ts_gripper", "gripper_position"})


def _check_obs_native_truncation(root: zarr.Group) -> _Result:
    if "obs_native_episode_ranges" not in root.get("meta", {}):
        return _Result("SKIP", "no obs_native_episode_ranges (pre-v2 session?)")
    rng = np.asarray(root["meta/obs_native_episode_ranges"][:])
    if rng.shape[0] == 0:
        return _Result("SKIP", "obs_native_episode_ranges is empty (0 episodes)")
    r_end = int(rng[-1, 0, 1])
    g_end = int(rng[-1, 1, 1])
    mismatches = []
    for k in sorted(root["obs_native"].array_keys()):
        actual = int(root["obs_native"][k].shape[0])
        if k in _NATIVE_ROBOT_KEYS and actual != r_end:
            mismatches.append(f"robot {k}: shape[0]={actual} != r_end={r_end}")
        elif k in _NATIVE_GRIPPER_KEYS and actual != g_end:
            mismatches.append(f"gripper {k}: shape[0]={actual} != g_end={g_end}")
        elif k not in _NATIVE_ROBOT_KEYS and k not in _NATIVE_GRIPPER_KEYS:
            mismatches.append(f"unknown key {k}: not in _NATIVE_{{ROBOT,GRIPPER}}_KEYS")
    if mismatches:
        return _Result("FAIL", " | ".join(mismatches))
    return _Result("PASS", f"r_end={r_end}, g_end={g_end}, 10/10 keys consistent")


def _check_v_frame(session: Path, n_eps: int) -> _Result:
    mismatches = []
    n_checked = 0
    for ep in range(n_eps):
        for cam in (0, 1):
            mp4 = session / "videos" / str(ep) / f"{cam}.mp4"
            npy = session / "videos" / str(ep) / f"cam_{cam}_frame_timestamps.npy"
            if not mp4.exists() or not npy.exists():
                mismatches.append(f"ep={ep} cam={cam}: missing artifact")
                continue
            try:
                res = subprocess.run(
                    ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-count_frames", "-show_entries", "stream=nb_read_frames",
                     "-of", "default=nokey=1:noprint_wrappers=1", str(mp4)],
                    capture_output=True, text=True,
                    env={"PATH": "/usr/bin:/bin"})
                mp4_n = int(res.stdout.strip())
            except Exception as e:
                mismatches.append(f"ep={ep} cam={cam}: ffprobe error: {e}")
                continue
            npy_n = int(np.load(str(npy)).shape[0])
            if mp4_n != npy_n:
                mismatches.append(f"ep={ep} cam={cam}: mp4={mp4_n} npy={npy_n} diff={mp4_n - npy_n:+d}")
            n_checked += 1
    if mismatches:
        return _Result("FAIL", " | ".join(mismatches[:5]) +
                       (f" ... (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""))
    return _Result("PASS", f"{n_checked}/{n_checked} ep*cam pairs match")


def _check_f2_path(session: Path, n_eps: int) -> _Result:
    missing = []
    n_checked = 0
    for ep in range(n_eps):
        calib_dir = session / "videos" / str(ep) / "calibration"
        if not calib_dir.is_dir():
            missing.append(f"ep={ep}: no calibration dir")
            continue
        for f in sorted(calib_dir.glob("cam_*.json")):
            with open(f) as fp:
                d = json.load(fp)
            ftp = d.get("frame_timestamps_path")
            if not ftp:
                missing.append(f"{f.name}: no frame_timestamps_path field")
                continue
            full = session / ftp
            if not full.exists():
                missing.append(f"{f.name}: target {full} does not exist")
                continue
            n_checked += 1
    if missing:
        return _Result("FAIL", " | ".join(missing[:5]))
    return _Result("PASS", f"{n_checked} cam JSONs resolve")


def _check_f4_dict_equality(session: Path, root: zarr.Group) -> _Result:
    zarr_attrs = dict(root["meta"].attrs)
    zarr_gc = zarr_attrs.get("gripper_convention")
    with open(session / "dataset_meta.json") as f:
        ds_meta = json.load(f)
    json_gc = ds_meta.get("gripper_convention")
    if zarr_gc is None:
        return _Result("FAIL", "zarr meta.attrs missing gripper_convention")
    if json_gc is None:
        return _Result("FAIL", "dataset_meta.json missing gripper_convention")
    if not isinstance(zarr_gc, dict) or not isinstance(json_gc, dict):
        return _Result("FAIL",
                       f"types: zarr={type(zarr_gc).__name__}, json={type(json_gc).__name__}")
    if zarr_gc != json_gc:
        return _Result("FAIL", f"dicts differ: zarr={zarr_gc} vs json={json_gc}")
    return _Result("PASS", "zarr ↔ json dict equality")


def _check_6_attrs_lengths(root: zarr.Group, n_eps: int) -> _Result:
    attrs = dict(root["meta"].attrs)
    list_keys = ("episode_tasks", "episode_ids", "episode_task_ids",
                 "episode_scene_ids", "episode_software_versions",
                 "episode_start_iso")
    mismatches = []
    for k in list_keys:
        v = list(attrs.get(k, []))
        if len(v) != n_eps:
            mismatches.append(f"{k}: len={len(v)} != n_eps={n_eps}")
    if mismatches:
        return _Result("FAIL", " | ".join(mismatches))
    return _Result("PASS", f"6/6 lists len == n_eps ({n_eps})")


def _check_language_json(session: Path, n_eps: int) -> _Result:
    missing = []
    for ep in range(n_eps):
        p = session / "videos" / str(ep) / "language.json"
        if not p.exists():
            missing.append(f"ep={ep}: file missing")
            continue
        with open(p) as f:
            d = json.load(f)
        if not d.get("instruction"):
            missing.append(f"ep={ep}: instruction null/empty")
        if not d.get("task_id"):
            missing.append(f"ep={ep}: task_id null/empty")
    if missing:
        return _Result("FAIL", " | ".join(missing[:5]))
    return _Result("PASS", f"{n_eps}/{n_eps} episodes with instruction + task_id")


def _check_episode_ends_consistency(root: zarr.Group) -> _Result:
    ee = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ee.shape[0] == 0:
        return _Result("SKIP", "episode_ends empty (0 episodes)")
    if np.any(np.diff(ee) <= 0):
        return _Result("FAIL", f"episode_ends not strictly increasing: {ee.tolist()}")
    last = int(ee[-1])
    data_lens = {}
    for k in sorted(root["data"].array_keys()):
        data_lens[k] = int(root["data"][k].shape[0])
    bad = {k: v for k, v in data_lens.items() if v != last}
    if bad:
        return _Result("FAIL", f"data/ arrays != episode_ends[-1]={last}: {bad}")
    return _Result("PASS", f"episode_ends[-1]={last}, all data/ arrays len={last}")


def _check_controller_log(tee_log: Path, jlimit_soft: int) -> _Result:
    if not tee_log.exists():
        return _Result("SKIP", f"log not found: {tee_log}")
    with open(tee_log) as f:
        lines = f.readlines()
    recovery = sum(1 for l in lines if "[FrankaPositionalController] Recovery #" in l)
    jlimit = sum(1 for l in lines if any(f"!! j{j} " in l for j in (3, 5, 6)))
    auto_home = sum(1 for l in lines
                    if "[FrankaPositionalController] !!" in l
                    and "recoveries in <10 s" in l)
    fail_msgs = []
    if recovery != 0:
        fail_msgs.append(f"Recovery={recovery}")
    if auto_home != 0:
        fail_msgs.append(f"AutoHOME={auto_home}")
    if jlimit > jlimit_soft:
        fail_msgs.append(f"j-limit={jlimit} > soft {jlimit_soft}")
    msg = f"Recovery={recovery}, AutoHOME={auto_home}, j-limit={jlimit}"
    if fail_msgs:
        return _Result("FAIL", f"{msg} | {' | '.join(fail_msgs)}")
    return _Result("PASS", msg)


# ============================== Reporting ==============================


_CHECKS = [
    (1, "obs_native truncation invariant"),
    (2, "V-frame mp4 ↔ npy 1:1"),
    (3, "F2-path post-hoc resolve"),
    (4, "F4-legacy dict equality"),
    (5, "6 attrs lists len == n_episodes"),
    (6, "language.json normality"),
    (7, "episode_ends consistency"),
    (8, "controller log baseline (optional)"),
]


def _print_report(session: Path, results: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f" Dataset health check: {session.name}")
    print(f"{'=' * 70}\n")
    for n, name in _CHECKS:
        if n not in results:
            continue
        r = results[n]
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[r.status]
        dots = "." * max(2, 45 - len(name))
        print(f"  [{n}] {name} {dots} {mark}  ({r.message})")
    pass_n = sum(1 for r in results.values() if r.status == "PASS")
    fail_n = sum(1 for r in results.values() if r.status == "FAIL")
    skip_n = sum(1 for r in results.values() if r.status == "SKIP")
    print(f"\nSummary: {pass_n} PASS, {fail_n} FAIL, {skip_n} SKIP")


# ============================== Main ==============================


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--session", required=True, type=Path,
                   help="Raw Polymetis_Franka_Teleop v2 session dir.")
    p.add_argument("--checks", default=None,
                   help="Comma-separated check numbers (1..8). Default: all.")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any check FAILs (CI gate).")
    p.add_argument("--tee-log", type=Path, default=None,
                   help="Controller stdout log (for check 8). Optional.")
    p.add_argument("--jlimit-soft", type=int, default=50,
                   help="Soft threshold for j-limit warns in check 8 (default 50).")
    args = p.parse_args()

    session = args.session.resolve()
    if not session.is_dir():
        print(f"ERROR: session not found: {session}", file=sys.stderr)
        return 1

    if args.checks:
        wanted = set(int(x) for x in args.checks.split(","))
    else:
        wanted = set(n for n, _ in _CHECKS)

    root = zarr.open(str(session / "replay_buffer.zarr"), mode="r")
    n_eps = int(np.asarray(root["meta/episode_ends"][:], dtype=np.int64).shape[0])

    results: dict[int, _Result] = {}
    if 1 in wanted: results[1] = _check_obs_native_truncation(root)
    if 2 in wanted: results[2] = _check_v_frame(session, n_eps)
    if 3 in wanted: results[3] = _check_f2_path(session, n_eps)
    if 4 in wanted: results[4] = _check_f4_dict_equality(session, root)
    if 5 in wanted: results[5] = _check_6_attrs_lengths(root, n_eps)
    if 6 in wanted: results[6] = _check_language_json(session, n_eps)
    if 7 in wanted: results[7] = _check_episode_ends_consistency(root)
    if 8 in wanted:
        if args.tee_log is None:
            results[8] = _Result("SKIP", "no --tee-log provided")
        else:
            results[8] = _check_controller_log(args.tee_log, args.jlimit_soft)

    _print_report(session, results)

    n_fail = sum(1 for r in results.values() if r.status == "FAIL")
    if args.strict and n_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
