#!/usr/bin/env python
"""Round-trip regression test for the GR00T-DROID and Diffusion Policy converters.

This tool reads a raw Polymetis_Franka_Teleop session, recomputes the
expected state/action values per converter's formula, and compares
element-wise against what the converter actually wrote into its output
dataset. Both converters' V-A round-trip checks live here so we have a
single regression entry point.

Invariant: with no code regression, every cell of the matrix prints
``0.0e+00 OK`` (bitwise match — the converter's formula and this tool's
formula are intentionally identical, sharing the same
``polymetis_franka_teleop.common.rotation_util`` and
``scripts_real._conversion_common`` helpers).

Usage
-----
    # Both converters at once (recommended for CI / pre-push):
    python scripts_real/tools/round_trip_test.py \\
        --converter both \\
        --input-session data/session_20260511_234242

    # Single converter:
    python scripts_real/tools/round_trip_test.py \\
        --converter gr00t \\
        --input-session data/session_20260511_234242

Exit code: 0 if every cell PASSes (max diff < ``--threshold``),
1 otherwise -- suitable for CI gating.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from scipy.spatial.transform import Rotation

from polymetis_franka_teleop.common.rotation_util import quat_to_rot6d


_GR00T_SLOTS = (
    "state.eef_pos",     "state.eef_rot6d",   "state.gripper",    "state.joint",
    "action.eef_pos",    "action.eef_rot6d",  "action.gripper",   "action.joint",
)
_DP_SLOTS = (
    "obs.eef_pos",       "obs.eef_rot6d",     "obs.eef_aa",       "obs.gripper",
    "action.eef_pos",    "action.eef_aa",     "action.gripper_next",
)


# ============================== GR00T round-trip ==============================


def _gr00t_matrix(
    session_dir: Path,
    dataset_dir: Path,
    max_width: float,
    episode_ends: np.ndarray,
    root: zarr.Group,
) -> np.ndarray:
    n_eps = int(episode_ends.shape[0])
    matrix = np.zeros((n_eps, 8), dtype=np.float64)
    for ep in range(n_eps):
        ep_start = 0 if ep == 0 else int(episode_ends[ep - 1])
        ep_end   = int(episode_ends[ep])
        s = slice(ep_start, ep_end)
        eef_pos   = np.asarray(root["data/robot0_eef_pos"][s],                 dtype=np.float64)
        eef_quat  = np.asarray(root["data/robot0_eef_orientation_quat"][s],    dtype=np.float64)
        joint_pos = np.asarray(root["data/robot0_joint_pos"][s],               dtype=np.float64)
        g_width   = np.asarray(root["data/robot0_gripper_width"][s],           dtype=np.float64).reshape(-1)
        cmd_pos   = np.asarray(root["data/action_ee_position_cmd"][s],         dtype=np.float64)
        cmd_quat  = np.asarray(root["data/action_ee_orientation_quat_cmd"][s], dtype=np.float64)
        cmd_jpos  = np.asarray(root["data/action_joint_position_cmd"][s],      dtype=np.float64)
        n = eef_pos.shape[0]

        exp_state = np.zeros((n, 17), dtype=np.float32)
        exp_state[:, 0:3]   = eef_pos
        exp_state[:, 3:9]   = quat_to_rot6d(eef_quat)
        exp_state[:, 9]     = g_width / max_width
        exp_state[:, 10:17] = joint_pos

        g_next = g_width.copy()
        if n > 1:
            g_next[:-1] = g_width[1:]
        exp_action = np.zeros((n, 17), dtype=np.float32)
        exp_action[:, 0:3]   = cmd_pos
        exp_action[:, 3:9]   = quat_to_rot6d(cmd_quat)
        exp_action[:, 9]     = g_next / max_width
        exp_action[:, 10:17] = cmd_jpos

        parquet = dataset_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        df = pd.read_parquet(parquet)
        state_par  = np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
        action_par = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]])
        s_diff = np.abs(state_par - exp_state)
        a_diff = np.abs(action_par - exp_action)

        matrix[ep, 0] = float(np.max(s_diff[:, 0:3]))
        matrix[ep, 1] = float(np.max(s_diff[:, 3:9]))
        matrix[ep, 2] = float(np.max(s_diff[:, 9:10]))
        matrix[ep, 3] = float(np.max(s_diff[:, 10:17]))
        matrix[ep, 4] = float(np.max(a_diff[:, 0:3]))
        matrix[ep, 5] = float(np.max(a_diff[:, 3:9]))
        matrix[ep, 6] = float(np.max(a_diff[:, 9:10]))
        matrix[ep, 7] = float(np.max(a_diff[:, 10:17]))
    return matrix


# ============================== DP round-trip ==============================


def _dp_matrix(
    session_dir: Path,
    dataset_dir: Path,
    max_width: float,
    episode_ends: np.ndarray,
    root: zarr.Group,
) -> np.ndarray:
    n_eps = int(episode_ends.shape[0])
    matrix = np.zeros((n_eps, 7), dtype=np.float64)
    hdf5_path = dataset_dir / "demos.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        for ep in range(n_eps):
            ep_start = 0 if ep == 0 else int(episode_ends[ep - 1])
            ep_end   = int(episode_ends[ep])
            s = slice(ep_start, ep_end)
            eef_pos    = np.asarray(root["data/robot0_eef_pos"][s],                  dtype=np.float64).astype(np.float32)
            eef_quat   = np.asarray(root["data/robot0_eef_orientation_quat"][s],     dtype=np.float64)
            eef_aa_raw = np.asarray(root["data/robot0_eef_rot_axis_angle"][s],       dtype=np.float64).astype(np.float32)
            g_width    = np.asarray(root["data/robot0_gripper_width"][s],            dtype=np.float64).reshape(-1, 1)
            cmd_pos    = np.asarray(root["data/action_ee_position_cmd"][s],          dtype=np.float64).astype(np.float32)
            cmd_quat   = np.asarray(root["data/action_ee_orientation_quat_cmd"][s],  dtype=np.float64)

            exp_eef_rot6d = quat_to_rot6d(eef_quat).astype(np.float32)
            exp_cmd_aa    = Rotation.from_quat(cmd_quat).as_rotvec().astype(np.float32)
            exp_gripper   = (g_width / max_width).astype(np.float32)
            n = eef_pos.shape[0]
            exp_g_next = exp_gripper.copy()
            if n > 1:
                exp_g_next[:-1] = exp_gripper[1:]

            demo = f[f"data/demo_{ep}"]
            h_eef_pos   = np.asarray(demo["obs/robot0_eef_pos"][:],            dtype=np.float32)
            h_eef_rot6d = np.asarray(demo["obs/robot0_eef_rot6d"][:],          dtype=np.float32)
            h_eef_aa    = np.asarray(demo["obs/robot0_eef_rot_axis_angle"][:], dtype=np.float32)
            h_gripper   = np.asarray(demo["obs/robot0_gripper_position"][:],   dtype=np.float32)
            h_actions   = np.asarray(demo["actions"][:],                       dtype=np.float32)

            matrix[ep, 0] = float(np.max(np.abs(h_eef_pos   - eef_pos)))
            matrix[ep, 1] = float(np.max(np.abs(h_eef_rot6d - exp_eef_rot6d)))
            matrix[ep, 2] = float(np.max(np.abs(h_eef_aa    - eef_aa_raw)))
            matrix[ep, 3] = float(np.max(np.abs(h_gripper   - exp_gripper)))
            matrix[ep, 4] = float(np.max(np.abs(h_actions[:, 0:3] - cmd_pos)))
            matrix[ep, 5] = float(np.max(np.abs(h_actions[:, 3:6] - exp_cmd_aa)))
            matrix[ep, 6] = float(np.max(np.abs(h_actions[:, 6:7] - exp_g_next)))
    return matrix


# ============================== Reporting ==============================


def _print_matrix(name: str, matrix: np.ndarray, slot_names: tuple[str, ...],
                  threshold: float) -> tuple[int, int]:
    """Returns (n_pass, n_total)."""
    n_eps, n_slots = matrix.shape
    print(f"\n=== {name} V-A {n_eps}*{n_slots}={n_eps*n_slots}-cell matrix"
          f" (threshold < {threshold:g}) ===\n")
    header = " ep | " + " | ".join(f"{n:^22s}" for n in slot_names)
    print(header)
    print("-" * len(header))
    n_pass = 0
    for ep in range(n_eps):
        cells = []
        for j in range(n_slots):
            v = matrix[ep, j]
            ok = v < threshold
            n_pass += int(ok)
            cells.append(f"{v:.2e}{'  OK' if ok else ' FAIL'}")
        print(f"  {ep} | " + " | ".join(f"{c:^22s}" for c in cells))
    total = n_eps * n_slots
    print(f"\n  {name}: {n_pass}/{total} PASS  "
          f"max={float(np.max(matrix)):.4e}  bitwise=0.0: {bool(np.all(matrix == 0.0))}")
    return n_pass, total


# ============================== Main ==============================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--converter", choices=["gr00t", "dp", "both"], default="both")
    parser.add_argument("--input-session", required=True, type=Path,
                        help="Raw Polymetis_Franka_Teleop session dir.")
    parser.add_argument("--output-base", type=Path, default=Path("data"),
                        help="Base dir where converted datasets live. Will look "
                             "for <base>/gr00t_droid/<session_name>/ and "
                             "<base>/diffusion_policy/<session_name>/ ."
                             "  (default: ./data)")
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="Per-cell max abs diff threshold (default: 1e-5).")
    args = parser.parse_args()

    session_dir = args.input_session.resolve()
    if not session_dir.is_dir():
        print(f"ERROR: input session not found: {session_dir}", file=sys.stderr)
        return 1

    # Load session once (zarr handle is cheap; we just need episode_ends + max_width).
    # We deliberately don't import scripts_real._conversion_common.load_session here
    # because the test must remain executable even if the converter package is
    # restructured -- inline the minimal load to stay self-contained.
    root = zarr.open(str(session_dir / "replay_buffer.zarr"), mode="r")
    with open(session_dir / "dataset_meta.json") as f:
        ds_meta = json.load(f)
    gtype = ds_meta["gripper_type"]
    max_width_key = (
        "art_gripper" if gtype == "art"
        else "franka_hand" if gtype == "franka"
        else f"{gtype}_gripper"
    )
    max_width = float(ds_meta["gripper_convention"]["max_width_m"][max_width_key])
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)

    print(f"Session: {session_dir}")
    print(f"  n_episodes = {episode_ends.shape[0]}")
    print(f"  max_width  = {max_width} ({max_width_key})")
    print(f"  threshold  = {args.threshold:g}")

    total_pass = 0
    total_cells = 0

    if args.converter in ("gr00t", "both"):
        gd = args.output_base / "gr00t_droid" / session_dir.name
        if not (gd / "data" / "chunk-000").exists():
            print(f"\nERROR: GR00T output not found at {gd}. "
                  f"Run convert_to_gr00t_droid.py first.", file=sys.stderr)
            return 1
        m = _gr00t_matrix(session_dir, gd, max_width, episode_ends, root)
        p, t = _print_matrix("GR00T-DROID", m, _GR00T_SLOTS, args.threshold)
        total_pass += p; total_cells += t

    if args.converter in ("dp", "both"):
        dp = args.output_base / "diffusion_policy" / session_dir.name
        if not (dp / "demos.hdf5").exists():
            print(f"\nERROR: DP output not found at {dp}/demos.hdf5. "
                  f"Run convert_to_diffusion_policy.py first.", file=sys.stderr)
            return 1
        m = _dp_matrix(session_dir, dp, max_width, episode_ends, root)
        p, t = _print_matrix("Diffusion-Policy", m, _DP_SLOTS, args.threshold)
        total_pass += p; total_cells += t

    print(f"\n{'=' * 70}")
    print(f" SUMMARY: {total_pass}/{total_cells} cells PASS")
    print(f"{'=' * 70}")
    return 0 if total_pass == total_cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
