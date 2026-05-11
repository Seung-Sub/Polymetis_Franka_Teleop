"""Rotation utilities for the data pipeline.

Convention (fixed across the dataset, see dataset_meta.json):
  * Quaternion order: ``[qx, qy, qz, qw]`` (scalar-last; matches scipy + ROS2).
  * Quaternion sign is normalised within each episode: for every consecutive
    pair ``(q[t-1], q[t])`` with ``dot < 0``, we flip ``q[t] *= -1``. The
    underlying rotation is preserved but the time-series is continuous.
  * 6D rotation = first two columns of the rotation matrix, flattened in
    column-major order: ``[m00, m10, m20, m01, m11, m21]``. Same convention
    as GR00T's eef_9d head and the Zhou et al. (2019) paper.
  * All rotations are in the robot base frame unless explicitly stated.

This module is the single source of truth for rotation conversions used by
``franka_vive_env.end_episode()`` and the various conversion scripts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def aa_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) or (N,3) → quaternion (4,) or (N,4) ``[qx,qy,qz,qw]``."""
    aa = np.asarray(axis_angle, dtype=np.float64)
    quat = R.from_rotvec(aa).as_quat()           # scipy returns scalar-last
    return quat


def quat_to_aa(quat: np.ndarray) -> np.ndarray:
    """Quaternion (4,) or (N,4) → axis-angle (3,) or (N,3)."""
    q = np.asarray(quat, dtype=np.float64)
    return R.from_quat(q).as_rotvec()


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Quaternion → 6D rotation (first two columns of R, column-major).

    Returns:
      (6,)  if input is (4,)
      (N,6) if input is (N,4)
    """
    q = np.asarray(quat, dtype=np.float64)
    mat = R.from_quat(q).as_matrix()             # (..., 3, 3)
    # Take first two columns, transpose so flatten gives column-major order
    # rot6d = [m00, m10, m20, m01, m11, m21]
    if mat.ndim == 2:
        return np.concatenate([mat[:, 0], mat[:, 1]])
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def quat_continuous_within_episode(quat: np.ndarray) -> np.ndarray:
    """Sign-normalise a quaternion time-series so consecutive samples are continuous.

    Iterates ``q[1:]``, flipping the sign of any entry whose dot product with
    its predecessor is negative. The underlying rotation is invariant under
    ``q ↔ -q`` so this is lossless; the only effect is removing the spurious
    sign flips that scipy / Eigen emit on rotation boundaries.

    Args:
        quat: (N, 4) array, scalar-last.

    Returns:
        (N, 4) sign-normalised, same dtype + memory layout as input.
    """
    q = np.asarray(quat, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f'expected (N,4) quaternion array, got {q.shape}')
    for t in range(1, q.shape[0]):
        if np.dot(q[t - 1], q[t]) < 0:
            q[t] = -q[t]
    return q


def slerp_at(times: np.ndarray, quats: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """SLERP a quaternion time-series ``(times, quats)`` at ``target_times``.

    Uses scipy.spatial.transform.Slerp. Caller is responsible for ensuring
    target_times is within [times[0], times[-1]] (Slerp raises otherwise).
    Wrapper exists so the rest of the pipeline doesn't import Slerp directly,
    keeping the convention uniform.

    Args:
        times: (N,) monotonic time stamps.
        quats: (N, 4) quaternions, scalar-last, ideally sign-normalised.
        target_times: (K,) timestamps to interpolate at.

    Returns:
        (K, 4) interpolated quaternions, scalar-last, sign-normalised.
    """
    t = np.asarray(times, dtype=np.float64)
    q = np.asarray(quats, dtype=np.float64)
    if t.shape[0] != q.shape[0]:
        raise ValueError(f'times length {t.shape[0]} != quats length {q.shape[0]}')
    if t.shape[0] < 2:
        # Degenerate: hold the single value
        return np.broadcast_to(q[0], (len(target_times), 4)).copy()
    slerp = Slerp(t, R.from_quat(q))
    out = slerp(np.clip(target_times, t[0], t[-1])).as_quat()
    return quat_continuous_within_episode(out)


# ----------------- self-tests (run with: python -m polymetis_franka_teleop.common.rotation_util) -----------------
if __name__ == '__main__':
    # Round-trip aa ↔ quat
    aa = np.array([0.5, -0.3, 0.7])
    q = aa_to_quat(aa)
    aa_back = quat_to_aa(q)
    assert np.allclose(aa, aa_back, atol=1e-12), 'aa↔quat round-trip failed'

    # Sign normalisation
    bad = np.array([
        [0, 0, 0, 1],
        [0.1, 0, 0, 0.99],
        [-0.2, 0, 0, -0.98],     # sign-flipped (rotation continuous)
        [-0.3, 0, 0, -0.95],
        [0.4, 0, 0, 0.91],       # sign-flipped again
    ])
    normed = quat_continuous_within_episode(bad)
    # All consecutive dots should be >= 0 after normalisation
    dots = (normed[1:] * normed[:-1]).sum(axis=1)
    assert (dots >= 0).all(), f'sign norm failed, dots={dots}'

    # rot6d shape
    r6 = quat_to_rot6d(q)
    assert r6.shape == (6,), f'rot6d shape {r6.shape}'
    r6_batch = quat_to_rot6d(np.tile(q, (4, 1)))
    assert r6_batch.shape == (4, 6), f'rot6d batch shape {r6_batch.shape}'

    # Slerp
    times = np.array([0.0, 1.0, 2.0])
    quats = np.array([
        [0, 0, 0, 1],
        aa_to_quat([0, 0, np.pi / 4]),
        aa_to_quat([0, 0, np.pi / 2]),
    ])
    slerped = slerp_at(times, quats, np.array([0.0, 0.5, 1.5, 2.0]))
    assert slerped.shape == (4, 4)
    # First sample is identity, last is full rotation
    assert np.allclose(slerped[0], [0, 0, 0, 1], atol=1e-9)
    assert np.allclose(np.abs(slerped[-1]), np.abs(quats[-1]), atol=1e-9)

    print('rotation_util: all self-tests passed')
