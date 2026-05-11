"""V-rpc bench: confirm single-RPC + local FK matches the legacy multi-RPC path.

Run once before starting production data collection. Compares each pair of
values that the data pipeline now sources from
``FrankaInterface.get_robot_state_batched()`` against the older per-method
polling path it replaced.

Prerequisites
-------------
- Polymetis arm server running on the NUC (default 192.168.1.12:50051).
- Robot in a STATIC pose (e.g. just after ``go_home``; gravity comp only,
  no impedance / move policy active). Bench polls only -- never sends a
  command -- but per-RPC sensor noise + tiny gravity-comp oscillation are
  expected to dominate the joint-velocity and torque deltas.

Usage
-----
    conda activate groot-client
    python -m polymetis_franka_teleop.scripts_real.bench_fk_consistency
        # or
    python /home/kist/Polymetis_Franka_Teleop/scripts_real/bench_fk_consistency.py

Margins
-------
Tuned for "no controller, static pose, same pinocchio model on both
sides". Loosened where two independent RPCs to a noisy sensor are being
compared (joint velocity, external torque -- ``tau_ext_hat_filtered``
is a server-side filter output and naturally jitters between snapshots).

    | metric         | margin       | rationale                            |
    |----------------|--------------|--------------------------------------|
    | TCP pos        | < 0.1 mm     | same pinocchio model => sub-um real  |
    | TCP rot        | < 0.01 deg   | same pinocchio model => sub-mdeg real|
    | flange pos     | < 0.1 mm     | rigid transform of TCP, same model   |
    | flange rot     | < 0.01 deg   | same                                 |
    | joint position | < 1e-5 rad   | franka encoder ~1e-6 rad             |
    | joint velocity | < 1e-2 rad/s | sensor noise + grav-comp wobble      |
    | torque_ext     | < 0.05 Nm    | filter output, different snapshots   |

If anything exceeds its margin, the most likely cause is that the
polymetis server and the in-process pinocchio model are loading
different URDFs (e.g. one has the Franka Hand attached, the other does
not). Inspect the URDF path the polymetis server reports at boot, and
``robot.robot.robot_model`` on the client side.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.spatial.transform as st
import torch

from polymetis_franka_teleop.real_world.franka_interpolation_controller import (
    FrankaInterface,
)


_MARGINS = {
    'TCP pos diff':    (0.1,    'mm'),
    'TCP rot diff':    (0.01,   'deg'),
    'joint pos diff':  (1e-5,   'rad'),
    'joint vel diff':  (1e-2,   'rad/s'),
    'flange pos diff': (0.1,    'mm'),
    'flange rot diff': (0.01,   'deg'),
    'torque_ext diff': (0.05,   'Nm'),
}


def report(name: str, vals, unit: str, scale: float = 1.0) -> bool:
    """Print mean/max/std and return True iff max < margin for ``name``."""
    arr = np.asarray(vals, dtype=np.float64) * scale
    margin, _ = _MARGINS[name]
    status = 'PASS' if np.max(arr) < margin else 'FAIL'
    print(f'  {name:18s}: '
          f'mean={np.mean(arr):.4e} {unit}, '
          f'max={np.max(arr):.4e} {unit}, '
          f'std={np.std(arr):.4e}   '
          f'[margin < {margin} {unit}: {status}]')
    return status == 'PASS'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ip', default='192.168.1.12',
                        help='polymetis arm server IP (default: NUC)')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('-n', type=int, default=100,
                        help='number of polling iterations (default: 100)')
    args = parser.parse_args()

    print(f'V-rpc: connecting to polymetis at {args.ip}:{args.port}')
    robot = FrankaInterface(ip=args.ip, port=args.port)
    print(f'V-rpc: connected. Sampling N={args.n} (static pose assumed).')
    time.sleep(0.5)  # let any post-connect transients settle

    deltas_tcp_pos    = []
    deltas_tcp_rot    = []
    deltas_q          = []
    deltas_qd         = []
    deltas_flange_pos = []
    deltas_flange_rot = []
    deltas_tau        = []

    for _ in range(args.n):
        # --- Old path (multi-RPC, server-side FK on the NUC) ---
        pose_old_tcp = robot.get_ee_pose()           # (6,) TCP, axis-angle
        q_old        = robot.get_joint_positions()
        qd_old       = robot.get_joint_velocities()
        flange_old   = robot._flange_pose_6d()       # (6,) flange, axis-angle
        rs_old       = robot.robot.get_robot_state()
        tau_direct   = np.asarray(rs_old.motor_torques_external,
                                  dtype=np.float64)

        # --- New path (single RPC + local pinocchio FK + Jacobian) ---
        state          = robot.get_robot_state_batched()
        tcp_pose_new   = state['ee_pose']
        q_new          = state['joint_position']
        qd_new         = state['joint_velocity']
        tau_batched    = state['joint_torque_external']

        # Re-derive flange from q_new with the same pinocchio call the
        # batched path uses, so the flange comparison isolates the
        # server-side FK vs client-side FK question (and is not confounded
        # by qd-induced motion between the two RPCs).
        flange_pos_t, flange_quat_t = robot.robot.robot_model.forward_kinematics(
            torch.from_numpy(q_new.astype(np.float32))
        )
        flange_pos_new    = flange_pos_t.numpy().astype(np.float64)
        flange_rotvec_new = st.Rotation.from_quat(
            flange_quat_t.numpy()).as_rotvec()
        flange_new        = np.concatenate([flange_pos_new, flange_rotvec_new])

        # ---- diffs ----
        deltas_tcp_pos.append(
            float(np.linalg.norm(pose_old_tcp[:3] - tcp_pose_new[:3])))
        r_old = st.Rotation.from_rotvec(pose_old_tcp[3:])
        r_new = st.Rotation.from_rotvec(tcp_pose_new[3:])
        deltas_tcp_rot.append(float((r_old.inv() * r_new).magnitude()))

        deltas_q.append(float(np.linalg.norm(q_old - q_new)))
        deltas_qd.append(float(np.linalg.norm(qd_old - qd_new)))

        deltas_flange_pos.append(
            float(np.linalg.norm(flange_old[:3] - flange_new[:3])))
        rf_old = st.Rotation.from_rotvec(flange_old[3:])
        rf_new = st.Rotation.from_rotvec(flange_new[3:])
        deltas_flange_rot.append(float((rf_old.inv() * rf_new).magnitude()))

        deltas_tau.append(float(np.linalg.norm(tau_direct - tau_batched)))

    print(f'\nV-rpc results (N={args.n}, static pose):')
    passes = [
        report('TCP pos diff',    deltas_tcp_pos,    'mm',    scale=1e3),
        report('TCP rot diff',    deltas_tcp_rot,    'deg',   scale=180.0/np.pi),
        report('joint pos diff',  deltas_q,          'rad'),
        report('joint vel diff',  deltas_qd,         'rad/s'),
        report('flange pos diff', deltas_flange_pos, 'mm',    scale=1e3),
        report('flange rot diff', deltas_flange_rot, 'deg',   scale=180.0/np.pi),
        report('torque_ext diff', deltas_tau,        'Nm'),
    ]
    print()
    if all(passes):
        print('V-rpc: ALL PASS -- single-RPC batched path matches legacy.')
        return 0
    print('V-rpc: at least one metric FAIL -- inspect URDF / robot_model '
          'loading on server vs client.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
