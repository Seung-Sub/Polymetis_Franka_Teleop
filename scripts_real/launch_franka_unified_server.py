#!/usr/bin/env python3
"""Unified ZeroRPC server for Franka — runs on the NUC, on top of Polymetis.

Architecture:
    NUC :50051 (polymetis launch_robot.py)        ← arm
    NUC :50052 (polymetis franka_hand_client.py)  ← Franka Hand (optional)
    NUC :4242  (this script)                      ← ZeroRPC bridge

The pro4000 controllers (`FrankaInterpolationController`,
`FrankaGripperController`) connect to :4242 and call the methods exposed
here. This indirection lets the pro4000 stay polymetis-agnostic.

If you are running the **ART gripper** (`gripper_backend='art'` in the
env) you don't need Franka Hand at all — pass `--no_gripper` so this
server does NOT try to connect to GripperInterface and will only expose
the arm methods. The pro4000's `ArtGripperController` talks directly to
the standalone TCP daemon on the pro4000 itself (`:50053`), bypassing
this server completely.

Usage on NUC (after `start_franka_arm.sh` is up):
    # ART gripper case (Franka Hand 미사용)
    python launch_franka_unified_server.py --no_gripper

    # Franka Hand case (also requires `start_franka_gripper.sh`)
    python launch_franka_unified_server.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import scipy.spatial.transform as st

try:
    import zerorpc
except ImportError:
    sys.exit("ERROR: zerorpc not installed in this conda env. Run: pip install zerorpc")

try:
    from polymetis import RobotInterface
except ImportError:
    sys.exit(
        "ERROR: polymetis not installed. This script must run inside the "
        "NUC's polymetis-local conda env (see Isaac-GR00T docs)."
    )


class FrankaUnifiedInterface:
    """ZeroRPC-exposed bridge over polymetis RobotInterface (+ optional GripperInterface)."""

    def __init__(self, robot_ip: str = 'localhost', grasp_force: float = 50.0,
                 enable_gripper: bool = True):
        print(f'[unified] connecting to polymetis arm at {robot_ip}:50051 ...')
        self.robot = RobotInterface(ip_address=robot_ip)
        print('[unified] arm connected')

        self.gripper = None
        self.grasp_force = grasp_force
        if enable_gripper:
            try:
                from polymetis import GripperInterface
                print(f'[unified] connecting to franka_hand at {robot_ip}:50052 ...')
                self.gripper = GripperInterface(ip_address=robot_ip)
                print('[unified] gripper connected')
            except Exception as e:
                print(f'[unified] gripper init failed: {e}')
                print('[unified] continuing without gripper (use --no_gripper to silence)')
                self.gripper = None

        ee = self.get_ee_pose()
        q = self.get_joint_positions()
        print('[unified] initial state:')
        print(f'  ee : {[f"{x:.3f}" for x in ee]}')
        print(f'  q  : {[f"{x:.2f}" for x in q]}')
        if self.gripper is not None:
            print(f'  gw : {self.get_gripper_state()["width"]:.3f}')

    # ---------- Arm ----------
    def get_ee_pose(self):
        pos, quat = self.robot.get_ee_pose()
        rot = st.Rotation.from_quat(quat.numpy()).as_rotvec()
        return np.concatenate([pos.numpy(), rot]).tolist()

    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()

    def get_robot_state(self):
        """Single-call replacement for the receive_keys loop in
        FrankaInterpolationController — returns ee_pose / joint_positions /
        joint_velocities in one ZeroRPC roundtrip instead of three.

        Returns: dict with keys 'ee_pose' (6,), 'q' (7,), 'qd' (7,).
        """
        pos, quat = self.robot.get_ee_pose()
        ee_pose = np.concatenate([
            pos.numpy(),
            st.Rotation.from_quat(quat.numpy()).as_rotvec(),
        ]).tolist()
        return {
            'ee_pose': ee_pose,
            'q':  self.robot.get_joint_positions().numpy().tolist(),
            'qd': self.robot.get_joint_velocities().numpy().tolist(),
        }

    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions), time_to_go=time_to_go,
        )

    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx), Kxd=torch.Tensor(Kxd),
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat()),
        )

    def terminate_current_policy(self):
        try:
            self.robot.terminate_current_policy()
        except Exception:
            pass  # idempotent — fine to call when no policy is active

    # ---------- Gripper (Franka Hand only) ----------
    def get_gripper_state(self):
        if self.gripper is None:
            return {'width': 0.0, 'is_grasped': False, 'is_moving': False}
        s = self.gripper.get_state()
        return {
            'width': float(s.width),
            'is_grasped': bool(getattr(s, 'is_grasped', False)),
            'is_moving': bool(getattr(s, 'is_moving', False)),
        }

    def gripper_goto(self, width, speed=0.2, force=None):
        if self.gripper is None:
            raise RuntimeError("Server started with --no_gripper — Franka Hand methods unavailable")
        width = max(0.0, min(0.08, float(width)))
        self.gripper.goto(width=width, speed=float(speed),
                          force=self.grasp_force if force is None else float(force))

    def gripper_grasp(self, speed=0.2, force=None, width=0.0):
        if self.gripper is None:
            raise RuntimeError("Server started with --no_gripper — Franka Hand methods unavailable")
        f = self.grasp_force if force is None else float(force)
        self.gripper.grasp(speed=float(speed), force=f)

    def gripper_move(self, width, speed=0.2):
        self.gripper_goto(width=width, speed=speed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=4242)
    p.add_argument('--robot_ip', type=str, default='localhost')
    p.add_argument('--grasp_force', type=float, default=50.0)
    p.add_argument('--no_gripper', action='store_true',
                   help='Skip GripperInterface init (use this for ART gripper workflows)')
    args = p.parse_args()

    iface = FrankaUnifiedInterface(
        robot_ip=args.robot_ip,
        grasp_force=args.grasp_force,
        enable_gripper=not args.no_gripper,
    )
    server = zerorpc.Server(iface)
    server.bind(f'tcp://0.0.0.0:{args.port}')
    print(f'[unified] listening on tcp://0.0.0.0:{args.port}')
    try:
        server.run()
    except KeyboardInterrupt:
        print('[unified] stopping')
    finally:
        server.close()


if __name__ == '__main__':
    main()
