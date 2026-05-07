"""
FrankaInterpolationController for Franka robot with configurable TCP offset.
Based on UMI's franka_interpolation_controller.py — but uses **direct
polymetis gRPC** (no ZeroRPC indirection on NUC), matching the
verified GR00T-side path (`Isaac-GR00T/examples/DROID/franka_env_kist.py`).

Key features:
- Runs in separate process to ensure predictable timing (avoiding Python GIL)
- Smooth trajectory interpolation via PoseTrajectoryInterpolator
- Configurable TCP offset for different grippers (Franka Hand vs ART vs WSG)
- Direct polymetis RobotInterface — pro4000 → NUC :50051 (raw gRPC)
- Teleop mode for direct control from ViveTeleopProcess

Requirements (NUC):
- polymetis launch_robot.py running on :50051
  (start via `sudo bash /usr/local/sbin/start_franka_arm.sh`)
- That's it — no extra ZeroRPC/wrapper layer needed.

Requirements (pro4000):
- polymetis Python client in the active conda env (already true in
  `groot-client` since Isaac-GR00T setup).

Usage:
    with SharedMemoryManager() as shm_manager:
        with FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip='172.16.0.3',
            tcp_offset=0.1034  # Franka Hand
        ) as robot_controller:
            # Get robot state
            state = robot_controller.get_state()
            # Schedule waypoint
            robot_controller.schedule_waypoint(pose, target_time)
"""

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from polymetis_franka_teleop.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from polymetis_franka_teleop.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from polymetis_franka_teleop.common.precise_sleep import precise_wait
from polymetis_franka_teleop.common.pose_util import pose_to_mat, mat_to_pose


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_HOME = 3  # Move to home joint positions


# Franka home joint positions (safe neutral position)
FRANKA_HOME_JOINTS = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])


def compute_tcp_transform(tcp_offset: float = 0.0, use_wsg_gripper: bool = False):
    """
    Compute TCP transformation matrix from flange to tool center point.

    Args:
        tcp_offset: TCP offset along Z-axis in meters (for Franka Hand)
        use_wsg_gripper: If True, use original WSG gripper transformation from UMI

    Returns:
        tx_flange_tip: 4x4 transformation matrix from flange to TCP
    """
    if use_wsg_gripper:
        # Original WSG gripper transformation from UMI
        tx_flangerot90_tip = np.identity(4)
        tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

        tx_flangerot45_flangerot90 = np.identity(4)
        tx_flangerot45_flangerot90[:3, :3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

        tx_flange_flangerot45 = np.identity(4)
        tx_flange_flangerot45[:3, :3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

        tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @ tx_flangerot90_tip
    else:
        # Simple Z-axis offset for Franka Hand (default gripper)
        # Franka Hand TCP is approximately 0.1034m from flange when closed
        tx_flange_tip = np.identity(4)
        tx_flange_tip[2, 3] = tcp_offset

    return tx_flange_tip


# Legacy global transforms for backward compatibility (WSG gripper)
tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    """TCP-frame wrapper supporting **two backend modes**:

    * ``mode='zerorpc'`` (DEFAULT, community-standard) — connects to a
      ZeroRPC bridge running on the NUC (port 4242) where the polymetis
      Python client is local. This is the architecture used by UMI,
      DROID/R2D2 and is the most reliable pattern for remote teleop —
      the 4 internal gRPCs that polymetis's ``update_desired_ee_pose``
      issues all stay loopback on the NUC.

    * ``mode='direct'`` — connects polymetis client directly to the arm
      gRPC server (port 50051). Simpler deployment (no NUC bridge needed)
      but capped at ~100 Hz from a remote workstation before the 1 s
      polymetis watchdog (THRESHOLD_NS) trips on NUC realtime stalls.

    Args:
        ip:   NUC host.
        port: 4242 for zerorpc, 50051 for direct (must match ``mode``).
        mode: 'zerorpc' (default) or 'direct'.
        tx_flange_tip_override: optional 4x4 flange→tool transform.
    """
    def __init__(self, ip='192.168.1.12', port=50051, mode='direct',
                 tx_flange_tip_override=None):
        self.mode = mode
        self._cached_q = None
        if mode == 'zerorpc':
            import zerorpc
            self._rpc = zerorpc.Client(heartbeat=20)
            self._rpc.connect(f'tcp://{ip}:{port}')
            self.robot = None  # not used in zerorpc mode
        elif mode == 'direct':
            from polymetis import RobotInterface
            self.robot = RobotInterface(ip_address=ip, port=port)
            self._rpc = None
        else:
            raise ValueError(f"FrankaInterface mode must be 'zerorpc' or 'direct', got {mode!r}")

        if tx_flange_tip_override is not None:
            self.tx_flange_tip = tx_flange_tip_override
            self.tx_tip_flange = np.linalg.inv(tx_flange_tip_override)
        else:
            self.tx_flange_tip = tx_flange_tip
            self.tx_tip_flange = tx_tip_flange

    # ----- low-level state -----
    def _flange_pose_6d(self):
        """Returns flange pose [x,y,z,rx,ry,rz] (axis-angle)."""
        if self.mode == 'zerorpc':
            return np.asarray(self._rpc.get_ee_pose(), dtype=np.float64)
        pos, quat = self.robot.get_ee_pose()
        rotvec = st.Rotation.from_quat(quat.numpy()).as_rotvec()
        return np.concatenate([pos.numpy(), rotvec])

    def get_ee_pose(self):
        flange = self._flange_pose_6d()
        return mat_to_pose(pose_to_mat(flange) @ self.tx_flange_tip)

    def get_joint_positions(self):
        if self.mode == 'zerorpc':
            return np.asarray(self._rpc.get_joint_positions(), dtype=np.float64)
        return self.robot.get_joint_positions().numpy()

    def get_joint_velocities(self):
        if self.mode == 'zerorpc':
            return np.asarray(self._rpc.get_joint_velocities(), dtype=np.float64)
        return self.robot.get_joint_velocities().numpy()

    # ----- control -----
    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        if self.mode == 'zerorpc':
            self._rpc.move_to_joint_positions(np.asarray(positions).tolist(), float(time_to_go))
            return
        import torch
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(np.asarray(positions, dtype=np.float32)),
            time_to_go=float(time_to_go),
        )

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        if self.mode == 'zerorpc':
            self._rpc.start_cartesian_impedance(np.asarray(Kx).tolist(), np.asarray(Kxd).tolist())
            return
        import torch
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(np.asarray(Kx, dtype=np.float32)),
            Kxd=torch.Tensor(np.asarray(Kxd, dtype=np.float32)),
        )

    def update_desired_ee_pose(self, tip_pose: np.ndarray):
        """Convert TCP-frame target → flange-frame target → robot.

        zerorpc mode: 1 ZeroRPC call to NUC bridge (which does IK + polymetis
            update locally on NUC — community-standard UMI/DROID path).
        direct mode: solve IK locally, then 1 polymetis gRPC for the joint
            update (3× fewer RPCs than polymetis's default update_desired_ee_pose).
        """
        flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ self.tx_tip_flange)
        if self.mode == 'zerorpc':
            self._rpc.update_desired_ee_pose(flange_pose.tolist())
            return
        import torch
        pos = flange_pose[:3]
        quat = st.Rotation.from_rotvec(flange_pose[3:]).as_quat()
        if self._cached_q is not None:
            q_current = self._cached_q
        else:
            q_current = self.robot.get_joint_positions()
        joint_target, success = self.robot.solve_inverse_kinematics(
            torch.Tensor(pos.astype(np.float32)),
            torch.Tensor(quat.astype(np.float32)),
            q_current,
        )
        if not success:
            return
        self.robot.update_desired_joint_positions(joint_target)

    def set_cached_q(self, q):
        """Optional: feed cached joint state for IK seed (direct mode only)."""
        if self.mode == 'direct':
            import torch
            self._cached_q = torch.as_tensor(q, dtype=torch.float32)

    def terminate_current_policy(self):
        try:
            if self.mode == 'zerorpc':
                self._rpc.terminate_current_policy()
            else:
                self.robot.terminate_current_policy()
        except Exception:
            pass

    def close(self):
        if self.mode == 'zerorpc':
            try: self._rpc.close()
            except Exception: pass
            self._rpc = None
        else:
            self.robot = None


class FrankaInterpolationController(mp.Process):
    """
    Franka robot controller with trajectory interpolation.
    Runs in a separate process to ensure predictable timing (avoiding Python GIL).

    This is the UMI-style controller that:
    1. Receives waypoint commands via SharedMemoryQueue
    2. Interpolates trajectory using PoseTrajectoryInterpolator
    3. Sends interpolated poses to robot at high frequency
    4. Publishes robot state to SharedMemoryRingBuffer

    Teleop Mode:
    When teleop_mode=True, the controller bypasses schedule_waypoint interpolation
    and reads target_pose directly from teleop_ring_buffer at 100Hz (from ViveTeleopProcess).
    This eliminates the 10Hz bottleneck in the main process for teleop applications.

    Args:
        shm_manager: SharedMemoryManager for inter-process communication
        robot_ip: IP address of the polymetis server (NUC)
        robot_port: Port of the polymetis arm gRPC server (default: 50051)
        frequency: Control frequency in Hz (default: 200)
        tcp_offset: TCP offset along Z-axis in meters (default: 0.1034 for Franka Hand)
        use_wsg_gripper: If True, use WSG gripper transformation (default: False)
        Kx_scale: Position gain scale (default: 1.0)
        Kxd_scale: Velocity gain scale (default: 1.0)
        launch_timeout: Timeout for controller startup (default: 3s)
        joints_init: Initial joint positions (7,) or None
        joints_init_duration: Duration for initial joint motion
        soft_real_time: Enable RT scheduling (requires rtprio_setup.sh)
        receive_latency: Latency compensation for state timestamps (default: 0.0)
        verbose: Enable verbose logging (default: False)
        teleop_mode: Enable teleop mode (bypass schedule_waypoint) (default: False)
        teleop_ring_buffer: SharedMemoryRingBuffer from ViveTeleopProcess (required if teleop_mode=True)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        robot_port=50051,                     # 50051 for direct, 4242 for zerorpc
        polymetis_mode: str = 'direct',       # 'direct' (default) or 'zerorpc' (UMI/DROID bridge)
        frequency=100,                         # 100 Hz is the empirically stable ceiling on this NUC
        tcp_offset: float = 0.1034,
        use_wsg_gripper: bool = False,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        teleop_mode: bool = False,
        teleop_ring_buffer: SharedMemoryRingBuffer = None,
        home_joints: np.ndarray = None,
        home_time: float = 2.0,
        ):

        # Home position configuration
        if home_joints is not None:
            self.home_joints = np.array(home_joints)
        else:
            self.home_joints = FRANKA_HOME_JOINTS.copy()

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        # Validate teleop mode configuration
        if teleop_mode and teleop_ring_buffer is None:
            raise ValueError("teleop_ring_buffer is required when teleop_mode=True")

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.polymetis_mode = polymetis_mode
        self.frequency = frequency
        self.tcp_offset = tcp_offset
        self.use_wsg_gripper = use_wsg_gripper
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.teleop_mode = teleop_mode
        self.teleop_ring_buffer = teleop_ring_buffer
        self.home_time = home_time

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def move_home(self):
        """
        Move robot to home position.
        This will stop impedance control, move to home joints, then restart impedance.
        Safe to call during teleop mode.
        """
        assert self.is_alive()
        message = {
            'cmd': Command.MOVE_HOME.value,
            'target_pose': np.zeros(6),  # dummy, not used
            'duration': self.home_time,
            'target_time': 0.0
        }
        self.input_queue.put(message)
        if self.verbose:
            print(f"[FrankaPositionalController] Home command sent (duration: {self.home_time}s)")

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # Compute TCP transformation
        tx_flange_tip_local = compute_tcp_transform(
            tcp_offset=self.tcp_offset,
            use_wsg_gripper=self.use_wsg_gripper
        )

        # start polymetis interface with configurable TCP transform + backend mode
        robot = FrankaInterface(
            ip=self.robot_ip,
            port=self.robot_port,
            mode=self.polymetis_mode,
            tx_flange_tip_override=tx_flange_tip_local,
        )

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
                print(f"[FrankaPositionalController] TCP offset: {self.tcp_offset}m, WSG mode: {self.use_wsg_gripper}")
                print(f"[FrankaPositionalController] Teleop mode: {self.teleop_mode}")

            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t

            # In teleop mode, we don't use PoseTrajectoryInterpolator for teleop targets
            # but we still need it for non-teleop commands (SERVOL, SCHEDULE_WAYPOINT)
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # start franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True

            # Teleop mode state
            teleop_target_pose = curr_pose.copy()
            last_teleop_timestamp = 0.0
            prev_clutch_active = False  # Track clutch state transitions
            wait_for_clutch_engage = False  # After recovery/HOME, wait for clutch 0->1

            # Pose offset compensation for clutch sync
            # This corrects for network delay between ViveTeleopProcess and this controller
            # When clutch is engaged, we calculate offset between local pose and teleop target
            # This offset is applied to all subsequent teleop targets during the clutch session
            clutch_pose_offset = None

            while keep_running:
                # Get current time
                t_now = time.monotonic()

                # === TELEOP MODE: Read target from teleop_ring_buffer ===
                if self.teleop_mode:
                    try:
                        teleop_state = self.teleop_ring_buffer.get()
                        teleop_timestamp = teleop_state.get('teleop_timestamp', 0.0)
                        clutch_active = bool(teleop_state.get('clutch_active', 0))
                        rotation_active = bool(teleop_state.get('rotation_active', 0))

                        # Check for clutch state transitions
                        clutch_just_engaged = clutch_active and not prev_clutch_active

                        # After recovery/HOME, we set wait_for_clutch_engage=True
                        # This means we need to see clutch go from 0->1 before accepting targets
                        if wait_for_clutch_engage:
                            if clutch_just_engaged:
                                # User just engaged clutch - now we can accept teleop targets
                                wait_for_clutch_engage = False
                                # Fetch fresh pose and sync
                                curr_pose = robot.get_ee_pose()
                                teleop_target_pose = curr_pose.copy()
                                target_pose = curr_pose.copy()

                                # Calculate pose offset between local pose and teleop target
                                # This compensates for network delay in ViveTeleopProcess
                                teleop_target_raw = teleop_state['target_pose']
                                clutch_pose_offset = curr_pose - teleop_target_raw
                                if self.verbose:
                                    offset_mag = np.linalg.norm(clutch_pose_offset[:3])
                                    print(f"[FrankaPositionalController] Clutch engaged - synced to: [{curr_pose[0]:.3f}, {curr_pose[1]:.3f}, {curr_pose[2]:.3f}] (offset: {offset_mag*1000:.1f}mm)")
                            else:
                                # Still waiting - hold current position (don't accept teleop targets)
                                target_pose = curr_pose.copy()
                        elif clutch_just_engaged:
                            # Normal clutch engage (not after HOME/recovery)
                            curr_pose = robot.get_ee_pose()
                            teleop_target_raw = teleop_state['target_pose']
                            clutch_pose_offset = curr_pose - teleop_target_raw
                            teleop_target_pose = curr_pose.copy()
                            target_pose = curr_pose.copy()
                            if self.verbose:
                                offset_mag = np.linalg.norm(clutch_pose_offset[:3])
                                print(f"[FrankaPositionalController] Clutch engaged (offset: {offset_mag*1000:.1f}mm)")
                        elif clutch_active:
                            # Normal teleop: Clutch is engaged - accept teleop target with offset
                            if teleop_timestamp > last_teleop_timestamp:
                                teleop_target_raw = teleop_state['target_pose']
                                # Apply pose offset compensation
                                if clutch_pose_offset is not None:
                                    teleop_target_pose = teleop_target_raw + clutch_pose_offset
                                else:
                                    teleop_target_pose = teleop_target_raw
                                last_teleop_timestamp = teleop_timestamp
                            target_pose = teleop_target_pose
                        elif rotation_active:
                            # === TRACKPAD ROTATION MODE ===
                            # Grip is OFF but trackpad rotation is active
                            # Follow target_pose directly (no offset compensation needed)
                            # ViveTeleopProcess computes rotation based on actual robot pose
                            if teleop_timestamp > last_teleop_timestamp:
                                teleop_target_pose = teleop_state['target_pose']
                                last_teleop_timestamp = teleop_timestamp
                            target_pose = teleop_target_pose
                            # Don't reset clutch_pose_offset here - keep it for next clutch engage
                        else:
                            # Clutch NOT engaged - hold current position
                            # This is critical: when clutch is off, robot should hold position
                            # and NOT follow stale teleop targets
                            try:
                                curr_pose = robot.get_ee_pose()
                            except:
                                pass
                            target_pose = curr_pose.copy()
                            teleop_target_pose = curr_pose.copy()  # Sync for next clutch engage
                            clutch_pose_offset = None  # Reset offset when clutch released

                        prev_clutch_active = clutch_active

                    except Exception as e:
                        if self.verbose and iter_idx % 100 == 0:
                            print(f"[FrankaPositionalController] Teleop read error: {e}")
                        # Fallback to interpolated pose
                        target_pose = pose_interp(t_now)
                else:
                    # === NORMAL MODE: Use interpolated pose ===
                    target_pose = pose_interp(t_now)

                # send command to robot (FrankaInterface handles TCP transform internally)
                # with automatic recovery on controller errors
                try:
                    robot.update_desired_ee_pose(target_pose)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for common recoverable errors
                    if any(err in error_str for err in [
                        'no controller running',
                        'communication_constraints_violation',
                        'cartesian_reflex',
                        'joint_velocity_violation',
                        'cartesian_velocity_violation',
                        'joint_position_limits',
                        'torque_discontinuity'
                    ]):
                        print(f"\n[FrankaPositionalController] Controller error detected: {e}")
                        print("[FrankaPositionalController] Attempting automatic recovery...")
                        try:
                            # Terminate any crashed policy
                            try:
                                robot.terminate_current_policy()
                            except:
                                pass
                            time.sleep(0.1)

                            # Get current robot pose (fresh)
                            curr_pose = robot.get_ee_pose()
                            teleop_target_pose = curr_pose.copy()

                            # Restart impedance controller
                            robot.start_cartesian_impedance(
                                Kx=self.Kx,
                                Kxd=self.Kxd
                            )
                            time.sleep(0.05)

                            # Update interpolator with fresh pose
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[time.monotonic()],
                                poses=[curr_pose]
                            )

                            # In teleop mode, require clutch re-engage before accepting targets
                            # This prevents jumping to stale target values
                            if self.teleop_mode:
                                wait_for_clutch_engage = True  # Wait for clutch 0->1 transition
                                prev_clutch_active = False  # Reset clutch state
                                teleop_target_pose = curr_pose.copy()  # Use current pose
                                clutch_pose_offset = None  # Reset offset

                            print("[FrankaPositionalController] Recovery successful! Impedance restarted. (Waiting for clutch engage)")
                        except Exception as recovery_error:
                            print(f"[FrankaPositionalController] Recovery failed: {recovery_error}")
                            # Continue running, will retry next iteration
                    else:
                        # Unknown error, log but continue
                        if self.verbose:
                            print(f"[FrankaPositionalController] Update error: {e}")

                # update robot state — under ZeroRPC, prefer single batched
                # call (get_robot_state) so the bridge collapses 3 polymetis
                # internal-gRPC bursts into one. Falls back to per-key fetch.
                state = dict()
                try:
                    if robot.mode == 'zerorpc':
                        rs = robot._rpc.get_robot_state()
                        state['ActualTCPPose'] = mat_to_pose(
                            pose_to_mat(np.asarray(rs['ee_pose'], dtype=np.float64))
                            @ robot.tx_flange_tip
                        )
                        state['ActualQ']  = np.asarray(rs['q'],  dtype=np.float64)
                        state['ActualQd'] = np.asarray(rs['qd'], dtype=np.float64)
                    else:
                        for key, func_name in self.receive_keys:
                            state[key] = getattr(robot, func_name)()
                except Exception as e:
                    if self.verbose and iter_idx % 100 == 0:
                        print(f"[FrankaPositionalController] State read error: {e}")
                    continue

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue (works in both modes)
                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose_cmd = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose_cmd,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose_cmd, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose_cmd = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose_cmd,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.MOVE_HOME.value:
                        # Move to home position
                        home_duration = float(command['duration'])
                        print(f"\n[FrankaPositionalController] Moving to HOME position ({home_duration}s)...")

                        try:
                            # 1. Terminate impedance controller
                            robot.terminate_current_policy()
                            time.sleep(0.05)

                            # 2. Move to home joints
                            robot.move_to_joint_positions(
                                positions=self.home_joints,
                                time_to_go=home_duration
                            )

                            # 3. Get new pose after home motion
                            curr_pose = robot.get_ee_pose()
                            teleop_target_pose = curr_pose.copy()

                            # 4. Restart impedance controller
                            robot.start_cartesian_impedance(
                                Kx=self.Kx,
                                Kxd=self.Kxd
                            )

                            # IMPORTANT: Wait for impedance controller to stabilize
                            # The controller needs time to settle after restart.
                            # Without this delay, immediate clutch engagement can cause jerk.
                            STABILIZATION_DELAY = 0.5  # seconds
                            time.sleep(STABILIZATION_DELAY)

                            # 5. Update interpolator with fresh pose
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[time.monotonic()],
                                poses=[curr_pose]
                            )

                            # Reset timing
                            t_start = time.monotonic()
                            iter_idx = 0
                            last_waypoint_time = t_start

                            # In teleop mode, require clutch re-engage before accepting targets
                            # This prevents jumping to positions moved during HOME
                            if self.teleop_mode:
                                wait_for_clutch_engage = True  # Wait for clutch 0->1 transition
                                prev_clutch_active = False  # Reset clutch state
                                clutch_pose_offset = None  # Reset offset

                            print(f"[FrankaPositionalController] HOME complete. Pose: [{curr_pose[0]:.3f}, {curr_pose[1]:.3f}, {curr_pose[2]:.3f}] (Waiting for clutch engage)")

                        except Exception as e:
                            print(f"[FrankaPositionalController] HOME failed: {e}")
                            # Try to restart impedance anyway
                            try:
                                robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
                            except:
                                pass
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose and iter_idx % 100 == 0:
                    actual_freq = 1 / (time.monotonic() - t_now) if (time.monotonic() - t_now) > 0 else 0
                    mode_str = "TELEOP" if self.teleop_mode else "NORMAL"
                    print(f"[FrankaPositionalController] [{mode_str}] Actual frequency: {actual_freq:.1f}Hz")

        finally:
            # mandatory cleanup
            print('[FrankaPositionalController] Terminating control policy...')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
