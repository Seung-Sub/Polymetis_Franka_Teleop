"""
FrankaInterpolationController for Franka robot with configurable TCP offset.
Based on UMI's franka_interpolation_controller.py — uses direct polymetis
gRPC to NUC :50051. The earlier ZeroRPC bridge path (UMI/DROID style) was
removed once we confirmed the direct path was reliable; see git log for
history if you need to revive it.

Key features:
- Runs in separate process to ensure predictable timing (avoiding Python GIL)
- Smooth trajectory interpolation via PoseTrajectoryInterpolator
- Configurable TCP offset for different grippers (Franka Hand vs ART vs WSG)
- Direct polymetis RobotInterface — pro4000 → NUC :50051 (raw gRPC)
- Teleop mode for direct control from ViveTeleopProcess

Requirements (NUC):
- polymetis launch_robot.py running on :50051
  (start via `sudo bash /usr/local/sbin/start_franka_arm.sh`)

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
    """TCP-frame wrapper around polymetis RobotInterface (direct gRPC :50051).

    Args:
        ip:   NUC host.
        port: polymetis arm gRPC port (default 50051).
        tx_flange_tip_override: optional 4x4 flange→tool transform.
    """
    def __init__(self, ip='192.168.1.12', port=50051,
                 tx_flange_tip_override=None):
        self._cached_q = None
        # Last joint target that successfully passed IK. We send this to
        # polymetis whenever a fresh IK call fails so the NUC keeps receiving
        # updates and its 1 s watchdog doesn't kill the impedance policy.
        # Without this fallback, sending a Vive target outside the workspace
        # for >1 s would cascade-kill the controller.
        self._last_good_joint_target = None
        self._ik_fail_streak = 0
        # Wallclock when the last visible "IK STUCK" warning was printed,
        # so we throttle the warning rate without throttling the recovery.
        self._last_ik_stuck_warn_t = 0.0

        from polymetis import RobotInterface
        self.robot = RobotInterface(ip_address=ip, port=port)

        if tx_flange_tip_override is not None:
            self.tx_flange_tip = tx_flange_tip_override
            self.tx_tip_flange = np.linalg.inv(tx_flange_tip_override)
        else:
            self.tx_flange_tip = tx_flange_tip
            self.tx_tip_flange = tx_tip_flange

    def wait_until_controller_ready(self, max_wait_s: float = 1.5,
                                    poll_period_s: float = 0.01) -> bool:
        """Poll until the active polymetis controller actually accepts joint
        updates, instead of using a blind ``time.sleep(1.5)``.

        Why: ``start_cartesian_impedance`` returns success as soon as polymetis
        QUEUES the new policy load, but the controller takes 50-400 ms more
        to be ready. During that window any ``update_desired_joint_positions``
        call returns "Use 'start_joint_impedance' to start a joint impedance
        controller" / "Tried to perform a controller update with no controller
        running" -- which our exception handler treats as a recovery and
        triggers another start_cartesian_impedance, perpetuating the race.

        This poll calls ``update_desired_joint_positions(current_q)`` (a
        no-op feed since target == current) repeatedly until it stops
        raising. The first successful call confirms the controller has
        finished loading. Returns True on success, False on timeout.

        Catalog #29.
        """
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            try:
                q_now = self.robot.get_joint_positions()
                self.robot.update_desired_joint_positions(q_now)
                return True
            except Exception:
                time.sleep(poll_period_s)
        return False

    def reset_ik_state(self):
        """Reset the IK seed + last-good-joint-target after a controller restart.

        After a libfranka reflex + automaticErrorRecovery + start_cartesian_
        impedance cycle, the robot is at a fresh pose but our cached IK seed
        (``_cached_q``) and silent-failure fallback (``_last_good_joint_
        target``) still hold the values from BEFORE the reflex. If the user
        commands a target near the previous reflex configuration, IK will
        seed from that bad pose and fail; the silent fallback then replays
        the same bad joint target every tick, leaving the robot frozen and
        the user unable to move with Grip (only trackpad-HOME, which uses
        a different code path, escapes). Calling this method on every
        successful recovery forces the next IK call to fetch fresh joints
        and breaks that lock.
        """
        self._cached_q = None
        self._last_good_joint_target = None
        self._ik_fail_streak = 0

    # ----- low-level state -----
    def _flange_pose_6d(self):
        """Returns flange pose [x,y,z,rx,ry,rz] (axis-angle)."""
        pos, quat = self.robot.get_ee_pose()
        rotvec = st.Rotation.from_quat(quat.numpy()).as_rotvec()
        return np.concatenate([pos.numpy(), rotvec])

    def get_ee_pose(self):
        flange = self._flange_pose_6d()
        return mat_to_pose(pose_to_mat(flange) @ self.tx_flange_tip)

    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy()

    # ----- control -----
    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        import torch
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(np.asarray(positions, dtype=np.float32)),
            time_to_go=float(time_to_go),
        )

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        import torch
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(np.asarray(Kx, dtype=np.float32)),
            Kxd=torch.Tensor(np.asarray(Kxd, dtype=np.float32)),
        )

    def update_desired_ee_pose(self, tip_pose: np.ndarray):
        """Convert TCP-frame target → flange-frame target → robot.

        Solves IK locally, then issues 1 polymetis gRPC for the joint
        update (3× fewer RPCs than polymetis's default update_desired_ee_pose).
        """
        flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ self.tx_tip_flange)
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
            # IK didn't converge — most often because the user pushed the Vive
            # target outside the reachable workspace. Re-send the LAST good
            # target so polymetis still receives a tick this iter; otherwise
            # the NUC's 1 s watchdog will kill the impedance policy after
            # ~100 consecutive failures and we get a recovery cascade.
            self._ik_fail_streak += 1
            if self._ik_fail_streak in (1, 50, 100):
                print(f"[FrankaInterface] IK failed {self._ik_fail_streak}x "
                      f"in a row — sending last good joint target to keep "
                      f"polymetis watchdog alive.", flush=True)
            # Visible warning when IK is stuck for >2 seconds (200 ticks at
            # 100 Hz) — tells the user to press trackpad HOME instead of
            # waiting in vain. Throttled to once per 2 s so the log isn't
            # spammed if the streak persists for minutes.
            now_t = time.monotonic()
            if (self._ik_fail_streak >= 200
                    and (now_t - self._last_ik_stuck_warn_t) > 2.0):
                print("[FrankaInterface] IK STUCK -- robot has not moved for "
                      ">2 s due to repeated IK failures. The cached IK seed "
                      "is likely from a near-singular configuration. **Press "
                      "trackpad HOME** (or press 'h') to recover.",
                      flush=True)
                self._last_ik_stuck_warn_t = now_t
            if self._last_good_joint_target is not None:
                self.robot.update_desired_joint_positions(self._last_good_joint_target)
            return
        self._ik_fail_streak = 0
        # NOTE: catalog #35's joint-velocity pre-clamp was REVERTED on
        # 2026-05-09. The 2.0 rad/s clamp was too close to Franka's per-joint
        # velocity limits (j0-j3 = 2.175 rad/s, j4-j6 = 2.61 rad/s) and made
        # ALL teleop noticeably slow, not just the bursts that would have
        # tripped libfranka's safety_controller. A future re-introduction
        # should either (a) use per-joint thresholds at ~90% of each Franka
        # limit, (b) only kick in when the IK delta is well above ordinary
        # teleop, or (c) be opt-in via a CLI flag for high-precision tasks.
        self._last_good_joint_target = joint_target
        self.robot.update_desired_joint_positions(joint_target)

    def set_cached_q(self, q):
        """Optional: feed cached joint state for IK seed."""
        import torch
        self._cached_q = torch.as_tensor(q, dtype=torch.float32)

    def terminate_current_policy(self):
        try:
            self.robot.terminate_current_policy()
        except Exception:
            pass

    def close(self):
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
        robot_port=50051,
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
        # Optional SharedNDArray reference (shape (1,) uint8). When this
        # controller decides on its own to MOVE_HOME (catalog #27 auto-HOME
        # escalation after consecutive recoveries), it writes 1 here so the
        # Vive process can mirror the move with a synthesized action lerp,
        # keeping the recorded action stream continuous through the
        # autonomous joint-space motion.
        external_home_request_array=None,
        ):

        # Home position configuration
        if home_joints is not None:
            self.home_joints = np.array(home_joints)
        else:
            self.home_joints = FRANKA_HOME_JOINTS.copy()
        self.external_home_request_array = external_home_request_array

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        # Validate teleop mode configuration
        if teleop_mode and teleop_ring_buffer is None:
            raise ValueError("teleop_ring_buffer is required when teleop_mode=True")

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
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
        # Always pin to dedicated cores away from NIC IRQ (cores 6,7 by default
        # on pro4000's 14-core CPU). This is the single most effective fix for
        # `communication_constraints_violation` reflex storms — see
        # docs/troubleshooting.md catalog #25.
        try:
            from polymetis_franka_teleop.common.realtime_util import apply_realtime, PRO4000_CORE_MAP
            apply_realtime(
                cores=PRO4000_CORE_MAP.get('franka_interp'),
                sched_priority=20,  # SCHED_RR/20 -- preempts SCHED_OTHER, well below NUC's RTPRIO 80
                name='FrankaInterp',
            )
        except Exception as e:
            print(f"[FrankaPositionalController] WARN: realtime tuning skipped: {e}")
        # Legacy SCHED_RR-only path (still honored if user passes soft_real_time=True
        # but the realtime_util import failed — kept for backwards compatibility)
        if self.soft_real_time and 'apply_realtime' not in dir():
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except Exception:
                pass

        # Compute TCP transformation
        tx_flange_tip_local = compute_tcp_transform(
            tcp_offset=self.tcp_offset,
            use_wsg_gripper=self.use_wsg_gripper
        )

        # start polymetis interface with configurable TCP transform
        robot = FrankaInterface(
            ip=self.robot_ip,
            port=self.robot_port,
            tx_flange_tip_override=tx_flange_tip_local,
        )

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
                print(f"[FrankaPositionalController] TCP offset: {self.tcp_offset}m, WSG mode: {self.use_wsg_gripper}")
                print(f"[FrankaPositionalController] Teleop mode: {self.teleop_mode}")

            # init pose. Polymetis silently no-ops move_to_joint_positions if
            # ANOTHER controller is already running on the NUC (a leftover
            # JointImpedance from a crashed demo, or one started by a parallel
            # client). The fix used by Isaac-GR00T's franka_env_kist.reset()
            # is to terminate any current policy first.
            if self.joints_init is not None:
                print(f"[FrankaPositionalController] joints_init={np.round(self.joints_init, 3).tolist()}"
                      f" — terminating any stale controller, then move_to_joint_positions"
                      f"(time_to_go={self.joints_init_duration}s)...", flush=True)
                try:
                    try:
                        robot.robot.terminate_current_policy()
                        time.sleep(0.3)
                    except Exception:
                        pass  # no policy was running — fine
                    q_before = robot.get_joint_positions()
                    robot.move_to_joint_positions(
                        positions=np.asarray(self.joints_init),
                        time_to_go=self.joints_init_duration
                    )
                    q_after = robot.get_joint_positions()
                    moved = float(np.linalg.norm(q_after - q_before))
                    err_to_tgt = float(np.linalg.norm(q_after - np.asarray(self.joints_init)))
                    print(f"[FrankaPositionalController] move_to_joint_positions returned. "
                          f"q_after={np.round(q_after, 3).tolist()}  ‖Δq‖={moved:.3f}  "
                          f"‖q_after-target‖={err_to_tgt:.3f}", flush=True)
                    # Real silent no-op only if BOTH barely moved AND still far from target.
                    # If err_to_tgt is small, the robot was already at home (e.g. previous
                    # demo left it there) — that's success.
                    if moved < 0.05 and err_to_tgt > 0.1:
                        print(f"[FrankaPositionalController] WARNING: robot barely moved "
                              f"AND not at target — polymetis may have silently no-op'd. "
                              f"Check: 1) Franka Desk FCI active 2) no external clients "
                              f"holding the impedance.", flush=True)
                except Exception as e:
                    print(f"[FrankaPositionalController] move_to_joint_positions FAILED: {e!r}",
                          flush=True)
            else:
                print(f"[FrankaPositionalController] joints_init is None — skipping startup move",
                      flush=True)

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
            # Wait until polymetis controller actually accepts updates
            # (poll-until-ready, see catalog #29). Replaces blind sleep(1.5).
            # Returns True when controller is taking joint updates -- usually
            # 100-300 ms. Times out at 1.5 s with False if polymetis is stuck.
            if not robot.wait_until_controller_ready(max_wait_s=1.5):
                print("[FrankaPositionalController] WARN: controller still not "
                      "ready after 1.5 s -- next iteration may re-trigger "
                      "recovery", flush=True)

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True

            # Recovery throttling — polymetis can return "no controller
            # running" briefly even during normal operation (watchdog races
            # under transient CPU load). Each recovery is silent except for a
            # summary every 30 s, plus the very first one.
            recovery_count = 0
            recovery_last_summary_t = time.monotonic()
            recovery_just_fired = False  # set True on the iter that recovers
            # Auto-HOME escalation: when start_cartesian_impedance silently
            # fails (polymetis returns "Use start_joint_impedance" because
            # the controller didn't actually take), simple recoveries cascade
            # forever. Track consecutive recoveries within a 10 s window and
            # auto-trigger a HOME (which forces a full polymetis state reset
            # via move_to_joint_positions) after 2 in a row.
            consecutive_recovery_count = 0
            last_clean_iter_t = time.monotonic()
            auto_home_pending = False

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
                        recovery_count += 1
                        # Track recoveries within a 10 s window. If two fire
                        # back-to-back without any clean run in between,
                        # start_cartesian_impedance is silently failing and
                        # we escalate to a full HOME (catalog #27).
                        now_rec_t = time.monotonic()
                        if (now_rec_t - last_clean_iter_t) < 10.0:
                            consecutive_recovery_count += 1
                        else:
                            consecutive_recovery_count = 1
                        last_clean_iter_t = now_rec_t
                        if consecutive_recovery_count >= 2 and self.teleop_mode:
                            print(f"[FrankaPositionalController] !! "
                                  f"{consecutive_recovery_count} recoveries in <10 s -- "
                                  f"start_cartesian_impedance not taking. "
                                  f"Auto-HOME to force polymetis state reset.",
                                  flush=True)
                            auto_home_pending = True
                        # Print full detail on first error so we can identify
                        # the trigger; subsequent ones print a one-liner so
                        # the user can see them happening (was previously
                        # silent and masked the true frequency).
                        if recovery_count == 1:
                            print(f"\n[FrankaPositionalController] Controller error: {e}")
                            print(f"[FrankaPositionalController] Attempting automatic recovery (count=1)...")
                        else:
                            short_err = str(e).split('\n')[0][:80]
                            print(f"[FrankaPositionalController] Recovery #{recovery_count}: {short_err}",
                                  flush=True)
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

                            # Restart impedance controller. Sleep 1.0 s before
                            # the loop's next update_desired_ee_pose call —
                            # polymetis takes 200-400 ms typical but can be
                            # 800+ ms under load; 0.5 s used to cascade into
                            # repeat recoveries. 1.5 s breaks the cascade
                            # observed at KIST 2026-05-09 (8 reflex trips
                            # in 1.5 min window with 1.0 s settle).
                            robot.start_cartesian_impedance(
                                Kx=self.Kx,
                                Kxd=self.Kxd
                            )
                            # Poll-until-ready instead of blind sleep (catalog #29).
                            if not robot.wait_until_controller_ready(max_wait_s=1.5):
                                print("[FrankaPositionalController] WARN: controller "
                                      "still not ready 1.5 s after recovery -- "
                                      "auto-HOME will fire next iter", flush=True)

                            # CRITICAL: reset the IK seed + last-good-joint-
                            # target. The previous values were captured BEFORE
                            # the reflex and may correspond to a near-singular
                            # configuration; without resetting, the next IK
                            # call seeds from the bad pose, fails, the silent-
                            # failure path replays the same bad joint target
                            # every tick, and the user sees Grip not respond
                            # while only HOME (which uses a different code
                            # path) escapes. Reset here forces the next IK
                            # call to fetch fresh joints. Catalog #26.
                            robot.reset_ik_state()

                            # CRITICAL: reset the loop timer so we do NOT
                            # "catch up" the 1.5 s sleep with a burst of
                            # 250+ Hz update_desired_joint_positions calls
                            # — that burst overwhelmed polymetis's
                            # HybridJointImpedanceController state machine
                            # and immediately re-tripped reflex (catalog #27,
                            # observed at KIST 2026-05-09: cumulative 6
                            # recoveries every 3.7 s in a 22 s window).
                            # HOME's code path already does this (line 1011);
                            # missing it here was the cascade root cause.
                            t_start = time.monotonic()
                            iter_idx = 0
                            last_waypoint_time = t_start

                            # Update interpolator with fresh pose
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[t_start],
                                poses=[curr_pose]
                            )

                            # Decide whether the user must release-and-re-press
                            # Grip after recovery. For "no controller running"
                            # (polymetis watchdog race) the robot stayed at its
                            # commanded pose and the offset between Vive and
                            # robot is unchanged — we can re-sync the offset
                            # in-place and keep teleop fluent. For reflex/limit
                            # violations the robot decelerated to an unknown
                            # pose, so we force grip release for safety.
                            is_watchdog_only = ('no controller running' in error_str
                                                and not any(x in error_str for x in (
                                                    'reflex', 'velocity_violation',
                                                    'joint_position_limits',
                                                    'torque_discontinuity')))
                            if self.teleop_mode:
                                if is_watchdog_only:
                                    # Recompute clutch offset against fresh pose
                                    # so the next teleop target lands at curr_pose.
                                    teleop_target_pose = curr_pose.copy()
                                    if clutch_pose_offset is not None:
                                        try:
                                            ts_now = self.teleop_ring_buffer.get()
                                            teleop_target_raw = ts_now['target_pose']
                                            clutch_pose_offset = curr_pose - teleop_target_raw
                                        except Exception:
                                            pass
                                    # Don't force wait_for_clutch_engage — keep teleop alive.
                                else:
                                    wait_for_clutch_engage = True
                                    prev_clutch_active = False
                                    teleop_target_pose = curr_pose.copy()
                                    clutch_pose_offset = None

                            recovery_just_fired = True
                            if recovery_count == 1:
                                print("[FrankaPositionalController] Recovery successful! "
                                      "Impedance restarted.")
                            # Rate summary every 30 s if recoveries keep firing
                            now_t = time.monotonic()
                            if now_t - recovery_last_summary_t >= 30.0:
                                rate = recovery_count / max(1.0, now_t - t_start)
                                print(f"[FrankaPositionalController] cumulative {recovery_count} "
                                      f"recoveries ({rate:.3f}/s avg) — investigate if rate>0.05/s "
                                      f"persists.", flush=True)
                                recovery_last_summary_t = now_t
                        except Exception as recovery_error:
                            print(f"[FrankaPositionalController] Recovery failed: {recovery_error}")
                            # Continue running, will retry next iteration
                    else:
                        # Unknown error, log but continue
                        if self.verbose:
                            print(f"[FrankaPositionalController] Update error: {e}")

                # update robot state via per-key polymetis fetch
                state = dict()
                try:
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

                # Feed the joint state we just read into FrankaInterface as the
                # IK seed for the upcoming update_desired_ee_pose. Without this
                # cache, ``update_desired_ee_pose`` falls back to making its
                # own get_joint_positions() gRPC every iter — at 100 Hz that's
                # 100 extra round-trips/s to the NUC, doubling polymetis load
                # and contributing to the "no controller running" watchdog
                # races we saw recover every ~10 s.
                try:
                    robot.set_cached_q(state['ActualQ'])
                except Exception:
                    pass

                # Joint-limit early-warning. Franka hard limits per joint:
                # j3 (elbow)  -> 2.97 rad   (warn at 2.5)
                # j5 (wrist2) -> 2.97 rad   (warn at 2.5)
                # j6 (wrist3) -> 2.89 rad   (warn at 2.5)
                # When teleop pushes past warning level the cartesian
                # impedance starts requiring high torque, and small extra
                # motion trips libfranka's safety_controller (Joint velocity
                # violation -> reflex). Warn (10 s throttled) so user can
                # back off before reflex fires. Catalog #28.
                if not hasattr(self, '_last_joint_warn_t'):
                    self._last_joint_warn_t = {3: 0.0, 5: 0.0, 6: 0.0}
                try:
                    qabs = [abs(float(state['ActualQ'][j])) for j in range(7)]
                except Exception:
                    qabs = [0.0] * 7
                now_warn_t = time.monotonic()
                _jlim = {3: 2.97, 5: 2.97, 6: 2.89}
                _jname = {3: 'j3 (elbow)', 5: 'j5 (wrist-flex)', 6: 'j6 (wrist-roll)'}
                for j in (3, 5, 6):
                    if (qabs[j] > 2.5
                            and (now_warn_t - self._last_joint_warn_t[j]) > 10.0):
                        margin = _jlim[j] - qabs[j]
                        print(f"[FrankaPositionalController] !! "
                              f"{_jname[j]}={qabs[j]:.3f} rad "
                              f"(limit {_jlim[j]}, margin {margin:.3f}) -- "
                              f"back off before reflex fires", flush=True)
                        self._last_joint_warn_t[j] = now_warn_t

                # fetch command from queue (works in both modes)
                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Auto-HOME escalation: when start_cartesian_impedance silently
                # failed and we hit 2+ recoveries within 10 s, inject a synthetic
                # MOVE_HOME command. HOME does a
                # move_to_joint_positions(home_joints) which forces polymetis
                # to cycle through its trajectory tracking controller, which in
                # turn forces a clean state transition that subsequent
                # start_cartesian_impedance can actually take. Any user
                # commands queued at the same instant are dropped; user can
                # re-issue after HOME completes (~3 s). Catalog #27.
                if auto_home_pending:
                    auto_home_pending = False
                    consecutive_recovery_count = 0
                    commands = {'cmd': [Command.MOVE_HOME.value],
                                'duration': [2.0]}
                    n_cmd = 1
                    # Mirror the move in the Vive process's HOME state machine
                    # so the synthesized action lerp fires (catalog #27 path
                    # used to bypass synthesis -- action would freeze at the
                    # user's last grip-on pose while obs moved to home).
                    if self.external_home_request_array is not None:
                        try:
                            self.external_home_request_array.get()[0] = 1
                        except Exception:
                            pass
                    print("[FrankaPositionalController] auto-HOME injected to "
                          "force polymetis state reset (catalog #27)", flush=True)

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
                        # Tag this iter so the slow-iter WARN doesn't fire — a
                        # MOVE_HOME iter legitimately takes 2 s motion + 0.5 s
                        # stabilize ≈ 3 s and is not a watchdog risk.
                        recovery_just_fired = True
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

                            # Poll-until-ready instead of blind sleep (catalog #29).
                            # 1.5 s timeout matches recovery path.
                            if not robot.wait_until_controller_ready(max_wait_s=1.5):
                                print("[FrankaPositionalController] WARN: post-HOME "
                                      "controller did not come ready in 1.5 s",
                                      flush=True)

                            # Reset IK seed/last_good cache so the post-HOME
                            # robot pose is the fresh seed for IK (catalog #26).
                            # Even though HOME got us to ready_pose, the cache
                            # may still hold stale values from before HOME.
                            robot.reset_ik_state()

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
                                robot.reset_ik_state()  # see catalog #26
                            except:
                                pass
                    else:
                        keep_running = False
                        break

                # Detect iters that take > 100 ms (= 10x normal). 100Hz iters
                # should be ~10 ms each. Anything > 100 ms is a stall worth
                # investigating; >1 s trips the polymetis watchdog. We
                # exclude recovery iters (which inherently sleep ~600 ms) by
                # comparing the previous-iter end timestamp.
                t_iter_end = time.monotonic()
                iter_duration = t_iter_end - t_now
                # Recovery iters legitimately spend 600+ ms in start_impedance
                # + sleep; quietly skip warning during the same iter as the
                # recovery code ran.
                # Suppress WARN when this iter ran a recovery or MOVE_HOME
                # (both legitimately sleep > 100 ms and aren't watchdog risks).
                # recovery_just_fired = True is set in BOTH cases.
                if iter_duration > 0.1 and not recovery_just_fired:
                    print(f"[FrankaPositionalController] WARN slow iter "
                          f"#{iter_idx}: {iter_duration*1000:.0f} ms "
                          f"(watchdog limit 1000 ms)", flush=True)
                recovery_just_fired = False

                # Track clean iters for the auto-HOME escalation: any iter
                # that reached this point without triggering recovery counts
                # as "clean". The consecutive_recovery_count window resets
                # itself naturally after 10 s of clean operation.
                last_clean_iter_t = time.monotonic()

                # regulate frequency + count overruns (iters where the
                # precise_wait deadline was already past when we got there).
                # High overrun rate = this loop is being scheduled-late by
                # the kernel, which is the proximate cause of NUC's
                # communication_constraints_violation reflex.
                t_wait_util = t_start + (iter_idx + 1) * dt
                if time.monotonic() > t_wait_util:
                    if not hasattr(self, '_overrun_count'):
                        self._overrun_count = 0
                    self._overrun_count += 1
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose and iter_idx % 100 == 0:
                    actual_freq = 1 / (time.monotonic() - t_now) if (time.monotonic() - t_now) > 0 else 0
                    mode_str = "TELEOP" if self.teleop_mode else "NORMAL"
                    overrun = getattr(self, '_overrun_count', 0)
                    print(f"[FrankaPositionalController] [{mode_str}] Actual frequency: "
                          f"{actual_freq:.1f}Hz  overruns={overrun}/100 (last 100 iters)")
                    self._overrun_count = 0

        finally:
            # mandatory cleanup
            print('[FrankaPositionalController] Terminating control policy...')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
