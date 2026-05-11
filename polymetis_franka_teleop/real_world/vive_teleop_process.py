"""
ViveTeleopProcess - High-frequency Vive teleop calculations in separate process.

This process runs at 100Hz and performs all teleop calculations:
- Clutch-based delta computation
- Controller-local to robot coordinate transformation
- Velocity clamping (optional)
- Outlier detection

Outputs to 3 SharedMemoryRingBuffers:
1. action_ring_buffer: For Main Process recording (target_pose + gripper)
2. robot_ring_buffer: For FrankaInterpolationController (direct target_pose)
3. gripper_ring_buffer: For FrankaGripperController (gripper commands)

Architecture:
    ViveTeleopProcess (100Hz)
    |-- Reads raw Vive data from ViveSharedMemory
    |-- Performs teleop calculations
    +-- Outputs to 3 RingBuffers
            |
            |-- action_ring_buffer -> Main Process (10Hz recording)
            |-- robot_ring_buffer -> FrankaInterpolationController (bypass schedule_waypoint)
            +-- gripper_ring_buffer -> FrankaGripperController

Usage:
    with SharedMemoryManager() as shm_manager:
        shm_manager.start()
        with ViveSharedMemory(shm_manager=shm_manager) as vive:
            with ViveTeleopProcess(
                shm_manager=shm_manager,
                vive_ring_buffer=vive.ring_buffer,
                robot_ip='192.168.1.10'
            ) as teleop:
                # Main process can read action from action_ring_buffer
                action = teleop.get_action()
"""

import multiprocessing as mp
import numpy as np
import time
import enum
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R, Slerp

from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from polymetis_franka_teleop.shared_memory.shared_ndarray import SharedNDArray


class GripperCommand(enum.Enum):
    NONE = 0
    OPEN = 1
    CLOSE = 2


class ViveTeleopProcess(mp.Process):
    """
    High-frequency Vive teleop calculations in separate process.

    Runs at 100Hz, performing all teleop calculations and outputting to
    SharedMemoryRingBuffers for consumption by controllers and main process.

    Args:
        shm_manager: SharedMemoryManager for shared memory allocation
        vive_ring_buffer: SharedMemoryRingBuffer from ViveSharedMemory for reading controller data
        robot_ip: IP address of robot (for getting initial pose)
        robot_port: polymetis arm gRPC port (default: 50051)
        frequency: Control frequency in Hz (default: 100)
        pos_scale: Position scaling factor (default: 1.0)
        rot_scale: Rotation scaling factor (default: 1.0)
        use_velocity_clamping: Enable velocity clamping (default: True)
        max_pos_velocity: Max position velocity in m/s (default: 1.5)
        max_rot_velocity: Max rotation velocity in rad/s (default: 2.5)
        get_max_k: Max states in ring buffer (default: 100)
        tcp_offset: TCP offset for Franka Hand (must match FrankaInterpolationController)
        verbose: Enable verbose logging (default: False)
    """

    # Default coordinate transformation matrix
    # Controller-local to Robot frame mapping:
    #   Pull toward user (Controller Y+) -> Robot X+ (forward)
    #   Move right (Controller X+) -> Robot Y+ (right)
    #   Move up (Controller Z-) -> Robot Z+ (up)
    TX_ROBOT_VIVE = np.array([
        [0, 1, 0],   # Robot X = Controller Y
        [1, 0, 0],   # Robot Y = Controller X
        [0, 0, -1]   # Robot Z = -Controller Z
    ], dtype=np.float32)

    # Outlier detection threshold
    POSITION_THRESHOLD = 1.0  # meters
    MAX_VELOCITY_THRESHOLD = 5.0  # m/s (anything faster is tracking error)

    def __init__(self,
            shm_manager: SharedMemoryManager,
            vive_ring_buffer,  # SharedMemoryRingBuffer from ViveSharedMemory
            robot_ip: str,
            robot_port: int = 50051,
            frequency: int = 100,
            pos_scale: float = 1.0,
            rot_scale: float = 1.0,
            use_velocity_clamping: bool = True,
            max_pos_velocity: float = 1.5,
            max_rot_velocity: float = 2.5,
            get_max_k: int = 100,
            receive_latency: float = 0.0,
            tcp_offset: float = 0.1034,  # TCP offset for Franka Hand
            gripper_open_width: float = 0.08,   # Gripper width when open (Franka Hand max)
            gripper_close_width: float = 0.005,   # Gripper width when closed (5mm - avoid 0.0 which causes libfranka exception)
            # Trackpad rotation parameters (Joint 7 / EE Z-axis rotation)
            enable_trackpad_rotation: bool = True,  # Enable trackpad touch rotation
            rotation_speed: float = 0.5,  # Rotation speed in degrees per update (at 100Hz)
            rotation_limit: float = 90.0,  # Max rotation from initial pose in degrees (±limit)
            trackpad_touch_threshold: float = 0.1,  # Threshold for detecting trackpad touch
            trackpad_rotation_threshold: float = 0.7,  # Trackpad Y threshold for rotation
            home_duration: float = 2.0,  # seconds — must match controller's home_time
            verbose: bool = False
        ):
        super().__init__(name="ViveTeleopProcess")

        # Trackpad rotation settings
        self.enable_trackpad_rotation = enable_trackpad_rotation
        self.rotation_speed = np.deg2rad(rotation_speed)  # Convert to radians
        self.rotation_limit = np.deg2rad(rotation_limit)  # Convert to radians
        self.trackpad_touch_threshold = trackpad_touch_threshold
        self.trackpad_rotation_threshold = trackpad_rotation_threshold

        self.vive_ring_buffer = vive_ring_buffer
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.use_velocity_clamping = use_velocity_clamping
        self.max_pos_velocity = max_pos_velocity
        self.max_rot_velocity = max_rot_velocity
        self.receive_latency = receive_latency
        self.tcp_offset = tcp_offset
        self.gripper_open_width = gripper_open_width
        self.gripper_close_width = gripper_close_width
        self.home_duration = float(home_duration)
        self.verbose = verbose

        # SharedNDArray for the synthesized-action HOME target. Main process
        # writes it once after env.start_wait() (when the robot has just
        # auto-homed); this process reads it on every trackpad-HOME to lerp
        # action target_pose from the user's last intent → home pose. A norm
        # of 0 means "not yet known", in which case we fall back to the
        # observation-following behaviour for that one HOME event.
        self.home_target_pose_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(6,), dtype=np.float64,
        )
        # Initialize the 6-vec to zeros (so .norm() == 0 means "not set").
        self.home_target_pose_array.get()[:] = 0.0

        # External HOME request flag (1 byte). Producers — env.move_home()
        # for keyboard 'h' / cv2 SIGHUP, FrankaInterpolationController for
        # catalog #27 auto-HOME escalation — set this to 1. The Vive process
        # picks it up on the next iteration, treats it as a synthetic
        # trackpad-press, and runs the same lerp/slerp HOME synthesis so the
        # recorded action stream stays continuous regardless of which path
        # triggered the move.
        self.external_home_request_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(1,), dtype=np.uint8,
        )
        self.external_home_request_array.get()[0] = 0

        # External re-sync flag: producer (env.start_wait) sets this after
        # auto-home completes so the Vive process replaces its stale
        # ``target_pose`` (initialised pre-auto-home) with the robot's
        # actual current TCP pose. Eliminates the pre-first-grip mismatch
        # between recorded action and obs.
        self.external_target_resync_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(1,), dtype=np.uint8,
        )
        self.external_target_resync_array.get()[0] = 0

        # Create ring buffer for action output (for Main Process - UMI style)
        # gripper_state: 0=open, 1=closed (discrete toggle state)
        # gripper_target_width: continuous width based on trigger_value (for smooth action data)
        action_example = {
            'target_pose': np.zeros((6,), dtype=np.float64),  # [x,y,z,rx,ry,rz]
            'gripper_state': np.float64(0.0),  # 0=open, 1=closed (discrete toggle)
            'gripper_target_width': np.float64(self.gripper_open_width),  # continuous width [0, 0.075]
            'clutch_active': np.uint8(0),
            'home_requested': np.uint8(0),  # Set to 1 when trackpad pressed to request home
            'home_active': np.uint8(0),  # Set to 1 during home movement (action = robot pose)
            'rotation_active': np.uint8(0),  # Set to 1 during trackpad rotation
            'timestamp': time.time()
        }
        self.action_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=action_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # Create ring buffer for robot target pose (for FrankaInterpolationController)
        robot_example = {
            'target_pose': np.zeros((6,), dtype=np.float64),
            'clutch_active': np.uint8(0),
            'rotation_active': np.uint8(0),  # Set to 1 during trackpad rotation
            'teleop_timestamp': time.time()
        }
        self.robot_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=robot_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # Create ring buffer for gripper commands (for FrankaGripperController)
        gripper_example = {
            'gripper_state': np.float64(0.0),  # 0=open, 1=closed
            'gripper_command': np.int32(GripperCommand.NONE.value),
            'teleop_timestamp': time.time()
        }
        self.gripper_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=gripper_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # Process synchronization
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

    # ========= Public APIs (called from Main Process) =========

    def get_action(self, k=None, out=None):
        """Get latest action (target_pose + gripper_state) for recording."""
        if k is None:
            return self.action_ring_buffer.get(out=out)
        else:
            return self.action_ring_buffer.get_last_k(k=k, out=out)

    def get_all_actions(self):
        """Get all buffered actions."""
        return self.action_ring_buffer.get_all()

    def is_clutch_active(self) -> bool:
        """Check if clutch (grip) is currently active."""
        state = self.action_ring_buffer.get()
        return bool(state['clutch_active'])

    def set_home_target_pose(self, pose: np.ndarray) -> None:
        """Provide the TCP pose corresponding to the canonical home joints.

        The env should call this *after* startup auto-home completes (i.e.
        right after ``robot.start_wait()`` returns), passing the robot's
        actual ``ActualTCPPose`` at that moment. The run loop then uses this
        as the lerp/slerp endpoint when synthesizing the action trajectory
        during trackpad-triggered HOME events.

        Args:
            pose: 6-vec [x, y, z, rx, ry, rz] (axis-angle), TCP frame.
        """
        pose = np.asarray(pose, dtype=np.float64)
        assert pose.shape == (6,), f'home pose shape {pose.shape} != (6,)'
        self.home_target_pose_array.get()[:] = pose

    def request_home_synthesis(self) -> None:
        """Ask the Vive process to enter HOME synthesis on its next iter.

        Callable from the main process / from controller subprocesses that
        hold a reference to ``external_home_request_array``. The flag is
        consumed (cleared) by the Vive process once seen, so callers don't
        need to reset it. Multiple back-to-back calls collapse — only one
        HOME synthesis fires per consumption."""
        self.external_home_request_array.get()[0] = 1

    def sync_target_pose_to_current_robot(self) -> None:
        """Force the Vive process's internal target_pose to match the
        robot's current TCP pose on its next iter. Used by env.start_wait()
        right after auto-home so the first recorded action sample doesn't
        carry the stale pre-auto-home pose."""
        self.external_target_resync_array.get()[0] = 1

    # ========= Start/Stop APIs =========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(timeout=5.0)
        if self.verbose:
            print(f"[ViveTeleopProcess] Started at PID {self.pid}")

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self, timeout=5.0):
        self.ready_event.wait(timeout=timeout)

    def stop_wait(self):
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= Helper methods =========

    def compute_local_delta(self, current_pos: np.ndarray, current_quat: np.ndarray,
                            start_pos: np.ndarray, start_quat: np.ndarray):
        """
        Compute delta in controller-local coordinates.

        Key operation: rotatedRelativePos = initialQuat.inverse() * relativePos

        Returns:
            local_pos_delta: Position delta in controller-local frame
            local_rot_delta: Rotation delta as scipy Rotation object
        """
        # Position delta in world frame
        world_delta = current_pos - start_pos

        # Rotate to controller-local frame
        r_start = R.from_quat(start_quat)
        local_pos_delta = r_start.inv().apply(world_delta)

        # Rotation delta
        r_current = R.from_quat(current_quat)
        local_rot_delta = r_start.inv() * r_current

        return local_pos_delta, local_rot_delta

    def local_to_robot(self, local_pos: np.ndarray, local_rot: R = None):
        """
        Convert controller-local delta to robot frame.

        Calibrated mapping:
        - Pull toward me (Controller Y+) -> Robot X+ (comes forward)
        - Move right (Controller X+) -> Robot Y+ (user's right)
        - Move up (Controller Z-) -> Robot Z+ (up)
        """
        robot_pos = np.array([
            local_pos[1],    # Robot X = Controller Y
            local_pos[0],    # Robot Y = Controller X
            -local_pos[2]    # Robot Z = -Controller Z
        ])

        if local_rot is None:
            return robot_pos, None

        # Transform rotation axis the same way as position
        local_rotvec = local_rot.as_rotvec()
        angle = np.linalg.norm(local_rotvec)

        if angle < 1e-8:
            return robot_pos, R.identity()

        axis = local_rotvec / angle
        robot_axis = np.array([
            axis[1],     # Robot X = Controller Y
            axis[0],     # Robot Y = Controller X
            -axis[2]     # Robot Z = -Controller Z
        ])
        robot_axis_norm = np.linalg.norm(robot_axis)
        if robot_axis_norm > 1e-8:
            robot_axis = robot_axis / robot_axis_norm

        return robot_pos, R.from_rotvec(robot_axis * angle)

    def validate_position_change(self, current_pos: np.ndarray,
                                  prev_pos: np.ndarray, prev_time: float) -> bool:
        """
        Check if position change is reasonable (outlier detection).
        Returns False if the change is too large (likely tracking error).
        """
        current_time = time.time()

        if prev_pos is None or prev_time is None:
            return True

        delta = np.linalg.norm(current_pos - prev_pos)
        dt = current_time - prev_time

        if dt > 0.001:  # Minimum 1ms to avoid division issues
            velocity = delta / dt
            if delta > self.POSITION_THRESHOLD and velocity > self.MAX_VELOCITY_THRESHOLD:
                return False

        return True

    def clamp_velocity(self, desired: np.ndarray, prev: np.ndarray,
                       max_velocity: float, dt: float) -> np.ndarray:
        """Apply velocity clamping to position or rotation."""
        if prev is None:
            return desired

        delta = desired - prev
        delta_mag = np.linalg.norm(delta)
        max_delta = max_velocity * dt

        if delta_mag > max_delta:
            return prev + delta / delta_mag * max_delta
        return desired

    # ========= Main Loop (runs in child process) =========

    def run(self):
        """Main teleop calculation loop running at 100Hz."""
        import scipy.spatial.transform as st
        from polymetis_franka_teleop.common.pose_util import pose_to_mat, mat_to_pose

        try:
            # Reuse FrankaInterface (direct polymetis :50051) for state reads.
            from polymetis_franka_teleop.real_world.franka_interpolation_controller import FrankaInterface
            robot_client = FrankaInterface(
                ip=self.robot_ip, port=self.robot_port,
            )

            def _flange_pose_6d():
                # FrankaInterface._flange_pose_6d() returns 6D axis-angle directly.
                return robot_client._flange_pose_6d()

            flange_pose = _flange_pose_6d()

            # Apply TCP offset to convert flange pose to TCP (tip) pose
            tx_flange_tip = np.identity(4)
            tx_flange_tip[2, 3] = self.tcp_offset
            initial_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
            target_pose = initial_pose.copy()

            if self.verbose:
                print(f"[ViveTeleopProcess] Connected to polymetis arm "
                      f"{self.robot_ip}:{self.robot_port}")
                print(f"[ViveTeleopProcess] Flange (raw): {flange_pose[:3]}")
                print(f"[ViveTeleopProcess] TCP (offset {self.tcp_offset}m): {initial_pose[:3]}")

            def get_current_robot_pose():
                try:
                    flange = _flange_pose_6d()
                    return mat_to_pose(pose_to_mat(flange) @ tx_flange_tip)
                except Exception:
                    return None

            # State variables
            clutch_active = False
            clutch_robot_start = np.zeros(6)
            clutch_vr_start_pos = np.zeros(3)
            clutch_vr_start_quat = np.array([0., 0., 0., 1.])

            gripper_closed = False
            gripper_state = 0.0  # 0=open, 1=closed
            gripper_command = GripperCommand.NONE

            # Gripper action continuity control
            # After toggle, wait for trigger release before applying new formula
            # This prevents discontinuity at toggle moment
            awaiting_trigger_release = False

            # Edge detection for buttons
            prev_grip = False
            prev_trigger = False
            prev_trackpad = False

            # Home mode state
            home_active = False
            home_requested = False
            home_start_time = 0.0   # wall-clock at HOME activation (lerp t=0)
            # Action duration of the synthesized HOME trajectory. Slightly
            # longer than the controller's actual move_to_joint_positions time
            # so that the action lerp comfortably brackets the real motion;
            # see start_franka_arm.sh + FrankaInterpolationController.home_time.
            HOME_DURATION = max(self.home_duration, 0.5)
            # Cartesian endpoints of the synthesized trajectory.
            # home_lerp_start_pose: the user's last-known intent at the moment
            #   trackpad is pressed. The lerp begins here so action signal
            #   is continuous (no jump from previous action sample).
            # home_lerp_end_pose: the TCP pose corresponding to home_joints,
            #   read out of self.home_target_pose_array (set by the env after
            #   startup auto-home). If unset (norm == 0), we fall back to
            #   obs-following for this one HOME — only impacts the *very
            #   first* HOME of a session that started before env injected it.
            home_lerp_start_pose = np.zeros(6, dtype=np.float64)
            home_lerp_end_pose = np.zeros(6, dtype=np.float64)
            home_lerp_use_synth = False  # toggled at trackpad-press

            # Trackpad rotation state
            rotation_active = False
            base_rotation = None  # Store initial EE rotation for limit checking
            accumulated_z_rotation = 0.0  # Accumulated rotation angle (radians) from base_rotation
            base_position = None  # Store position when rotation started

            # Grip release requirement (after HOME or recovery)
            # When True, user must release grip before clutch can be activated again
            # This ensures proper 0→1 transition detection in FrankaInterpolationController
            require_grip_release = False

            # Velocity clamping state
            prev_target_pos = None
            prev_target_rot = None

            # Outlier detection state
            prev_vr_pos = None
            prev_vr_time = None

            # Timing
            dt = 1.0 / self.frequency
            t_start = time.monotonic()
            iter_idx = 0

            # Gripper target width (continuous value based on trigger_value)
            # Initialized to open width (0.08m)
            gripper_target_width = self.gripper_open_width

            # Put initial state
            action_state = {
                'target_pose': target_pose,
                'gripper_state': np.float64(gripper_state),
                'gripper_target_width': np.float64(gripper_target_width),
                'clutch_active': np.uint8(0),
                'home_requested': np.uint8(0),
                'home_active': np.uint8(0),
                'rotation_active': np.uint8(0),
                'timestamp': time.time()
            }
            self.action_ring_buffer.put(action_state)

            robot_state = {
                'target_pose': target_pose,
                'clutch_active': np.uint8(0),
                'rotation_active': np.uint8(0),
                'teleop_timestamp': time.time()
            }
            self.robot_ring_buffer.put(robot_state)

            gripper_rb_state = {
                'gripper_state': np.float64(gripper_state),
                'gripper_command': np.int32(GripperCommand.NONE.value),
                'teleop_timestamp': time.time()
            }
            self.gripper_ring_buffer.put(gripper_rb_state)

            self.ready_event.set()

            while not self.stop_event.is_set():
                t_now = time.monotonic()

                # Get Vive state
                vive_state = self.vive_ring_buffer.get()
                vive_pos = vive_state['position']
                vive_quat = vive_state['quaternion']
                grip_pressed = bool(vive_state['grip'])
                trigger_pressed = bool(vive_state['trigger'])
                trackpad_pressed = bool(vive_state['trackpad'])
                trigger_value = float(vive_state.get('trigger_value', 0.0))  # 0.0 ~ 1.0

                # Trackpad touch detection (for rotation)
                trackpad_x = float(vive_state.get('trackpad_x', 0.0))
                trackpad_y = float(vive_state.get('trackpad_y', 0.0))
                trackpad_touched = (abs(trackpad_x) > self.trackpad_touch_threshold or
                                   abs(trackpad_y) > self.trackpad_touch_threshold)

                # === External HOME request synthesis ===
                # Consumed once-per-rising-edge. Producers:
                #   * env.move_home()  -- covers cv2 'h' key / SIGHUP
                #   * FrankaInterpolationController catalog #27 auto-HOME
                # By spoofing a trackpad rising edge here we reuse the exact
                # same HOME state-machine + lerp/slerp synthesis that the
                # user-triggered trackpad-press path uses, so action stream
                # stays continuous regardless of which path fired the move.
                if (not home_active) and bool(self.external_home_request_array.get()[0]):
                    self.external_home_request_array.get()[0] = 0  # consume
                    trackpad_pressed = True
                    prev_trackpad = False  # force the rising-edge detector to fire below
                    if self.verbose:
                        print("[ViveTeleopProcess] external HOME request received "
                              "— firing synthetic trackpad press")

                # === Target pose re-sync (env.start_wait → after auto-home) ===
                # Replaces the stale initial target_pose (captured at Vive
                # process boot, before auto-home moved the robot) with the
                # robot's actual current TCP pose. Removes the first-grip
                # discontinuity between action[0] and obs[0].
                if bool(self.external_target_resync_array.get()[0]):
                    self.external_target_resync_array.get()[0] = 0  # consume
                    curr = get_current_robot_pose()
                    if curr is not None:
                        target_pose = curr.copy()
                        if self.verbose:
                            print(f"[ViveTeleopProcess] target_pose re-synced to "
                                  f"current robot pose {curr[:3].round(3).tolist()}")

                # === TRACKPAD: Home pose request ===
                if trackpad_pressed and not prev_trackpad:
                    # Trackpad just pressed - request home
                    home_requested = True
                    home_active = True
                    home_start_time = time.time()
                    clutch_active = False  # Disable clutch during home
                    prev_target_pos = None
                    prev_target_rot = None
                    # CRITICAL: clear gripper_command at HOME START so the
                    # 2-sec HOME window does not keep publishing the stale
                    # pre-HOME CLOSE/OPEN command into gripper_ring_buffer.
                    # Without this, env.move_home()'s explicit goto(open_w)
                    # gets immediately overridden in the same ART controller
                    # iter by the stale CLOSE from Vive teleop -- gripper
                    # closes again right after opening, and the user sees
                    # the 3-press regression. Catalog #34.
                    gripper_command = GripperCommand.NONE

                    # === Capture endpoints for the synthesized action trajectory ===
                    # Start of lerp = whatever the user's last action target was
                    # (this is target_pose right now: either the last Vive-driven
                    # value while grip was held, or the held-after-release value).
                    # End of lerp = home TCP pose (set once by env at startup).
                    home_lerp_start_pose = target_pose.copy()
                    end_candidate = self.home_target_pose_array.get().copy()
                    if np.linalg.norm(end_candidate[:3]) > 1e-6:
                        home_lerp_end_pose = end_candidate
                        home_lerp_use_synth = True
                        if self.verbose:
                            print(
                                f"[ViveTeleopProcess] HOME requested — synthesizing "
                                f"cartesian trajectory from "
                                f"{home_lerp_start_pose[:3].round(3).tolist()} -> "
                                f"{home_lerp_end_pose[:3].round(3).tolist()} "
                                f"over {HOME_DURATION:.1f}s"
                            )
                    else:
                        # First HOME of the session before env has injected
                        # the home target pose. Fall back to obs-following so
                        # the action at least matches the robot's actual motion.
                        home_lerp_use_synth = False
                        if self.verbose:
                            print(
                                "[ViveTeleopProcess] HOME requested — home_target_pose "
                                "not yet set (first HOME?); using obs-follow fallback"
                            )

                prev_trackpad = trackpad_pressed

                # === HOME MODE: Synthesize cartesian trajectory as action ===
                if home_active:
                    elapsed = time.time() - home_start_time
                    s = min(1.0, max(0.0, elapsed / HOME_DURATION))

                    if home_lerp_use_synth:
                        # Position: straight-line lerp
                        target_pose[:3] = (
                            (1.0 - s) * home_lerp_start_pose[:3]
                            + s * home_lerp_end_pose[:3]
                        )
                        # Rotation: slerp on the unit-quaternion form
                        try:
                            r_start = R.from_rotvec(home_lerp_start_pose[3:6])
                            r_end   = R.from_rotvec(home_lerp_end_pose[3:6])
                            slerp = Slerp(
                                [0.0, 1.0],
                                R.concatenate([r_start, r_end]),
                            )
                            r_now = slerp([s])[0]
                            target_pose[3:6] = r_now.as_rotvec()
                        except Exception:
                            # Slerp can fail on near-identical quats (rare);
                            # fall back to rotvec lerp which is acceptable for
                            # small angular gaps.
                            target_pose[3:6] = (
                                (1.0 - s) * home_lerp_start_pose[3:6]
                                + s * home_lerp_end_pose[3:6]
                            )
                    else:
                        # First-HOME fallback (no endpoint cached yet): mirror
                        # the actual robot pose so action ≈ obs for this one
                        # event. Subsequent HOMEs will use the synthesized
                        # path because we cache the endpoint at HOME end below.
                        current_robot_pose = get_current_robot_pose()
                        if current_robot_pose is not None:
                            target_pose = current_robot_pose.copy()

                    # Check if home mode should end
                    if elapsed > HOME_DURATION:
                        home_active = False
                        home_requested = False
                        clutch_active = False
                        prev_target_pos = None
                        prev_target_rot = None
                        require_grip_release = True
                        base_rotation = None
                        base_position = None
                        accumulated_z_rotation = 0.0

                        # Reset GRIPPER toggle state to OPEN to match the
                        # physical state. env.move_home() opens the gripper
                        # as part of HOME, but ViveTeleopProcess's
                        # gripper_closed latch was retaining its pre-HOME
                        # value, causing the next trigger press to fall on
                        # the wrong half of the toggle. Catalog #34.
                        gripper_closed = False
                        gripper_state = 0.0
                        gripper_command = GripperCommand.NONE
                        awaiting_trigger_release = True
                        gripper_target_width = self.gripper_open_width

                        # Final action target = the canonical home pose. This
                        # is what subsequent record_action samples will see
                        # until the user re-engages grip and starts driving
                        # the EE elsewhere.
                        if home_lerp_use_synth:
                            target_pose = home_lerp_end_pose.copy()
                        else:
                            # Fallback path: cache the end-of-HOME pose so
                            # the *next* HOME event has a synthesizable
                            # endpoint, then sync action target there.
                            curr = get_current_robot_pose()
                            if curr is not None:
                                target_pose = curr.copy()
                                self.home_target_pose_array.get()[:] = curr
                        home_lerp_use_synth = False

                        if self.verbose:
                            print(
                                f"[ViveTeleopProcess] HOME ended — action target "
                                f"now at home pose {target_pose[:3].round(3).tolist()}, "
                                f"gripper reset to OPEN. Release grip to continue."
                            )

                    # Skip ALL clutch processing during HOME mode so the user
                    # waving the Vive around mid-HOME doesn't corrupt the
                    # synthesized trajectory.
                    prev_grip = grip_pressed
                    prev_trigger = trigger_pressed

                # === GRIP RELEASE DETECTION ===
                # If require_grip_release is True (after HOME), wait for grip to be released
                if require_grip_release and not grip_pressed:
                    require_grip_release = False
                    if self.verbose:
                        print("[ViveTeleopProcess] Grip released - clutch can now be activated")

                # === TRACKPAD TOUCH: EE Z-axis rotation (Joint 7) ===
                # Conditions: trackpad touched (not pressed) + grip OFF + not in HOME mode
                # This allows rotating the gripper while clutch is released
                rotation_active = False
                if (self.enable_trackpad_rotation and
                    trackpad_touched and not trackpad_pressed and
                    not grip_pressed and not home_active):

                    # Determine rotation direction based on trackpad Y position
                    delta_angle = 0.0
                    if trackpad_y > self.trackpad_rotation_threshold:
                        # Top of trackpad → clockwise rotation (negative angle)
                        delta_angle = -self.rotation_speed
                    elif trackpad_y < -self.trackpad_rotation_threshold:
                        # Bottom of trackpad → counter-clockwise rotation (positive angle)
                        delta_angle = self.rotation_speed

                    if delta_angle != 0.0:
                        rotation_active = True

                        # Get current robot pose for initialization
                        current_robot_pose = get_current_robot_pose()
                        if current_robot_pose is not None:
                            # Extract current EE rotation
                            current_rotvec = current_robot_pose[3:6]
                            current_rot = R.from_rotvec(current_rotvec)

                            # Initialize base_rotation and base_position on first rotation
                            if base_rotation is None:
                                base_rotation = current_rot
                                base_position = current_robot_pose[:3].copy()
                                accumulated_z_rotation = 0.0
                                if self.verbose:
                                    print(f"[ViveTeleopProcess] Rotation started - base set")

                            # Check rotation limits using accumulated angle (not current robot angle)
                            can_rotate = True
                            if delta_angle < 0 and accumulated_z_rotation + delta_angle <= -self.rotation_limit:
                                can_rotate = False
                                if self.verbose:
                                    print(f"[ViveTeleopProcess] Rotation limit reached: {np.rad2deg(accumulated_z_rotation):.1f}° (min: {-np.rad2deg(self.rotation_limit):.1f}°)")
                            elif delta_angle > 0 and accumulated_z_rotation + delta_angle >= self.rotation_limit:
                                can_rotate = False
                                if self.verbose:
                                    print(f"[ViveTeleopProcess] Rotation limit reached: {np.rad2deg(accumulated_z_rotation):.1f}° (max: {np.rad2deg(self.rotation_limit):.1f}°)")

                            if can_rotate:
                                # Accumulate rotation angle
                                accumulated_z_rotation += delta_angle

                                # Compute target rotation from base_rotation + accumulated angle
                                # This ensures the target keeps increasing even if robot hasn't caught up
                                accumulated_rot = R.from_rotvec([0, 0, accumulated_z_rotation])
                                new_rot = base_rotation * accumulated_rot  # Apply accumulated rotation to base
                                new_rotvec = new_rot.as_rotvec()

                                # Update target_pose with new rotation (keep base position for stability)
                                target_pose[:3] = base_position  # Use base position to avoid drift
                                target_pose[3:6] = new_rotvec  # Apply accumulated rotation

                                if self.verbose:
                                    direction = "CW" if delta_angle < 0 else "CCW"
                                    print(f"[ViveTeleopProcess] Rotating {direction}: target={np.rad2deg(accumulated_z_rotation):.1f}°")
                    else:
                        # Trackpad touched but not in rotation zone (center area)
                        # Reset base_rotation so next rotation starts fresh
                        if base_rotation is not None:
                            base_rotation = None
                            base_position = None
                            accumulated_z_rotation = 0.0
                            if self.verbose:
                                print(f"[ViveTeleopProcess] Rotation ended - base reset")
                else:
                    # Trackpad not touched or other conditions not met
                    # Reset base_rotation so next rotation starts fresh
                    if base_rotation is not None:
                        base_rotation = None
                        base_position = None
                        accumulated_z_rotation = 0.0
                        if self.verbose:
                            print(f"[ViveTeleopProcess] Rotation ended - base reset")

                # === GRIP: Clutch control (disabled during home mode) ===
                # Added: require_grip_release must be False to activate clutch
                if grip_pressed and not home_active and not require_grip_release:
                    if not clutch_active:
                        # Grip just pressed - start clutch
                        clutch_active = True

                        # Fetch CURRENT robot pose for proper sync after Home or position changes
                        current_robot_pose = get_current_robot_pose()
                        if current_robot_pose is not None:
                            target_pose = current_robot_pose.copy()
                            clutch_robot_start = current_robot_pose.copy()
                        else:
                            # Fallback to internal target_pose if fetch fails
                            clutch_robot_start = target_pose.copy()

                        clutch_vr_start_pos = vive_pos.copy()
                        clutch_vr_start_quat = vive_quat.copy()
                        prev_target_pos = None
                        prev_target_rot = None
                        prev_vr_pos = vive_pos.copy()
                        prev_vr_time = time.time()

                        if self.verbose:
                            print(f"[ViveTeleopProcess] Clutch ON (start: [{clutch_robot_start[0]:.3f}, {clutch_robot_start[1]:.3f}, {clutch_robot_start[2]:.3f}])")
                    else:
                        # Grip held - compute delta and update target

                        # Outlier detection
                        if not self.validate_position_change(vive_pos, prev_vr_pos, prev_vr_time):
                            prev_vr_time = time.time()
                            # Skip this frame
                            continue

                        prev_vr_pos = vive_pos.copy()
                        prev_vr_time = time.time()

                        # Compute local delta
                        local_pos, local_rot = self.compute_local_delta(
                            vive_pos, vive_quat,
                            clutch_vr_start_pos, clutch_vr_start_quat
                        )

                        # Convert to robot frame
                        robot_pos, robot_rot = self.local_to_robot(local_pos, local_rot)
                        robot_pos *= self.pos_scale

                        # Desired target position
                        desired_pos = clutch_robot_start[:3] + robot_pos

                        # Position velocity clamping
                        if self.use_velocity_clamping:
                            target_pose[:3] = self.clamp_velocity(
                                desired_pos, prev_target_pos,
                                self.max_pos_velocity, dt
                            )
                        else:
                            target_pose[:3] = desired_pos

                        prev_target_pos = target_pose[:3].copy()

                        # Rotation
                        if robot_rot is not None:
                            rotvec = robot_rot.as_rotvec() * self.rot_scale
                            robot_rot_scaled = R.from_rotvec(rotvec)
                            robot_r_start = R.from_rotvec(clutch_robot_start[3:])
                            robot_r_desired = robot_rot_scaled * robot_r_start
                            desired_rotvec = robot_r_desired.as_rotvec()

                            # Rotation velocity clamping
                            if self.use_velocity_clamping and prev_target_rot is not None:
                                prev_r = R.from_rotvec(prev_target_rot)
                                r_diff = robot_r_desired * prev_r.inv()
                                diff_rotvec = r_diff.as_rotvec()
                                diff_mag = np.linalg.norm(diff_rotvec)
                                max_rot = self.max_rot_velocity * dt

                                if diff_mag > max_rot:
                                    clamped = diff_rotvec / diff_mag * max_rot
                                    clamped_r = R.from_rotvec(clamped)
                                    robot_r_target = clamped_r * prev_r
                                    target_pose[3:] = robot_r_target.as_rotvec()
                                else:
                                    target_pose[3:] = desired_rotvec
                            else:
                                target_pose[3:] = desired_rotvec

                            prev_target_rot = target_pose[3:].copy()

                elif not grip_pressed:
                    if clutch_active:
                        # Grip released - stop clutch
                        clutch_active = False
                        prev_target_pos = None
                        prev_target_rot = None

                        if self.verbose:
                            print(f"[ViveTeleopProcess] Clutch OFF - keeping last target_pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")

                    # When clutch is NOT pressed:
                    # - KEEP the last target_pose (from clutch release or initial pose)
                    # - This is the user's intended action before releasing clutch
                    # - Robot controller holds current position independently (via clutch_active=0)
                    # - Action recording will use this preserved target_pose
                    #
                    # NOTE: We do NOT update target_pose here anymore.
                    # The robot maintains position via impedance control, not from target_pose.
                    # This ensures recorded action = user's intended teleop target (not robot drift)

                # === TRIGGER: Gripper control ===
                # 1. Toggle gripper state when trigger is fully pressed (edge detection)
                # 2. Compute continuous gripper_target_width based on trigger_value
                #
                # Continuous width logic (for action recording):
                #   - OPEN state (gripper_closed=False):
                #       trigger 0→1 = action 0.075→0.00 (closing direction)
                #   - CLOSE state (gripper_closed=True):
                #       trigger 0→1 = action 0.00→0.075 (opening direction)
                #
                # Continuity through toggle:
                #   - After toggle, hold at target value until trigger is released to 0
                #   - This prevents discontinuity (e.g., 0.008 jumping to 0.08)
                #   - Only when trigger is fully released, start responding to new presses
                #
                # Example flow (OPEN → CLOSE):
                #   trigger=0.0 → action=0.075 (open)
                #   trigger=0.5 → action=0.0375 (halfway)
                #   trigger=1.0 → action=0.00 (closed) → TOGGLE!
                #   [trigger still held] → action=0.00 (hold)
                #   trigger released to 0.0 → action=0.00 (ready for next)
                #   trigger=0.5 → action=0.0375 (now opening direction)
                #   trigger=1.0 → action=0.075 (open) → TOGGLE!

                # Toggle on edge (trigger fully pressed)
                if trigger_pressed and not prev_trigger:
                    gripper_closed = not gripper_closed
                    awaiting_trigger_release = True  # Wait for trigger release before new input
                    if gripper_closed:
                        gripper_state = 1.0
                        gripper_command = GripperCommand.CLOSE
                        if self.verbose:
                            print("[ViveTeleopProcess] Gripper CLOSE")
                    else:
                        gripper_state = 0.0
                        gripper_command = GripperCommand.OPEN
                        if self.verbose:
                            print("[ViveTeleopProcess] Gripper OPEN")

                # Detect trigger release after toggle
                TRIGGER_RELEASE_THRESHOLD = 0.05  # Consider released when < 5%
                if awaiting_trigger_release and trigger_value < TRIGGER_RELEASE_THRESHOLD:
                    awaiting_trigger_release = False

                # Compute continuous gripper target width based on trigger_value
                width_range = self.gripper_open_width - self.gripper_close_width

                if awaiting_trigger_release:
                    # After toggle, hold at target value until trigger is fully released
                    # This prevents discontinuity at toggle moment
                    if gripper_closed:
                        gripper_target_width = self.gripper_close_width  # 0.00
                    else:
                        gripper_target_width = self.gripper_open_width   # 0.075
                else:
                    # Normal operation: trigger_value controls continuous width
                    if gripper_closed:
                        # CLOSE state (base=0.00): trigger press → open direction
                        # trigger_value 0→1 = width 0.00→0.075
                        gripper_target_width = self.gripper_close_width + width_range * trigger_value
                    else:
                        # OPEN state (base=0.075): trigger press → close direction
                        # trigger_value 0→1 = width 0.075→0.00
                        gripper_target_width = self.gripper_open_width - width_range * trigger_value

                prev_grip = grip_pressed
                prev_trigger = trigger_pressed

                # === Output to ring buffers ===
                t_recv = time.time()
                # UMI convention: timestamp is when action will be executed (future time)
                # In teleop mode, action is executed immediately, so we use current time
                # But for compatibility with UMI training pipeline, we add a small offset
                t_command_target = t_recv + dt  # Next cycle execution time (UMI style)

                # Keep home_requested True while home_active is True
                # This ensures main loop (10Hz) can detect it even though teleop runs at 100Hz
                home_requested_output = home_requested or home_active

                # Action buffer (for Main Process recording - UMI style)
                action_state = {
                    'target_pose': target_pose.copy(),
                    'gripper_state': np.float64(gripper_state),  # Discrete toggle state (0/1)
                    'gripper_target_width': np.float64(gripper_target_width),  # Continuous width
                    'clutch_active': np.uint8(1 if clutch_active else 0),
                    'home_requested': np.uint8(1 if home_requested_output else 0),
                    'home_active': np.uint8(1 if home_active else 0),
                    'rotation_active': np.uint8(1 if rotation_active else 0),
                    'timestamp': t_command_target  # UMI style: future execution time
                }
                self.action_ring_buffer.put(action_state)

                # Robot buffer (for FrankaInterpolationController)
                # Note: clutch_active and rotation_active are separate flags
                # - clutch_active: grip button held (normal teleop with offset compensation)
                # - rotation_active: trackpad rotation (direct target following, no offset)
                # FrankaInterpolationController handles these separately
                robot_state = {
                    'target_pose': target_pose.copy(),
                    'clutch_active': np.uint8(1 if clutch_active else 0),
                    'rotation_active': np.uint8(1 if rotation_active else 0),
                    'teleop_timestamp': t_recv
                }
                self.robot_ring_buffer.put(robot_state)

                # Gripper buffer (for FrankaGripperController)
                gripper_rb_state = {
                    'gripper_state': np.float64(gripper_state),
                    'gripper_command': np.int32(gripper_command.value),
                    'teleop_timestamp': t_recv
                }
                self.gripper_ring_buffer.put(gripper_rb_state)

                # === Regulate frequency ===
                iter_idx += 1
                t_wait_until = t_start + iter_idx * dt
                t_sleep = t_wait_until - time.monotonic()
                if t_sleep > 0:
                    time.sleep(t_sleep)

        except Exception as e:
            print(f"[ViveTeleopProcess] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close robot client connection
            try:
                robot_client.close()
            except:
                pass
            self.ready_event.set()
            if self.verbose:
                print("[ViveTeleopProcess] Process terminated")
