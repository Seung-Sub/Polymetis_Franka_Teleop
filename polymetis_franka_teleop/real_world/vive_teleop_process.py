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
from scipy.spatial.transform import Rotation as R

from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


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
        self.verbose = verbose

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
            home_start_time = 0.0
            HOME_DURATION = 3.0  # seconds to stay in home mode

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
                    if self.verbose:
                        print("[ViveTeleopProcess] HOME requested via trackpad")

                prev_trackpad = trackpad_pressed

                # === HOME MODE: Output actual robot pose as action ===
                if home_active:
                    # Always fetch and use actual robot pose during HOME
                    current_robot_pose = get_current_robot_pose()
                    if current_robot_pose is not None:
                        target_pose = current_robot_pose.copy()

                    # Check if home mode should end
                    if (time.time() - home_start_time > HOME_DURATION):
                        home_active = False
                        home_requested = False

                        # CRITICAL: Force clutch re-sync when HOME ends
                        # This ensures target_pose starts fresh from current robot pose
                        clutch_active = False
                        prev_target_pos = None
                        prev_target_rot = None

                        # CRITICAL: Require grip release before clutch can be activated
                        # This ensures FrankaInterpolationController sees proper 0->1 transition
                        require_grip_release = True

                        # Reset rotation state after HOME (robot position changed)
                        base_rotation = None
                        base_position = None
                        accumulated_z_rotation = 0.0

                        # Reset GRIPPER toggle state to OPEN to match the
                        # physical state. env.move_home() opens the gripper as
                        # part of HOME, but ViveTeleopProcess's gripper_closed
                        # latch was retaining its pre-HOME value, causing the
                        # next trigger press to fall on the wrong half of the
                        # toggle (had to press TWICE to actually close).
                        gripper_closed = False
                        gripper_state = 0.0
                        gripper_command = GripperCommand.NONE
                        awaiting_trigger_release = True   # ignore stale trigger state
                        gripper_target_width = self.gripper_open_width

                        # Sync target_pose to current robot pose
                        if current_robot_pose is not None:
                            target_pose = current_robot_pose.copy()

                        if self.verbose:
                            print(f"[ViveTeleopProcess] HOME mode ended (synced to: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}], "
                                  f"gripper reset to OPEN) - Release grip to continue")

                    # Skip ALL clutch processing during HOME mode
                    # (prevents Vive movement from affecting target_pose)
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
