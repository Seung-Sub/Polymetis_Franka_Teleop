"""
FrankaGripperController for Franka Panda default gripper.
Based on UMI's WSGController structure, using ZeroRPC to communicate with gripper server.

Key features:
- Runs in separate process to avoid Python GIL
- SIMPLIFIED: No interpolation - direct OPEN/CLOSE commands (matches data collection)
- Compatible with UMI's schedule_waypoint API
- SharedMemory-based inter-process communication
- Teleop mode for direct gripper control from ViveTeleopProcess

IMPORTANT DESIGN DECISION (v2.0):
- Data collection records binary gripper states (OPEN or CLOSE)
- Policy outputs are also essentially binary
- Therefore, interpolation is UNNECESSARY and adds latency
- This version detects state transitions and sends single commands

Requirements:
- ZeroRPC interface server running on NUC
- zerorpc Python package installed

Gripper specifications (Franka Hand):
- Width range: 0.0 to 0.08 meters (0 to 80mm) physical max
- Usable range for training: 0.0 to 0.075 meters (0 to 75mm) to match observation
- Max speed: ~0.2 m/s
- Grasp force: 0.01 to 70 N (recommended: 5-40 N)

Usage:
    with SharedMemoryManager() as shm_manager:
        with FrankaGripperController(
            shm_manager=shm_manager,
            robot_ip='172.16.0.2'
        ) as gripper_controller:
            # Get gripper state
            state = gripper_controller.get_state()
            # Schedule command (only executes on state change)
            gripper_controller.schedule_waypoint(width=0.04, target_time=time.time()+1)
"""

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from polymetis_franka_teleop.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from polymetis_franka_teleop.common.precise_sleep import precise_wait


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2
    GOTO = 3
    GRASP = 4


class FrankaGripperController(mp.Process):
    """
    Controller for Franka Panda default gripper using ZeroRPC.

    This controller runs in a separate process and provides DISCRETE
    command execution (OPEN or CLOSE) matching data collection behavior.

    v2.0 Changes:
    - Removed PoseTrajectoryInterpolator (unnecessary for binary gripper)
    - Commands are queued with timestamps and executed at scheduled time
    - Single OPEN or CLOSE command per state transition
    - Reduced latency by removing interpolation overhead

    Teleop Mode:
    When teleop_mode=True, the controller reads gripper commands directly
    from teleop_ring_buffer (from ViveTeleopProcess) for immediate response.

    Args:
        shm_manager: SharedMemoryManager for inter-process communication
        robot_ip: IP address of the ZeroRPC server (same as robot NUC)
        gripper_port: Port of the ZeroRPC gripper server (default: 4242)
        frequency: Control frequency in Hz (default: 30)
        move_max_speed: Maximum gripper speed in m/s (default: 0.2)
        get_max_k: Maximum number of states to buffer (default: frequency * 10)
        launch_timeout: Timeout for controller startup (default: 3s)
        receive_latency: Latency compensation for state timestamps (default: 0.0)
        verbose: Enable verbose logging (default: False)
        teleop_mode: Enable teleop mode (bypass schedule_waypoint) (default: False)
        teleop_ring_buffer: SharedMemoryRingBuffer from ViveTeleopProcess (required if teleop_mode=True)
    """

    # Franka Hand specifications
    MAX_WIDTH = 0.08  # meters (80mm)
    MIN_WIDTH = 0.0   # meters
    DEFAULT_SPEED = 0.2   # m/s (maximum speed)
    DEFAULT_FORCE = 50.0  # N (strong grip for holding objects)

    # Gripper command enum values (must match ViveTeleopProcess.GripperCommand)
    GRIPPER_CMD_NONE = 0
    GRIPPER_CMD_OPEN = 1
    GRIPPER_CMD_CLOSE = 2

    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip: str,
            gripper_port: int = 4242,
            frequency: int = 30,
            move_max_speed: float = 0.2,
            get_max_k: int = None,
            command_queue_size: int = 1024,
            launch_timeout: float = 3.0,
            receive_latency: float = 0.0,
            use_unified_server: bool = True,
            verbose: bool = False,
            teleop_mode: bool = False,
            teleop_ring_buffer: SharedMemoryRingBuffer = None,
            gripper_open_width: float = 0.075,  # Gripper width when open (default: 75mm, matches obs max)
            gripper_close_width: float = 0.005,   # Gripper width when closed (default: 5mm, avoid 0 which causes libfranka exception)
            ):
        # Validate teleop mode configuration
        if teleop_mode and teleop_ring_buffer is None:
            raise ValueError("teleop_ring_buffer is required when teleop_mode=True")

        super().__init__(name="FrankaGripperController")
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
        self.use_unified_server = use_unified_server
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.teleop_mode = teleop_mode
        self.teleop_ring_buffer = teleop_ring_buffer
        self.gripper_open_width = gripper_open_width
        self.gripper_close_width = gripper_close_width

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # Build input queue for commands
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,      # Target width in meters
            'target_time': 0.0,     # Target time (time.time())
            'speed': self.DEFAULT_SPEED,
            'force': self.DEFAULT_FORCE,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # Build ring buffer for gripper state
        example = {
            'gripper_width': 0.0,           # Current width in meters
            'gripper_is_grasped': False,    # Whether object is grasped
            'gripper_is_moving': False,     # Whether gripper is moving
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
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

    # ========= Launch methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaGripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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

    # ========= Context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= Command methods ============
    def schedule_waypoint(self, width: float, target_time: float):
        """
        Schedule gripper command to reach target width at target_time.

        Unlike robot arm, gripper uses DISCRETE commands:
        - If width > threshold: schedule OPEN
        - If width <= threshold: schedule CLOSE

        Args:
            width: Target gripper width in meters (0.0 to 0.08)
            target_time: Absolute time (time.time()) to execute the command
        """
        # Clamp width to valid range
        width = np.clip(width, self.MIN_WIDTH, self.MAX_WIDTH)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': width,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def goto(self, width: float, speed: float = None):
        """
        Move gripper to target width immediately.

        Args:
            width: Target gripper width in meters
            speed: Movement speed in m/s (default: 0.2)
        """
        if speed is None:
            speed = self.DEFAULT_SPEED

        width = np.clip(width, self.MIN_WIDTH, self.MAX_WIDTH)

        message = {
            'cmd': Command.GOTO.value,
            'target_pos': width,
            'speed': speed
        }
        self.input_queue.put(message)

    def grasp(self, speed: float = None, force: float = None):
        """
        Close gripper and grasp object.

        Args:
            speed: Grasping speed in m/s (default: 0.2)
            force: Grasping force in N (default: 50.0)
        """
        if speed is None:
            speed = self.DEFAULT_SPEED
        if force is None:
            force = self.DEFAULT_FORCE

        message = {
            'cmd': Command.GRASP.value,
            'speed': speed,
            'force': force
        }
        self.input_queue.put(message)

    def restart_put(self, start_time: float):
        """Restart state buffer from new start time."""
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= Receive APIs =============
    def get_state(self, k=None, out=None):
        """Get current gripper state."""
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        """Get all buffered gripper states."""
        return self.ring_buffer.get_all()

    # ========= Main loop in process ============
    def run(self):
        try:
            # Use ZeroRPC to communicate with gripper server on NUC
            import zerorpc

            # Connect to gripper ZeroRPC server
            gripper = zerorpc.Client(heartbeat=20)
            gripper.connect(f"tcp://{self.robot_ip}:{self.gripper_port}")

            if self.verbose:
                print(f"[FrankaGripperController] Connected to gripper at {self.robot_ip}:{self.gripper_port}")
                print(f"[FrankaGripperController] Teleop mode: {self.teleop_mode}")
                print(f"[FrankaGripperController] v2.0: Simplified discrete mode (no interpolation)")

            # Define method names based on server type
            if self.use_unified_server:
                get_state_fn = gripper.get_gripper_state
                goto_fn = gripper.gripper_goto
                grasp_fn = gripper.gripper_grasp
            else:
                get_state_fn = gripper.get_state
                goto_fn = gripper.goto
                grasp_fn = gripper.grasp

            # Get initial state
            gripper_state = get_state_fn()
            curr_width = gripper_state['width']

            # Threshold for OPEN/CLOSE decision
            width_threshold = (self.gripper_open_width + self.gripper_close_width) / 2

            # Track current state (True = OPEN, False = CLOSE)
            current_is_open = curr_width > width_threshold

            # Pending commands queue: [(target_is_open, target_time_monotonic), ...]
            pending_commands = []

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0

            # Teleop mode state
            last_teleop_timestamp = 0.0
            last_teleop_command = self.GRIPPER_CMD_NONE

            while keep_running:
                t_now = time.monotonic()
                dt = 1 / self.frequency

                # === TELEOP MODE: Read from teleop_ring_buffer ===
                if self.teleop_mode:
                    try:
                        teleop_state = self.teleop_ring_buffer.get()
                        teleop_timestamp = teleop_state.get('teleop_timestamp', 0.0)
                        gripper_command = int(teleop_state.get('gripper_command', self.GRIPPER_CMD_NONE))
                        # ViveTeleopProcess outputs gripper_state (0.0=open, 1.0=closed)
                        gripper_state_val = float(teleop_state.get('gripper_state', 0.0))
                        # Map discrete state to actual width
                        target_width = self.gripper_close_width if gripper_state_val > 0.5 else self.gripper_open_width

                        # Only process new commands
                        if teleop_timestamp > last_teleop_timestamp:
                            last_teleop_timestamp = teleop_timestamp

                            # Execute gripper command if it changed
                            if gripper_command != self.GRIPPER_CMD_NONE and gripper_command != last_teleop_command:
                                try:
                                    if gripper_command == self.GRIPPER_CMD_CLOSE:
                                        grasp_fn(self.move_max_speed, self.DEFAULT_FORCE, target_width)
                                        current_is_open = False
                                        if self.verbose:
                                            print(f"[FrankaGripperController] TELEOP CLOSE to {target_width:.3f}m")
                                    elif gripper_command == self.GRIPPER_CMD_OPEN:
                                        goto_fn(target_width, self.move_max_speed, self.DEFAULT_FORCE)
                                        current_is_open = True
                                        if self.verbose:
                                            print(f"[FrankaGripperController] TELEOP OPEN to {target_width:.3f}m")
                                    last_teleop_command = gripper_command
                                except Exception as e:
                                    if self.verbose:
                                        print(f"[FrankaGripperController] Teleop gripper error: {e}")

                    except Exception as e:
                        if self.verbose and iter_idx % 100 == 0:
                            print(f"[FrankaGripperController] Teleop read error: {e}")

                else:
                    # === NORMAL MODE: Process pending commands ===
                    # First, read current gripper state BEFORE executing commands
                    try:
                        pre_state = get_state_fn()
                        pre_is_moving = pre_state['is_moving']
                        pre_is_grasped = pre_state['is_grasped']
                        pre_width = pre_state['width']
                    except Exception as e:
                        pre_is_moving = False
                        pre_is_grasped = False
                        pre_width = self.gripper_open_width if current_is_open else self.gripper_close_width

                    # Execute commands whose time has arrived
                    commands_to_remove = []
                    for i, (target_is_open, target_time_mono) in enumerate(pending_commands):
                        if t_now >= target_time_mono:
                            # Skip if gripper is currently moving (wait for it to finish)
                            if pre_is_moving:
                                if self.verbose:
                                    print(f"[FrankaGripperController] Gripper moving, deferring command")
                                continue  # Don't remove, try again next loop

                            # Time to execute this command
                            if target_is_open != current_is_open:
                                try:
                                    if target_is_open:
                                        # OPEN command
                                        goto_fn(self.gripper_open_width, self.move_max_speed, self.DEFAULT_FORCE)
                                        if self.verbose:
                                            print(f"[FrankaGripperController] OPEN to {self.gripper_open_width:.3f}m")
                                        current_is_open = True
                                    else:
                                        # CLOSE command (grasp with force)
                                        # Skip if already grasped to prevent duplicate grasp errors
                                        if pre_is_grasped:
                                            if self.verbose:
                                                print(f"[FrankaGripperController] Already grasped, skipping CLOSE")
                                            current_is_open = False  # Update state anyway
                                        else:
                                            grasp_fn(self.move_max_speed, self.DEFAULT_FORCE, self.gripper_close_width)
                                            if self.verbose:
                                                print(f"[FrankaGripperController] CLOSE (grasp) to {self.gripper_close_width:.3f}m")
                                            current_is_open = False
                                except Exception as e:
                                    if self.verbose:
                                        print(f"[FrankaGripperController] Command error: {e}")
                                        import traceback
                                        traceback.print_exc()
                            commands_to_remove.append(i)

                    # Remove executed commands (in reverse order to maintain indices)
                    for i in reversed(commands_to_remove):
                        pending_commands.pop(i)

                # Get current gripper state and sync current_is_open with actual state
                try:
                    gripper_state = get_state_fn()
                    current_width = gripper_state['width']
                    is_grasped = gripper_state['is_grasped']
                    is_moving = gripper_state['is_moving']

                    # Sync current_is_open with actual gripper state
                    # This prevents sending duplicate commands to already open/closed gripper
                    actual_is_open = current_width > width_threshold
                    if actual_is_open != current_is_open and not is_moving:
                        if self.verbose:
                            print(f"[FrankaGripperController] State sync: {current_is_open} -> {actual_is_open} (width={current_width:.3f})")
                        current_is_open = actual_is_open
                except Exception as e:
                    if self.verbose:
                        print(f"[FrankaGripperController] State read error: {e}")
                    current_width = self.gripper_open_width if current_is_open else self.gripper_close_width
                    is_grasped = False
                    is_moving = False

                # Store state in ring buffer
                t_recv = time.time()
                state = {
                    'gripper_width': current_width,
                    'gripper_is_grasped': is_grasped,
                    'gripper_is_moving': is_moving,
                    'gripper_receive_timestamp': t_recv,
                    'gripper_timestamp': t_recv - self.receive_latency
                }
                self.ring_buffer.put(state)

                # Fetch commands from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Process queued commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # Translate global time to monotonic time
                        target_time_mono = time.monotonic() - time.time() + target_time

                        # Determine target state (OPEN or CLOSE)
                        target_is_open = target_pos > width_threshold

                        if self.verbose:
                            state_str = "OPEN" if target_is_open else "CLOSE"
                            time_delta = target_time_mono - t_now
                            print(f"[FrankaGripperController] Scheduled {state_str} in {time_delta:.3f}s")

                        # Add to pending commands
                        pending_commands.append((target_is_open, target_time_mono))

                        # Keep commands sorted by time
                        pending_commands.sort(key=lambda x: x[1])

                    elif cmd == Command.GOTO.value:
                        target_width = command['target_pos']
                        speed = command.get('speed', self.DEFAULT_SPEED)
                        try:
                            goto_fn(target_width, speed, self.DEFAULT_FORCE)
                            current_is_open = target_width > width_threshold
                            if self.verbose:
                                print(f"[FrankaGripperController] Immediate GOTO {target_width:.3f}m")
                        except Exception as e:
                            if self.verbose:
                                print(f"[FrankaGripperController] Goto error: {e}")
                    elif cmd == Command.GRASP.value:
                        speed = command.get('speed', self.DEFAULT_SPEED)
                        force = command.get('force', self.DEFAULT_FORCE)
                        try:
                            grasp_fn(speed, force, self.gripper_close_width)
                            current_is_open = False
                            if self.verbose:
                                print(f"[FrankaGripperController] Immediate GRASP")
                        except Exception as e:
                            if self.verbose:
                                print(f"[FrankaGripperController] Grasp error: {e}")
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                        # Clear pending commands on restart
                        pending_commands.clear()
                    else:
                        keep_running = False
                        break

                # Signal ready after first successful loop
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # Regulate frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        except ImportError as e:
            print(f"[FrankaGripperController] Failed to import zerorpc: {e}")
            print("[FrankaGripperController] Install with: pip install zerorpc")
            self.ready_event.set()
        except Exception as e:
            print(f"[FrankaGripperController] Error: {e}")
            import traceback
            traceback.print_exc()
            self.ready_event.set()
        finally:
            self.ready_event.set()
            try:
                gripper.close()
            except:
                pass
            if self.verbose:
                print(f"[FrankaGripperController] Disconnected from gripper")
