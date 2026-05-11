"""FrankaGripperController for the Franka Panda default gripper (Franka Hand).

Talks to fairo-polymetis' GripperServerLauncher over gRPC :50052 (NOT zerorpc;
see /home/kist/fairo/polymetis/polymetis/conf/launch_gripper.yaml — port: 50052
and gripper_interface.py which uses ``grpc.insecure_channel``).

Key features:
- Runs in separate process to avoid Python GIL.
- DISCRETE OPEN/CLOSE commands (matches recorded action semantics — see
  ``ArtGripperController`` for the matching pattern).
- Same UMI ``schedule_waypoint(width, target_time)`` API as ArtGripperController.
- Non-blocking ``goto`` / ``grasp`` (``blocking=False``) so the controller loop
  does not stall during gripper motion (catalog #30 pattern from ART).

Wire-up (NUC):
    sudo bash /usr/local/sbin/start_franka_gripper.sh
    -> python launch_gripper.py gripper=franka_hand
    -> GripperServerLauncher on :50052 (gRPC)

Gripper specifications (Franka Hand):
- Width range: 0.0 to 0.08 meters (80 mm) physical max; usable 0.005-0.075 m
  (libfranka raises on width=0.0 → 5 mm safety margin on close_width).
- Max speed: ~0.2 m/s.
- Grasp force: 0.01 to 70 N.

Backend metadata (port, width envelope, default force, TCP offset) lives in
``polymetis_franka_teleop.common.gripper_specs.GRIPPER_SPECS['franka']``;
this controller honours whatever is passed in.
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
    Controller for the Franka Panda default gripper via fairo-polymetis gRPC :50052.

    Discrete OPEN/CLOSE state machine matching recorded action semantics,
    non-blocking command issue, teleop-mode hot path that reads directly
    from ViveTeleopProcess's ring buffer.

    Args:
        shm_manager: SharedMemoryManager for inter-process communication
        robot_ip: IP of the polymetis Franka Hand gRPC server (NUC, default 192.168.1.12)
        gripper_port: gRPC port (default 50052, matches conf/launch_gripper.yaml)
        frequency: Control loop frequency in Hz (default 60)
        move_max_speed: Default gripper speed in m/s
        get_max_k: Max ring-buffer history (default frequency * 10)
        launch_timeout: Spawn timeout (s)
        receive_latency: Subtracted from state timestamps (latency compensation)
        verbose: Verbose logging
        teleop_mode: Read commands from teleop_ring_buffer (bypass input queue)
        teleop_ring_buffer: required iff teleop_mode=True
        gripper_open_width / gripper_close_width: Backend-specific width envelope
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
            gripper_port: int = 50052,         # fairo-polymetis launch_gripper.yaml default
            frequency: int = 60,
            move_max_speed: float = 0.2,
            get_max_k: int = None,
            command_queue_size: int = 1024,
            launch_timeout: float = 3.0,
            receive_latency: float = 0.0,
            verbose: bool = False,
            teleop_mode: bool = False,
            teleop_ring_buffer: SharedMemoryRingBuffer = None,
            gripper_open_width: float = 0.075,  # Franka Hand obs max
            gripper_close_width: float = 0.005, # 5 mm safety (libfranka raises on 0.0)
            ):
        # Validate teleop mode configuration
        if teleop_mode and teleop_ring_buffer is None:
            raise ValueError("teleop_ring_buffer is required when teleop_mode=True")

        super().__init__(name="FrankaGripperController")
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
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
            # fairo-polymetis Franka Hand uses gRPC :50052. The
            # ``GripperInterface`` thin wrapper handles connect + protobuf.
            from polymetis import GripperInterface

            gripper = GripperInterface(
                ip_address=self.robot_ip, port=self.gripper_port,
            )

            if self.verbose:
                print(f"[FrankaGripperController] Connected to polymetis Franka Hand "
                      f"at {self.robot_ip}:{self.gripper_port} (gRPC)")
                print(f"[FrankaGripperController] Teleop mode: {self.teleop_mode}")

            # GripperInterface state has protobuf accessors (.width, .is_moving,
            # .is_grasped). Wrap into uniform helpers so the rest of the loop
            # treats them the same regardless of backend.
            def get_state_fn():
                st = gripper.get_state()
                return {
                    'width': float(st.width),
                    'is_moving': bool(st.is_moving),
                    'is_grasped': bool(st.is_grasped),
                }

            def goto_fn(width: float, speed: float, force: float):
                gripper.goto(width=float(width), speed=float(speed),
                             force=float(force), blocking=False)

            def grasp_fn(speed: float, force: float, grasp_width: float):
                gripper.grasp(speed=float(speed), force=float(force),
                              grasp_width=float(grasp_width), blocking=False)

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
            overrun_count = 0  # iters where precise_wait was already past t_end

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
                if time.monotonic() > t_end:
                    overrun_count += 1
                precise_wait(t_end=t_end, time_func=time.monotonic)
                if self.verbose and iter_idx % (self.frequency * 5) == 0:
                    # every ~5 s, report loop health
                    print(f"[FrankaGripperController] iter={iter_idx} target={self.frequency}Hz "
                          f"overruns={overrun_count}/{self.frequency * 5} (last 5s)")
                    overrun_count = 0

        except ImportError as e:
            print(f"[FrankaGripperController] Failed to import polymetis: {e}")
            print("[FrankaGripperController] Install fairo-polymetis client "
                  "(should already be present in groot-client env).")
            self.ready_event.set()
        except Exception as e:
            print(f"[FrankaGripperController] Error: {e}")
            import traceback
            traceback.print_exc()
            self.ready_event.set()
        finally:
            self.ready_event.set()
            # GripperInterface holds a grpc.Channel; closing is best-effort.
            try:
                gripper.channel.close()
            except Exception:
                pass
            if self.verbose:
                print(f"[FrankaGripperController] Disconnected from polymetis Franka Hand")
