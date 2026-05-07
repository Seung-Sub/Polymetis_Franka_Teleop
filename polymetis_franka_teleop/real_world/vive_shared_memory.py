"""
ViveSharedMemory - HTC Vive Controller input via SharedMemory.

Based on UMI's SpacemouseSharedMemory pattern, adapted for HTC Vive controller.
Communicates with vive_input TCP server (port 12345) and provides pose/button state
via SharedMemoryRingBuffer for lock-free inter-process communication.

Features:
- mp.Process based async operation
- SharedMemoryRingBuffer for state sharing (lock-free FILO)
- Position (3D) + Quaternion (4D) pose tracking
- Button states (grip, trigger, trackpad, menu)
- Analog values (trigger_value, trackpad_x/y)
- Haptic feedback support via UDP (port 12346)
- Coordinate transformation for robot frame

Usage:
    with SharedMemoryManager() as shm_manager:
        shm_manager.start()
        with ViveSharedMemory(shm_manager=shm_manager) as vive:
            state = vive.get_motion_state_transformed()
            if vive.is_button_pressed('grip'):
                # do something
"""

import multiprocessing as mp
import numpy as np
import time
import socket
import json
from multiprocessing.managers import SharedMemoryManager
from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class ViveSharedMemory(mp.Process):
    """
    HTC Vive Controller input handler running in separate process.

    Connects to vive_input TCP server and publishes controller state
    via SharedMemoryRingBuffer.

    Args:
        shm_manager: SharedMemoryManager for shared memory allocation
        host: IP address of vive_input server (default: '127.0.0.1')
        port: TCP port of vive_input server (default: 12345)
        get_max_k: Maximum number of states to store in ring buffer
        frequency: Target polling frequency in Hz
        dtype: Data type for floating point values
        enable_haptic: Enable haptic feedback via UDP
        haptic_port: UDP port for haptic feedback
        verbose: Enable verbose logging
    """

    # Default TCP/UDP ports for vive_input
    DEFAULT_PORT = 12345
    HAPTIC_PORT = 12346

    def __init__(self,
            shm_manager: SharedMemoryManager,
            host: str = '127.0.0.1',
            port: int = 12345,
            get_max_k: int = 30,
            frequency: int = 200,
            dtype=np.float32,
            enable_haptic: bool = True,
            haptic_port: int = 12346,
            verbose: bool = False
        ):
        super().__init__(name="ViveSharedMemory")

        self.host = host
        self.port = port
        self.frequency = frequency
        self.dtype = dtype
        self.enable_haptic = enable_haptic
        self.haptic_port = haptic_port
        self.verbose = verbose

        # Coordinate transformation matrix
        # VR World (SteamVR) to Robot base frame
        # This transforms Vive controller motion to robot-centric motion
        # Calibrated mapping:
        #   Pull toward user (Vive Y+) -> Robot X+ (forward)
        #   Move right (Vive X+) -> Robot Y+ (right)
        #   Move up (Vive Z-) -> Robot Z+ (up)
        self.tx_robot_vive = np.array([
            [0, 1, 0],   # Robot X = Vive Y (pull = forward)
            [1, 0, 0],   # Robot Y = Vive X (right = right)
            [0, 0, -1]   # Robot Z = -Vive Z (up = up)
        ], dtype=dtype)

        # Create ring buffer with example data structure
        example = {
            # Pose: position (3) + quaternion (4) = 7
            'position': np.zeros((3,), dtype=dtype),
            'quaternion': np.array([0., 0., 0., 1.], dtype=dtype),  # [x,y,z,w]

            # Button states (boolean as uint8)
            'grip': np.uint8(0),
            'trigger': np.uint8(0),
            'trackpad': np.uint8(0),
            'menu': np.uint8(0),

            # Analog values
            'trigger_value': np.float32(0.0),
            'trackpad_x': np.float32(0.0),
            'trackpad_y': np.float32(0.0),

            # Controller info
            'controller_role': np.int32(0),  # 0=right, 1=left
            'connected': np.uint8(0),

            # Timestamp
            'receive_timestamp': time.time()
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # Shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

        # Haptic socket (created in child process)
        self.haptic_sock = None

    # ======= Get State APIs ==========

    def get_state(self, k=None, out=None):
        """Get raw state from ring buffer."""
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        """Get all buffered states."""
        return self.ring_buffer.get_all()

    def get_motion_state(self):
        """
        Get raw motion state as 6D vector [dx, dy, dz, drx, dry, drz].

        For Vive controller, we compute delta from initial pose.
        Unlike SpaceMouse which reports velocities, Vive reports absolute pose.

        Returns:
            6D numpy array: [x, y, z, rx, ry, rz] in Vive frame
        """
        state = self.ring_buffer.get()

        # Position
        pos = np.array(state['position'], dtype=self.dtype)

        # Quaternion to rotation vector
        quat = state['quaternion']  # [x, y, z, w]
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(quat)  # scipy uses [x,y,z,w]
        rotvec = rot.as_rotvec().astype(self.dtype)

        # Combine into 6D state
        motion_state = np.concatenate([pos, rotvec])
        return motion_state

    def get_motion_state_transformed(self):
        """
        Get motion state transformed to robot coordinate frame.

        Applies coordinate transformation:
        - Vive Y+ (pull) -> Robot X+ (forward)
        - Vive X+ (right) -> Robot Y+ (right)
        - Vive Z- (up) -> Robot Z+ (up)

        Returns:
            6D numpy array: [x, y, z, rx, ry, rz] in robot frame
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_robot_vive @ state[:3]
        tf_state[3:] = self.tx_robot_vive @ state[3:]
        return tf_state

    def get_pose(self):
        """
        Get current pose as (position, quaternion).

        Returns:
            position: (3,) array [x, y, z]
            quaternion: (4,) array [x, y, z, w]
        """
        state = self.ring_buffer.get()
        return state['position'].copy(), state['quaternion'].copy()

    def get_pose_transformed(self):
        """
        Get pose transformed to robot coordinate frame.

        Returns:
            position: (3,) array in robot frame
            quaternion: (4,) array in robot frame
        """
        state = self.ring_buffer.get()
        pos = self.tx_robot_vive @ state['position']

        # Transform quaternion
        from scipy.spatial.transform import Rotation
        quat = state['quaternion']
        rot = Rotation.from_quat(quat)
        rot_matrix = rot.as_matrix()
        # Apply coordinate transformation to rotation matrix
        tf_rot_matrix = self.tx_robot_vive @ rot_matrix @ self.tx_robot_vive.T
        tf_rot = Rotation.from_matrix(tf_rot_matrix)
        tf_quat = tf_rot.as_quat()  # [x,y,z,w]

        return pos.astype(self.dtype), tf_quat.astype(self.dtype)

    def get_button_state(self):
        """
        Get current button states.

        Returns:
            dict with keys: grip, trigger, trackpad, menu (all bool)
        """
        state = self.ring_buffer.get()
        return {
            'grip': bool(state['grip']),
            'trigger': bool(state['trigger']),
            'trackpad': bool(state['trackpad']),
            'menu': bool(state['menu'])
        }

    def is_button_pressed(self, button_name: str) -> bool:
        """Check if specific button is pressed."""
        state = self.ring_buffer.get()
        return bool(state.get(button_name, 0))

    def get_trigger_value(self) -> float:
        """Get analog trigger value (0.0 to 1.0)."""
        state = self.ring_buffer.get()
        return float(state['trigger_value'])

    def get_trackpad_position(self):
        """Get trackpad touch position (-1.0 to 1.0 for each axis)."""
        state = self.ring_buffer.get()
        return float(state['trackpad_x']), float(state['trackpad_y'])

    def is_connected(self) -> bool:
        """Check if controller is connected."""
        state = self.ring_buffer.get()
        return bool(state['connected'])

    # ======= Haptic Feedback APIs ==========

    def send_haptic(self, role: int = 0, duration_us: int = 500, intensity: float = 1.0):
        """
        Send haptic feedback command to vive_input.

        Args:
            role: 0=right controller, 1=left controller
            duration_us: Duration in microseconds (max 5000)
            intensity: 0.0 to 1.0
        """
        if not self.enable_haptic or self.haptic_sock is None:
            return

        cmd = {
            "role": role,
            "duration": min(duration_us, 5000),
            "intensity": max(0.0, min(1.0, intensity))
        }

        try:
            msg = json.dumps(cmd).encode('utf-8')
            self.haptic_sock.sendto(msg, (self.host, self.haptic_port))
        except Exception:
            pass

    def haptic_pulse(self, duration_us: int = 200):
        """Quick haptic pulse for button press feedback."""
        self.send_haptic(role=0, duration_us=duration_us, intensity=1.0)

    # ======= Start/Stop APIs ==========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self, timeout=5.0):
        """Wait for process to be ready."""
        self.ready_event.wait(timeout=timeout)

    def stop_wait(self):
        """Wait for process to terminate."""
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ======= Main Loop (runs in child process) ==========

    def run(self):
        """Main loop running in child process."""
        sock = None
        buffer = ""

        # Create haptic UDP socket
        if self.enable_haptic:
            try:
                self.haptic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception as e:
                if self.verbose:
                    print(f"[ViveSharedMemory] Warning: Could not create haptic socket: {e}")
                self.haptic_sock = None

        try:
            # Connect to vive_input TCP server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.host, self.port))
            sock.setblocking(False)

            if self.verbose:
                print(f"[ViveSharedMemory] Connected to vive_input at {self.host}:{self.port}")

            # Initialize state
            state = {
                'position': np.zeros((3,), dtype=self.dtype),
                'quaternion': np.array([0., 0., 0., 1.], dtype=self.dtype),
                'grip': np.uint8(0),
                'trigger': np.uint8(0),
                'trackpad': np.uint8(0),
                'menu': np.uint8(0),
                'trigger_value': np.float32(0.0),
                'trackpad_x': np.float32(0.0),
                'trackpad_y': np.float32(0.0),
                'controller_role': np.int32(0),
                'connected': np.uint8(0),
                'receive_timestamp': time.time()
            }

            # Send initial state so client can start reading
            self.ring_buffer.put(state)
            self.ready_event.set()

            while not self.stop_event.is_set():
                # Read from socket
                try:
                    data = sock.recv(4096).decode('utf-8')
                    if data:
                        buffer += data
                except BlockingIOError:
                    pass
                except Exception as e:
                    if self.verbose:
                        print(f"[ViveSharedMemory] Socket error: {e}")
                    time.sleep(0.01)
                    continue

                # Parse JSON lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue

                    try:
                        msg = json.loads(line)
                        receive_timestamp = time.time()

                        # Prefer right controller, fallback to left
                        ctrl = msg.get('right_controller')
                        controller_role = 0
                        if not (ctrl and ctrl.get('connected', False)):
                            ctrl = msg.get('left_controller')
                            controller_role = 1

                        if ctrl and ctrl.get('connected', False):
                            # Parse pose
                            pose = ctrl.get('pose', {})
                            state['position'] = np.array([
                                pose.get('x', 0),
                                pose.get('y', 0),
                                pose.get('z', 0)
                            ], dtype=self.dtype)
                            state['quaternion'] = np.array([
                                pose.get('qx', 0),
                                pose.get('qy', 0),
                                pose.get('qz', 0),
                                pose.get('qw', 1)
                            ], dtype=self.dtype)

                            # Parse buttons
                            buttons = ctrl.get('buttons', {})
                            state['grip'] = np.uint8(buttons.get('grip', False))
                            state['trigger'] = np.uint8(buttons.get('trigger', False))
                            state['trackpad'] = np.uint8(buttons.get('trackpad_button', False))
                            state['menu'] = np.uint8(buttons.get('menu', False))

                            # Parse analog values
                            # Note: trigger analog value is at controller level, not in buttons
                            state['trigger_value'] = np.float32(ctrl.get('trigger', 0.0))
                            # Trackpad position is under 'trackpad' key, not 'buttons'
                            trackpad = ctrl.get('trackpad', {})
                            state['trackpad_x'] = np.float32(trackpad.get('x', 0.0))
                            state['trackpad_y'] = np.float32(trackpad.get('y', 0.0))

                            state['controller_role'] = np.int32(controller_role)
                            state['connected'] = np.uint8(1)
                        else:
                            state['connected'] = np.uint8(0)

                        state['receive_timestamp'] = receive_timestamp
                        self.ring_buffer.put(state)

                    except json.JSONDecodeError:
                        continue

                # Regulate frequency
                time.sleep(1.0 / self.frequency)

        except Exception as e:
            print(f"[ViveSharedMemory] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass
            if self.haptic_sock:
                try:
                    self.haptic_sock.close()
                except:
                    pass
            self.ready_event.set()

            if self.verbose:
                print("[ViveSharedMemory] Process terminated")
