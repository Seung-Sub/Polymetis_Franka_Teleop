"""
FrankaPolicyEnv - Environment for policy inference on Franka robot.

Based on FrankaViveEnv, but without Vive teleoperation components.
Designed for deploying trained policies on Franka robot with Intel RealSense cameras.

Architecture:
- FrankaInterpolationController: Runs at 200Hz for smooth trajectory execution
- FrankaGripperController: Runs at 30Hz for gripper control
- MultiRealsense: Captures RGB from RealSense cameras
- Main loop: Runs at 10Hz for policy inference and action execution

Key differences from FrankaViveEnv:
- No ViveTeleopProcess (no teleoperation)
- Robot/gripper controllers run in normal mode (teleop_mode=False)
- Actions are scheduled via schedule_waypoint() from policy output
- Optional human fallback mode via keyboard

Usage:
    with FrankaPolicyEnv(
        output_dir='./data/eval',
        robot_ip='192.168.1.10'
    ) as env:
        # Policy inference loop
        obs = env.get_obs()
        # ... run policy ...
        env.exec_actions(actions, timestamps)
"""

from typing import Optional, List
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
from multiprocessing.managers import SharedMemoryManager

from polymetis_franka_teleop.real_world.franka_interpolation_controller import FrankaInterpolationController
from polymetis_franka_teleop.real_world.franka_gripper_controller import FrankaGripperController
from polymetis_franka_teleop.real_world.art_gripper_controller import ArtGripperController
from polymetis_franka_teleop.real_world.video_recorder import VideoRecorder
from polymetis_franka_teleop.real_world.image_transform import ImageTransform
from polymetis_franka_teleop.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from polymetis_franka_teleop.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from polymetis_franka_teleop.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from polymetis_franka_teleop.common.interpolation_util import get_interp1d, PoseInterpolator


class FrankaPolicyEnv:
    """
    Environment for policy inference on Franka robot.

    This environment is designed for deploying trained diffusion policies
    on a Franka robot with Intel RealSense cameras. It provides:
    - Timestamp-aligned observations (camera + robot state)
    - Action execution via schedule_waypoint (smooth interpolation)
    - Episode recording for logging/debugging

    Unlike FrankaViveEnv, this class does NOT include:
    - ViveTeleopProcess (no teleop)
    - Vive controller input
    - Direct teleop mode for controllers

    Args:
        output_dir: Directory for saving episode recordings
        robot_ip: IP address of NUC running Franka interface server
        robot_port: ZeroRPC port for robot control (default: 4242)
        gripper_port: ZeroRPC port for gripper control (default: 4242)
        frequency: Main loop frequency in Hz (default: 10)
        camera_serial_numbers: List of RealSense serial numbers
        camera_resolution: Camera capture resolution (W, H)
        camera_fps: Camera capture FPS
        obs_image_resolution: Observation image size for policy (H, W)
        obs_float32: If True, return images as float32 [0,1]
        camera_obs_horizon: Number of observation timesteps for cameras
        robot_obs_horizon: Number of observation timesteps for robot state
        gripper_obs_horizon: Number of observation timesteps for gripper
        camera_obs_latency: Camera observation latency in seconds
        robot_obs_latency: Robot state observation latency
        gripper_obs_latency: Gripper state observation latency
        robot_action_latency: Robot action execution latency (for compensation)
        gripper_action_latency: Gripper action execution latency
        tcp_offset: TCP offset for Franka Hand (default: 0.1034m)
        init_joints: Initial joint positions (7,) or None
        gripper_open_width: Gripper width when open (default: 0.075m)
        gripper_close_width: Gripper width when closed (default: 0.0m)
        enable_multi_cam_vis: Enable multi-camera visualization
        shm_manager: SharedMemoryManager instance
        verbose: Enable verbose logging
    """

    def __init__(self,
            # required params
            output_dir,
            robot_ip,
            # env params
            robot_port=50051,
            gripper_port=4242,         # Franka Hand polymetis service port (zerorpc); ART ignores this
            frequency=10,
            # camera params
            camera_serial_numbers=None,
            camera_resolution=(1280, 720),  # ZED HD720 default; (640,480) works for RealSense backend
            camera_fps=60,
            # obs params
            obs_image_resolution=(224, 224),
            max_obs_buffer_size=60,
            obs_float32=True,  # For policy inference, typically float32
            # timing (all in seconds) — None defers to install/latency_calibration.json
            # (with V3 2026-01-25 hardcoded fallback inside latency_config.py).
            align_camera_idx=0,
            camera_obs_latency=None,
            robot_obs_latency=None,
            gripper_obs_latency=None,
            # action latency for compensation (None → JSON config / hardcoded fallback)
            robot_action_latency=None,
            gripper_action_latency=None,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # observation horizons
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # robot params
            robot_frequency=100,  # KIST stable ceiling; raise on a faster RT NUC (UMI uses 200)
            tcp_offset=None,  # auto: 0.1034m for Franka Hand, 0.216m for ART
            init_joints=None,
            # gripper params
            gripper_open_width=0.075,
            gripper_close_width=0.005,
            # backend selection (KIST extension)
            camera_backend='zed',
            gripper_backend='art',
            art_gripper_host='127.0.0.1',
            art_gripper_port=50053,
            # action latency override for ART (used only if gripper_action_latency=None
            # AND gripper_backend='art' AND no JSON entry — hardcoded last-resort default)
            art_gripper_action_latency=0.085,
            # speed limits
            max_pos_speed=2.0,
            max_rot_speed=6.0,
            # Cartesian impedance gain scaling (UMI defaults Kx=[750,750,750,15,15,15], Kxd=[37,37,37,2,2,2])
            Kx_scale=1.0,
            Kxd_scale=1.0,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,
            verbose=False
            ):

        # Backend-specific defaults
        if tcp_offset is None:
            tcp_offset = 0.216 if gripper_backend == 'art' else 0.1034
        if gripper_backend == 'art':
            if gripper_open_width <= 0.075 + 1e-9:
                gripper_open_width = 0.095

        # Resolve latency constants from install/latency_calibration.json (or
        # hardcoded V3 fallback inside latency_config) when caller passed None.
        # Explicit kwargs always win.
        from polymetis_franka_teleop.common.latency_config import (
            get_camera_obs_latency, get_robot_obs_latency, get_gripper_obs_latency,
            get_robot_action_latency, get_gripper_action_latency,
        )
        if camera_obs_latency is None:
            camera_obs_latency = get_camera_obs_latency(camera_backend)
        if robot_obs_latency is None:
            robot_obs_latency = get_robot_obs_latency()
        if gripper_obs_latency is None:
            gripper_obs_latency = get_gripper_obs_latency(gripper_backend)
        if robot_action_latency is None:
            robot_action_latency = get_robot_action_latency()
        if gripper_action_latency is None:
            gripper_action_latency = get_gripper_action_latency(gripper_backend)
        # Legacy ART override: if the caller passed art_gripper_action_latency
        # AND gripper_backend=='art' AND gripper_action_latency wasn't set
        # explicitly, honour the legacy override path.
        if gripper_backend == 'art' and art_gripper_action_latency is not None:
            # Only apply if this kwarg was explicitly set to a non-default-looking value;
            # otherwise the JSON-resolved value above is preferred.
            if abs(art_gripper_action_latency - 0.085) > 1e-9:
                gripper_action_latency = art_gripper_action_latency

        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # === Setup Cameras ===
        n_cameras = len(camera_serial_numbers) if camera_serial_numbers else 2

        # Compute resolution for visualization
        rw, rh, col, row = optimal_row_cols(
            n_cameras=n_cameras,
            in_wh_ratio=camera_resolution[0] / camera_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )

        # Camera transforms
        transform = []
        vis_transform = []
        video_recorder = []

        for i in range(n_cameras):
            # Observation transform (resize and convert to RGB)
            transform.append(ImageTransform(
                input_res=camera_resolution,
                output_res=obs_image_resolution,
                bgr_to_rgb=True,
                float32=obs_float32
            ))

            # Visualization transform
            vis_transform.append(ImageTransform(
                input_res=camera_resolution,
                output_res=(rw, rh),
                bgr_to_rgb=False,
                float32=False
            ))

            # Video recorder
            video_recorder.append(VideoRecorder.create_h264(
                fps=camera_fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=18,
                thread_type='FRAME',
                thread_count=1
            ))

        if camera_backend == 'zed':
            from polymetis_franka_teleop.real_world.multi_zed import MultiZed
            camera = MultiZed(
                shm_manager=shm_manager,
                serial_numbers=camera_serial_numbers,
                resolution=camera_resolution,
                capture_fps=camera_fps,
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                transform=transform,
                vis_transform=None,
                video_recorder=video_recorder,
                verbose=verbose,
            )
        elif camera_backend == 'realsense':
            from polymetis_franka_teleop.real_world.multi_realsense import MultiRealsense
            camera = MultiRealsense(
                shm_manager=shm_manager,
                serial_numbers=camera_serial_numbers,
                resolution=camera_resolution,
                capture_fps=camera_fps,
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                transform=transform,
                vis_transform=None,
                video_recorder=video_recorder,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown camera_backend={camera_backend!r}")

        # Multi-camera visualizer
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        # === Setup Robot Controller (normal mode, NOT teleop) ===
        robot = FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            robot_port=robot_port,
            frequency=robot_frequency,
            tcp_offset=tcp_offset,
            use_wsg_gripper=False,
            Kx_scale=Kx_scale,
            Kxd_scale=Kxd_scale,
            joints_init=init_joints,
            verbose=verbose,
            receive_latency=robot_obs_latency,
            teleop_mode=False,  # Normal mode for policy execution
            teleop_ring_buffer=None
        )

        # === Setup Gripper Controller (normal mode, NOT teleop) ===
        if gripper_backend == 'art':
            gripper = ArtGripperController(
                shm_manager=shm_manager,
                host=art_gripper_host,
                port=art_gripper_port,
                frequency=60,  # raised from 30 to match camera (matches teleop env)
                verbose=verbose,
                receive_latency=gripper_obs_latency,
                teleop_mode=False,
                teleop_ring_buffer=None,
                gripper_open_width=gripper_open_width,
                gripper_close_width=gripper_close_width,
            )
        elif gripper_backend == 'franka':
            gripper = FrankaGripperController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                gripper_port=gripper_port,
                frequency=60,  # raised from 30 (matches teleop env)
                verbose=verbose,
                receive_latency=gripper_obs_latency,
                teleop_mode=False,
                teleop_ring_buffer=None,
                gripper_open_width=gripper_open_width,
                gripper_close_width=gripper_close_width,
            )
        else:
            raise ValueError(f"Unknown gripper_backend={gripper_backend!r}")
        self.gripper_backend = gripper_backend
        self.camera_backend = camera_backend

        # Store references
        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.multi_cam_vis = multi_cam_vis
        self.shm_manager = shm_manager

        # Parameters
        self.frequency = frequency
        self.camera_fps = camera_fps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.verbose = verbose
        self.n_cameras = n_cameras

        # Timing parameters
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon

        # Speed limits
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed

        # Gripper parameters
        self.gripper_open_width = gripper_open_width
        self.gripper_close_width = gripper_close_width

        # Recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer

        # Temp memory buffers
        self.last_camera_data = None

        # Recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.start_time = None

        # Persist eval-time config to zarr meta so eval rollouts are
        # self-describing. Policy eval has no data_format (always runs a
        # fixed policy ckpt), so that key is omitted.
        try:
            import zarr
            zarr_path = str(self.output_dir.joinpath('replay_buffer.zarr').absolute())
            root = zarr.open(zarr_path, mode='a')
            meta = root.require_group('meta')
            cfg = {
                'mode': 'policy_eval',
                'gripper_backend': self.gripper_backend,
                'camera_backend': self.camera_backend,
                'frequency': int(self.frequency),
                'gripper_max_width': float(self.gripper.gripper_open_width),
                'gripper_close_width': float(getattr(self.gripper, 'gripper_close_width', 0.0)),
                'camera_obs_latency': float(self.camera_obs_latency),
                'robot_obs_latency': float(self.robot_obs_latency),
                'gripper_obs_latency': float(self.gripper_obs_latency),
                'robot_action_latency': float(self.robot_action_latency),
                'gripper_action_latency': float(self.gripper_action_latency),
            }
            for k, v in cfg.items():
                meta.attrs[k] = v
        except Exception as e:
            print(f"[FrankaPolicyEnv] WARN: could not write zarr meta config: {e}")

    # ======== Start/Stop API =============
    @property
    def is_ready(self):
        return (
            self.camera.is_ready and
            self.robot.is_ready and
            self.gripper.is_ready
        )

    def start(self, wait=True):
        """Start all processes."""
        # Start robot and gripper first
        self.robot.start(wait=False)
        self.gripper.start(wait=False)

        # Small delay
        time.sleep(0.5)

        # Start camera
        self.camera.start(wait=False)

        # Start visualizer
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)

        if wait:
            self.start_wait()

    def stop(self, wait=True):
        """Stop all processes."""
        self.end_episode()

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)

        self.camera.stop(wait=False)
        self.gripper.stop(wait=False)
        self.robot.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        self.robot.start_wait()
        self.gripper.start_wait()
        self.camera.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
        # Wait for camera buffers to fill up
        time.sleep(2.0)

    def stop_wait(self):
        self.camera.stop_wait()
        self.gripper.stop_wait()
        self.robot.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    def move_home(self, wait_time=3.0):
        """
        Move robot to home position.

        This calls the robot controller's move_home() method which:
        1. Stops current impedance control
        2. Moves to home joint positions
        3. Restarts impedance control

        Args:
            wait_time: Time to wait after sending home command (default: 3.0s)
        """
        if self.verbose:
            print("[FrankaPolicyEnv] Moving to home position...")
        self.robot.move_home()
        time.sleep(wait_time)
        if self.verbose:
            print("[FrankaPolicyEnv] Home position reached.")

    # ========= Context Manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= Async Env API ===========
    def get_obs(self) -> dict:
        """
        Get observation dict with timestamp-aligned data.

        Returns dict with:
            - camera{i}_rgb: (horizon, H, W, C) camera images
            - robot0_eef_pos: (horizon, 3) end-effector position
            - robot0_eef_rot_axis_angle: (horizon, 3) end-effector rotation
            - robot0_gripper_width: (horizon, 1) gripper width
            - timestamp: (horizon,) observation timestamps
        """
        assert self.is_ready

        # Get camera data
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps
            * (self.camera_fps / self.frequency))
        self.last_camera_data = self.camera.get(
            k=k,
            out=self.last_camera_data)

        # Get robot state
        last_robot_data = self.robot.get_all_state()

        # Get gripper state
        last_gripper_data = self.gripper.get_all_state()

        # Get last camera timestamp as reference
        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # Align camera observations
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = []
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # Align robot observations
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'],
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[..., :3],
            'robot0_eef_rot_axis_angle': robot_pose[..., 3:]
        }

        # Align gripper observations
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_width'][..., None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }

        # Accumulate observations for recording
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_width'][..., None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # Combine all observations
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data

    def exec_actions(self,
            actions: np.ndarray,
            timestamps: np.ndarray,
            compensate_latency=True):
        """
        Execute actions on robot.

        This is the main method for policy deployment. Actions are scheduled
        as waypoints with target timestamps, and the robot controller
        interpolates smoothly between them.

        GRIPPER HANDLING (v2.0):
        - Gripper uses discrete OPEN/CLOSE commands (no interpolation)
        - Only schedules command when state transition is detected
        - This matches data collection behavior and reduces latency

        Args:
            actions: (N, 7) array of actions [x, y, z, rx, ry, rz, gripper_width]
                     - [:6]: absolute TCP pose (position + axis-angle rotation)
                     - [6]: gripper target width in meters
            timestamps: (N,) array of target execution times (wall clock)
            compensate_latency: If True, subtract action latency from timestamps
        """
        assert self.is_ready

        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # Filter only future actions
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        if len(new_actions) == 0:
            return

        # Latency compensation
        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # Gripper threshold for OPEN/CLOSE decision
        gripper_threshold = (self.gripper_open_width + self.gripper_close_width) / 2

        # Track gripper state to detect transitions
        if not hasattr(self, '_last_gripper_is_open'):
            # Initialize from current gripper state
            gripper_state = self.gripper.get_state()
            self._last_gripper_is_open = gripper_state['gripper_width'] > gripper_threshold

        # Schedule waypoints on robot
        for i in range(len(new_actions)):
            r_action = new_actions[i, :6]  # TCP pose

            # Schedule robot waypoint (every step - needs smooth interpolation)
            self.robot.schedule_waypoint(
                pose=r_action,
                target_time=new_timestamps[i] - r_latency
            )

        # === GRIPPER: Only schedule on state transition ===
        # Analyze all action steps to find transitions
        gripper_actions = new_actions[:, 6]
        gripper_is_open = gripper_actions > gripper_threshold

        # Find the first transition point
        transition_idx = None
        for i in range(len(gripper_is_open)):
            if i == 0:
                prev_is_open = self._last_gripper_is_open
            else:
                prev_is_open = gripper_is_open[i - 1]

            if gripper_is_open[i] != prev_is_open:
                transition_idx = i
                break

        # If transition found, schedule single command at transition time
        if transition_idx is not None:
            target_is_open = gripper_is_open[transition_idx]
            target_width = self.gripper_open_width if target_is_open else self.gripper_close_width
            target_time = new_timestamps[transition_idx] - g_latency

            if self.verbose:
                state_str = "OPEN" if target_is_open else "CLOSE"
                print(f"  [exec_actions] Gripper transition: {state_str} at t+{(target_time - receive_time)*1000:.0f}ms")

            self.gripper.schedule_waypoint(
                width=target_width,
                target_time=target_time
            )

            # Update state to match the scheduled command
            # This ensures next inference correctly detects transitions
            self._last_gripper_is_open = target_is_open
        else:
            # No transition: keep tracking the current intent
            self._last_gripper_is_open = gripper_is_open[-1]

        # Debug: Show gripper intent summary
        if self.verbose:
            n_open = np.sum(gripper_is_open)
            n_close = len(gripper_is_open) - n_open
            intent = "OPEN" if n_open > n_close else "CLOSE"
            print(f"  [exec_actions] Gripper intent: {intent} (O:{n_open}/C:{n_close})")

        # Record actions (if recording is active)
        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)

    def get_robot_state(self):
        """Get current robot state."""
        return self.robot.get_state()

    def get_gripper_state(self):
        """Get current gripper state."""
        return self.gripper.get_state()

    def servoL(self, pose, duration=0.1):
        """
        Move robot to target pose immediately.
        Used for human control mode.
        """
        self.robot.servoL(pose, duration=duration)

    def gripper_goto(self, width):
        """
        Move gripper to target width immediately.
        Used for human control mode.
        """
        self.gripper.goto(width)

    def gripper_grasp(self):
        """Close gripper and grasp."""
        self.gripper.grasp()

    # ========= Recording API =============
    def start_episode(self, start_time=None):
        """Start recording an episode."""
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # Prepare video directory
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)

        video_paths = []
        for i in range(self.n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # Start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # Create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )

        print(f'Episode {episode_id} started!')

    def end_episode(self):
        """Stop recording and save episode."""
        assert self.is_ready

        # Stop video recording
        self.camera.stop_recording()

        if self.obs_accumulator is not None:
            assert self.action_accumulator is not None

            # Find end time
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1] + 1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]

                # Interpolate robot pose
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)

                final_actions = actions[:n_steps].copy()

                episode = {
                    'timestamp': timestamps,
                    'action': final_actions,
                }

                episode['robot0_eef_pos'] = robot_pose[:, :3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:, 3:]

                # Interpolate joint positions
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                # Interpolate gripper width
                gripper_interpolator = get_interp1d(
                    t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                    x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                )
                episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')

            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        """Drop current episode without saving."""
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
