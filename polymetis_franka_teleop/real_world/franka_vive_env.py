"""
FrankaViveEnv - Data collection environment with HTC Vive controller teleoperation.

Based on UMI's UmiEnv, adapted for:
- HTC Vive Controller teleoperation (instead of SpaceMouse)
- Franka Panda robot with Franka Hand gripper (discrete open/close)
- Intel RealSense cameras (instead of UVC cameras)
- Direct teleop mode: ViveTeleopProcess runs at 100Hz, bypassing main 10Hz loop

Architecture:
- ViveSharedMemory: Reads Vive controller input at ~200Hz
- ViveTeleopProcess: Computes target poses at 100Hz
- FrankaInterpolationController: Reads from teleop_ring_buffer directly (teleop_mode)
- FrankaGripperController: Reads from teleop_ring_buffer directly (teleop_mode)
- Main loop: Records data at 10Hz using UMI's timestamp alignment

Usage:
    with FrankaViveEnv(
        output_dir='./data',
        robot_ip='172.16.0.3'
    ) as env:
        # Start episode
        env.start_episode()

        # Main loop for data recording
        while not done:
            obs = env.get_obs()
            env.record_action()  # Records action from ViveTeleopProcess

        env.end_episode()
"""

from typing import Optional
import pathlib
import numpy as np


# === Ready-pose presets ===
# 6-joint base (joints 1..6); joint 7 (gripper yaw) is appended per-gripper.
#
# DROID base — matches the GR00T DROID training data state distribution.
# The pose is slightly tilted forward (-π/5 vs -π/4) so the wrist camera sees
# the table workspace by default.
_READY_BASE_DROID  = [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5]
# Standard Franka home — the panda factory default.
_READY_BASE_FRANKA = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0,     np.pi / 2]


def compute_ready_pose(data_format: str, gripper_backend: str) -> np.ndarray:
    """Pick the joint-7 (gripper yaw) value and base 6-joint pose from the
    data format + gripper combination.

    | data_format        | gripper | joints 1..6     | joint 7 |
    |--------------------|---------|-----------------|---------|
    | groot (DROID)      | art     | DROID tilt      | 0       |
    | groot (DROID)      | franka  | DROID tilt      | π/4     |
    | umi / diffusion    | art     | Franka home     | 0       |  ← yaw matches GR00T
    | umi / diffusion    | franka  | Franka home     | π/4     |  ← stock Franka pose

    GR00T uses joint-7=0 with the DROID-tilted base because its training data
    was collected that way; switching base poses would put state observations
    out of distribution for the policy.
    """
    df = data_format.lower()
    gb = gripper_backend.lower()
    if df == 'groot':
        base = _READY_BASE_DROID
    elif df in ('umi', 'diffusion'):
        base = _READY_BASE_FRANKA
    else:
        raise ValueError(
            f"Unknown data_format={data_format!r} (use 'groot', 'umi', or 'diffusion')")
    j7 = 0.0 if gb == 'art' else np.pi / 4
    return np.array(list(base) + [j7], dtype=np.float64)
import time
import shutil
import math
import cv2
from multiprocessing.managers import SharedMemoryManager

from polymetis_franka_teleop.real_world.franka_interpolation_controller import FrankaInterpolationController
from polymetis_franka_teleop.real_world.franka_gripper_controller import FrankaGripperController
from polymetis_franka_teleop.real_world.art_gripper_controller import ArtGripperController
from polymetis_franka_teleop.real_world.vive_shared_memory import ViveSharedMemory
from polymetis_franka_teleop.real_world.vive_teleop_process import ViveTeleopProcess
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


class FrankaViveEnv:
    """
    Environment for data collection with Vive controller teleoperation.

    Key architectural features:
    - ViveTeleopProcess runs at 100Hz for responsive teleoperation
    - Robot/gripper controllers read directly from teleop_ring_buffer (teleop_mode)
    - Main loop runs at 10Hz for data recording only
    - Gripper uses discrete state: 0.0 (open) or 1.0 (closed)
    """

    def __init__(self,
            # required params
            output_dir,
            robot_ip,
            # env params
            frequency=10,
            # camera params
            camera_serial_numbers=None,
            camera_resolution=(1280, 720),  # ZED HD720 default; (640,480) works for RealSense backend
            camera_fps=60,
            # obs
            obs_image_resolution=(224, 224),
            max_obs_buffer_size=60,
            obs_float32=False,
            # timing (all in seconds)
            # Latency values calibrated via V3 direct measurement (2026-01-25)
            # See scripts_real/LATENCY_CALIBRATION_GUIDE.md for details
            align_camera_idx=0,  # Which camera's timestamp to use as reference
            camera_obs_latency=0.015,   # V3 HW timestamp (was: 0.125)
            robot_obs_latency=0.001,    # V3 round-trip/2 (was: 0.0001)
            gripper_obs_latency=0.001,  # V3 round-trip/2 (was: 0.01)
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # robot params
            # KIST default: polymetis-direct mode = simpler deployment (no NUC bridge needed),
            # cleaner architecture for our hardware. ZeroRPC bridge is preserved as opt-in via
            # polymetis_mode='zerorpc' + robot_port=4242 (UMI/DROID-style remote teleop).
            polymetis_mode='direct',   # 'direct' (default, polymetis :50051) or 'zerorpc' (UMI/DROID bridge :4242)
            robot_port=50051,          # 50051 for direct, 4242 for zerorpc
            gripper_port=4242,
            robot_frequency=100,       # KIST: 100 Hz is the empirically stable ceiling on this NUC
                                       # (both zerorpc and direct modes; >150 Hz trips the polymetis 1s watchdog
                                       # because NUC realtime thread can't keep up under combined gRPC+IK load).
                                       # UMI uses 200 Hz on a faster RT NUC — bump if your NUC supports it.
            tcp_offset=None,
            init_joints=None,             # explicit ready-pose override; None → compute from data_format
            # data_format determines the ready (home) joint pose:
            #   'groot'  → DROID-tilted base, matches nvidia/GR00T-N1.7-DROID training data
            #   'umi' / 'diffusion' → standard Franka home
            # joint-7 (gripper yaw) is set by gripper_backend regardless.
            data_format='groot',
            auto_home_on_start=True,      # auto-move to ready pose when controller boots
            # backend selection (KIST extension)
            camera_backend='zed',          # 'zed' or 'realsense'
            gripper_backend='art',         # 'art' (Hyundai) or 'franka' (Franka Hand)
            art_gripper_host='127.0.0.1',
            art_gripper_port=50053,
            # teleop params
            vive_host='127.0.0.1',
            vive_port=12345,
            teleop_frequency=100,
            pos_scale=1.0,
            rot_scale=1.0,
            # Velocity limits for ViveTeleopProcess
            # These are applied in the 100Hz teleop process, not the 10Hz main loop
            # Higher values = faster/more responsive, but may hit hardware limits
            use_velocity_clamping=False,  # Disable by default for 1:1 mapping
            max_pos_velocity=2.0,  # m/s - Franka max cartesian velocity is ~1.7m/s
            max_rot_velocity=2.5,  # rad/s - Franka max angular velocity
            # Cartesian impedance gain scaling — multiplied onto the UMI defaults
            # Kx=[750,750,750, 15,15,15], Kxd=[37,37,37, 2,2,2]. >1 = stiffer/snappier
            # tracking but more chance of overshoot/reflex; <1 = softer.
            Kx_scale=1.0,
            Kxd_scale=1.0,
            gripper_open_width=0.075,  # Match actual Franka Hand observation max (was 0.08)
            gripper_close_width=None,  # auto: 0.0 for ART (full close), 0.005 for Franka Hand
            grip_force=None,           # auto: 60N for ART (firm grip on thin objects), 30N Franka
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,
            verbose=False
            ):

        # Backend-specific defaults
        if tcp_offset is None:
            # Franka Hand ~10.34cm, ART tip ~21.6cm
            tcp_offset = 0.216 if gripper_backend == 'art' else 0.1034
        if gripper_backend == 'art':
            # ART width envelope is wider than Franka Hand's
            if gripper_open_width <= 0.075 + 1e-9:
                gripper_open_width = 0.095
            # Full mechanical close — ART firmware accepts 0mm. Lets us pinch
            # thin objects (paper, cards) that won't trigger contact at 5mm.
            if gripper_close_width is None:
                gripper_close_width = 0.0
            # Stronger default grip than Franka Hand — enough to hold smooth
            # objects without slipping, well below the 100N firmware cap.
            if grip_force is None:
                grip_force = 60.0
        else:  # franka hand
            # Franka libfranka throws if width=0.0 — leave a 5mm safety margin
            if gripper_close_width is None:
                gripper_close_width = 0.005
            if grip_force is None:
                grip_force = 30.0

        # Resolve ready pose from data_format + gripper_backend if not explicitly given.
        # ``ready_pose`` is the canonical home for both trackpad-HOME and (when
        # auto_home_on_start) the controller's startup move-to.
        if init_joints is None:
            ready_pose = compute_ready_pose(data_format, gripper_backend)
        else:
            ready_pose = np.asarray(init_joints, dtype=np.float64)
        if verbose:
            print(f"[FrankaViveEnv] data_format={data_format} gripper={gripper_backend} "
                  f"auto_home_on_start={auto_home_on_start} "
                  f"→ ready pose (rad): {np.round(ready_pose, 3).tolist()}")
        self.data_format = data_format
        self.ready_pose = ready_pose
        # joints_init: triggers a single move at controller boot. None = skip.
        # home_joints: where MOVE_HOME (trackpad / 'h' key) goes during teleop;
        #              always set so HOME keeps working even with --no_auto_home.
        controller_joints_init = ready_pose if auto_home_on_start else None

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

        # === Setup Vive Controller ===
        vive = ViveSharedMemory(
            shm_manager=shm_manager,
            host=vive_host,
            port=vive_port,
            frequency=200,
            verbose=verbose
        )

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
            # Use picklable class instead of closure for multiprocessing
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

            # Video recorder.
            # buffer_size=512 gives ~8.5 s of grace at 60 fps before queue.Full
            # crashes the SingleZed worker when the encoder briefly stalls
            # (disk write hiccup, GIL contention, etc.). Default 128 = 2.1 s,
            # which we observed empty out under recording load.
            # thread_count=4 lets one camera's H.264 encoder use multiple cores
            # — single-thread libx264 at HD720@60 is borderline on this CPU
            # and can't drain the queue at line rate when both cameras record.
            video_recorder.append(VideoRecorder.create_h264(
                fps=camera_fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=18,
                thread_type='FRAME',
                thread_count=4,
                buffer_size=512,
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

        # === Setup ViveTeleopProcess ===
        # This creates 3 ring buffers for robot, gripper, and action recording
        # Note: ViveTeleopProcess outputs discrete gripper_state (0/1)
        # The actual width mapping is handled by FrankaGripperController
        teleop = ViveTeleopProcess(
            shm_manager=shm_manager,
            vive_ring_buffer=vive.ring_buffer,
            robot_ip=robot_ip,
            robot_port=robot_port,
            polymetis_mode=polymetis_mode,
            frequency=teleop_frequency,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            use_velocity_clamping=use_velocity_clamping,
            max_pos_velocity=max_pos_velocity,
            max_rot_velocity=max_rot_velocity,
            tcp_offset=tcp_offset,
            gripper_open_width=gripper_open_width,
            gripper_close_width=gripper_close_width,
            verbose=verbose
        )

        # === Setup Robot Controller (teleop mode) ===
        # joints_init: one-shot move-to at controller boot (before impedance
        #              starts). None = skip; user passed --no_auto_home.
        # home_joints: target for MOVE_HOME (trackpad / 'h' key). Always set
        #              so HOME keeps working.
        # joints_init_duration matches GR00T DROID's reset move (4 s) — fast
        # enough for routine startup, slow enough not to startle anyone.
        robot = FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            robot_port=robot_port,
            polymetis_mode=polymetis_mode,
            frequency=robot_frequency,
            tcp_offset=tcp_offset,
            use_wsg_gripper=False,
            Kx_scale=Kx_scale,
            Kxd_scale=Kxd_scale,
            joints_init=controller_joints_init,
            joints_init_duration=4.0,
            home_joints=ready_pose,
            home_time=2.0,
            verbose=verbose,
            receive_latency=robot_obs_latency,
            teleop_mode=True,
            teleop_ring_buffer=teleop.robot_ring_buffer
        )

        # === Setup Gripper Controller (teleop mode) ===
        if gripper_backend == 'art':
            gripper = ArtGripperController(
                shm_manager=shm_manager,
                host=art_gripper_host,
                port=art_gripper_port,
                frequency=30,
                verbose=verbose,
                receive_latency=gripper_obs_latency,
                teleop_mode=True,
                teleop_ring_buffer=teleop.gripper_ring_buffer,
                gripper_open_width=gripper_open_width,
                gripper_close_width=gripper_close_width,
                default_force=grip_force,
            )
        elif gripper_backend == 'franka':
            gripper = FrankaGripperController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                gripper_port=gripper_port,
                frequency=30,
                verbose=verbose,
                receive_latency=gripper_obs_latency,
                teleop_mode=True,
                teleop_ring_buffer=teleop.gripper_ring_buffer,
                gripper_open_width=gripper_open_width,
                gripper_close_width=gripper_close_width,
            )
        else:
            raise ValueError(f"Unknown gripper_backend={gripper_backend!r}")
        self.gripper_backend = gripper_backend
        self.camera_backend = camera_backend

        # Store references
        self.vive = vive
        self.teleop = teleop
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

        # Timing parameters
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon

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

        # Last robot observation (for non-clutch action recording)
        # Used when clutch is inactive or during HOME motion
        self._last_robot_obs_pose = None
        self._last_gripper_obs_width = None

    # ======== Start/Stop API =============
    @property
    def is_ready(self):
        return (
            self.vive.is_ready and
            self.teleop.is_ready and
            self.camera.is_ready and
            self.robot.is_ready and
            self.gripper.is_ready
        )

    def start(self, wait=True):
        """Start all processes."""
        import time

        # Start Vive first (others depend on it)
        self.vive.start(wait=True)

        # Start teleop (creates ring buffers for robot/gripper)
        self.teleop.start(wait=True)

        # Start robot and gripper (they read from teleop ring buffers)
        self.robot.start(wait=False)
        self.gripper.start(wait=False)

        # Small delay to let robot/gripper initialize before camera
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
        self.teleop.stop(wait=False)
        self.vive.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        self.robot.start_wait()
        self.gripper.start_wait()
        self.camera.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
        # Wait for camera buffers to fill up
        # At 30fps, 2 seconds gives us 60 frames which is enough for obs_horizon
        import time
        time.sleep(2.0)

    def stop_wait(self):
        self.camera.stop_wait()
        self.gripper.stop_wait()
        self.robot.stop_wait()
        self.teleop.stop_wait()
        self.vive.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

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

        Timestamp alignment policy:
        - 'current' time is the last camera timestamp
        - Robot/gripper observations are interpolated to match camera timestamps
        """
        assert self.is_ready

        # Get camera data
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps
            * (self.camera_fps / self.frequency))  # camera_fps / main_freq
        self.last_camera_data = self.camera.get(
            k=k,
            out=self.last_camera_data)

        # Get robot state (from controller ring buffer)
        last_robot_data = self.robot.get_all_state()

        # Get gripper state
        last_gripper_data = self.gripper.get_all_state()

        # Get last camera timestamp as reference (use align_camera_idx)
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

        # Store last robot observation for use in record_action when clutch is inactive or HOME
        # During these states, action = observation for continuous trajectory
        self._last_robot_obs_pose = robot_pose[-1].copy()  # Latest interpolated pose
        # gripper_obs['robot0_gripper_width'] shape is (horizon, 1), get the last value as scalar
        self._last_gripper_obs_width = float(gripper_obs['robot0_gripper_width'][-1, 0])  # Latest gripper width

        # Combine all observations
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data

    def record_action(self, timestamp=None):
        """
        Record action from ViveTeleopProcess.

        In teleop mode, actions are computed by ViveTeleopProcess at 100Hz.
        This method reads the action from the action_ring_buffer and records it.

        IMPORTANT: For UMI-compatible data collection, the timestamp should be
        provided by the main loop to maintain a precise time grid:
            timestamp = t_command_target - time.monotonic() + time.time()

        This ensures actions are recorded at exact intervals (e.g., 0.1s for 10Hz)
        aligned with the episode start_time.

        Action Source Logic:
            The action pose source depends on the teleop state:

            1. Clutch ACTIVE (grip button held):
               - Use Vive teleop target_pose (user is actively controlling)

            2. Clutch INACTIVE (grip button released, not HOME):
               - Use observation pose (robot holding position via impedance control)
               - This ensures action = obs for continuous trajectory

            3. HOME motion active:
               - Use observation pose (robot following its own trajectory to home)
               - This ensures action = obs during autonomous HOME motion

            This logic ensures:
            - Before first clutch activation: action = obs (robot stationary)
            - During teleop: action = vive target (user intent)
            - During clutch release: action = obs (robot holding position)
            - During HOME: action = obs (robot autonomous motion)
            - After HOME, waiting for clutch: action = obs (robot stationary)

        Args:
            timestamp: Wall-clock time when this action will be executed.
                       If None, uses the timestamp from ViveTeleopProcess (not recommended).
        """
        if self.action_accumulator is None:
            return

        # Get action from teleop process
        teleop_action = self.teleop.action_ring_buffer.get()

        # Check teleop states
        clutch_active = bool(teleop_action.get('clutch_active', 0))
        home_active = bool(teleop_action.get('home_active', 0))

        # Action format: [x, y, z, rx, ry, rz, gripper_width]
        # - target_pose: Absolute TCP pose (NOT delta!)
        # - gripper_width: Target gripper width in meters (continuous value)
        #   - Computed from trigger_value in ViveTeleopProcess
        #   - Provides smooth transition during trigger press for better Diffusion Policy training

        # Determine target pose based on teleop state
        if clutch_active and not home_active:
            # Clutch ACTIVE: Use Vive teleop target_pose (user actively controlling)
            target_pose = teleop_action['target_pose']  # (6,) absolute pose
        else:
            # Clutch INACTIVE or HOME: Use observation pose
            # This ensures action = obs for:
            # - Before first clutch activation
            # - During clutch release (robot holding position)
            # - During HOME motion (robot following autonomous trajectory)
            # - After HOME, waiting for clutch re-activation
            if hasattr(self, '_last_robot_obs_pose') and self._last_robot_obs_pose is not None:
                target_pose = self._last_robot_obs_pose.copy()
            else:
                # Fallback to teleop action if no observation available yet
                target_pose = teleop_action['target_pose']

        # Get gripper target width
        # Gripper action is INDEPENDENT of clutch state:
        #   - Always use teleop gripper_target_width (trigger_value based)
        #   - This is different from pose, which depends on clutch
        #
        # Rationale:
        #   - Pose: clutch controls whether user is moving the robot
        #   - Gripper: trigger controls gripper regardless of robot movement
        #   - User may want to operate gripper while clutch is released
        #
        # ViveTeleopProcess computes gripper_target_width based on trigger analog value:
        #   - OPEN state + trigger pressed: 0.08 → 0.00 (closing)
        #   - CLOSE state + trigger pressed: 0.00 → 0.08 (opening)
        #   - After toggle, holds at target until trigger is released
        gripper_target_width = teleop_action['gripper_target_width']

        # Use provided timestamp (UMI-style grid) or fallback to teleop timestamp
        if timestamp is None:
            # Fallback: use teleop timestamp (may cause timing irregularities)
            action_timestamp = teleop_action['timestamp']
        else:
            # Preferred: use main loop's precise timestamp
            action_timestamp = timestamp

        # Combine into single action array
        action = np.concatenate([target_pose, [gripper_target_width]])

        # Record action
        self.action_accumulator.put(
            action[None, :],  # (1, 7)
            np.array([action_timestamp])
        )

    def exec_actions(self,
            actions: np.ndarray,
            timestamps: np.ndarray,
            compensate_latency=False):
        """
        Execute actions on robot (for policy deployment / eval).

        This method is the counterpart to UmiEnv.exec_actions() and is used
        during policy inference (eval_real.py) to send actions to the robot.

        NOTE: During teleoperation (data collection), this method is NOT used.
        Actions are sent directly by ViveTeleopProcess to the robot controller.

        Args:
            actions: (N, 7) array of actions [x, y, z, rx, ry, rz, gripper_width]
                     - [:6]: absolute TCP pose
                     - [6]: gripper target width in meters (0.0 ~ gripper_open_width)
                           Values below threshold → grasp (close)
                           Values above threshold → move to open position
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

        # Latency compensation (if enabled)
        r_latency = getattr(self, 'robot_action_latency', 0.1) if compensate_latency else 0.0
        g_latency = getattr(self, 'gripper_action_latency', 0.1) if compensate_latency else 0.0

        # Schedule waypoints on robot and gripper
        for i in range(len(new_actions)):
            r_action = new_actions[i, :6]  # TCP pose
            g_action = new_actions[i, 6]   # gripper state

            # Schedule robot waypoint
            self.robot.schedule_waypoint(
                pose=r_action,
                target_time=new_timestamps[i] - r_latency
            )

            # Schedule gripper action
            # Convert continuous width prediction to discrete grasp/move command
            # Threshold at midpoint between open and close widths
            # - Below threshold: intent to close → grasp command
            # - Above threshold: intent to open → move to open position
            gripper_threshold = (self.gripper.gripper_open_width + self.gripper.gripper_close_width) / 2
            if g_action < gripper_threshold:
                gripper_width = self.gripper.gripper_close_width  # grasp
            else:
                gripper_width = self.gripper.gripper_open_width   # open
            self.gripper.schedule_waypoint(
                width=gripper_width,
                target_time=new_timestamps[i] - g_latency
            )

        # Record actions (if recording is active)
        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)

    def get_robot_state(self):
        """Get current robot state."""
        return self.robot.get_state()

    def get_gripper_state(self):
        """Get current gripper state."""
        return self.gripper.get_state()

    def get_teleop_state(self):
        """Get current teleop state (from ViveTeleopProcess)."""
        return self.teleop.action_ring_buffer.get()

    def is_clutch_engaged(self):
        """Check if Vive controller clutch (grip button) is engaged."""
        state = self.vive.get_button_state()
        return state.get('grip', False)

    def move_home(self, wait=False):
        """
        Reset to ready pose: open gripper + move robot to home joints.

        Mirrors Isaac-GR00T's franka_env_kist.reset() — gripper opens
        BEFORE the arm moves so anything currently held is released, and
        the robot starts from a known fully-open state.

        Sequence:
        1. Open gripper (non-blocking — happens in parallel with arm move)
        2. Robot controller MOVE_HOME (terminates impedance, moves joints,
           restarts impedance, requires clutch re-engage)
        3. Optional wait for arm motion to complete

        Args:
            wait: If True, block until home motion completes (default: False)
        """
        print("[FrankaViveEnv] Moving to HOME position (gripper open + arm to ready pose)...")

        # 1. Open gripper. The ArtGripperController polls input_queue in both
        # teleop and normal modes, so this works during active teleop too.
        try:
            open_w = float(self.gripper.gripper_open_width)
            self.gripper.goto(width=open_w)
        except Exception as e:
            print(f"[FrankaViveEnv] gripper open failed (continuing arm move): {e}")

        # 2. Send home command to robot controller (asynchronous)
        self.robot.move_home()

        if wait:
            # Wait for arm motion to complete (approximate — arm move time
            # is set in FrankaInterpolationController.home_time = 2.0s).
            import time
            time.sleep(2.5)
            print("[FrankaViveEnv] HOME complete. Re-engage clutch (grip) to continue teleop.")

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

        n_cameras = self.camera.n_cameras
        video_paths = []
        for i in range(n_cameras):
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

            # Find end time (minimum of all timestamps)
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

                # Interpolate robot pose for observations
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)

                # Actions are already correct from record_action():
                # - Clutch active: Vive teleop target_pose
                # - Clutch inactive / HOME: observation pose
                # No post-processing needed.
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
