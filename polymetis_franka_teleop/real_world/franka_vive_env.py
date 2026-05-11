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

    | data_format    | gripper | joints 1..6  | joint 7 |
    |----------------|---------|--------------|---------|
    | groot (DROID)  | art     | DROID tilt   | 0       |
    | groot (DROID)  | franka  | DROID tilt   | π/4     |
    | diffusion      | art     | Franka home  | 0       |
    | diffusion      | franka  | Franka home  | π/4     |

    GR00T uses joint-7=0 with the DROID-tilted base because its training
    data was collected that way; switching base poses would put state
    observations out of distribution for the policy. The Diffusion-Policy /
    UMI lineage uses the stock Franka home — same ready-pose either way
    since both consume our zarr stream identically.
    """
    df = data_format.lower()
    gb = gripper_backend.lower()
    if df == 'groot':
        base = _READY_BASE_DROID
    elif df == 'diffusion':
        base = _READY_BASE_FRANKA
    else:
        raise ValueError(
            f"Unknown data_format={data_format!r} (use 'groot' or 'diffusion')")
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
            # Latency values default to install/latency_calibration.json (or to
            # the V3 2026-01-25 hardcoded fallback if the JSON is missing).
            # Pass an explicit value here to override per-call. Backend-aware
            # resolution happens below after camera_backend / gripper_backend
            # are known.
            align_camera_idx=0,  # Which camera's timestamp to use as reference
            camera_obs_latency=None,
            robot_obs_latency=None,
            gripper_obs_latency=None,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # robot params
            robot_port=50051,
            gripper_port=4242,         # Franka Hand polymetis service port (zerorpc); ART ignores this
            robot_frequency=100,       # 100 Hz is the empirically stable ceiling on this NUC
                                       # (>150 Hz trips the polymetis 1s watchdog because the NUC RT
                                       # thread can't keep up under combined gRPC+IK load).
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

        # Backend-specific defaults from the central registry.
        from polymetis_franka_teleop.common.gripper_specs import get_spec
        spec = get_spec(gripper_backend)
        if tcp_offset is None:
            tcp_offset = spec.tcp_offset
        # Width envelope: caller overrides win, otherwise spec applies.
        # The ``gripper_open_width=0.075`` default is the Franka-Hand value;
        # treat it as "unset" for the ART path and override to ART's 95 mm.
        if gripper_backend == 'art' and gripper_open_width <= 0.075 + 1e-9:
            gripper_open_width = spec.open_width
        if gripper_close_width is None:
            gripper_close_width = spec.close_width
        if grip_force is None:
            grip_force = spec.default_force

        # Resolve ready pose from data_format + gripper_backend if not explicitly given.
        # Resolve latency constants early so sub-controllers (gripper, etc.)
        # can be constructed with the right ``receive_latency``. JSON config
        # > explicit kwarg > backend-specific fallback.
        from polymetis_franka_teleop.common.latency_config import (
            get_camera_obs_latency, get_robot_obs_latency, get_gripper_obs_latency,
        )
        if camera_obs_latency is None:
            camera_obs_latency = get_camera_obs_latency(camera_backend)
        if robot_obs_latency is None:
            robot_obs_latency = get_robot_obs_latency()
        if gripper_obs_latency is None:
            gripper_obs_latency = get_gripper_obs_latency(gripper_backend)

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
            frequency=teleop_frequency,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            use_velocity_clamping=use_velocity_clamping,
            max_pos_velocity=max_pos_velocity,
            max_rot_velocity=max_rot_velocity,
            tcp_offset=tcp_offset,
            gripper_open_width=gripper_open_width,
            gripper_close_width=gripper_close_width,
            # Match FrankaInterpolationController.home_time so the synthesized
            # action lerp closes exactly when the joint-space move finishes.
            home_duration=2.0,
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
            teleop_ring_buffer=teleop.robot_ring_buffer,
            # Share the Vive process's external-HOME flag so the controller's
            # catalog #27 auto-HOME escalation also fires the action lerp.
            external_home_request_array=teleop.external_home_request_array,
        )

        # === Setup Gripper Controller (teleop mode) ===
        if gripper_backend == 'art':
            gripper = ArtGripperController(
                shm_manager=shm_manager,
                host=art_gripper_host,
                port=art_gripper_port,
                frequency=60,  # raised from 30 to match camera 60 fps; ART daemon PDO 100Hz handles this comfortably
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
                frequency=60,  # raised from 30; libfranka commands stay rate-limited via transition-only logic, but state polling at 60 Hz aligns with camera
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

        # Timing parameters — already resolved at top of __init__
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

        # Write recording-time config to the zarr meta. The converters
        # (convert_to_gr00t_lerobot, convert_franka_vive_to_umi_format) read
        # these so the user can't accidentally pass mismatched
        # --gripper_max_width / --fps / --data_format at conversion time.
        self._write_recording_meta()

    def _write_recording_meta(self):
        """Persist data-collection config into replay_buffer.zarr/meta/.attrs.

        Idempotent — overwrites on every __init__. Stored as a JSON-serializable
        dict so that converters can `zarr.open(...).require_group("meta").attrs`.
        """
        try:
            import zarr
            zarr_path = str(self.output_dir.joinpath('replay_buffer.zarr').absolute())
            root = zarr.open(zarr_path, mode='a')
            meta = root.require_group('meta')
            cfg = {
                'data_format': self.data_format,
                'gripper_backend': self.gripper_backend,
                'camera_backend': self.camera_backend,
                'frequency': int(self.frequency),
                'gripper_max_width': float(self.gripper.gripper_open_width),
                'gripper_close_width': float(getattr(self.gripper, 'gripper_close_width', 0.0)),
                # Latency constants in effect at record time (so post-hoc
                # latency analysis can use the right reference values).
                'camera_obs_latency': float(self.camera_obs_latency),
                'robot_obs_latency': float(self.robot_obs_latency),
                'gripper_obs_latency': float(self.gripper_obs_latency),
            }
            for k, v in cfg.items():
                meta.attrs[k] = v

            # === Schema v2 conventions (data_pipeline_spec.md §4) ===
            # These never change within a dataset; persisted so converters
            # and downstream tooling can rely on them without re-deriving.
            meta.attrs['schema_version']           = '2.0.0'
            meta.attrs['quaternion_order']         = '[qx, qy, qz, qw]'
            meta.attrs['quaternion_sign_normalised'] = True
            meta.attrs['ee_pose_frame']            = 'base'
            meta.attrs['gripper_convention']       = '0=open, 1=closed (binary toggle); width in meters (continuous)'
            meta.attrs['robot_type']               = 'franka_panda'
            meta.attrs['teleop_device']            = 'vive_pro_controller'
            meta.attrs['obs_native_layout']        = 'contiguous + episode_ranges'
            meta.attrs['obs_native_ranges_shape']  = '(E, 2, 2) — [[r_start, r_end], [g_start, g_end]] per episode'
        except Exception as e:
            # never block recording on a meta-write failure
            print(f"[FrankaViveEnv] WARN: could not write zarr meta config: {e}")

        # Also write a dataset_meta.json side-car at the session dir level
        # (more discoverable for casual inspection than poking into zarr
        # attrs). Layout matches data_pipeline_spec.md §4.
        try:
            import json
            ds_meta = {
                'dataset_name': self.output_dir.name,
                'version': '2.0.0',
                'robot_type': 'franka_panda',
                'gripper_type': self.gripper_backend,
                'frame_conventions': {
                    'ee_pose_frame': 'base',
                    'quaternion_order': '[qx, qy, qz, qw]',
                    'quaternion_sign_normalized': True,
                    'gripper_convention_discrete': '0=open, 1=closed',
                    'gripper_convention_continuous': 'width in meters [close_width, open_width]',
                },
                'control_loop_rate_hz': 100,
                'state_sample_rate_hz': 100,
                'camera_fps_target': int(getattr(self, 'camera_fps', 60)),
                'main_loop_rate_hz': int(self.frequency),
                'data_format_target': self.data_format,
                'storage_container': 'zarr',
                'obs_native_layout': 'contiguous + episode_ranges (mirrors data/ + episode_ends)',
                'calibration_method': 'none yet (extrinsics placeholder; future apriltag)',
            }
            (self.output_dir / 'dataset_meta.json').write_text(json.dumps(ds_meta, indent=2))
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: dataset_meta.json write failed: {e}")

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

        # Pin the main demo process AND any child that doesn't pin itself to
        # the "general pool" cores 0-5, 10-13. Cores 6-9 are reserved for
        # FrankaInterpolationController (6,7) and ArtGripperController (8,9).
        # Without this, video recorder threads / multiprocessing helpers can
        # land on cores 6-9 and steal cycles from the timing-critical loops,
        # causing libfranka communication_constraints_violation reflex storms.
        try:
            import os
            general_pool = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13}
            os.sched_setaffinity(0, general_pool)
            print(f"[FrankaViveEnv] main + general children pinned to {sorted(general_pool)} "
                  f"(cores 6-9 reserved for FrankaInterp/ArtGripper)")
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: main affinity pin failed: {e}")

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

        # After auto-home: the robot is now physically at ``ready_pose``.
        # Sample its TCP pose and feed it to the Vive teleop process so the
        # synthesized cartesian action trajectory during trackpad-HOME can
        # lerp/slerp to a known endpoint (instead of falling back to
        # obs-following). One-shot, runs once per session.
        try:
            state = self.robot.get_state()
            home_tcp = state.get('ActualTCPPose')
            if home_tcp is not None and np.linalg.norm(home_tcp[:3]) > 1e-6:
                self.teleop.set_home_target_pose(np.asarray(home_tcp, dtype=np.float64))
                # Also tell the Vive process to drop its stale pre-auto-home
                # ``target_pose`` and resample the robot's current pose. This
                # eliminates the pre-first-grip action ≠ obs discontinuity.
                self.teleop.sync_target_pose_to_current_robot()
                if self.verbose:
                    print(f"[FrankaViveEnv] home_target_pose seeded "
                          f"{np.asarray(home_tcp)[:3].round(3).tolist()} -> teleop process; "
                          f"target_pose resync requested")
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: could not seed home_target_pose "
                  f"({type(e).__name__}: {e}); first HOME of this session will "
                  f"fall back to obs-following.")

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

        # Accumulate observations for recording. Schema v2:
        #   - measured state at native rate (controller ~100Hz)
        #   - includes joint_torque_external + ee linear/angular velocity
        #     + sign-normalised quaternion (via end_episode post-process)
        #   - includes per-step commanded values from the controller
        # All keys here have aligned timestamps == ``robot_timestamp`` (the
        # batched-fetch receipt time minus receive_latency).
        if self.obs_accumulator is not None:
            robot_data_for_acc = {
                'robot0_eef_pose':              last_robot_data['ActualTCPPose'],
                'robot0_eef_orientation_quat':  last_robot_data['ActualTCPQuat'],
                'robot0_eef_linear_velocity':   last_robot_data['ActualEELinVel'],
                'robot0_eef_angular_velocity':  last_robot_data['ActualEEAngVel'],
                'robot0_joint_pos':             last_robot_data['ActualQ'],
                'robot0_joint_vel':             last_robot_data['ActualQd'],
                'robot0_joint_torque_external': last_robot_data['ActualTorquesExt'],
                # Command stream (recorded alongside state since the
                # controller emits both at the same 100Hz tick).
                'cmd_joint_position':           last_robot_data['LastJointCmd'],
                'cmd_joint_velocity':           last_robot_data['LastJointVelCmd'],
                'cmd_ee_pose':                  last_robot_data['LastEEPoseCmd'],
                'cmd_ee_orientation_quat':      last_robot_data['LastEEQuatCmd'],
                'cmd_ee_linear_velocity':       last_robot_data['LastEELinVelCmd'],
                'cmd_ee_angular_velocity':      last_robot_data['LastEEAngVelCmd'],
                'control_mode':                 last_robot_data['ControlMode'].astype(np.float64)
                                                if hasattr(last_robot_data['ControlMode'], 'astype')
                                                else np.asarray(last_robot_data['ControlMode'],
                                                                dtype=np.float64),
            }
            self.obs_accumulator.put(
                data=robot_data_for_acc,
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

    def record_action(self, timestamp=None):
        """Record one action sample from the Vive teleop process.

        The action stream is intentionally a *synthetic continuous-grip
        trajectory*: regardless of clutch / HOME / trackpad-rotation state,
        ``teleop_action['target_pose']`` always carries the user's last
        expressed intent in TCP coordinates, and ``gripper_target_width``
        carries the analog trigger value. This method just snapshots both.

        State-by-state semantics (handled entirely inside ViveTeleopProcess —
        env.record_action is a passive reader):

        * **Before first grip-on**: target_pose initialised to the robot's
          startup pose, so action ≈ obs while the user is idle.
        * **Grip held**: target_pose = current Vive-driven target.
        * **Grip released**: target_pose unchanged — the cartesian-impedance
          controller keeps holding that target with restoring force, and the
          action stream reflects the user's last intent.
        * **Trackpad touch (joint-7 rotation)**: target_pose rotates in place
          around joint-7, even though grip is off.
        * **Trackpad press (HOME)**: target_pose lerps/slerps from the user's
          last intent to home_TCP_pose over ``home_duration`` seconds —
          synthesizing what the action would look like if the user had
          dragged the gripper to home with grip held.

        Args:
            timestamp: wall-clock time when this action will be executed.
                       Provided by the main loop for UMI-precise gridding;
                       falls back to the teleop process's own timestamp.
        """
        if self.action_accumulator is None:
            return

        teleop_action = self.teleop.action_ring_buffer.get()
        target_pose = teleop_action['target_pose']
        gripper_target_width = teleop_action['gripper_target_width']

        action_timestamp = (
            float(teleop_action['timestamp']) if timestamp is None else float(timestamp)
        )
        action = np.concatenate([target_pose, [gripper_target_width]])
        self.action_accumulator.put(
            action[None, :],                         # (1, 7)
            np.array([action_timestamp]),
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

        # 0. Signal the Vive process to enter HOME synthesis state. The
        # trackpad-press path does this internally inside ViveTeleopProcess;
        # for the other two entry points -- ``h`` key / SIGHUP via this
        # method, and the controller-internal catalog #27 escalation via
        # the same shared flag -- the synthesis won't fire unless we tell
        # the Vive process explicitly. Without this, action stays pinned
        # to the user's last grip-on target during the autonomous move.
        try:
            self.teleop.request_home_synthesis()
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: could not signal home synthesis: {e}")

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
    def start_episode(self, start_time=None, task=None, task_id=None, scene_id=None):
        """Start recording an episode.

        Writes the per-episode metadata atomically (zarr meta + language.json
        side-car) so a drop_episode call can roll both back consistently.

        Args:
            start_time: optional float wall-clock anchor (defaults to now).
            task:       optional language instruction. Required for converters
                        to populate LeRobot ``tasks`` field; if missing the
                        converter falls back to its ``--task`` CLI argument.
            task_id:    optional explicit task identifier. If None, derived
                        as sha256(normalized(task))[:12] so identical
                        instructions across sessions collapse to one
                        LeRobot ``task_index``.
            scene_id:   optional scene/setup identifier (groups episodes
                        from the same physical environment).
        """
        import hashlib, subprocess, uuid, json, datetime
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        # Episode metadata block — written atomically below.
        ep_uuid = str(uuid.uuid4())
        ts_start_iso = datetime.datetime.fromtimestamp(
            start_time, tz=datetime.timezone.utc).isoformat()

        # git hash for software_version
        git_hash = None
        try:
            import os
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            git_hash = subprocess.check_output(
                ['git', '-C', repo_root, 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            pass

        # task_id derivation (hash-based default, explicit override)
        if task is not None and task_id is None:
            normalised = ' '.join(str(task).split()).strip().lower()
            task_id = hashlib.sha256(normalised.encode('utf-8')).hexdigest()[:12]
            task_id_source = 'auto_hash'
        else:
            task_id_source = 'explicit' if task_id else 'none'

        # Atomic write: zarr meta.attrs + language.json side-car.
        # If either fails we abort so the two stay in sync (or both empty).
        try:
            import zarr
            zarr_path = str(self.output_dir.joinpath('replay_buffer.zarr').absolute())
            root = zarr.open(zarr_path, mode='a')
            meta = root.require_group('meta')

            tasks = list(meta.attrs.get('episode_tasks', []))
            task_ids = list(meta.attrs.get('episode_task_ids', []))
            ep_ids = list(meta.attrs.get('episode_ids', []))
            scene_ids = list(meta.attrs.get('episode_scene_ids', []))
            sw_versions = list(meta.attrs.get('episode_software_versions', []))
            start_iso_list = list(meta.attrs.get('episode_start_iso', []))

            tasks.append(str(task) if task is not None else '')
            task_ids.append(task_id or '')
            ep_ids.append(ep_uuid)
            scene_ids.append(str(scene_id) if scene_id is not None else '')
            sw_versions.append(git_hash or '')
            start_iso_list.append(ts_start_iso)

            meta.attrs['episode_tasks']             = tasks
            meta.attrs['episode_task_ids']          = task_ids
            meta.attrs['episode_ids']               = ep_ids
            meta.attrs['episode_scene_ids']         = scene_ids
            meta.attrs['episode_software_versions'] = sw_versions
            meta.attrs['episode_start_iso']         = start_iso_list

            # language.json side-car (per-episode dir created later in this
            # function; we put the JSON next to the videos folder for clarity)
            episode_id = self.replay_buffer.n_episodes
            ep_dir = self.video_dir.joinpath(str(episode_id))
            ep_dir.mkdir(parents=True, exist_ok=True)
            lang_path = ep_dir.joinpath('language.json')
            lang_path.write_text(json.dumps({
                'episode_id': ep_uuid,
                'instruction': str(task) if task is not None else None,
                'task_id': task_id,
                'task_id_source': task_id_source,
                'scene_id': str(scene_id) if scene_id is not None else None,
                'subtask_annotations': None,
                'timestamp_start_utc': ts_start_iso,
                'software_version': git_hash,
            }, indent=2))
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: episode metadata write failed: {e}")

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

                # === Schema v2 new fields (linear-interp scalar/vector) ===
                # Helper closure so we don't repeat get_interp1d 9 times.
                from polymetis_franka_teleop.common.rotation_util import (
                    quat_continuous_within_episode, slerp_at,
                )
                def _interp_linear(key):
                    if key not in self.obs_accumulator.data:
                        return None
                    return get_interp1d(
                        np.array(self.obs_accumulator.timestamps[key]),
                        np.array(self.obs_accumulator.data[key]),
                    )(timestamps)

                # Sign-normalise the recorded quaternion stream BEFORE
                # interpolating, then SLERP into the action grid.
                quat_ts = np.array(self.obs_accumulator.timestamps.get(
                    'robot0_eef_orientation_quat', []))
                quat_data = np.array(self.obs_accumulator.data.get(
                    'robot0_eef_orientation_quat', []))
                if quat_data.size > 0:
                    quat_data = quat_continuous_within_episode(quat_data)
                    episode['robot0_eef_orientation_quat'] = slerp_at(
                        quat_ts, quat_data, timestamps)

                cmd_quat_ts = np.array(self.obs_accumulator.timestamps.get(
                    'cmd_ee_orientation_quat', []))
                cmd_quat_data = np.array(self.obs_accumulator.data.get(
                    'cmd_ee_orientation_quat', []))
                if cmd_quat_data.size > 0:
                    cmd_quat_data = quat_continuous_within_episode(cmd_quat_data)
                    episode['action_ee_orientation_quat_cmd'] = slerp_at(
                        cmd_quat_ts, cmd_quat_data, timestamps)

                # Linear-interp scalar / vector fields
                for src_key, dst_key in [
                    ('robot0_eef_linear_velocity',   'robot0_eef_linear_velocity'),
                    ('robot0_eef_angular_velocity',  'robot0_eef_angular_velocity'),
                    ('robot0_joint_torque_external', 'robot0_joint_torque_external'),
                    ('cmd_joint_position',           'action_joint_position_cmd'),
                    ('cmd_joint_velocity',           'action_joint_velocity_cmd'),
                    ('cmd_ee_linear_velocity',       'action_ee_linear_velocity_cmd'),
                    ('cmd_ee_angular_velocity',      'action_ee_angular_velocity_cmd'),
                ]:
                    v = _interp_linear(src_key)
                    if v is not None:
                        episode[dst_key] = v

                # action_ee_position_cmd: same source as cmd_ee_pose first 3
                if 'cmd_ee_pose' in self.obs_accumulator.data:
                    cmd_pose = _interp_linear('cmd_ee_pose')
                    episode['action_ee_position_cmd'] = cmd_pose[:, :3]

                # control_mode: nearest-neighbour rather than linear (it's
                # categorical). At least one value per action timestep.
                if 'control_mode' in self.obs_accumulator.data:
                    cm_ts   = np.array(self.obs_accumulator.timestamps['control_mode'])
                    cm_data = np.array(self.obs_accumulator.data['control_mode']).flatten()
                    # nearest-neighbour lookup
                    idxs = np.searchsorted(cm_ts, timestamps)
                    idxs = np.clip(idxs, 0, len(cm_data) - 1)
                    episode['action_control_mode'] = cm_data[idxs].astype(np.int8)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')

                # === Native-rate stream dump (contiguous + episode_ranges) ===
                # Schema v2: persist the raw 100Hz proprioception streams
                # alongside the action-grid view, so a future ACT 50Hz
                # converter doesn't have to back-fill from a 15Hz signal.
                # Storage: ~700 KB / 60 s episode << mp4 (~10 MB).
                self._append_native_rate_stream(episode_id)

                # Camera calibration sidecar (intrinsics now, extrinsics
                # placeholder for future calib pass).
                self._write_camera_calib(episode_id)

                # Mark episode success + end timestamp in language.json
                # (matches start_episode's atomic-write pattern).
                self._finalise_language_json(episode_id, success=True)

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
        # Also roll back the matching native-rate range row, if present.
        try:
            import zarr
            root = zarr.open(str(self.output_dir.joinpath('replay_buffer.zarr').absolute()), 'a')
            if 'obs_native_episode_ranges' in root.get('meta', {}):
                rng = root['meta']['obs_native_episode_ranges']
                if rng.shape[0] > episode_id:
                    # Truncate every obs_native key back to the row's start.
                    start = int(rng[episode_id, 0])
                    if 'obs_native' in root:
                        for k in list(root['obs_native'].array_keys()):
                            root['obs_native'][k].resize(start, *root['obs_native'][k].shape[1:])
                    rng.resize(episode_id, 2)
        except Exception as e:
            print(f'[FrankaViveEnv] WARN: drop_episode native-rate rollback failed: {e}')
        print(f'Episode {episode_id} dropped!')

    def _finalise_language_json(self, episode_id: int, success: bool) -> None:
        """Stamp end_iso + success on the per-episode language.json. The
        start half was written in ``start_episode`` so this just rounds out
        the record. Idempotent — safe to call multiple times."""
        import datetime, json
        try:
            ep_dir = self.video_dir.joinpath(str(episode_id))
            path = ep_dir.joinpath('language.json')
            if not path.exists():
                return
            doc = json.loads(path.read_text())
            doc['timestamp_end_utc'] = datetime.datetime.now(
                tz=datetime.timezone.utc).isoformat()
            doc['success'] = bool(success)
            path.write_text(json.dumps(doc, indent=2))
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: language.json finalise failed: {e}")

    def _write_camera_calib(self, episode_id: int) -> None:
        """Dump per-camera intrinsics + extrinsics (placeholder) JSON.

        Schema is defined now so downstream (rerun replay, calib tooling)
        has a stable target. ``extrinsics`` is filled with ``null`` and
        marked ``measured=false`` until a real calibration is performed.
        Intrinsics are pulled from the camera worker's ``intrinsics_array``
        SHM (populated at boot from the ZED SDK / RealSense SDK).
        """
        import json
        try:
            ep_dir = self.video_dir.joinpath(str(episode_id))
            calib_dir = ep_dir.joinpath('calibration')
            calib_dir.mkdir(parents=True, exist_ok=True)
            # MultiZed / MultiRealsense exposes per-cam SingleZed workers via
            # ``.cameras`` (a dict / list). Each worker has ``intrinsics_array``
            # (fx, fy, cx, cy, h, w, baseline) and ``serial_number``.
            cams = getattr(self.camera, 'cameras', None) or {}
            for i, cam_idx in enumerate(sorted(cams.keys()) if isinstance(cams, dict) else range(len(cams))):
                worker = cams[cam_idx] if isinstance(cams, dict) else cams[i]
                try:
                    intr = worker.intrinsics_array.get().copy()
                    fx, fy, cx, cy, h, w, baseline = [float(x) for x in intr[:7]]
                except Exception:
                    fx = fy = cx = cy = h = w = baseline = 0.0
                serial = int(getattr(worker, 'serial_number', 0))
                calib = {
                    'index': int(cam_idx) if isinstance(cam_idx, int) else i,
                    'model': type(worker).__name__,                  # SingleZed / SingleRealsense
                    'serial': serial,
                    'intrinsics': {
                        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                        'width': int(w), 'height': int(h),
                        'baseline_m': baseline,                       # stereo baseline (ZED)
                        'distortion_model': 'none',                   # ZED LEFT eye is rectified
                        'distortion_coeffs': [],
                    },
                    # Camera-to-base 4x4 SE(3). To be filled by a future
                    # calibration step (apriltag / checkerboard); the schema
                    # is fixed now so downstream tools (rerun-io, etc.)
                    # don't need a follow-up migration.
                    'extrinsics': {
                        'measured': False,
                        'position_xyz_m': None,                       # [x, y, z]
                        'orientation_quat_xyzw': None,                # [qx, qy, qz, qw]
                    },
                }
                fname = calib_dir.joinpath(f'cam_{i}.json')
                fname.write_text(json.dumps(calib, indent=2))
        except Exception as e:
            print(f"[FrankaViveEnv] WARN: camera calib export failed: {e}")

    def _append_native_rate_stream(self, episode_id: int) -> None:
        """Append the current ObsAccumulator's raw streams to obs_native/.

        Layout decision (see ``dataset_meta.json``): **contiguous arrays
        + episode_ranges**, matching the existing ``data/`` group's
        episode_ends pattern. Each obs_native key is one variable-length
        sequence concatenated across episodes; ``meta/obs_native_episode_ranges``
        (E, 2) gives ``[start, end]`` per episode for slicing.
        """
        import zarr
        if self.obs_accumulator is None:
            return
        try:
            root = zarr.open(str(self.output_dir.joinpath('replay_buffer.zarr').absolute()), 'a')
            native = root.require_group('obs_native')
            meta = root.require_group('meta')

            # The fields we persist at native rate. Quaternion is the
            # already-sign-normalised raw stream (no SLERP — caller does
            # that at conversion time).
            from polymetis_franka_teleop.common.rotation_util import (
                quat_continuous_within_episode,
            )
            ts_robot = np.array(self.obs_accumulator.timestamps.get('robot0_joint_pos', []))
            if ts_robot.size == 0:
                return

            quat_raw = np.array(self.obs_accumulator.data.get(
                'robot0_eef_orientation_quat', np.zeros((0, 4))))
            if quat_raw.size > 0:
                quat_raw = quat_continuous_within_episode(quat_raw)

            robot_streams = {
                'ts_robot':                     ts_robot.astype(np.float64),
                'joint_position':               np.array(self.obs_accumulator.data['robot0_joint_pos']),
                'joint_velocity':               np.array(self.obs_accumulator.data['robot0_joint_vel']),
                'joint_torque_external':        np.array(self.obs_accumulator.data.get(
                    'robot0_joint_torque_external', np.zeros((len(ts_robot), 7)))),
                'ee_pose_axis_angle':           np.array(self.obs_accumulator.data['robot0_eef_pose']),
                'ee_orientation_quat':          quat_raw,
                'ee_linear_velocity':           np.array(self.obs_accumulator.data.get(
                    'robot0_eef_linear_velocity', np.zeros((len(ts_robot), 3)))),
                'ee_angular_velocity':          np.array(self.obs_accumulator.data.get(
                    'robot0_eef_angular_velocity', np.zeros((len(ts_robot), 3)))),
            }
            ts_gripper = np.array(self.obs_accumulator.timestamps.get('robot0_gripper_width', []))
            gripper_streams = {
                'ts_gripper':       ts_gripper.astype(np.float64),
                'gripper_position': np.array(self.obs_accumulator.data['robot0_gripper_width']).reshape(-1, 1),
            }

            # Append (or create) each contiguous array.
            for key, arr in {**robot_streams, **gripper_streams}.items():
                arr = np.asarray(arr)
                if key in native:
                    ds = native[key]
                    old_n = ds.shape[0]
                    new_shape = (old_n + arr.shape[0],) + arr.shape[1:]
                    ds.resize(*new_shape)
                    ds[old_n:] = arr
                else:
                    native.create_dataset(key, data=arr, chunks=(min(4096, arr.shape[0] or 1),)
                                          + arr.shape[1:])

            # Update episode_ranges (E, 2) for both robot + gripper streams.
            # Two ranges per episode because robot + gripper have different
            # lengths. Store as a single (E, 2, 2) so each episode row gives
            # [[r_start, r_end], [g_start, g_end]].
            r_end = native['ts_robot'].shape[0]
            r_start = r_end - ts_robot.shape[0]
            g_end = native['ts_gripper'].shape[0]
            g_start = g_end - ts_gripper.shape[0]
            row = np.array([[r_start, r_end], [g_start, g_end]], dtype=np.int64)

            if 'obs_native_episode_ranges' in meta:
                rng = meta['obs_native_episode_ranges']
                # Pad to episode_id+1 if needed (in case of out-of-order
                # writes — shouldn't happen but defensive).
                if rng.shape[0] < episode_id + 1:
                    rng.resize(episode_id + 1, 2, 2)
                rng[episode_id] = row
            else:
                ranges = np.zeros((episode_id + 1, 2, 2), dtype=np.int64)
                ranges[episode_id] = row
                meta.create_dataset('obs_native_episode_ranges', data=ranges,
                                    chunks=(64, 2, 2))
        except Exception as e:
            print(f'[FrankaViveEnv] WARN: native-rate dump failed: {e}')
