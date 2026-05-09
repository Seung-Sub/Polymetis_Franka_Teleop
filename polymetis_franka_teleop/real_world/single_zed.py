"""SingleZed — UMI-style ZED camera worker process.

Mirrors `SingleRealsense` API surface so it can be a drop-in replacement
inside `franka_vive_env` / `franka_policy_env`. Uses pyzed.sl for capture,
LEFT-eye only (matching DROID + GR00T training convention).

ZED cameras report `BGRA` from `retrieve_image(VIEW.LEFT)` and a per-frame
hardware timestamp in nanoseconds; we convert both to UMI's expected
`bgr24 + seconds float64` so downstream ImageTransform / VideoRecorder /
RingBuffer all stay binary-compatible with the RealSense path.
"""
from typing import Optional, Callable, Dict
import enum
import time
import numpy as np
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

import pyzed.sl as sl

from polymetis_franka_teleop.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from polymetis_franka_teleop.shared_memory.shared_ndarray import SharedNDArray
from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from polymetis_franka_teleop.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from polymetis_franka_teleop.real_world.video_recorder import VideoRecorder


_RES_MAP = {
    (1280, 720):  sl.RESOLUTION.HD720,
    (1920, 1080): sl.RESOLUTION.HD1080,
    (2208, 1242): sl.RESOLUTION.HD2K,
    (672, 376):   sl.RESOLUTION.VGA,
}


class Command(enum.Enum):
    SET_BRIGHTNESS = 0
    SET_EXPOSURE = 1
    SET_GAIN = 2
    SET_WHITEBALANCE = 3
    START_RECORDING = 4
    STOP_RECORDING = 5
    RESTART_PUT = 6


class SingleZed(mp.Process):
    MAX_PATH_LENGTH = 4096

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            get_max_k=30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            receive_latency: float = 0.0,
            verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps
        resolution = tuple(resolution)
        if resolution not in _RES_MAP:
            raise ValueError(
                f"unsupported ZED resolution {resolution!r}; pick one of {list(_RES_MAP.keys())}"
            )

        # Ring buffer schema — matches SingleRealsense ('color' = HxWx3 bgr8).
        h, w = resolution[1], resolution[0]
        examples = {
            'color': np.empty(shape=(h, w, 3), dtype=np.uint8),
            'camera_capture_timestamp': 0.0,
            'camera_receive_timestamp': 0.0,
            'timestamp': 0.0,
            'step_idx': 0,
        }

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # Command queue (string fields use fixed-size numpy strings)
        cmd_examples = {
            'cmd': Command.SET_EXPOSURE.value,
            'option_value': 0.0,
            'video_path': np.array('a' * self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0,
        }
        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=cmd_examples, buffer_size=128,
        )

        # Intrinsics: [fx, fy, cx, cy, height, width, baseline]
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64,
        )
        intrinsics_array.get()[:] = 0

        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, codec='h264', input_pix_fmt='bgr24',
                crf=18, thread_type='FRAME', thread_count=1,
            )

        # Persisted attributes
        self.serial_number = int(serial_number)
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.put_start_time = None

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices_serial():
        """Return ZED serial numbers physically attached and in AVAILABLE state."""
        return sorted(
            int(d.serial_number)
            for d in sl.Camera.get_device_list()
            if d.camera_state == sl.CAMERA_STATE.AVAILABLE
        )

    # ---------- context manager ----------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ---------- user API ----------
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self, timeout=10.0):
        ok = self.ready_event.wait(timeout=timeout)
        if not ok:
            print(f'[SingleZed {self.serial_number}] Warning: start_wait timed out after {timeout}s')

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        return (self.ring_buffer.get(out=out)
                if k is None else self.ring_buffer.get_last_k(k, out=out))

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def set_exposure(self, exposure=None, gain=None):
        if exposure is None and gain is None:
            self.command_queue.put({'cmd': Command.SET_EXPOSURE.value, 'option_value': -1.0})
            self.command_queue.put({'cmd': Command.SET_GAIN.value, 'option_value': -1.0})
        else:
            if exposure is not None:
                self.command_queue.put({'cmd': Command.SET_EXPOSURE.value, 'option_value': float(exposure)})
            if gain is not None:
                self.command_queue.put({'cmd': Command.SET_GAIN.value, 'option_value': float(gain)})

    def set_white_balance(self, white_balance=None):
        self.command_queue.put({
            'cmd': Command.SET_WHITEBALANCE.value,
            'option_value': float(-1 if white_balance is None else white_balance),
        })

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, cx, cy = self.intrinsics_array.get()[:4]
        m = np.eye(3); m[0, 0] = fx; m[1, 1] = fy; m[0, 2] = cx; m[1, 2] = cy
        return m

    def start_recording(self, video_path: str, start_time: float = -1):
        if len(video_path.encode('utf-8')) > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time,
        })

    def stop_recording(self):
        self.command_queue.put({'cmd': Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time,
        })

    # ---------- subprocess loop ----------
    def run(self):
        threadpool_limits(1)
        cv2.setNumThreads(1)

        # Per-process VideoRecorder spawn (matches SingleRealsense pattern).
        video_shm_manager = SharedMemoryManager()
        video_shm_manager.start()
        h, w = self.resolution[1], self.resolution[0]
        self.video_recorder.start(
            shm_manager=video_shm_manager,
            data_example=np.empty(shape=(h, w, 3), dtype=np.uint8),
        )
        self.video_recorder.start_wait()

        cam = sl.Camera()
        try:
            init = sl.InitParameters()
            init.set_from_serial_number(self.serial_number)
            init.camera_resolution = _RES_MAP[self.resolution]
            init.camera_fps = self.capture_fps
            init.depth_mode = sl.DEPTH_MODE.NONE
            init.coordinate_units = sl.UNIT.METER
            err = cam.open(init)
            # POTENTIAL_CALIBRATION_ISSUE is a *warning*, not an error in the
            # ZED SDK: the camera opens and streams just fine, the SDK is only
            # cautioning that stereo depth precision may be off (calibration
            # drift, temperature, or a different sensor batch than the cal
            # file expected). We use LEFT eye only and never compute depth,
            # so this warning is safe to ignore. Treating it as fatal causes
            # MultiZed (concurrent open of two cameras) to crash one of them
            # -- observed on KIST ZED 2i (sn 33538770) on 2026-05-09.
            if err == sl.ERROR_CODE.SUCCESS:
                pass
            elif str(err).strip().upper() == 'POTENTIAL CALIBRATION ISSUE':
                print(f'[SingleZed {self.serial_number}] WARN: POTENTIAL CALIBRATION '
                      f'ISSUE -- continuing (LEFT eye only, no depth needed). '
                      f'If you need depth, refresh /usr/local/zed/settings/SN*.conf '
                      f'(install_from_scratch.md Phase F-2).')
            else:
                raise RuntimeError(f'ZED open failed sn={self.serial_number}: {err}')

            ci = cam.get_camera_information()
            cp = ci.camera_configuration.calibration_parameters.left_cam
            arr = self.intrinsics_array.get()
            arr[0] = cp.fx; arr[1] = cp.fy; arr[2] = cp.cx; arr[3] = cp.cy
            arr[4] = h; arr[5] = w
            try:
                arr[6] = ci.camera_configuration.calibration_parameters.stereo_transform.get_translation().get()[0]
            except Exception:
                arr[6] = 0.0

            put_idx = None
            put_start_time = self.put_start_time or time.time()
            iter_idx = 0
            t_loop = time.time()
            mat = sl.Mat()
            rt = sl.RuntimeParameters()
            # 1-second moving-window FPS counter — instantaneous 1/Δt prints
            # are useless because USB hiccups create backlogs that the next few
            # iters drain in milliseconds, making the rate appear to jump
            # 5↔80 even when the average is steady at the configured rate.
            _fps_window_start = t_loop
            _fps_window_iters = 0

            while not self.stop_event.is_set():
                if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                    continue
                receive_time = time.time()
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                bgra = mat.get_data()  # HxWx4
                color = bgra[..., :3]  # BGR

                hw_ts_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                capture_ts = hw_ts_ns / 1e9 if hw_ts_ns > 0 else receive_time

                data = {
                    'color': color,
                    'camera_capture_timestamp': capture_ts,
                    'camera_receive_timestamp': receive_time,
                }

                put_data = data
                if self.transform is not None:
                    try:
                        put_data = self.transform(dict(data))
                    except Exception as e:
                        print(f'[SingleZed {self.serial_number}] Transform error: {e}')

                # Latency-compensated timestamp (UMI convention)
                calibrated_time = receive_time - self.receive_latency

                if self.put_downsample:
                    _, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time], start_time=put_start_time,
                        dt=1 / self.put_fps, next_global_idx=put_idx, allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = calibrated_time
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = calibrated_time
                    self.ring_buffer.put(put_data, wait=False)

                if iter_idx == 0:
                    self.ready_event.set()

                vis_data = put_data if self.vis_transform == self.transform \
                    else (data if self.vis_transform is None else self.vis_transform(dict(data)))
                self.vis_ring_buffer.put(vis_data, wait=False)

                rec_data = put_data if self.recording_transform == self.transform \
                    else (data if self.recording_transform is None else self.recording_transform(dict(data)))
                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], frame_time=calibrated_time)

                # Commands
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                for i in range(n_cmd):
                    cmd = commands['cmd'][i]
                    val = float(commands['option_value'][i])
                    if cmd == Command.SET_EXPOSURE.value:
                        if val < 0:
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
                        else:
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(val))
                    elif cmd == Command.SET_GAIN.value:
                        if val >= 0:
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(val))
                    elif cmd == Command.SET_BRIGHTNESS.value:
                        cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, int(val))
                    elif cmd == Command.SET_WHITEBALANCE.value:
                        if val < 0:
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
                        else:
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
                            cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, int(val))
                    elif cmd == Command.START_RECORDING.value:
                        # SharedMemoryQueue stores string fields as fixed-width
                        # numpy <U4096 with trailing null padding — strip it.
                        raw = commands['video_path'][i]
                        path = str(raw).rstrip('\x00').rstrip()
                        st = float(commands['recording_start_time'][i])
                        print(f'[SingleZed {self.serial_number}] START_RECORDING path={path!r} (raw type={type(raw).__name__}) start_time={st}', flush=True)
                        self.video_recorder.start_recording(path, start_time=None if st < 0 else st)
                        print(f'[SingleZed {self.serial_number}] video_recorder.is_ready()={self.video_recorder.is_ready()}', flush=True)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = float(commands['put_start_time'][i])

                if self.verbose:
                    _fps_window_iters += 1
                    now = time.time()
                    if now - _fps_window_start >= 1.0:
                        avg_fps = _fps_window_iters / (now - _fps_window_start)
                        print(f'[SingleZed {self.serial_number}] FPS {avg_fps:.1f} (1s avg)')
                        _fps_window_start = now
                        _fps_window_iters = 0
                iter_idx += 1
        finally:
            # Ensure any in-progress recording is finalised (writes mp4 trailer)
            # BEFORE we tear down the encoder subprocess. The main loop above
            # exits as soon as stop_event is set, so a STOP_RECORDING command
            # queued just before stop() may not have been pulled — call it
            # idempotently here.
            try:
                if self.video_recorder is not None and self.video_recorder.is_ready():
                    self.video_recorder.stop_recording()
            except Exception:
                pass
            try:
                if self.video_recorder is not None and self.video_recorder.is_started:
                    self.video_recorder.stop()
                    self.video_recorder.end_wait()
            except Exception:
                pass
            try:
                video_shm_manager.shutdown()
            except Exception:
                pass
            cam.close()
            self.ready_event.set()

        if self.verbose:
            print(f'[SingleZed {self.serial_number}] Exiting worker process.')
