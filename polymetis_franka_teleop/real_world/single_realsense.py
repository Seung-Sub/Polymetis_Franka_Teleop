"""
SingleRealsense - Intel RealSense camera handler for UMI.

Based on UMI's UvcCamera pattern, adapted for Intel RealSense cameras.
Uses pyrealsense2 for camera capture and provides UMI-compatible interface.

Features:
- mp.Process based async operation
- SharedMemoryRingBuffer for frame sharing (lock-free FILO)
- Color, depth, and infrared stream support
- UMI-style timestamp handling with latency compensation
- Video recording support
- Advanced mode configuration

Usage:
    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number='123456789'
        ) as camera:
            frames = camera.get(k=5)
"""

from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from polymetis_franka_teleop.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from polymetis_franka_teleop.shared_memory.shared_ndarray import SharedNDArray
from polymetis_franka_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from polymetis_franka_teleop.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from polymetis_franka_teleop.real_world.video_recorder import VideoRecorder


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            receive_latency: float = 0.0,  # Latency compensation for timestamps
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAME
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
        # Note: Don't store shm_manager - it contains weakrefs that can't be pickled for spawn
        # Video recorder creates its own SharedMemoryManager inside run()

    @staticmethod
    def get_connected_devices_serial(include_l500=True):
        """Get serial numbers of connected RealSense devices.

        Args:
            include_l500: If True, include L500 series (L515). Default True.
        """
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                # Support D400 series (D435, D435i, D405, etc.)
                # and optionally L500 series (L515)
                if product_line == 'D400':
                    serials.append(serial)
                elif include_l500 and product_line == 'L500':
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        # Note: With spawn method, we cannot start video_recorder here because
        # SharedMemoryQueue contains weakrefs that can't be pickled.
        # Video recorder will be started inside run() method instead.
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        # Note: video_recorder is managed inside the subprocess (run method)
        # Setting stop_event will cause run() to exit and clean up video_recorder
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self, timeout=10.0):
        result = self.ready_event.wait(timeout=timeout)
        if not result:
            print(f'[SingleRealsense {self.serial_number}] Warning: start_wait timed out after {timeout}s')
        # Note: video_recorder is started inside run(), so it's already ready when ready_event is set

    def end_wait(self):
        self.join()
        # Note: video_recorder cleanup happens inside run() finally block

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale

    def start_recording(self, video_path: str, start_time: float = -1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })

    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)

        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color,
                w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth,
                w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                w, h, rs.format.y8, fps)

        # Start video recorder inside subprocess (required for spawn method)
        # Create a new SharedMemoryManager for the video recorder
        video_shm_manager = SharedMemoryManager()
        video_shm_manager.start()
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=video_shm_manager,
            data_example=data_example)
        self.video_recorder.start_wait()

        try:
            print(f'[SingleRealsense {self.serial_number}] Enabling device...')
            rs_config.enable_device(self.serial_number)

            # start pipeline
            print(f'[SingleRealsense {self.serial_number}] Starting pipeline...')
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            print(f'[SingleRealsense {self.serial_number}] Pipeline started!')

            # report global time (may not be available on all cameras like L515)
            try:
                d = pipeline_profile.get_device().first_color_sensor()
                d.set_option(rs.option.global_time_enabled, 1)
            except RuntimeError as e:
                print(f'[SingleRealsense {self.serial_number}] Warning: Could not enable global time: {e}')

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get intrinsics
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale

            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)

                # grab data
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                if self.enable_depth:
                    data['depth'] = np.asarray(
                        frameset.get_depth_frame().get_data())
                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())

                # apply transform
                put_data = data
                if self.transform is not None:
                    try:
                        put_data = self.transform(dict(data))
                    except Exception as e:
                        print(f'[SingleRealsense {self.serial_number}] Transform error: {e}')
                        import traceback
                        traceback.print_exc()

                # Calibrated timestamp with latency compensation (UMI style)
                calibrated_time = receive_time - self.receive_latency

                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            next_global_idx=put_idx,
                            allow_negative=True
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

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'],
                        frame_time=calibrated_time)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1) if duration > 0 else 0
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                iter_idx += 1
        finally:
            # Only stop video_recorder if it was started (stop_event is created in start())
            if self.video_recorder is not None and self.video_recorder.is_started:
                self.video_recorder.stop()
                self.video_recorder.end_wait()
            # Shutdown video shared memory manager
            try:
                video_shm_manager.shutdown()
            except Exception:
                pass
            rs_config.disable_all_streams()
            self.ready_event.set()

        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')
