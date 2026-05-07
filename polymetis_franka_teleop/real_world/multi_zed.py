"""MultiZed — manager for multiple SingleZed processes.

Drop-in replacement for `MultiRealsense` with the same `__enter__/__exit__`,
`get`, `start_recording`, etc. so that `franka_vive_env`/`franka_policy_env`
can swap camera backend by changing one import.
"""
from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from polymetis_franka_teleop.real_world.single_zed import SingleZed
from polymetis_franka_teleop.real_world.video_recorder import VideoRecorder


def _repeat_to_list(x, n, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x


class MultiZed:
    def __init__(
            self,
            serial_numbers: Optional[List[int]] = None,
            shm_manager: Optional[SharedMemoryManager] = None,
            resolution=(1280, 720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            get_max_k=30,
            transform: Optional[Union[Callable, List[Callable]]] = None,
            vis_transform: Optional[Union[Callable, List[Callable]]] = None,
            recording_transform: Optional[Union[Callable, List[Callable]]] = None,
            video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
            receive_latency: float = 0.0,
            verbose=False,
    ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleZed.get_connected_devices_serial()
        n = len(serial_numbers)
        if n == 0:
            raise RuntimeError('MultiZed: no ZED cameras detected — check USB power and pyzed.')

        transform = _repeat_to_list(transform, n, Callable)
        vis_transform = _repeat_to_list(vis_transform, n, Callable)
        recording_transform = _repeat_to_list(recording_transform, n, Callable)
        video_recorder = _repeat_to_list(video_recorder, n, VideoRecorder)

        cameras = {}
        for i, sn in enumerate(serial_numbers):
            cameras[int(sn)] = SingleZed(
                shm_manager=shm_manager,
                serial_number=sn,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                get_max_k=get_max_k,
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                receive_latency=receive_latency,
                verbose=verbose,
            )
        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        return all(c.is_ready for c in self.cameras.values())

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for c in self.cameras.values():
            c.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for c in self.cameras.values():
            c.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        for c in self.cameras.values():
            c.start_wait()

    def stop_wait(self):
        for c in self.cameras.values():
            c.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = dict()
        for i, c in enumerate(self.cameras.values()):
            out[i] = c.get(k=k, out=out.get(i))
        return out

    def get_vis(self, out=None):
        results = []
        for i, c in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = {k: v[i:i + 1].reshape(v.shape[1:]) for k, v in out.items()}
            this_out = c.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = {k: np.stack([x[k] for x in results]) for k in results[0].keys()}
        return out

    def set_exposure(self, exposure=None, gain=None):
        for c in self.cameras.values():
            c.set_exposure(exposure=exposure, gain=gain)

    def set_white_balance(self, white_balance=None):
        for c in self.cameras.values():
            c.set_white_balance(white_balance=white_balance)

    def get_intrinsics(self):
        return np.array([c.get_intrinsics() for c in self.cameras.values()])

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = [str(video_dir / f'{i}.mp4') for i in range(self.n_cameras)]
        assert len(video_path) == self.n_cameras
        for i, c in enumerate(self.cameras.values()):
            c.start_recording(video_path[i], start_time)

    def stop_recording(self):
        for c in self.cameras.values():
            c.stop_recording()

    def restart_put(self, start_time):
        for c in self.cameras.values():
            c.restart_put(start_time)
