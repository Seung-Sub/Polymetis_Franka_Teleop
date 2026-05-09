"""ArtGripperController — Hyundai Motors ART gripper controller.

Drop-in replacement for `FrankaGripperController` so `franka_vive_env` /
`franka_policy_env` can switch between Franka Hand and ART by selecting
the controller class. The public API (start/stop, schedule_waypoint,
goto, grasp, get_state, ring_buffer schema) is **identical** to
FrankaGripperController so downstream code is unchanged.

Implementation differences:
  - Talks to the ART standalone daemon over TCP localhost:50053 via
    `art_gripper_client.ArtGripperInterface` (no ZeroRPC/NUC hop).
  - Width range 0..0.100m (vs Franka 0..0.080m) — passed through verbatim
    to the firmware which clamps internally.
  - Pose DOF locked at boot value (KIST runs the 2-finger variant).
"""
from __future__ import annotations

import os
import sys
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


class ArtGripperController(mp.Process):
    """Discrete (OPEN/CLOSE) controller for the Hyundai Motors ART gripper.

    Same interface as FrankaGripperController so callers can swap. Internally
    it bridges to the standalone TCP daemon via art_gripper_client.
    """

    # ART hardware envelope
    MAX_WIDTH = 0.100   # meters
    MIN_WIDTH = 0.000   # meters
    DEFAULT_SPEED = 0.15   # m/s (= 150 mm/s — matches client's DEFAULT_WIDTH_SPEED_MM_S)
    DEFAULT_FORCE = 30.0   # N

    # Match ViveTeleopProcess.GripperCommand enum
    GRIPPER_CMD_NONE = 0
    GRIPPER_CMD_OPEN = 1
    GRIPPER_CMD_CLOSE = 2

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            host: str = "127.0.0.1",
            port: int = 50053,
            frequency: int = 60,
            move_max_speed: float = 0.15,
            default_force: float = 30.0,
            get_max_k: int = None,
            command_queue_size: int = 1024,
            launch_timeout: float = 3.0,
            receive_latency: float = 0.0,
            verbose: bool = False,
            teleop_mode: bool = False,
            teleop_ring_buffer: SharedMemoryRingBuffer = None,
            gripper_open_width: float = 0.095,
            gripper_close_width: float = 0.005,
            art_pypath: str = None,
    ):
        if teleop_mode and teleop_ring_buffer is None:
            raise ValueError("teleop_ring_buffer is required when teleop_mode=True")
        super().__init__(name="ArtGripperController")

        self.host = host
        self.port = port
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.default_force = default_force
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.teleop_mode = teleop_mode
        self.teleop_ring_buffer = teleop_ring_buffer
        self.gripper_open_width = gripper_open_width
        self.gripper_close_width = gripper_close_width
        # Custom python path for art_gripper_client (in case package isn't pip-installed)
        self.art_pypath = art_pypath or os.environ.get(
            "ART_GRIPPER_PYPATH", os.path.expanduser("~/Hyundai_motors_Gripper/python"))

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # Command queue
        cmd_example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0,
            'speed': self.DEFAULT_SPEED,
            'force': self.DEFAULT_FORCE,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=cmd_example, buffer_size=command_queue_size)

        # State ring buffer — KEY NAMES MUST MATCH FrankaGripperController
        # so franka_vive_env's get_obs() doesn't need to branch.
        state_example = {
            'gripper_width': 0.0,
            'gripper_is_grasped': False,
            'gripper_is_moving': False,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, examples=state_example,
            get_max_k=get_max_k, get_time_budget=0.2,
            put_desired_frequency=frequency)

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ---------- launch ----------
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ArtGripperController] spawned PID={self.pid}")

    def stop(self, wait=True):
        self.input_queue.put({'cmd': Command.SHUTDOWN.value})
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ---------- commands ----------
    def schedule_waypoint(self, width: float, target_time: float):
        width = float(np.clip(width, self.MIN_WIDTH, self.MAX_WIDTH))
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': width, 'target_time': target_time,
        })

    def goto(self, width: float, speed: float = None):
        speed = self.DEFAULT_SPEED if speed is None else speed
        width = float(np.clip(width, self.MIN_WIDTH, self.MAX_WIDTH))
        self.input_queue.put({
            'cmd': Command.GOTO.value,
            'target_pos': width, 'speed': speed,
        })

    def grasp(self, speed: float = None, force: float = None):
        self.input_queue.put({
            'cmd': Command.GRASP.value,
            'speed': self.DEFAULT_SPEED if speed is None else speed,
            'force': self.default_force if force is None else force,
        })

    def restart_put(self, start_time: float):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value, 'target_time': start_time,
        })

    # ---------- state ----------
    def get_state(self, k=None, out=None):
        return (self.ring_buffer.get(out=out)
                if k is None else self.ring_buffer.get_last_k(k=k, out=out))

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ---------- subprocess loop ----------
    def run(self):
        gripper = None
        try:
            # Lazy import + sys.path inject so the daemon repo can live anywhere.
            try:
                from art_gripper_client import ArtGripperInterface
            except ImportError:
                if self.art_pypath and self.art_pypath not in sys.path:
                    sys.path.insert(0, self.art_pypath)
                from art_gripper_client import ArtGripperInterface

            gripper = ArtGripperInterface(ip_address=self.host, port=self.port,
                                          auto_motor_on=True)
            if self.verbose:
                print(f"[ArtGripperController] connected {self.host}:{self.port} "
                      f"max_width={gripper.metadata.max_width:.3f}m teleop={self.teleop_mode}")

            # Open the gripper at boot — matches Isaac-GR00T's franka_env_kist.reset()
            # behavior, where the gripper is reset alongside the arm's auto-home so
            # the robot starts each session from a known fully-open state.
            try:
                gripper.goto(width=self.gripper_open_width,
                             speed=self.move_max_speed,
                             force=self.default_force,
                             blocking=True)
                if self.verbose:
                    print(f"[ArtGripperController] startup: opened to "
                          f"{self.gripper_open_width:.3f}m", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"[ArtGripperController] startup open failed: {e}", flush=True)

            width_threshold = (self.gripper_open_width + self.gripper_close_width) / 2

            init = gripper.get_state()
            current_is_open = init.width > width_threshold

            pending_commands = []  # [(target_is_open, mono_t)]
            last_teleop_ts = 0.0
            last_teleop_cmd = self.GRIPPER_CMD_NONE

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            dt = 1 / self.frequency
            overrun_count = 0  # iters where precise_wait was already past t_end

            while keep_running:
                t_now = time.monotonic()

                # ---- explicit goto/grasp/shutdown from main process. Polled
                # in BOTH teleop and normal modes so the demo can override
                # regardless. Critical: must also honour SHUTDOWN here, or
                # stop() will hang forever in teleop mode (the legacy NORMAL-
                # mode handler at the bottom of this loop never sees it
                # because we drained it up here).
                try:
                    explicit_batch = self.input_queue.get_all()
                    n_explicit = len(explicit_batch.get('cmd', []))
                    for j in range(n_explicit):
                        ec = int(explicit_batch['cmd'][j])
                        if ec == Command.SHUTDOWN.value:
                            keep_running = False
                            break
                        if ec == Command.GOTO.value:
                            target_pos = float(explicit_batch['target_pos'][j])
                            speed = float(explicit_batch.get('speed', [self.move_max_speed])[j])
                            try:
                                gripper.goto(width=target_pos, speed=speed,
                                             force=self.default_force, blocking=False)
                                current_is_open = target_pos > width_threshold
                                # CRITICAL: Reset last_teleop_cmd so the next
                                # teleop CLOSE/OPEN from Vive is honoured even
                                # if it matches the pre-HOME value. Without
                                # this, the user has to press trigger 3 times
                                # after a HOME-while-CLOSED to actually close
                                # again (1st = stale-equal, ignored; 2nd =
                                # OPEN to already-open, no change; 3rd =
                                # actually closes).
                                last_teleop_cmd = self.GRIPPER_CMD_NONE
                                if self.verbose:
                                    print(f"[ArtGripperController] explicit "
                                          f"GOTO {target_pos:.3f}m "
                                          f"(reset teleop cmd state)",
                                          flush=True)
                            except Exception as e:
                                if self.verbose:
                                    print(f"[ArtGripperController] explicit GOTO err: {e}",
                                          flush=True)
                    if not keep_running:
                        break  # exit while-loop immediately
                except Empty:
                    pass

                # ---- TELEOP ----
                if self.teleop_mode:
                    try:
                        ts = self.teleop_ring_buffer.get()
                        teleop_ts = ts.get('teleop_timestamp', 0.0)
                        cmd = int(ts.get('gripper_command', self.GRIPPER_CMD_NONE))
                        gstate = float(ts.get('gripper_state', 0.0))
                        target_w = (self.gripper_close_width if gstate > 0.5
                                    else self.gripper_open_width)
                        if teleop_ts > last_teleop_ts:
                            last_teleop_ts = teleop_ts
                            # If cmd just transitioned to NONE (e.g. ViveTeleopProcess
                            # cleared it on HOME end, or after one-shot toggle),
                            # also reset our latch so the next non-NONE cmd is
                            # honoured even if it equals the pre-HOME value.
                            # Without this, the first trigger press after a HOME
                            # taken from a CLOSED state ends up as a no-op
                            # because cmd==CLOSE matches the stale last_teleop_cmd.
                            if cmd == self.GRIPPER_CMD_NONE:
                                if last_teleop_cmd != self.GRIPPER_CMD_NONE:
                                    last_teleop_cmd = self.GRIPPER_CMD_NONE
                            elif cmd != last_teleop_cmd:
                                try:
                                    if cmd == self.GRIPPER_CMD_CLOSE:
                                        gripper.grasp(width=target_w,
                                                      speed=self.move_max_speed,
                                                      force=self.default_force,
                                                      timeout_s=2.0)
                                        current_is_open = False
                                    elif cmd == self.GRIPPER_CMD_OPEN:
                                        gripper.goto(width=target_w,
                                                     speed=self.move_max_speed,
                                                     force=self.default_force,
                                                     blocking=False)
                                        current_is_open = True
                                    last_teleop_cmd = cmd
                                except Exception as e:
                                    if self.verbose:
                                        print(f"[ArtGripperController] teleop err: {e}")
                    except Exception as e:
                        if self.verbose and iter_idx % 100 == 0:
                            print(f"[ArtGripperController] teleop read err: {e}")

                # ---- NORMAL: pending command execution ----
                else:
                    try:
                        pre_state = gripper.get_state()
                        pre_is_moving = pre_state.is_in_motion
                        pre_is_grasped = pre_state.is_grasped
                    except Exception:
                        pre_is_moving = False
                        pre_is_grasped = False

                    to_remove = []
                    for i, (target_is_open, t_mono) in enumerate(pending_commands):
                        if t_now < t_mono:
                            continue
                        if pre_is_moving:
                            continue  # keep, retry next cycle
                        if target_is_open != current_is_open:
                            try:
                                if target_is_open:
                                    gripper.goto(
                                        width=self.gripper_open_width,
                                        speed=self.move_max_speed,
                                        force=self.default_force,
                                        blocking=False,
                                    )
                                    current_is_open = True
                                else:
                                    if not pre_is_grasped:
                                        gripper.grasp(
                                            width=self.gripper_close_width,
                                            speed=self.move_max_speed,
                                            force=self.default_force,
                                            timeout_s=2.0,
                                        )
                                    current_is_open = False
                            except Exception as e:
                                if self.verbose:
                                    print(f"[ArtGripperController] cmd err: {e}")
                        to_remove.append(i)
                    for i in reversed(to_remove):
                        pending_commands.pop(i)

                # ---- state publish ----
                try:
                    s = gripper.get_state()
                    cw = s.width
                    is_grasped = s.is_grasped
                    is_moving = s.is_in_motion
                    # state-sync to avoid duplicate cmd
                    actual_open = cw > width_threshold
                    if actual_open != current_is_open and not is_moving:
                        current_is_open = actual_open
                except Exception:
                    cw = self.gripper_open_width if current_is_open else self.gripper_close_width
                    is_grasped = False
                    is_moving = False
                t_recv = time.time()
                self.ring_buffer.put({
                    'gripper_width': cw,
                    'gripper_is_grasped': bool(is_grasped),
                    'gripper_is_moving': bool(is_moving),
                    'gripper_receive_timestamp': t_recv,
                    'gripper_timestamp': t_recv - self.receive_latency,
                })

                # ---- pull commands ----
                try:
                    cmds = self.input_queue.get_all()
                    n = len(cmds['cmd'])
                except Empty:
                    n = 0
                for j in range(n):
                    cmd = int(cmds['cmd'][j])
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    if cmd == Command.SCHEDULE_WAYPOINT.value:
                        tp = float(cmds['target_pos'][j])
                        tt = float(cmds['target_time'][j])
                        target_open = tp > width_threshold
                        t_mono = time.monotonic() - time.time() + tt
                        pending_commands.append((target_open, t_mono))
                        pending_commands.sort(key=lambda x: x[1])
                    elif cmd == Command.GOTO.value:
                        tp = float(cmds['target_pos'][j])
                        sp = float(cmds.get('speed', [self.DEFAULT_SPEED] * n)[j])
                        try:
                            gripper.goto(width=tp, speed=sp,
                                         force=self.default_force, blocking=False)
                            current_is_open = tp > width_threshold
                        except Exception as e:
                            if self.verbose:
                                print(f"[ArtGripperController] goto err: {e}")
                    elif cmd == Command.GRASP.value:
                        sp = float(cmds.get('speed', [self.DEFAULT_SPEED] * n)[j])
                        fc = float(cmds.get('force', [self.default_force] * n)[j])
                        try:
                            gripper.grasp(width=self.gripper_close_width,
                                          speed=sp, force=fc, timeout_s=2.0)
                            current_is_open = False
                        except Exception as e:
                            if self.verbose:
                                print(f"[ArtGripperController] grasp err: {e}")
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = float(cmds['target_time'][j]) - time.time() + time.monotonic()
                        iter_idx = 1
                        pending_commands.clear()

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                t_end = t_start + dt * iter_idx
                if time.monotonic() > t_end:
                    overrun_count += 1
                precise_wait(t_end=t_end, time_func=time.monotonic)
                if self.verbose and iter_idx % (self.frequency * 5) == 0:
                    # every ~5 s, report loop health
                    print(f"[ArtGripperController] iter={iter_idx} target={self.frequency}Hz "
                          f"overruns={overrun_count}/{self.frequency * 5} "
                          f"(last 5s)")
                    overrun_count = 0

        except Exception as e:
            print(f"[ArtGripperController] fatal: {e}")
            import traceback; traceback.print_exc()
            self.ready_event.set()
        finally:
            self.ready_event.set()
            if gripper is not None:
                try:
                    gripper.close()
                except Exception:
                    pass
            if self.verbose:
                print("[ArtGripperController] disconnected")
