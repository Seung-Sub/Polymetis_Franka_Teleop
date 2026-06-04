#!/usr/bin/env python3
"""KIST Franka + ART + ZED real-robot DP eval (UMI-style, latency-matched).

This script runs in the ``groot-client`` env to keep the Polymetis stack
binary-compatible. The DP policy lives in a separate ``umi`` env (torch
2.11+cu128, sm_120 GPU) and is contacted over ZMQ on localhost. Launch the
daemon via ``bin/run_eval_dp.sh`` which manages both processes.

Latency-handling logic (mirrors universal_manipulation_interface/eval_real.py
and the appendix in ideal_DP_eval.md):

  (1) Observation latency matching — BimanualUmiEnv.get_obs style
      a. Each Polymetis controller is constructed with its own
         ``receive_latency`` from install/latency_calibration.json so that
         every sample's stored ``timestamp`` is the true sensor-capture
         time (host_receive - latency).
      b. The pair-mutual-best-error rule picks the camera whose latest
         frame timestamp leaves the smallest summed time-gap to a "<=t"
         frame in the other cameras (ZED 60Hz pair has ≤16ms misalignment).
      c. ``obs_grid = last_t - [N-1, ..., 0] * dt`` defines the obs
         timestamps.
      d. Cameras: nearest-neighbor frame index.
      e. Robot pose: PoseInterpolator (SE(3) linear+SLERP) on robot_timestamp.
      f. Gripper width: 1D linear interp on gripper_timestamp.

  (2) Action latency matching — eval_real.py main-loop style
      a. ``action_timestamps = obs_grid[-1] + k*dt`` — the policy is told
         its first action lands at the moment of the last observation.
      b. ``is_new = action_timestamps > time.time() + action_exec_latency``
         discards actions that are already in the past (~0.01s margin).
      c. ``schedule_waypoint(pose, target_time - robot_action_latency)``
         and ``gripper.schedule_waypoint(width, target_time -
         gripper_action_latency)`` so each device starts moving early and
         arrives exactly at the intended ``target_time``.
      d. ``precise_wait(t_cycle_end - frame_latency)`` yields ~16ms early
         so the next ``get_obs()`` grabs a fresh ZED frame.

Action/obs preprocessing is imported directly from
``scripts_real.convert_to_dp_image`` so the eval-time encoding is
byte-identical to what was used for training (no re-implementation drift).

Usage (auto via wrapper):
    bash bin/run_eval_dp.sh checkpoints/dp/unet_epoch100.ckpt --max_duration 60

Direct (after daemon is already running on port 5555):
    python scripts_real/eval_dp_real.py --daemon_port 5555 --max_duration 60
"""
import sys, os, time, json
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ---- DEBUG: force unbuffered stdout/stderr + traceback hooks ----
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import faulthandler, signal, traceback, atexit
import quant.calib.dp_globalvar as globalvar 
import torch

faulthandler.enable()

def _dbg_sig_handler(signum, frame):
    print(f"\n[!!! eval_dp_real] got signal {signum} ({signal.Signals(signum).name})", flush=True, file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    if signum in (signal.SIGTERM, signal.SIGINT):
        sys.exit(128 + signum)

for _sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGPIPE, signal.SIGUSR1):
    try:
        signal.signal(_sig, _dbg_sig_handler)
    except Exception:
        pass

def _on_exit():
    print(f"[!!! eval_dp_real] atexit fired (pid={os.getpid()})", flush=True, file=sys.stderr)
atexit.register(_on_exit)

def _excepthook(exctype, value, tb):
    print(f"[!!! eval_dp_real] UNCAUGHT EXCEPTION", flush=True, file=sys.stderr)
    traceback.print_exception(exctype, value, tb, file=sys.stderr)
    sys.stderr.flush()
sys.excepthook = _excepthook
# ---- end DEBUG ----

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import subprocess, signal, atexit

from multiprocessing.managers import SharedMemoryManager
import click
import numpy as np
import cv2
import zmq
import msgpack
import msgpack_numpy as mnp
mnp.patch()
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image

from polymetis_franka_teleop.real_world.franka_interpolation_controller import (
    FrankaInterpolationController,
)
from polymetis_franka_teleop.real_world.multi_zed import MultiZed
from polymetis_franka_teleop.real_world.art_gripper_controller import ArtGripperController
from polymetis_franka_teleop.common.interpolation_util import (
    PoseInterpolator, get_interp1d,
)
from polymetis_franka_teleop.common.precise_sleep import precise_wait
from polymetis_franka_teleop.common.latency_config import (
    get_camera_obs_latency, get_robot_obs_latency, get_gripper_obs_latency,
    get_robot_action_latency, get_gripper_action_latency,
)

def resolve_latencies(camera_backend: str, gripper_backend: str) -> dict:
    return {
        'camera_obs_latency':   get_camera_obs_latency(camera_backend),
        'robot_obs_latency':    get_robot_obs_latency(),
        'gripper_obs_latency':  get_gripper_obs_latency(gripper_backend),
        'robot_action_latency': get_robot_action_latency(),
        'gripper_action_latency': get_gripper_action_latency(gripper_backend),
    }

# CRITICAL: matches compute_ready_pose('diffusion', 'art') from franka_vive_env.
# DP data collection (start_teleop.sh --data_format diffusion --gripper_backend art)
# uses the standard Franka home for joints 1-6 + joint-7=0 (ART) = the values
# below. A previous hard-coded version had j7=π/4 which put the wrist 45°
# off the training distribution — policy obs OOD => bad actions.
# Inlining (not importing compute_ready_pose) avoids pulling in franka_vive_env's
# heavy vive_shared_memory / vive_teleop_process dependencies.
FRANKA_HOME_JOINTS = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, 0.0]
ART_MAX_WIDTH = 0.095
VIS_JPEG_PATH = '/tmp/dp_eval_vis.jpg'

# Workspace bounds (Franka reachable, KIST table-mount): xyz in robot base frame.
# These should bracket every action sample produced during teleop. Action values
# outside trigger a safe stop. Update if you change ready pose / table layout.
WORKSPACE_BOUNDS_M = {
    'x_min': 0.10, 'x_max': 0.75,
    'y_min': -0.40, 'y_max': 0.40,
    'z_min': 0.02, 'z_max': 0.55,
}
MAX_POS_STEP_PER_DT_M = 0.300   # 300mm/100ms — accept training-distribution first-action jumps
# Original 60mm rejected normal policy output: franka_peg task has action mean ~200mm below ready,
# so first inference at ready legitimately commands a 200mm+ jump toward the peg.
# Degenerate-chunk guard (2026-06-04): identity-normalizer ckpts run the DDIM
# sampler with clip_sample=False, so an OOD obs can yield an out-of-range chunk
# (grip>1, 300-580mm CONSECUTIVE steps vs the normal <=~45mm). Holding these
# keeps calibration data drawn from in-distribution deploy behaviour (and the
# arm safe). Matches eval_dp_real.py.
DEGENERATE_INTRA_STEP_M = 0.10


# =========================================================================
# Observation pipeline — UMI latency-matched
# =========================================================================
def gather_latency_matched_obs(camera, ctrl, gripper, n_obs_steps, dt, image_size):
    K = max(n_obs_steps * 4, 16)

    cam_data = camera.get(k=K)
    # MultiZed.get returns {0: {color, timestamp, ...}, 1: {...}}; index in
    # serial_numbers order — for us [0]=exterior, [1]=wrist.
    cam_keys = list(cam_data.keys())
    ts_list = [np.asarray(cam_data[k]['timestamp']) for k in cam_keys]

    # ---- UMI mutual-best-error: align camera with smallest summed time-gap
    align_idx = 0
    best_err = float('inf')
    for cand_i, ts_cand in enumerate(ts_list):
        last_cand = float(ts_cand[-1])
        err = 0.0
        for j, ts_other in enumerate(ts_list):
            if j == cand_i:
                continue
            # smallest |t_other - last_cand| with t_other <= last_cand
            below = ts_other[ts_other <= last_cand]
            if len(below) == 0:
                err = float('inf'); break
            err += last_cand - below[-1]
        if err < best_err:
            best_err = err
            align_idx = cand_i

    last_t = float(ts_list[align_idx][-1])
    obs_ts = last_t - (np.arange(n_obs_steps)[::-1] * dt)

    # ---- Camera frames at each grid t — nearest neighbor index
    def _select_frames(ts_frames, color_frames, grid):
        idxs = [int(np.argmin(np.abs(ts_frames - g))) for g in grid]
        return color_frames[idxs]

    ext_color = _select_frames(ts_list[0], np.asarray(cam_data[cam_keys[0]]['color']), obs_ts)
    wri_color = _select_frames(ts_list[1], np.asarray(cam_data[cam_keys[1]]['color']), obs_ts)

    def _preprocess(frames):
        # MATCH training EXACTLY (convert_to_dp_image.py:51-69):
        #   PyAV decodes video as RGB (rgb24), then PIL.Image.BILINEAR resize.
        # SingleZed ring buffer stores BGR (single_zed.py:325), so:
        #   raw BGR -> cv2.cvtColor BGR2RGB -> PIL.Image.fromarray -> PIL.BILINEAR
        # cv2.INTER_LINEAR ≠ PIL.BILINEAR byte-exact: PIL anchors pixels at
        # corners, cv2 at center, and they round differently. Using PIL here
        # removes the only remaining image-preprocess mismatch with training.
        out = np.empty((frames.shape[0], 3, image_size, image_size), dtype=np.float32)
        for i, f in enumerate(frames):
            bgr = f[..., :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            small = np.asarray(
                Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR))
            out[i] = small.astype(np.float32).transpose(2, 0, 1) / 255.0
        return out
    ext = _preprocess(ext_color)
    wri = _preprocess(wri_color)

    # ---- Robot pose stream — SE3 interp at obs_ts.
    # Mirrors training-side semantics (convert_to_dp_image.py v2.1):
    #   1. SE(3) SLERP on (pos, axis-angle) via PoseInterpolator — same class
    #      used during recording (franka_vive_env.end_episode line 1202-1206)
    #      and identical to scipy.spatial.transform.Slerp underneath.
    #   2. Apply alternative-branch flip on the resulting axis-angle so
    #      axis-x ≥ 0 at the first obs sample (same lossless transform as
    #      training's per-episode flip). This matches the v2.1 dataset's
    #      "axis-x ≥ 0 at episode start" invariant.
    #   3. Derive obs.quat = R.from_rotvec(eef_aa).as_quat() — single source
    #      of truth for rotation so obs.aa and obs.quat express the same
    #      rotation in every frame (training dataset has 100% rotation
    #      consistency: |dot(obs.quat, R.from_rotvec(obs.aa).as_quat())| = 1.0
    #      across all 26037 frames).
    # Earlier evals SLERPed ActualTCPQuat independently of the pose
    # interpolator and the two streams disagreed in sign on ~68% of frames
    # — that mismatch made the policy's axis-angle output collapse toward
    # the distribution mean. Do NOT revert this back to dual-stream SLERP.
    rs = ctrl.get_state(k=K)
    robot_ts = np.asarray(rs['robot_timestamp'])
    pose6 = np.asarray(rs['ActualTCPPose'])
    pose_interp = PoseInterpolator(t=robot_ts, x=pose6)
    sampled_pose = pose_interp(obs_ts)
    eef_pos = sampled_pose[:, 0:3].astype(np.float32)
    eef_aa  = sampled_pose[:, 3:6].astype(np.float64)

    # === Match training v2.1 axis-angle branch alignment ===
    # convert_to_dp_image.py forces axis-x ≥ 0 at episode start by applying
    # the lossless alternative-branch transform (axis, θ) → (-axis, 2π-θ).
    # The new training dataset (franka_peg_80ep_v2.1) was built that way, so
    # eval-time obs.aa must follow the same convention or the policy sees an
    # OOD distribution.
    if len(eef_aa) > 0 and eef_aa[0, 0] < 0:
        _norms = np.linalg.norm(eef_aa, axis=1, keepdims=True)
        _nz = (_norms > 1e-6).flatten()
        _flipped = eef_aa.copy()
        _flipped[_nz] = -eef_aa[_nz] * ((2*np.pi - _norms[_nz]) / _norms[_nz])
        eef_aa = _flipped

    # === Match training v2.1: obs.quat DERIVED from obs.aa ===
    # Single source of truth for rotation. The training-side conversion does
    # the same (Rotation.from_rotvec(eef_aa).as_quat()) so that obs.aa and
    # obs.quat always express the same rotation with the same sign
    # convention. Earlier code SLERPed ActualTCPQuat independently and got
    # ~68% sign-disagreement per frame — model saw two redundant rotation
    # streams whose sign relation was noise.
    eef_quat = R.from_rotvec(eef_aa).as_quat().astype(np.float32)
    eef_aa = eef_aa.astype(np.float32)

    # ---- Gripper width — 1D linear interp
    gs = gripper.get_state(k=K)
    gripper_ts = np.asarray(gs['gripper_timestamp'])
    gw = np.asarray(gs['gripper_width']).reshape(-1, 1)
    g_interp = get_interp1d(gripper_ts, gw)
    sampled_w = g_interp(obs_ts).reshape(-1)
    g_norm = (sampled_w / ART_MAX_WIDTH).astype(np.float32).reshape(-1, 1)

    obs = {
        'exterior_image_1_left': ext,
        'wrist_image_left':      wri,
        'robot0_eef_pos':        eef_pos,
        'robot0_eef_quat':       eef_quat,
        'robot0_eef_rot_axis_angle': eef_aa,
        'robot0_gripper_qpos':   g_norm,
    }

    # ---- Per-camera frame-age diagnostics (freeze detection) ----
    # 'timestamp' is the calibrated capture time (receive_time -
    # receive_latency) on the SAME wall clock as the eval loop's time.time(),
    # so (now - ts) ≈ how long ago that frame was captured. Returns, per camera
    # (0=exterior, 1=wrist/gripper):
    #   'newest'   = ring-buffer's freshest frame timestamp — balloons and keeps
    #                growing if the SingleZed process died and stopped writing.
    #   'selected' = timestamp of the frame actually fed to the model for the
    #                last (freshest) obs step (nearest-neighbor to obs_ts[-1]).
    # The caller subtracts these from time.time() at the predict() call so the
    # log shows exactly how old each image is at the moment it enters the model.
    cam_ts_info = {}
    for ci in range(len(cam_keys)):
        ts_c = ts_list[ci]
        sel_i = int(np.argmin(np.abs(ts_c - obs_ts[-1])))
        cam_ts_info[ci] = {
            'newest':   float(ts_c[-1]),
            'selected': float(ts_c[sel_i]),
        }
    return obs, obs_ts, cam_ts_info


# Viewer logging throttle: emit at most one diagnostic per ~5s of failures
# so the operator notices but the terminal doesn't get spammed.
_VIEWER_LAST_WARN_T = [0.0]
_VIEWER_FAIL_STREAK = [0]


def write_viewer_jpeg(camera, status_lines: list, image_size: int = 320) -> None:
    """Compose ext+wrist camera + status overlay into a single JPEG for the
    out-of-process cv2 viewer (bin/cv2_viewer.py).

    Atomic write via temp-file + os.replace. cv2_viewer polls the file with
    cv2.imread; without atomic write the viewer occasionally caught a partial
    JPEG (logged "Premature end of JPEG file") and rendered partial / black
    frames. Critical: the temp file MUST keep a .jpg suffix because
    cv2.imwrite picks the encoder via extension — a "*.jpg.tmp" name yields
    "could not find a writer for the specified extension".
    """
    ext = None
    wri = None
    ext_age = None   # seconds since the exterior cam's last frame (None = unknown)
    wri_age = None   # seconds since the wrist cam's last frame
    fail_reason = None
    try:
        cam = camera.get(k=1)
        keys = list(cam.keys())
        if len(keys) >= 2:
            ext_raw = np.asarray(cam[keys[0]]['color'])
            wri_raw = np.asarray(cam[keys[1]]['color'])
            if ext_raw.ndim == 4:
                ext_raw = ext_raw[-1]
            if wri_raw.ndim == 4:
                wri_raw = wri_raw[-1]
            ext = cv2.resize(ext_raw[..., :3], (image_size, image_size))
            wri = cv2.resize(wri_raw[..., :3], (image_size, image_size))
            # Per-camera staleness: a frozen stream keeps returning its LAST
            # frame, so camera.get() still succeeds but the timestamp stops
            # advancing. age = now - last_frame_ts grows -> that is exactly the
            # ">1s frozen image" the operator sees. (ts are wall-clock time.time
            # baked at capture minus receive_latency, see single_zed.py.)
            _now_age = time.time()
            try:
                ext_age = _now_age - float(np.asarray(cam[keys[0]]['timestamp']).reshape(-1)[-1])
            except Exception:
                ext_age = None
            try:
                wri_age = _now_age - float(np.asarray(cam[keys[1]]['timestamp']).reshape(-1)[-1])
            except Exception:
                wri_age = None
        else:
            fail_reason = f"camera.get returned {len(keys)} streams (expected ≥2)"
    except Exception as e:
        fail_reason = f"{type(e).__name__}: {e}"
    if fail_reason is not None or ext is None or wri is None:
        _VIEWER_FAIL_STREAK[0] += 1
        now = time.time()
        if (now - _VIEWER_LAST_WARN_T[0]) > 5.0:
            _VIEWER_LAST_WARN_T[0] = now
            print(f"[viewer/dbg] camera frame fetch failed "
                  f"(streak={_VIEWER_FAIL_STREAK[0]}): {fail_reason or 'unknown'}",
                  flush=True)
    else:
        _VIEWER_FAIL_STREAK[0] = 0
    if ext is None or wri is None:
        ext = np.full((image_size, image_size, 3), 32, dtype=np.uint8)
        wri = np.full((image_size, image_size, 3), 32, dtype=np.uint8)
        cv2.putText(ext, 'no camera frame yet', (8, image_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    # label
    cv2.putText(ext, 'exterior (35766817)', (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(wri, 'wrist (11667817)', (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 220, 220), 1, cv2.LINE_AA)

    # ---- Per-camera "INPUT STALLED" overlay ----------------------------------
    # If a camera's last frame is older than STALE_INPUT_S, its live stream has
    # frozen (USB drop, ZED grab stall, cable). Mark THAT panel red so the
    # operator sees *which* of the two cameras died instead of guessing from a
    # frozen image. Threshold = 1.0s per request ("1초 이상 멈춘 것 같으면").
    STALE_INPUT_S = 1.0
    def _mark_stale(img, age):
        if age is None or age <= STALE_INPUT_S:
            return
        h, w = img.shape[:2]
        cv2.rectangle(img, (1, 1), (w - 2, h - 2), (0, 0, 255), 6)
        cv2.putText(img, 'NO INPUT', (8, h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f'frozen {age:4.1f}s', (8, h // 2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    _mark_stale(ext, ext_age)
    _mark_stale(wri, wri_age)

    # status panel
    panel = np.full((image_size, 320, 3), 24, dtype=np.uint8)
    for i, line in enumerate(status_lines[:8]):
        cv2.putText(panel, line, (8, 22 + 22*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Always-on camera age readout at the bottom (green=live, red=stalled).
    def _age_line(name, age):
        if age is None:
            return f'{name}: n/a', (0, 0, 255)
        stalled = age > STALE_INPUT_S
        col = (0, 0, 255) if stalled else (0, 220, 120)
        return f'{name} {age:4.2f}s' + (' STALLED' if stalled else ''), col
    _l1, _c1 = _age_line('cam ext', ext_age)
    _l2, _c2 = _age_line('cam wri', wri_age)
    cv2.putText(panel, _l1, (8, image_size - 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _c1, 1, cv2.LINE_AA)
    cv2.putText(panel, _l2, (8, image_size - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _c2, 1, cv2.LINE_AA)

    combined = np.concatenate([ext, wri, panel], axis=1)
    # Atomic write: cv2.imwrite picks encoder via extension, so the temp
    # file MUST keep the .jpg suffix. Put the unique part before .jpg, not
    # after. Earlier "VIS_JPEG_PATH + '.tmp'" produced "*.jpg.tmp" which cv2
    # could not encode → "could not find a writer for the specified extension".
    base, ext_suffix = os.path.splitext(VIS_JPEG_PATH)
    tmp_path = f"{base}.tmp{ext_suffix or '.jpg'}"
    ok = cv2.imwrite(tmp_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        try:
            os.replace(tmp_path, VIS_JPEG_PATH)
        except OSError:
            cv2.imwrite(VIS_JPEG_PATH, combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
    else:
        # fall back to in-place if temp write failed
        cv2.imwrite(VIS_JPEG_PATH, combined, [cv2.IMWRITE_JPEG_QUALITY, 80])


def launch_viewer():
    """Spawn bin/cv2_viewer.py reading VIS_JPEG_PATH, with SIGINT relay on 'q'."""
    # Seed image so viewer doesn't error on first read. Use a clearly visible
    # background (mid-grey 80, not 24) and large warning text so it's obvious
    # the viewer is alive and waiting for real camera frames — earlier the
    # near-black 24-grey background made operators report "black screen".
    seed = np.full((320, 320*3, 3), 80, dtype=np.uint8)
    cv2.putText(seed, 'eval starting up - waiting for cameras...',
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(seed, '(if you see this for >10s the writer is failing)',
                (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imwrite(VIS_JPEG_PATH, seed)
    viewer_path = os.path.join(ROOT_DIR, 'bin', 'cv2_viewer.py')
    if not os.path.exists(viewer_path):
        print(f'[viewer] {viewer_path} missing; running headless')
        return None
    try:
        # No env= override: inherit DISPLAY/XAUTHORITY as-is and leave
        # QT_QPA_PLATFORM UNSET, exactly like data collection
        # (demo_franka_vive.py launches this same cv2_viewer.py with no env=).
        # An earlier version here forced QT_QPA_PLATFORM=xcb (a 5080 workaround
        # where Qt otherwise fell back to 'minimal'/no-window); on pro4000 that
        # forced xcb made cv2's Qt paint the image surface BLACK while the
        # window + title still rendered. Collection never forced it and renders
        # fine, so inheriting the env unifies eval with collection and fixes
        # the black canvas.
        print(f'[viewer] DISPLAY={os.environ.get("DISPLAY", "<unset>")} '
              f'XAUTHORITY={os.environ.get("XAUTHORITY", "<unset>")} '
              f'QT_QPA_PLATFORM={os.environ.get("QT_QPA_PLATFORM", "<unset>")}',
              flush=True)
        proc = subprocess.Popen([
            sys.executable, viewer_path, VIS_JPEG_PATH,
            '--signal-pid', str(os.getpid()),
            '--poll-ms', '50',
            '--win-name', 'DP Eval - KIST Franka',
        ])
        time.sleep(0.5)
        if proc.poll() is not None:
            print(f'[viewer] exited rc={proc.returncode}; running headless',
                  flush=True)
            return None
        print(f'[viewer] PID {proc.pid} — JPEG @ {VIS_JPEG_PATH}', flush=True)
        atexit.register(lambda p=proc: (
            p.terminate(), time.sleep(0.5),
            p.kill() if p.poll() is None else None))
        return proc
    except Exception as e:
        print(f'[viewer] launch failed: {e}', flush=True)
        return None


def safety_ok(pos: np.ndarray, prev_pos: np.ndarray, dt: float) -> bool:
    """Reject actions outside workspace bounds or too large per-step.

    DEBUG: prints the exact reason on reject so we can tell whether reject was
    WORKSPACE or STEP-norm. Earlier 2026-05-19 runs showed all 5 action timesteps
    rejected — the print below reveals whether prev_pos was actually None, what
    step-norm value was used, and which dimension was outside workspace.
    """
    x, y, z = pos
    in_ws = (WORKSPACE_BOUNDS_M['x_min'] <= x <= WORKSPACE_BOUNDS_M['x_max']
             and WORKSPACE_BOUNDS_M['y_min'] <= y <= WORKSPACE_BOUNDS_M['y_max']
             and WORKSPACE_BOUNDS_M['z_min'] <= z <= WORKSPACE_BOUNDS_M['z_max'])
    if not in_ws:
        print('  [safety/dbg] WORKSPACE reject: pos=' + str(pos.round(3).tolist())
              + ' x_ok=' + str(WORKSPACE_BOUNDS_M['x_min'] <= x <= WORKSPACE_BOUNDS_M['x_max'])
              + ' y_ok=' + str(WORKSPACE_BOUNDS_M['y_min'] <= y <= WORKSPACE_BOUNDS_M['y_max'])
              + ' z_ok=' + str(WORKSPACE_BOUNDS_M['z_min'] <= z <= WORKSPACE_BOUNDS_M['z_max']),
              flush=True)
        return False
    if prev_pos is not None:
        step = float(np.linalg.norm(pos - prev_pos))
        if step > MAX_POS_STEP_PER_DT_M:
            print('  [safety/dbg] STEP reject: step_norm=' + str(round(step*1000, 1))
                  + 'mm > MAX=' + str(round(MAX_POS_STEP_PER_DT_M*1000, 1))
                  + 'mm  prev=' + str(prev_pos.round(3).tolist())
                  + ' -> pos=' + str(pos.round(3).tolist()), flush=True)
            return False
    else:
        print('  [safety/dbg] prev_pos is None — only WORKSPACE checked  pos='
              + str(pos.round(3).tolist()), flush=True)
    return True


# =========================================================================
# ZMQ inference client (REQ side)
# =========================================================================
class InferenceClient:
    def __init__(self, port, ctx=None):
        self.ctx = ctx or zmq.Context.instance()
        self._endpoint = f"tcp://127.0.0.1:{port}"
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 10000)   # 10s safety
        self.sock.connect(self._endpoint)

    def ping(self) -> bool:
        self.sock.send(msgpack.packb({'op': 'ping'}))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        return bool(reply.get('pong'))

    def predict(self, obs_np: dict):
        # 2026-05-21: ZMQ REQ socket can occasionally EAGAIN on RCVTIMEO=10s
        # right after daemon startup (observed: daemon log stopped at 'listening',
        # never recv'd, while client send appeared to succeed). REQ-REP state
        # machine forbids send() before recv() completes, so if we got EAGAIN
        # the socket is locked in 'recv-pending' state -- we must rebuild it.
        import time as _time
        t_send = _time.time()
        try:
            self.sock.send(msgpack.packb({'op': 'predict', 'obs': obs_np}))
            reply = msgpack.unpackb(self.sock.recv(), raw=False)
        except zmq.error.Again as e:
            print(f"[client] zmq EAGAIN at predict() recv after "
                  f"{(_time.time()-t_send)*1000:.0f}ms - daemon unresponsive. "
                  f"Rebuilding REQ socket and retrying once.", flush=True)
            # REQ socket is stuck in recv-pending; close and reopen
            try:
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.close()
            except Exception:
                pass
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, 10000)
            # find original endpoint by inspecting prior connect arg
            self.sock.connect(self._endpoint)
            self.sock.send(msgpack.packb({'op': 'predict', 'obs': obs_np}))
            reply = msgpack.unpackb(self.sock.recv(), raw=False)
        if not reply.get('ok'):
            raise RuntimeError(f"daemon error: {reply.get('error', '?')}")
        # FIX A (match eval_dp_real): prefer the full predicted horizon when the
        # daemon ships it. Returns (action_array, action_offset, inference_ms)
        # where action_offset = n_obs_steps-1 places step k at
        # t_obs + (k - action_offset)*dt. Falls back to the short 'action'.
        if 'action_pred' in reply:
            return (reply['action_pred'],
                    int(reply.get('n_obs_steps', 1)) - 1,
                    reply.get('inference_ms', 0.0))
        return reply['action'], 0, reply.get('inference_ms', 0.0)

    def save_calib(self, out_path: str):
        # Calibration inputs are appended (globalvar.appendInput) inside the
        # diffusion UNet sampling loop, which runs in the DAEMON process — not
        # here. So we cannot torch.save them locally (this process's globalvar
        # list is always empty). Ask the daemon, which owns the populated list,
        # to write the .pth itself.
        self.sock.send(msgpack.packb({'op': 'save_calib', 'path': out_path}))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        if not reply.get('ok'):
            raise RuntimeError(f"daemon save_calib error: {reply.get('error', '?')}")
        return int(reply.get('n', 0)), reply.get('path', out_path)

    def shutdown(self):
        try:
            self.sock.send(msgpack.packb({'op': 'shutdown'}))
            self.sock.recv()
        except Exception:
            pass
        self.sock.close()


# =========================================================================
# Main loop
# =========================================================================
@click.command()
@click.option('--output_dir', required=True, type=str)
@click.option('--daemon_port', default=5555, type=int)
@click.option('--robot_ip', default='192.168.1.14')
@click.option('--steps_per_inference', default=8, type=int)
@click.option('--max_duration', default=60.0, type=float)
@click.option('--frequency', default=10.0, type=float)
@click.option('--tcp_offset', default=0.216)
@click.option('--image_size', default=224, type=int)
@click.option('--art_host', default='127.0.0.1')
@click.option('--art_port', default=50053, type=int)
@click.option('--n_obs_steps', default=2, type=int,
              help='Match cfg.n_obs_steps (default 2 for our DP configs).')
@click.option('--auto_start_after', default=0.0, type=float,
              help='Auto-start eval after N seconds of PREP, no ENTER needed. 0=disable (default).')
def main(output_dir, daemon_port, robot_ip, steps_per_inference, max_duration, frequency,
         tcp_offset, image_size, art_host, art_port, n_obs_steps, auto_start_after):

    # Resolve latency constants from install/latency_calibration.json
    lat = resolve_latencies(camera_backend='zed', gripper_backend='art')
    cam_obs_lat   = float(lat['camera_obs_latency'])
    robot_obs_lat = float(lat['robot_obs_latency'])
    grip_obs_lat  = float(lat['gripper_obs_latency'])
    robot_act_lat = float(lat['robot_action_latency'])
    grip_act_lat  = float(lat['gripper_action_latency'])
    action_exec_lat = 0.010  # UMI eval_real.py default safety margin
    print(f"[eval] resolved latencies (s): cam_obs={cam_obs_lat:.4f} "
          f"robot_obs={robot_obs_lat:.4f} grip_obs={grip_obs_lat:.4f} "
          f"robot_act={robot_act_lat:.4f} grip_act={grip_act_lat:.4f}")

    # ---- 1. Connect to inference daemon ----
    client = InferenceClient(daemon_port)
    print(f"[eval] connected to daemon on port {daemon_port}, pinging…")
    if not client.ping():
        raise RuntimeError("daemon did not respond to ping")
    print(f"[eval] daemon alive.")

    dt = 1.0 / frequency
    frame_latency = 1.0 / 60.0  # ZED 60fps headroom

    with SharedMemoryManager() as shm:
        # FrankaViveEnv-pattern init: create all hardware objects first, then
        # start robot/gripper async, sleep 0.5s, start cameras async, then
        # batched start_wait(). Original code did camera.start(wait=True)
        # standalone which hung on the ZED ready_event before the other
        # subprocesses had a chance to spin up — that 2026-05-19 hang was
        # reproduced multiple times. This now matches demo_franka_vive.py
        # (FrankaViveEnv.start, lines 662-684) exactly.
        print("[eval] creating hardware objects (camera + robot + gripper)...")
        camera = MultiZed(
            shm_manager=shm,
            serial_numbers=[35766817, 11667817],   # ext, wrist (same as collection)
            resolution=(672, 376),
            capture_fps=60,
            receive_latency=cam_obs_lat,
        )
        ctrl = FrankaInterpolationController(
            shm_manager=shm,
            robot_ip=robot_ip,
            frequency=100.0,
            tcp_offset=tcp_offset,
            launch_timeout=10.0,
            joints_init=FRANKA_HOME_JOINTS,
            joints_init_duration=4.0,
            # CRITICAL: home_joints MUST match joints_init.  Without this, the
            # controller defaults FRANKA_HOME_JOINTS to j7=π/4 (Franka panda
            # gripper convention).  Then move_home() drives j7 from 0 (ART
            # convention used during DP data collection) to π/4, producing
            # exactly the joint jerk that triggers power_limit_violation reflex
            # and the OOD obs the policy then sees.  FrankaViveEnv.__init__
            # (line 394) sets home_joints=ready_pose for the same reason.
            home_joints=FRANKA_HOME_JOINTS,
            receive_latency=robot_obs_lat,
            verbose=False,
            teleop_mode=False,
        )
        gripper = ArtGripperController(
            shm_manager=shm,
            host=art_host, port=art_port,
            frequency=60.0,
            receive_latency=grip_obs_lat,
            # Width envelope + grip force MUST match the spec used by data
            # collection (gripper_specs.py:'art'): open=0.095m, close=0.0m
            # (ART firmware accepts full mechanical close), force=60N.
            # Earlier eval used close=0.005 (Franka-Hand default to avoid
            # libfranka exception) and dropped force=60 (controller default
            # is 30N) — both diverge from training-distribution grasp
            # behaviour. ART has no libfranka so width=0.0 is safe.
            gripper_open_width=ART_MAX_WIDTH,
            gripper_close_width=0.0,
            default_force=60.0,
        )

        # Spawn order matches FrankaViveEnv: robot+gripper first, then 0.5s
        # delay, then camera. Camera is last so it doesn't sit alone in a
        # blocking ready-wait while the other USB/CPU resources are still idle.
        print("[eval] starting Franka controller (async)...")
        ctrl.start(wait=False)
        print("[eval] starting ART gripper (async)...")
        gripper.start(wait=False)
        time.sleep(0.5)
        print("[eval] starting cameras (async)...")
        camera.start(wait=False)

        # Batched ready-wait — each prints whether it actually came up.
        # SingleZed/Franka/ART all use ready_event.wait(launch_timeout) under
        # the hood (10s default), so this returns within 10s even on failure
        # rather than hanging silently.
        print("[eval] waiting for Franka controller ready...")
        ctrl.start_wait()
        print(f"[eval]   ctrl.is_ready = {ctrl.is_ready}")
        print("[eval] waiting for ART gripper ready...")
        gripper.start_wait()
        print(f"[eval]   gripper.is_ready = {gripper.is_ready}")
        print("[eval] waiting for ZED cameras ready...")
        camera.start_wait()
        print(f"[eval]   camera.is_ready = {camera.is_ready}")

        if not (ctrl.is_ready and gripper.is_ready and camera.is_ready):
            raise RuntimeError(
                f"hardware not ready: ctrl={ctrl.is_ready} "
                f"gripper={gripper.is_ready} camera={camera.is_ready}"
            )

        # Boot already moved the arm to FRANKA_HOME_JOINTS via the
        # controller's joints_init (joints_init_duration=4.0s) and completed
        # start_cartesian_impedance + wait_until_controller_ready before
        # ready_event fired — see FrankaInterpolationController.run lines
        # 664-728. Calling move_home() here triggered a second
        # terminate/start cycle that raced with the first schedule_waypoint
        # and tripped "no controller running" on iter=0 (recorded in
        # /tmp/eval_unet.log @ 2026-05-19 21:01:42). Match data-collection
        # behaviour (demo_franka_vive.py / FrankaViveEnv.start): no extra
        # move_home in startup; trust joints_init.
        print("[eval] arm already at FRANKA_HOME_JOINTS (via joints_init); "
              "letting cartesian impedance settle...")
        time.sleep(1.0)

        # Gripper open: ArtGripperController.run() already issues an opening
        # goto at startup (art_gripper_controller.py:231), but we explicitly
        # re-issue here so the gripper is guaranteed at ART_MAX_WIDTH at the
        # exact start-of-eval baseline regardless of any prior teleop state.
        print("[eval] opening gripper to known state...")
        gripper.goto(ART_MAX_WIDTH)
        time.sleep(1.2)

        # ---- 1.5. Prep / safety gate ----
        # Out-of-process cv2_viewer.py subprocess (demo_franka_vive.py pattern).
        # In-process cv2.imshow deadlocks/busy-loops under multi-subprocess
        # load (ZED grab + polymetis + ART all contend for the X mutex via
        # Qt5; documented in demo_franka_vive.py lines 197-202).
        cv2.setNumThreads(2)
        print("[eval] launching camera viewer (cv2_viewer.py subprocess)")
        viewer = launch_viewer()

        # Live status loop until operator confirms
        ready_acked = False
        try:
            # quick 5s pre-check loop with status update so the operator can
            # visually verify both cameras and the robot at ready pose
            for tick in range(20):
                rs = ctrl.get_state(k=1)
                pos = np.asarray(rs['ActualTCPPose'][-1, 0:3])
                gs = gripper.get_state(k=1)
                gw = float(gs['gripper_width'][-1])
                status = [
                    'PREP — verify hardware then',
                    'press ENTER to start eval',
                    "or 'q' in viewer to abort",
                    '',
                    f'eef pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}',
                    f'gripper width: {gw*1000:.0f} mm',
                    f'cameras: 2x ZED 60fps',
                    f'latency cam_obs={cam_obs_lat*1000:.0f} robot_act={robot_act_lat*1000:.0f}ms',
                ]
                write_viewer_jpeg(camera, status)
                time.sleep(0.1)

            print()
            print('=' * 60)
            print('[PREP] Robot at READY pose, cameras streaming, gripper open.')
            print('[PREP] Verify the viewer window shows both camera feeds.')
            if auto_start_after > 0:
                print(f'[PREP] Auto-starting eval in {auto_start_after:.1f}s. Press Ctrl-C to abort.')
                print('=' * 60)
                t_auto = time.time()
                while time.time() - t_auto < auto_start_after:
                    rs = ctrl.get_state(k=1)
                    pos = np.asarray(rs['ActualTCPPose'][-1, 0:3])
                    gs = gripper.get_state(k=1)
                    gw = float(gs['gripper_width'][-1])
                    remaining = auto_start_after - (time.time() - t_auto)
                    status = [
                        f'PREP auto-start in {remaining:4.1f}s',
                        f'(or Ctrl-C abort)',
                        '',
                        f'eef pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}',
                        f'gripper width: {gw*1000:.0f} mm',
                    ]
                    write_viewer_jpeg(camera, status)
                    time.sleep(0.1)
                ready_acked = True
            else:
                print("[PREP] Press ENTER to start eval, Ctrl-C to abort.")
                print('=' * 60)
                try:
                    input()
                    ready_acked = True
                except (KeyboardInterrupt, EOFError):
                    print('[PREP] aborted by operator before eval start.')
                    return
        except KeyboardInterrupt:
            print('[PREP] aborted by operator (Ctrl-C in prep loop).')
            return

        if not ready_acked:
            return

        # ---- 1.6 Controller liveness check + cartesian-impedance wake-up ----
        # /tmp/eval_unet.log @ 2026-05-19 22:13 still shows "no controller
        # running" on iter=0 even after we removed the redundant ctrl.move_home
        # call. Cause: polymetis can silently terminate cartesian_impedance
        # during the long PREP window; the controller worker's main loop then
        # eats the error on the next update_desired_ee_pose and goes through
        # its 1.5 s recovery — but the recovery overlaps with our very first
        # eval-loop schedule_waypoint, producing the visible jerk.
        #
        # Mitigation: probe the controller with a dummy schedule_waypoint to
        # the robot's CURRENT pose, then verify the controller ring buffer is
        # still ticking. If we trip recovery here, it completes BEFORE the
        # eval loop schedules any real action.
        print("[eval] probing controller liveness before eval loop start...")
        try:
            rs_pre = ctrl.get_state(k=1)
            ping_pose = np.asarray(rs_pre['ActualTCPPose'][-1]).astype(np.float64)
            ts_before = float(rs_pre['robot_timestamp'][-1])
            # Schedule a no-op waypoint 400ms ahead -- 400ms > robot_act_lat
            # (148ms) so the controller actually processes it instead of
            # discarding as past.
            ctrl.schedule_waypoint(ping_pose, target_time=time.time() + 0.4)
            time.sleep(0.6)
            rs_post = ctrl.get_state(k=1)
            ts_after = float(rs_post['robot_timestamp'][-1])
            ring_dt_ms = (ts_after - ts_before) * 1000.0
            print(f"[eval]   controller ring advanced {ring_dt_ms:.0f}ms over 600ms wait")
            if ring_dt_ms < 100.0:
                print("[eval]   WARN: controller ring barely advanced — controller "
                      "may be stuck. Giving extra 1.5s for recovery to settle.")
                time.sleep(1.5)
        except Exception as e:
            print(f"[eval] liveness probe failed: {type(e).__name__}: {e}")

        # Final settle so any auto-recovery from the probe completes before
        # the eval loop fires its first real schedule_waypoint burst.
        time.sleep(0.5)

        # ---- 2. Eval loop ----
        t_start = time.time()
        eval_t_start = t_start + 1.0
        precise_wait(eval_t_start - frame_latency, time_func=time.time)

        iter_idx = 0
        n_actions_total = 0
        rejected_by_safety = 0
        # Seed prev_action_pos from the robot's current TCP position so the
        # first inference's first action step gets a real STEP-norm guard.
        # Previously prev_action_pos started as None and safety_ok only ran
        # WORKSPACE check on iter=0/step=0 — a policy that jumps 0.5m+ from
        # ready pose at first step would bypass the per-dt motion cap.
        rs_seed = ctrl.get_state(k=1)
        prev_action_pos = np.asarray(rs_seed['ActualTCPPose'][-1, 0:3]).astype(np.float64)
        print(f"[eval] seeded prev_action_pos from robot TCP: "
              f"{prev_action_pos.round(3).tolist()}")

        try:
            while time.time() - eval_t_start < max_duration:
                t_cycle_end = eval_t_start + (iter_idx + steps_per_inference) * dt

                # 2026-05-21 01:34 failure: iter=0 completed (7/7 actions
                # applied), then on iter=1 the main thread blocked forever
                # in select() while the multiprocessing children
                # (FrankaInterpController, ArtGripperController, MultiZed)
                # had silently died. The children inherit our stdout via
                # spawn, but their crash mode (native segfault or external
                # SIGKILL) left no Python traceback, so we couldn't see WHY
                # they died. Detecting their death here at least lets us
                # exit cleanly via finally (instead of deadlocking on a
                # ring-buffer read of a dead writer).
                for _name, _p in (('FrankaInterp', ctrl),
                                  ('ArtGripper', gripper),
                                  ('MultiZed', camera)):
                    if hasattr(_p, 'is_alive') and not _p.is_alive():
                        ec = getattr(_p, 'exitcode', '?')
                        raise RuntimeError(
                            f"{_name} child process died (exitcode={ec}) "
                            f"before iter={iter_idx}. See its inherited "
                            f"stdout in {os.environ.get('EVAL_LOG','this log')} "
                            f"for any traceback.")

                # On the first 2 iters we print which substep we're on so
                # that a stall / native crash leaves a paper trail.
                _verbose_step = (iter_idx < 2)
                if _verbose_step:
                    print(f"[eval/step] iter={iter_idx} (a) gather_latency_matched_obs ...",
                          flush=True)

                # (a) latency-matched obs
                obs_np, obs_ts, cam_ts_info = gather_latency_matched_obs(
                    camera, ctrl, gripper, n_obs_steps, dt, image_size)
                t_obs = float(obs_ts[-1])
                obs_lat_now = time.time() - t_obs
                if _verbose_step:
                    print(f"[eval/step] iter={iter_idx} (a) done. (b) client.predict ...",
                          flush=True)

                # (b) inference via daemon (batch=1 prepend)
                obs_batched = {k: v[None] for k, v in obs_np.items()}
                t_model_in = time.time()   # instant the obs is handed to the model
                action_seq, action_offset, inf_ms = client.predict(obs_batched)
                # writable copy: msgpack_numpy deserialises into a READ-ONLY array.
                action_seq = np.array(action_seq[0], dtype=np.float32)   # (horizon, 7)
                # Action sanity / degenerate-chunk guard (match eval_dp_real):
                action_seq[:, 6] = np.clip(action_seq[:, 6], 0.0, 1.0)
                _intra = (np.linalg.norm(np.diff(action_seq[:, :3], axis=0), axis=1)
                          if len(action_seq) > 1 else np.array([0.0]))
                _degenerate_chunk = bool(_intra.max() > DEGENERATE_INTRA_STEP_M)
                if _verbose_step:
                    print(f"[eval/step] iter={iter_idx} (b) done. (c)..(e) schedule_waypoint ...",
                          flush=True)
                print(f"[eval] iter={iter_idx:4d} obs_lat={obs_lat_now*1000:5.1f}ms "
                      f"inf={inf_ms:5.1f}ms action_shape={action_seq.shape}")
                # How old is each camera image at the instant it enters the model?
                #   in=age of the frame actually fed to the policy (selected),
                #   newest=age of the freshest frame currently in the buffer.
                # If a camera's SingleZed process froze, both grow every iter and
                # never recover — that's the "wrist/gripper camera frozen" signal.
                _cam_names = {0: 'ext', 1: 'wrist'}
                _age_parts = []
                for _ci in sorted(cam_ts_info):
                    _ti = cam_ts_info[_ci]
                    _age_parts.append(
                        f"{_cam_names.get(_ci, f'cam{_ci}')}: "
                        f"in={(t_model_in - _ti['selected'])*1000:6.1f}ms "
                        f"newest={(t_model_in - _ti['newest'])*1000:6.1f}ms")
                print(f"  [cam/age] iter={iter_idx:4d} " + "  ".join(_age_parts))
                # On iter 0, dump the obs/action values so we can compare
                # against training_audit/data/sample_first3_eps.npz offline.
                # Training demo_0 first frame:
                #   obs.pos=[0.309, 0.007, 0.375]  obs.aa=[-3.135, 0.015, -0.013]
                #   obs.quat=[+1.0, -0.005, +0.004, -0.003]  obs.g=0.989
                #   action[0]=[0.305, -0.0, 0.375, +3.124, +0.048, -0.012, 0.989]
                # If the eval-time obs significantly disagrees with this, the
                # robot is in a different start pose / the dataset reset
                # routine is mismatched.
                if iter_idx == 0:
                    print("[eval/dbg] === FIRST INFERENCE OBS DUMP ===")
                    for k in ('robot0_eef_pos', 'robot0_eef_quat',
                              'robot0_eef_rot_axis_angle', 'robot0_gripper_qpos'):
                        v = obs_np[k]
                        print(f"  obs.{k}  shape={v.shape}  last_t={v[-1].round(4).tolist()}")
                    for k in ('exterior_image_1_left', 'wrist_image_left'):
                        v = obs_np[k]
                        print(f"  obs.{k}  shape={v.shape}  dtype={v.dtype}  "
                              f"min={float(v.min()):.3f} max={float(v.max()):.3f} "
                              f"mean={float(v.mean()):.3f}")
                    print("[eval/dbg] --- ACTION SEQUENCE OUTPUT ---")
                    print(f"  action_seq.shape={action_seq.shape}")
                    for s_i in range(min(action_seq.shape[0], 8)):
                        a = action_seq[s_i]
                        print(f"  step {s_i}: pos={a[:3].round(3).tolist()}  "
                              f"aa={a[3:6].round(3).tolist()} (norm={np.linalg.norm(a[3:6]):.3f})  "
                              f"g={a[6]:.3f}")
                    print("[eval/dbg] === END ===")

                # (c) UMI action timestamps (FIX A: step k at t_obs+(k-offset)*dt).
                action_timestamps = (np.arange(len(action_seq), dtype=np.float64)
                                     - action_offset) * dt + t_obs

                # (d) discard outdated
                curr_time = time.time()
                is_new = action_timestamps > (curr_time + action_exec_lat)
                if not is_new.any():
                    next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                    action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                    action_seq = action_seq[[-1]]
                    print(f"  Over budget — sending last action only at t+{action_timestamps[0]-curr_time:.3f}s")
                else:
                    action_seq = action_seq[is_new]
                    action_timestamps = action_timestamps[is_new]

                # Degenerate chunk -> hold the current pose for one dt instead of
                # executing an OOD diffusion sample (match eval_dp_real).
                if _degenerate_chunk:
                    rs_now = ctrl.get_state(k=1)
                    cur6 = np.asarray(rs_now['ActualTCPPose'][-1]).astype(np.float64)
                    try:
                        cur_g = float(gripper.get_state(k=1)['gripper_width'][-1]) / ART_MAX_WIDTH
                    except Exception:
                        cur_g = 1.0
                    action_seq = np.concatenate([cur6, [cur_g]])[None]   # (1,7) hold
                    action_timestamps = np.array([time.time() + dt])
                    print(f"  [degenerate-chunk] intra_step_max={_intra.max()*1000:.0f}mm "
                          f"-> HOLD this cycle (OOD diffusion sample, not executed)")

                # (e) ahead-of-time schedule_waypoint
                # Robot: every step. Gripper: ONLY on detected OPEN<->CLOSE transition
                # (FrankaPolicyEnv.exec_actions:592 pattern; ART is binary).
                applied = 0
                n_act = len(action_seq)
                gripper_is_open_per_step = []
                # Raw policy gripper output (continuous, [0,1]) for each action
                # step — captured BEFORE the safety check so rejected poses still
                # contribute their gripper value. Used by the [gripper/raw] log
                # below to diagnose 0.5-threshold dithering.
                grip_norms_per_step = []
                # Keep-alive: if every action is rejected, the cartesian impedance
                # controller starves and polymetis drops the policy (see
                # FrankaPositionalController auto-recovery). Tracking the
                # last-known-good pose lets us send a hold waypoint instead,
                # mirroring eval_franka_policy.py which simply trusts the
                # policy. We are stricter: reject the action AND hold position.
                first_step_pose6_for_keepalive = None
                # Re-anchor the per-step motion guard to the robot's CURRENT TCP
                # each cycle (FIX A long horizon makes a stale prev_action_pos
                # manufacture false 200-400mm STEP rejects). Match eval_dp_real.
                rs_seed_cycle = ctrl.get_state(k=1)
                prev_action_pos = np.asarray(
                    rs_seed_cycle['ActualTCPPose'][-1, 0:3]).astype(np.float64)
                for i in range(len(action_seq)):
                    act = action_seq[i]
                    pos = act[0:3].astype(np.float64)
                    aa  = act[3:6].astype(np.float64)
                    grip_norm = float(act[6])
                    grip_norms_per_step.append(grip_norm)

                    if not safety_ok(pos, prev_action_pos, dt):
                        rejected_by_safety += 1
                        print(f"  [safety] rejected action_pos={pos.round(3)} -> hold-in-place")
                        # Send a hold waypoint at the current robot pose so the
                        # cartesian-impedance loop keeps eating commands and
                        # does not get auto-recovered by FrankaInterpController.
                        if first_step_pose6_for_keepalive is None:
                            rs_now = ctrl.get_state(k=1)
                            cur_pose = np.asarray(rs_now["ActualTCPPose"][-1]).astype(np.float64)
                            first_step_pose6_for_keepalive = cur_pose
                        ctrl.schedule_waypoint(
                            first_step_pose6_for_keepalive,
                            target_time=action_timestamps[i] - robot_act_lat)
                        gripper_is_open_per_step.append(None)
                        continue

                    pose6 = np.concatenate([pos, aa])
                    ctrl.schedule_waypoint(
                        pose6, target_time=action_timestamps[i] - robot_act_lat)
                    gripper_is_open_per_step.append(grip_norm > 0.5)
                    prev_action_pos = pos
                    applied += 1
                n_actions_total += applied
                if _verbose_step:
                    print(f"[eval/step] iter={iter_idx} (e) done. applied={applied}/{n_act}. "
                          f"(f) viewer + (g) precise_wait ...", flush=True)

                # ---- Gripper raw-output diagnostics -------------------------
                # The policy emits a CONTINUOUS gripper value act[6] in [0,1]
                # for each of the steps_per_inference (=8) action steps. eval
                # thresholds it at 0.5 into a binary ART open/close and issues a
                # schedule_waypoint only on the FIRST open<->close transition in
                # the burst. Because the gripper runs at 60Hz while inference is
                # 10Hz (8 future steps per burst), at most one binary command is
                # sent per inference even though 8 candidate decisions exist.
                # Logging the full per-step vector + how many steps sit within
                # ±0.1 of 0.5 reveals whether the policy is decisive (values
                # near 0/1) or dithering around the threshold (which would make
                # the binary command chatter open/close every inference).
                if grip_norms_per_step:
                    _g = np.asarray(grip_norms_per_step, dtype=np.float32)
                    _near = int(np.sum(np.abs(_g - 0.5) < 0.1))
                    _mask = (_g > 0.5).astype(int).tolist()
                    _prev_open = getattr(main, "_last_gripper_is_open", None)
                    _state = ('OPEN' if _prev_open else
                              'CLOSE' if _prev_open is not None else '?')
                    try:
                        _gw_now = float(gripper.get_state(k=1)['gripper_width'][-1])
                        _gw_str = f" width={_gw_now*1000:4.0f}mm"
                    except Exception:
                        _gw_str = ""
                    print(f"  [gripper/raw] iter={iter_idx:4d} "
                          f"g=[{', '.join(f'{x:.3f}' for x in _g)}] "
                          f"thr=0.5 open_mask={_mask} "
                          f"range=[{float(_g.min()):.3f},{float(_g.max()):.3f}] "
                          f"near0.5={_near}/{len(_g)} "
                          f"cmd_state={_state}{_gw_str}",
                          flush=True)

                # Gripper transition detection — single binary command per cycle.
                if not hasattr(main, "_last_gripper_is_open"):
                    gs0 = gripper.get_state(k=1)
                    main._last_gripper_is_open = float(gs0["gripper_width"][-1]) > 0.05
                transition_idx = None
                prev_is_open = main._last_gripper_is_open
                for i, is_open in enumerate(gripper_is_open_per_step):
                    if is_open is None:
                        continue
                    if is_open != prev_is_open:
                        transition_idx = i
                        break
                    prev_is_open = is_open
                if transition_idx is not None:
                    target_is_open = gripper_is_open_per_step[transition_idx]
                    # 0.0 close-width matches gripper_specs.py 'art' = data
                    # collection convention. 0.005 was a Franka-Hand carryover.
                    target_width = ART_MAX_WIDTH if target_is_open else 0.0
                    gripper.schedule_waypoint(
                        target_width,
                        target_time=action_timestamps[transition_idx] - grip_act_lat)
                    main._last_gripper_is_open = target_is_open
                    print(f"  [gripper] {'OPEN' if target_is_open else 'CLOSE'} @ action[{transition_idx}]")
                else:
                    valid = [v for v in gripper_is_open_per_step if v is not None]
                    if valid:
                        main._last_gripper_is_open = valid[-1]

                # (f) viewer update (light — every cycle)
                status = [
                    f'EVAL — iter {iter_idx}',
                    f't elapsed {time.time()-eval_t_start:5.1f}s / {max_duration:.0f}s',
                    f'obs lat   {obs_lat_now*1000:5.1f} ms',
                    f'inference {inf_ms:5.1f} ms',
                    f'actions   {applied}/{n_act} applied',
                    f'rejected (safety) {rejected_by_safety}',
                ]
                try:
                    write_viewer_jpeg(camera, status)
                except Exception:
                    pass

                # (g) wait for next cycle with one camera-frame headroom
                if _verbose_step:
                    print(f"[eval/step] iter={iter_idx} (f) done. (g) precise_wait ...",
                          flush=True)
                # CRITICAL (2026-05-27): pass time_func=time.time.
                # t_cycle_end is wall-clock (derived from eval_t_start =
                # time.time()+1), but precise_wait defaults to time.monotonic
                # (precise_sleep.py:16). Without this arg, t_wait =
                # t_cycle_end(~1.78e9) - monotonic(~1e5) ~= 1.78e9 s, so
                # time.sleep() blocked for ~56 years after iter 0 -- the loop
                # appeared to "run once then hang", which starved the Franka
                # 1s watchdog and killed the child controllers. The pre-loop
                # wait at line ~759 already passes time_func=time.time; this
                # in-loop call had simply missed it.
                precise_wait(t_cycle_end - frame_latency, time_func=time.time)
                iter_idx += steps_per_inference
                if _verbose_step:
                    print(f"[eval/step] iter={iter_idx - steps_per_inference} cycle complete; "
                          f"next iter starting", flush=True)

        except KeyboardInterrupt:
            print("\n[eval] KeyboardInterrupt — stopping", flush=True)
        except SystemExit as e:
            # signal_handler 가 sys.exit() 호출 시 SystemExit. log 에 명시.
            print(f"\n[eval] SystemExit code={e.code} — stopping", flush=True)
            raise
        except BaseException as e:
            # Catch-all so a sudden non-Keyboard exception still logs
            # before finally runs.
            print(f"\n[eval] unhandled exception in main loop: "
                  f"{type(e).__name__}: {e}", flush=True)
            import traceback as _tb
            _tb.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            raise
        finally:
            # Every print here gets flush=True so a kill-9 mid-finally still
            # leaves a paper trail of which step was running. Earlier 2026-05-21
            # 00:34 run terminated with NO finally output despite the daemon
            # log showing client.shutdown() reached -- suggesting Python's
            # default line-buffer was discarded with the process. Forcing
            # flush after each line addresses that.
            print("[eval] returning to ready pose", flush=True)
            # CRITICAL: the calibration inputs live in the DAEMON process
            # (diffusion_unet_hybrid_image_policy calls globalvar.appendInput
            # inside policy.predict_action, which runs there). This process's
            # globalvar list is ALWAYS empty, so the old local torch.save wrote
            # an empty file. Ask the daemon to save before we shut it down.
            out_path = os.path.join(output_dir, 'DiffusionInput_calib.pth')
            try:
                n_saved, saved_path = client.save_calib(out_path)
                print(f"[INFO] daemon saved {n_saved} calib inputs -> {saved_path}",
                      flush=True)
            except Exception as e:
                print(f"[eval] WARN save_calib failed: {type(e).__name__}: {e}",
                      flush=True)

            try:
                # FrankaInterpolationController exposes move_home() (not
                # move_to_joint_positions, which lives on the inner polymetis
                # RobotInterface). Verified 2026-05-21 by failure log:
                #   "'FrankaInterpolationController' object has no attribute
                #    'move_to_joint_positions'"
                if hasattr(ctrl, 'move_home'):
                    # move_home() takes NO args (duration = self.home_time);
                    # time_to_go=4.0 raised TypeError so the arm never homed.
                    ctrl.move_home()
                elif hasattr(ctrl, 'move_to_joint_positions'):
                    ctrl.move_to_joint_positions(FRANKA_HOME_JOINTS, time_to_go=4.0)
                else:
                    print("[eval] WARN: no move_home/move_to_joint_positions; "
                          "letting ctrl.stop() bring robot to a soft halt", flush=True)
            except Exception as e:
                print(f"[eval] WARN move_home: {e}", flush=True)
            print("[eval] stopping FrankaInterpController...", flush=True)
            ctrl.stop(wait=True)
            print("[eval] stopping ArtGripperController...", flush=True)
            gripper.stop(wait=True)
            print("[eval] stopping MultiZed...", flush=True)
            camera.stop(wait=True)
            print("[eval] shutting down DP daemon (client.shutdown)...", flush=True)
            client.shutdown()
            elapsed = time.time() - eval_t_start
            print(f"[eval] done. actions={n_actions_total} "
                  f"rejected_safety={rejected_by_safety} elapsed={elapsed:.1f}s",
                  flush=True)


if __name__ == '__main__':
    main()
