#!/usr/bin/env python3
"""Calibrate ZED camera observation latency (HW timestamp method).

ZED SDK's TIME_REFERENCE.IMAGE returns the camera's internal capture
timestamp at sub-microsecond resolution. With ``setSVOPosition(0)`` /
default settings the SDK syncs that clock to the system clock so:

    obs_latency = receive_time (sys) - capture_time (HW)

includes the full pipeline: sensor exposure end ➜ USB 3.0 frame ➜
SDK demux ➜ ``cam.grab()`` returns ➜ Python ``cv2.cvtColor`` (we don't
include the cvtColor; it happens after the timestamp snapshot).

Run for each camera you use. KIST has two:
    33538770  ZED 2i    (exterior)
    11667817  ZED Mini  (wrist)

Both cameras' obs latency goes into ``camera_obs_latency.zed`` because
the env code does not currently distinguish per-camera-model. If
calibration shows the two differ by > a few ms (USB topology / cable
/ exposure differences), report the larger value (worst-case latency
is what the latency-compensation algorithm should subtract).

Usage:
    # Calibrate the exterior camera
    python scripts_real/calibrate_zed_latency.py --serial 33538770

    # Both, then take the max
    python scripts_real/calibrate_zed_latency.py --serial 33538770 --serial 11667817

    # Just measure, don't write
    python scripts_real/calibrate_zed_latency.py --serial 33538770 --no_patch

    # More frames for tighter percentiles
    python scripts_real/calibrate_zed_latency.py --serial 33538770 --frames 1000
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from statistics import mean, stdev

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import click
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print('[ERROR] pyzed.sl not installed -- this calibrator must run inside '
          'the groot-client conda env (or any env with the ZED SDK Python '
          'wrapper installed).', file=sys.stderr)
    sys.exit(1)


def measure_one(serial: int, frames: int, fps: int, resolution: str) -> dict:
    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.depth_mode = sl.DEPTH_MODE.NONE  # we don't need depth, only RGB latency
    init.coordinate_units = sl.UNIT.METER
    res_map = {
        'VGA': sl.RESOLUTION.VGA,
        'HD720': sl.RESOLUTION.HD720,
        'HD1080': sl.RESOLUTION.HD1080,
        'HD2K': sl.RESOLUTION.HD2K,
    }
    init.camera_resolution = res_map[resolution]
    init.camera_fps = fps
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        # Mirror single_zed.py: POTENTIAL_CALIBRATION_ISSUE is a non-fatal
        # warning. Other errors are fatal.
        if str(err).strip().upper() == 'POTENTIAL CALIBRATION ISSUE':
            print(f'  [WARN] ZED {serial} reported POTENTIAL CALIBRATION ISSUE -- '
                  f'continuing (LEFT eye only, no depth used).')
        else:
            raise SystemExit(f'  [ERROR] could not open ZED with serial {serial}: {err}')

    rt = sl.RuntimeParameters()
    mat = sl.Mat()

    # Throw away first N frames (USB warm-up + auto-exposure stabilization)
    for _ in range(30):
        cam.grab(rt)

    samples_capture_to_receive = []  # seconds
    samples_receive_dt = []          # seconds (dt between consecutive frames, for fps verify)
    last_recv = None

    print(f'[zed {serial}] capturing {frames} frames at {resolution}@{fps}fps...')
    for i in range(frames):
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue
        receive_time = time.time()
        # Pull HW timestamp before we do any work (matches single_zed.py order)
        hw_ts_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        if hw_ts_ns <= 0:
            continue
        capture_ts = hw_ts_ns / 1e9
        lat = receive_time - capture_ts
        if 0 < lat < 1.0:  # sanity filter: discard outliers (clock skew, USB stalls)
            samples_capture_to_receive.append(lat)
        if last_recv is not None:
            samples_receive_dt.append(receive_time - last_recv)
        last_recv = receive_time
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{frames}] last latency = {lat*1000:6.2f} ms')
    cam.close()

    if not samples_capture_to_receive:
        return {'serial': serial, 'n': 0}
    arr = np.asarray(samples_capture_to_receive)
    dt_arr = np.asarray(samples_receive_dt) if samples_receive_dt else np.asarray([0.0])
    return {
        'serial': serial,
        'n': len(arr),
        'median_s': float(np.median(arr)),
        'mean_s': float(np.mean(arr)),
        'std_s': float(np.std(arr)),
        'p10_s': float(np.percentile(arr, 10)),
        'p90_s': float(np.percentile(arr, 90)),
        'p99_s': float(np.percentile(arr, 99)),
        'min_s': float(arr.min()),
        'max_s': float(arr.max()),
        'measured_fps': float(1.0 / dt_arr.mean()) if dt_arr.size else 0.0,
    }


@click.command()
@click.option('--serial', '-s', multiple=True, type=int, required=True,
              help='ZED serial number (repeat for multiple cameras)')
@click.option('--frames', default=600, type=int,
              help='Frames to capture per camera (default 600 = ~10 s @ 60 fps)')
@click.option('--fps', default=60, type=int,
              help='Camera fps for the measurement (matches recording fps)')
@click.option('--resolution', default='VGA', type=click.Choice(['VGA', 'HD720', 'HD1080', 'HD2K']),
              help='Camera resolution. KIST default = VGA (672x376) for max bandwidth headroom.')
@click.option('--patch/--no_patch', default=True,
              help='Write the worst-case (highest median) latency into install/latency_calibration.json')
def main(serial, frames, fps, resolution, patch):
    print('=' * 60)
    print(f'ZED camera obs_latency calibration')
    print('=' * 60)
    print(f'serials: {list(serial)}')
    print(f'frames per camera: {frames}')
    print(f'capture: {resolution} @ {fps} fps')
    print()

    results = []
    for sn in serial:
        try:
            r = measure_one(sn, frames, fps, resolution)
        except SystemExit as e:
            print(str(e))
            continue
        if r['n'] == 0:
            print(f'  [zed {sn}] NO SAMPLES (HW timestamps unavailable?)')
            continue
        print()
        print(f'  ZED {sn}: {r["n"]} samples, measured fps = {r["measured_fps"]:.1f}')
        print(f'    median = {r["median_s"]*1000:6.2f} ms')
        print(f'    mean   = {r["mean_s"]*1000:6.2f} ms ± {r["std_s"]*1000:.2f} ms')
        print(f'    p10/p90/p99 = {r["p10_s"]*1000:.2f} / '
              f'{r["p90_s"]*1000:.2f} / {r["p99_s"]*1000:.2f} ms')
        print(f'    range  = {r["min_s"]*1000:.2f} ~ {r["max_s"]*1000:.2f} ms')
        results.append(r)

    if not results:
        print('  no successful measurements; nothing to write.')
        return

    # Use the *worst-case* median across cameras as the env-wide camera_obs_latency.
    # Reasoning: the latency-compensation algorithm subtracts ``camera_obs_latency``
    # from receive_time to estimate the actual capture moment. Under-compensating
    # (using a too-small latency) is worse than over-compensating, because it
    # creates obs/action timing mismatches the policy can't recover from.
    worst = max(results, key=lambda r: r['median_s'])
    chosen = round(worst['median_s'], 4)
    print()
    print(f'  → chosen camera_obs_latency.zed = {chosen*1000:.2f} ms '
          f'(worst-case median, ZED {worst["serial"]})')

    if not patch:
        print('  --no_patch given; not writing JSON.')
        return

    from polymetis_franka_teleop.common.latency_config import patch_calibration
    updates = {
        'camera_obs_latency': {'zed': float(chosen)},
        '_calibration_dates': {'camera_zed': date.today().isoformat()},
    }
    p = patch_calibration(updates)
    print(f'  patched: {p}')


if __name__ == '__main__':
    main()
