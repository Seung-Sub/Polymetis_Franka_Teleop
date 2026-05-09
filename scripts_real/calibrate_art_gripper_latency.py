#!/usr/bin/env python3
"""Calibrate ART gripper observation + action latency (direct TCP path).

The ART gripper does NOT go through Polymetis / ZeroRPC like the Franka
Hand. It speaks raw TCP to a daemon on :50053 (see
Hyundai_motors_Gripper/src/server.cpp). The existing
``calibrate_franka_gripper_latency.py`` measures the ZeroRPC path and
its numbers are NOT representative of the ART path.

This script:

  1. Opens an ArtGripperClient TCP connection.
  2. Measures **obs_latency** as median round-trip / 2 of get_state() calls.
     The daemon's GET_STATUS handler is a mutex-locked memcpy of the
     latest PDO frame (server.cpp:135-151), so the only real cost is
     the localhost TCP round-trip.
  3. Measures **action_latency** by sending alternating goto(open ↔ close)
     commands non-blocking and timing how long until the daemon-reported
     finger_width starts to change. This captures the full path:
     send ➜ TCP ➜ daemon control thread ➜ EtherCAT M2S ➜ slave actuator
     onset.

After printing stats it offers to write the measured numbers into
``install/latency_calibration.json`` so subsequent FrankaViveEnv /
FrankaPolicyEnv constructions pick them up automatically.

Usage:
    # ART daemon must be running (systemctl is-active art-gripper-daemon)
    python scripts_real/calibrate_art_gripper_latency.py
    python scripts_real/calibrate_art_gripper_latency.py --no_patch       # measure only
    python scripts_real/calibrate_art_gripper_latency.py --action_n 50    # more samples
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from statistics import median, mean, stdev

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import click


def _connect(host: str, port: int):
    # art_gripper_client lives in the Hyundai_motors_Gripper sister repo.
    # If the user has set ART_GRIPPER_PYPATH (start_teleop.sh does), respect it.
    extra = os.environ.get('ART_GRIPPER_PYPATH', os.path.expanduser(
        '~/Hyundai_motors_Gripper/python'))
    if os.path.isdir(extra) and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        from art_gripper_client import ArtGripperClient
    except ImportError as e:
        print(f'[ERROR] art_gripper_client not importable: {e}', file=sys.stderr)
        print(f'        Set ART_GRIPPER_PYPATH or pip install -e '
              f'~/Hyundai_motors_Gripper/python', file=sys.stderr)
        sys.exit(1)
    c = ArtGripperClient(host=host, port=port)
    c.connect()
    return c


def measure_obs_latency(gripper, n: int) -> dict:
    """One get_state() round-trip per iter; report median, p10, p50, p90."""
    samples = []
    for _ in range(n):
        t0 = time.monotonic()
        _ = gripper.get_state()
        t1 = time.monotonic()
        samples.append((t1 - t0) / 2.0)
        # micro pause to let the EtherCAT cycle run between polls
        time.sleep(0.005)
    samples.sort()
    return {
        'samples': samples,
        'median_s': samples[len(samples) // 2],
        'mean_s': mean(samples),
        'std_s': stdev(samples) if len(samples) > 1 else 0.0,
        'p10_s': samples[max(0, int(0.1 * len(samples)))],
        'p90_s': samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        'n': len(samples),
    }


def measure_action_latency(gripper, n: int, open_w: float, close_w: float,
                           settle_s: float) -> dict:
    """Send non-blocking goto, poll state, time until measurable motion onset.

    Threshold = 5 mm absolute change. Records the wallclock between
    'goto returned' and 'state.width crossed threshold'. Excludes
    samples where motion was never detected within ``timeout_s``.
    """
    samples = []
    timeouts = 0
    timeout_s = 1.0
    poll_period_s = 0.0  # busy poll for finest resolution

    # Start at known open state
    print(f'[action] homing to open={open_w*1000:.0f} mm ...')
    gripper.goto(width=open_w, blocking=True)
    time.sleep(settle_s)

    going_to_close = True
    for i in range(n):
        target = close_w if going_to_close else open_w
        try:
            initial = gripper.get_state().width
        except Exception as e:
            print(f'  [warn] get_state pre-iter {i}: {e}')
            continue
        t_send = time.monotonic()
        try:
            gripper.goto(width=target, blocking=False)
        except Exception as e:
            print(f'  [warn] goto iter {i}: {e}')
            continue
        # Poll until measurable motion
        t_observe = None
        while True:
            try:
                st = gripper.get_state()
            except Exception:
                break
            if abs(st.width - initial) > 0.005:
                t_observe = time.monotonic()
                break
            if time.monotonic() - t_send > timeout_s:
                break
            if poll_period_s > 0:
                time.sleep(poll_period_s)
        if t_observe is None:
            timeouts += 1
        else:
            dt = t_observe - t_send
            samples.append(dt)
            print(f'  [action #{i:02d}] {dt*1000:6.1f} ms  '
                  f'({initial*1000:.1f} → toward {target*1000:.0f} mm)')
        going_to_close = not going_to_close
        # let the gripper finish the motion before next iter
        time.sleep(settle_s)
    if not samples:
        return {'samples': [], 'n': 0, 'timeouts': timeouts}
    samples.sort()
    return {
        'samples': samples,
        'median_s': samples[len(samples) // 2],
        'mean_s': mean(samples),
        'std_s': stdev(samples) if len(samples) > 1 else 0.0,
        'min_s': samples[0],
        'max_s': samples[-1],
        'n': len(samples),
        'timeouts': timeouts,
    }


@click.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=50053, type=int)
@click.option('--obs_n', default=200, type=int,
              help='Number of get_state round-trips for obs_latency')
@click.option('--action_n', default=20, type=int,
              help='Number of open/close cycles for action_latency')
@click.option('--open_width', default=0.095, type=float,
              help='Open width in metres (KIST default 95 mm)')
@click.option('--close_width', default=0.005, type=float,
              help='Close width in metres')
@click.option('--settle_s', default=0.8, type=float,
              help='Seconds to wait between cycles for full settling')
@click.option('--patch/--no_patch', default=True,
              help='Write measured numbers into install/latency_calibration.json')
def main(host, port, obs_n, action_n, open_width, close_width, settle_s, patch):
    print('=' * 60)
    print(f'ART gripper latency calibration  (target {host}:{port})')
    print('=' * 60)

    g = _connect(host, port)
    init = g.get_state()
    print(f'[connect] initial width = {init.width*1000:.1f} mm '
          f'in_motion={init.is_in_motion} fault={init.is_fault}')

    obs = measure_obs_latency(g, obs_n)
    print()
    print(f'  obs_latency (round-trip / 2) over {obs["n"]} samples:')
    print(f'    median = {obs["median_s"]*1000:6.3f} ms')
    print(f'    mean   = {obs["mean_s"]*1000:6.3f} ms ± {obs["std_s"]*1000:.3f} ms')
    print(f'    p10/p90 = {obs["p10_s"]*1000:.3f} / {obs["p90_s"]*1000:.3f} ms')

    print()
    act = measure_action_latency(g, action_n, open_width, close_width, settle_s)
    print()
    if act['n'] == 0:
        print(f'  action_latency: NO SAMPLES (timeouts={act["timeouts"]}). '
              f'Increase --action_n or check gripper.')
    else:
        print(f'  action_latency over {act["n"]} samples '
              f'(timeouts={act["timeouts"]}):')
        print(f'    median = {act["median_s"]*1000:6.3f} ms')
        print(f'    mean   = {act["mean_s"]*1000:6.3f} ms ± {act["std_s"]*1000:.3f} ms')
        print(f'    range  = {act["min_s"]*1000:.1f} ~ {act["max_s"]*1000:.1f} ms')

    g.disconnect()

    if not patch:
        print()
        print('  --no_patch given; not writing JSON.')
        return

    # Decide which value to record. Median is more robust to outliers than mean.
    obs_lat = obs['median_s']
    act_lat = act.get('median_s')
    updates = {
        'gripper_obs_latency': {'art': float(round(obs_lat, 4))},
        '_calibration_dates': {'gripper_art': date.today().isoformat()},
    }
    if act_lat is not None:
        updates['gripper_action_latency'] = {'art': float(round(act_lat, 4))}

    print()
    print('  Writing to install/latency_calibration.json:')
    for k, v in updates.items():
        print(f'    {k} = {v}')

    from polymetis_franka_teleop.common.latency_config import patch_calibration
    p = patch_calibration(updates)
    print(f'  patched: {p}')


if __name__ == '__main__':
    main()
