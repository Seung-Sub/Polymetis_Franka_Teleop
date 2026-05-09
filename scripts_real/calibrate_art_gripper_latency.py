#!/usr/bin/env python3
"""Calibrate ART gripper observation + action latency (direct TCP path).

The ART gripper does NOT go through Polymetis like the Franka Hand. It
speaks raw TCP to a daemon on :50053 (see
Hyundai_motors_Gripper/src/server.cpp). For Franka Hand latency
measurement, copy this file as a template and substitute the polymetis
Franka Hand zerorpc protocol (``gripper.goto`` / ``gripper.get_state``
on :4242).

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
        # Match the production code path: art_gripper_controller.py uses
        # ArtGripperInterface, the Polymetis-style wrapper. Older snapshots
        # exported ArtGripperClient; fall back to it for forward-compat.
        try:
            from art_gripper_client import ArtGripperInterface as _Cls  # noqa: N813
        except ImportError:
            from art_gripper_client import ArtGripperClient as _Cls
    except ImportError as e:
        print(f'[ERROR] art_gripper_client not importable: {e}', file=sys.stderr)
        print(f'        Set ART_GRIPPER_PYPATH or pip install -e '
              f'~/Hyundai_motors_Gripper/python', file=sys.stderr)
        sys.exit(1)
    # ArtGripperInterface accepts ip_address=, ArtGripperClient accepts host=.
    try:
        c = _Cls(ip_address=host, port=port)
    except TypeError:
        c = _Cls(host=host, port=port)
    if hasattr(c, 'connect'):
        c.connect()
    return c


def measure_obs_latency(gripper, n: int, pdo_period_s: float) -> dict:
    """get_state() round-trip + EtherCAT PDO cache-age model term.

    The ART daemon's GET_STATUS handler is a mutex-locked memcpy of the
    *latest PDO frame* — server.cpp:135-151. The PDO frame itself is updated
    once every ``pdo_period_s`` seconds (10 ms at KIST default; see
    /home/user/Hyundai_motors_Gripper/src/thread/procCtrl.cpp:15
    targetPeriod_ms=10.0). When we read at a random moment inside that
    period, the data we get is between 0 and pdo_period_s old, with a uniform
    distribution -> mean cache age = pdo_period_s / 2.

    The ART daemon does NOT expose a server-side timestamp in its TCP wire
    protocol, so unlike the polymetis arm path we cannot directly subtract
    a reference clock. We add pdo_period_s/2 as a model term instead.

    Total obs_latency = round_trip / 2  +  pdo_period_s / 2
                        ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                        TCP transport   PDO cache age (avg)
    """
    rtt_samples = []
    for _ in range(n):
        t0 = time.monotonic()
        _ = gripper.get_state()
        t1 = time.monotonic()
        rtt_samples.append(t1 - t0)
        # micro pause so we sample uniformly across PDO periods
        time.sleep(0.005)
    rtt_samples.sort()
    rtt_half_median = rtt_samples[len(rtt_samples) // 2] / 2.0
    cache_age_model = pdo_period_s / 2.0
    return {
        'samples': rtt_samples,
        'rtt_half_median_s': rtt_half_median,
        'cache_age_model_s': cache_age_model,
        'total_obs_latency_s': rtt_half_median + cache_age_model,
        'mean_s': mean(rtt_samples) / 2.0,
        'std_s': stdev(rtt_samples) / 2.0 if len(rtt_samples) > 1 else 0.0,
        'p10_s': rtt_samples[max(0, int(0.1 * len(rtt_samples)))] / 2.0,
        'p90_s': rtt_samples[min(len(rtt_samples) - 1, int(0.9 * len(rtt_samples)))] / 2.0,
        'n': len(rtt_samples),
    }


def measure_action_latency(gripper, n: int, open_w: float, close_w: float,
                           settle_s: float, threshold_mm: float = 1.0,
                           speed_m_s: float = 0.20) -> dict:
    """Send non-blocking goto, poll state, time until measurable motion onset.

    The action_latency for compensation is "schedule -> actuator starts moving".
    Our metric (first detectable width change) inevitably includes a small
    traversal-to-threshold component:
        traversal_s = (threshold_mm / 1000) / speed_m_s
    With threshold_mm=1.0 and speed_m_s=0.20 that's 5 ms of traversal,
    dwarfed by the actuator onset itself (~100 ms). Smaller threshold or
    higher speed reduce this further but hit sensor-noise / safety limits.
    """
    samples = []
    timeouts = 0
    timeout_s = 1.0
    poll_period_s = 0.0  # busy poll for finest resolution

    threshold_m = threshold_mm / 1000.0
    traversal_offset_s = threshold_m / speed_m_s   # we'll subtract this from the raw measurement
    # Start at known open state
    print(f'[action] homing to open={open_w*1000:.0f} mm ...')
    gripper.goto(width=open_w, blocking=True)
    time.sleep(settle_s)
    print(f'[action] threshold={threshold_mm:.1f} mm, speed={speed_m_s*1000:.0f} mm/s, '
          f'traversal-offset={traversal_offset_s*1000:.1f} ms (subtracted)')

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
            gripper.goto(width=target, speed=speed_m_s, blocking=False)
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
            if abs(st.width - initial) > threshold_m:
                t_observe = time.monotonic()
                break
            if time.monotonic() - t_send > timeout_s:
                break
            if poll_period_s > 0:
                time.sleep(poll_period_s)
        if t_observe is None:
            timeouts += 1
        else:
            raw_dt = t_observe - t_send
            dt = max(raw_dt - traversal_offset_s, 0.0)  # subtract traversal time
            samples.append(dt)
            print(f'  [action #{i:02d}] raw={raw_dt*1000:6.1f} ms - '
                  f'traversal={traversal_offset_s*1000:.1f} ms = {dt*1000:6.1f} ms  '
                  f'({initial*1000:.1f} -> {target*1000:.0f} mm)')
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
@click.option('--threshold_mm', default=1.0, type=float,
              help='Width-change threshold for action onset detection. Smaller = '
                   'tighter onset measurement, larger = robust against sensor noise. '
                   'Traversal time is subtracted from the result automatically.')
@click.option('--action_speed', default=0.20, type=float,
              help='goto speed in m/s for the action measurement')
@click.option('--pdo_period_ms', default=10.0, type=float,
              help='Daemon PDO loop period in ms (controls obs cache-age model term). '
                   'Default 10 ms matches procCtrl.cpp:15 targetPeriod_ms.')
@click.option('--patch/--no_patch', default=True,
              help='Write measured numbers into install/latency_calibration.json')
def main(host, port, obs_n, action_n, open_width, close_width, settle_s,
         threshold_mm, action_speed, pdo_period_ms, patch):
    print('=' * 60)
    print(f'ART gripper latency calibration  (target {host}:{port})')
    print('=' * 60)

    g = _connect(host, port)
    init = g.get_state()
    print(f'[connect] initial width = {init.width*1000:.1f} mm '
          f'in_motion={init.is_in_motion} fault={init.is_fault}')

    obs = measure_obs_latency(g, obs_n, pdo_period_ms / 1000.0)
    print()
    print(f'  obs_latency over {obs["n"]} samples:')
    print(f'    TCP RTT/2 median  = {obs["rtt_half_median_s"]*1000:6.3f} ms '
          f'(p10/p90 = {obs["p10_s"]*1000:.3f} / {obs["p90_s"]*1000:.3f})')
    print(f'    PDO cache-age model = {obs["cache_age_model_s"]*1000:6.3f} ms '
          f'(= {pdo_period_ms:.1f} ms / 2)')
    print(f'    -> total obs_latency = {obs["total_obs_latency_s"]*1000:6.3f} ms')

    print()
    act = measure_action_latency(g, action_n, open_width, close_width, settle_s,
                                 threshold_mm=threshold_mm, speed_m_s=action_speed)
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

    # ArtGripperInterface uses .close(); legacy ArtGripperClient used .disconnect()
    if hasattr(g, 'close'):
        g.close()
    elif hasattr(g, 'disconnect'):
        g.disconnect()

    if not patch:
        print()
        print('  --no_patch given; not writing JSON.')
        return

    # Decide which value to record. Median is more robust to outliers than mean.
    obs_lat = obs['total_obs_latency_s']
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
