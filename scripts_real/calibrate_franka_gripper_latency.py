#!/usr/bin/env python3
"""Calibrate Franka Hand observation + action latency (polymetis gRPC :50052).

Mirrors :file:`calibrate_art_gripper_latency.py` but talks to fairo-polymetis'
``GripperInterface`` (gRPC :50052, configured in
``/home/kist/fairo/polymetis/polymetis/conf/launch_gripper.yaml``) instead of
the ART daemon's TCP server.

obs path:
    polymetis ``GripperInterface.get_state()`` returns a ``GripperState``
    protobuf with ``.width`` / ``.is_grasped`` / ``.is_moving``. Each call is
    a single gRPC round-trip. We report ``RTT/2`` median + the gripper's
    internal sample-period cache-age model term (Franka Hand publishes ~100 Hz
    per fairo's gripper_server impl) so the result is comparable to the ART
    calibrator's number.

action path:
    Send alternating non-blocking ``goto(open ↔ close)`` commands, busy-poll
    ``get_state()`` until ``|width − width_initial| > threshold``. The first
    width sample exceeding threshold marks actuator onset; we subtract the
    threshold-traversal time so the reported number reflects pure onset
    latency (network + libfranka command queue + actuator wake-up).

Pre-req:
    NUC running ``sudo bash /usr/local/sbin/start_franka_gripper.sh`` which
    launches ``python launch_gripper.py gripper=franka_hand`` and binds the
    gRPC server to :50052. Sanity:

        ssh kist@192.168.1.12 'ss -tlnp | grep 50052'

Usage:
    python scripts_real/calibrate_franka_gripper_latency.py
    python scripts_real/calibrate_franka_gripper_latency.py --no_patch
    python scripts_real/calibrate_franka_gripper_latency.py --robot_ip 192.168.1.12
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from statistics import mean, stdev

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import click


def _connect(ip: str, port: int):
    try:
        from polymetis import GripperInterface
    except ImportError as e:
        print(f'[ERROR] cannot import polymetis: {e}', file=sys.stderr)
        sys.exit(1)
    return GripperInterface(ip_address=ip, port=port)


def _state_dict(gripper):
    """Uniform dict view of the GripperState protobuf."""
    st = gripper.get_state()
    return {
        'width': float(st.width),
        'is_grasped': bool(st.is_grasped),
        'is_moving': bool(st.is_moving),
    }


def measure_obs_latency(gripper, n: int, sample_period_s: float) -> dict:
    """Round-trip / 2 + Franka Hand server sample-period cache-age model term.

    libfranka publishes joint/gripper state at a fixed rate; the polymetis
    gripper server polls it and serves the latest snapshot. Average cache age
    is ``sample_period_s / 2``.
    """
    rtt_samples = []
    for _ in range(n):
        t0 = time.monotonic()
        _ = gripper.get_state()
        t1 = time.monotonic()
        rtt_samples.append(t1 - t0)
        time.sleep(0.005)
    rtt_samples.sort()
    rtt_half_median = rtt_samples[len(rtt_samples) // 2] / 2.0
    cache_age_model = sample_period_s / 2.0
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
                           speed_m_s: float = 0.10, force_n: float = 20.0) -> dict:
    """Send non-blocking goto, poll state, time until measurable motion onset.

    Subtracts a small ``threshold/speed`` traversal offset so the reported
    number reflects just the onset path (network + libfranka command queue +
    actuator wake-up).
    """
    samples = []
    timeouts = 0
    timeout_s = 1.5
    threshold_m = threshold_mm / 1000.0
    traversal_offset_s = threshold_m / speed_m_s

    # Home to the open width first
    print(f'[action] homing to open={open_w*1000:.0f} mm ...')
    gripper.goto(width=open_w, speed=speed_m_s, force=force_n, blocking=True)
    time.sleep(settle_s)
    print(f'[action] threshold={threshold_mm:.1f} mm, speed={speed_m_s*1000:.0f} mm/s, '
          f'traversal-offset={traversal_offset_s*1000:.1f} ms (subtracted)')

    going_to_close = True
    for i in range(n):
        target = close_w if going_to_close else open_w
        try:
            initial = _state_dict(gripper)['width']
        except Exception as e:
            print(f'  [warn] get_state pre-iter {i}: {e}')
            continue
        t_send = time.monotonic()
        try:
            if going_to_close:
                # grasp toward close_w
                gripper.grasp(speed=speed_m_s, force=force_n,
                              grasp_width=close_w, blocking=False)
            else:
                gripper.goto(width=open_w, speed=speed_m_s, force=force_n,
                             blocking=False)
        except Exception as e:
            print(f'  [warn] command iter {i}: {e}')
            continue

        t_observe = None
        while True:
            try:
                st = _state_dict(gripper)
            except Exception:
                break
            if abs(st['width'] - initial) > threshold_m:
                t_observe = time.monotonic()
                break
            if time.monotonic() - t_send > timeout_s:
                break
        if t_observe is None:
            timeouts += 1
        else:
            raw_dt = t_observe - t_send
            dt = max(raw_dt - traversal_offset_s, 0.0)
            samples.append(dt)
            print(f'  [action #{i:02d}] raw={raw_dt*1000:6.1f} ms - '
                  f'traversal={traversal_offset_s*1000:.1f} ms = {dt*1000:6.1f} ms  '
                  f'({initial*1000:.1f} -> {target*1000:.0f} mm)')
        going_to_close = not going_to_close
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
@click.option('--robot_ip', default='192.168.1.12', help='polymetis gripper server IP (NUC)')
@click.option('--port', default=50052, type=int, help='polymetis gripper gRPC port')
@click.option('--obs_n', default=200, type=int)
@click.option('--action_n', default=20, type=int)
@click.option('--open_width', default=0.075, type=float, help='Open width (m), Franka Hand max')
@click.option('--close_width', default=0.005, type=float, help='Close width (m), libfranka safety')
@click.option('--settle_s', default=1.0, type=float, help='Settle time between cycles')
@click.option('--threshold_mm', default=1.0, type=float, help='Motion-onset detection threshold')
@click.option('--action_speed', default=0.10, type=float, help='gripper speed (m/s)')
@click.option('--action_force', default=20.0, type=float, help='gripper force (N)')
@click.option('--sample_period_ms', default=10.0, type=float,
              help='Franka Hand server publish period (ms) — controls cache-age term')
@click.option('--patch/--no_patch', default=True,
              help='Write measured numbers into install/latency_calibration.json')
def main(robot_ip, port, obs_n, action_n, open_width, close_width, settle_s,
         threshold_mm, action_speed, action_force, sample_period_ms, patch):
    print('=' * 60)
    print(f'Franka Hand latency calibration  (polymetis gRPC {robot_ip}:{port})')
    print('=' * 60)

    g = _connect(robot_ip, port)
    init = _state_dict(g)
    print(f'[connect] initial width = {init["width"]*1000:.1f} mm '
          f'is_moving={init["is_moving"]} is_grasped={init["is_grasped"]}')

    obs = measure_obs_latency(g, obs_n, sample_period_ms / 1000.0)
    print()
    print(f'  obs_latency over {obs["n"]} samples:')
    print(f'    gRPC RTT/2 median       = {obs["rtt_half_median_s"]*1000:6.3f} ms '
          f'(p10/p90 = {obs["p10_s"]*1000:.3f} / {obs["p90_s"]*1000:.3f})')
    print(f'    sample-period cache-age = {obs["cache_age_model_s"]*1000:6.3f} ms '
          f'(= {sample_period_ms:.1f} ms / 2)')
    print(f'    -> total obs_latency    = {obs["total_obs_latency_s"]*1000:6.3f} ms')

    print()
    act = measure_action_latency(g, action_n, open_width, close_width, settle_s,
                                 threshold_mm=threshold_mm,
                                 speed_m_s=action_speed,
                                 force_n=action_force)
    print()
    if act['n'] == 0:
        print(f'  action_latency: NO SAMPLES (timeouts={act["timeouts"]}). '
              f'Check NUC :50052 reachable and gripper is initialised.')
    else:
        print(f'  action_latency over {act["n"]} samples (timeouts={act["timeouts"]}):')
        print(f'    median = {act["median_s"]*1000:6.3f} ms')
        print(f'    mean   = {act["mean_s"]*1000:6.3f} ms ± {act["std_s"]*1000:.3f} ms')
        print(f'    range  = {act["min_s"]*1000:.1f} ~ {act["max_s"]*1000:.1f} ms')

    if not patch:
        print()
        print('  --no_patch given; not writing JSON.')
        return

    obs_lat = obs['total_obs_latency_s']
    act_lat = act.get('median_s')
    updates = {
        'gripper_obs_latency': {'franka': float(round(obs_lat, 4))},
        '_calibration_dates': {'gripper_franka': date.today().isoformat()},
    }
    if act_lat is not None:
        updates['gripper_action_latency'] = {'franka': float(round(act_lat, 4))}

    print()
    print('  Writing to install/latency_calibration.json:')
    for k, v in updates.items():
        print(f'    {k} = {v}')

    from polymetis_franka_teleop.common.latency_config import patch_calibration
    p = patch_calibration(updates)
    print(f'  patched: {p}')


if __name__ == '__main__':
    main()
