#!/usr/bin/env python3
"""Calibrate Franka arm latency on the polymetis gRPC path.

KIST production setup runs ``FrankaInterpolationController`` against
``polymetis launch_robot.py`` directly: pro4000 → NUC :50051 (raw gRPC).

This script measures the production path:

  obs_latency  = round-trip / 2 of polymetis.RobotInterface.get_robot_state()
  action_latency = wall-clock between robot.update_desired_joint_positions(t)
                   and the moment that target shows up in the next
                   get_robot_state() (lower bound on the libfranka servo onset).

After printing stats it (with ``--patch``, default) writes the obs result
into ``install/latency_calibration.json``. Action latency is *not* auto-
patched — measuring it requires a small joint perturbation (motion!), so
it's behind ``--measure_action`` and writes only when explicitly chosen.

Pre-req:
  * NUC polymetis arm server running on :50051 (sudo bash
    /usr/local/sbin/start_franka_arm.sh).
  * Robot at a safe joint configuration (e-stop in hand for the action
    measurement path).

Usage:
    # obs only (no robot motion, ~3 s)
    python scripts_real/calibrate_franka_arm_direct.py
    # obs + action (small ~5 mm perturbation; ~30 s)
    python scripts_real/calibrate_franka_arm_direct.py --measure_action

    # don't write JSON
    python scripts_real/calibrate_franka_arm_direct.py --no_patch
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from statistics import mean, stdev

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import click
import numpy as np


def measure_obs_latency(robot, n: int) -> dict:
    """Measure obs latency in two ways and report both:

    (1) ROUND-TRIP / 2: pure network RPC time. Misses libfranka 1 kHz cache
        residency (avg 0.5 ms) — the data we receive is from the *last*
        libfranka tick, which can be up to one tick period (1 ms) old.

    (2) FLOOR-OFFSET METHOD using server-side state.timestamp:
        a) For each sample, compute diff = receive_time - state.timestamp.
        b) The clock offset between the NUC (server) and pro4000 (client) is
           an unknown constant Δ. min(diff) ≈ Δ + minimum_possible_latency.
        c) The minimum latency happens when the libfranka tick runs *exactly*
           when our request arrives, so cache age ≈ 0 and only network return
           contributes. That floor ≈ Δ + RTT/2.
        d) Subtracting that floor from each diff gives **clock-offset-free
           latency above the network minimum**, i.e. the cache-age component.
        e) median(latencies) + RTT/2 = total obs_latency from libfranka tick
           to our reception, which is what UMI's latency-comp expects.

    This is robust to clock drift between pro4000 and the NUC.
    """
    rtt_samples = []
    diffs = []  # receive_time - state.timestamp (server ts) per sample
    for _ in range(n):
        t0 = time.monotonic()
        state = robot.get_robot_state()
        t1 = time.monotonic()
        recv_wall = time.time()
        rtt_samples.append(t1 - t0)
        ts = state.timestamp.seconds + state.timestamp.nanos * 1e-9
        diffs.append(recv_wall - ts)
        time.sleep(0.005)
    rtt_samples.sort()
    diffs_arr = np.asarray(diffs)
    rtt = np.asarray(rtt_samples)

    # Method 1: pure RTT/2
    rtt_half_median = float(np.median(rtt) / 2.0)
    # Method 2: floor-offset
    floor = float(diffs_arr.min())
    cache_ages = diffs_arr - floor              # >= 0; how much older than floor
    cache_age_median = float(np.median(cache_ages))
    # Total obs_latency = network return (RTT/2 floor) + cache age above floor
    total_latency = rtt_half_median + cache_age_median

    return {
        'n': len(diffs),
        'rtt_half_median_s': rtt_half_median,
        'cache_age_median_s': cache_age_median,
        'total_obs_latency_s': total_latency,
        'rtt_p10_s': float(np.percentile(rtt, 10) / 2.0),
        'rtt_p90_s': float(np.percentile(rtt, 90) / 2.0),
        'cache_age_p90_s': float(np.percentile(cache_ages, 90)),
        'estimated_clock_skew_s': floor - rtt_half_median,  # diagnostic
    }


def measure_action_latency(robot, n: int, perturbation_rad: float) -> dict:
    """Send a small joint perturbation; time wall-clock until state catches up.

    Uses cartesian_impedance start + update_desired_joint_positions to mirror
    exactly what ``franka_interpolation_controller.py`` does on every tick.

    For each iteration:
      1. Read current q.
      2. Compute perturbed target q' = q + ε * pattern.
      3. t_send = now()
      4. update_desired_joint_positions(q')
      5. Poll get_joint_positions() at 200 Hz; record t_observe when
         |q_obs - q| > 0.5 * ε.
      6. action_latency_sample = t_observe - t_send.
      7. Send q back so the robot stays put.

    Lower bound: this counts gRPC + libfranka command queue + servo onset,
    same path the env's update_desired_joint_positions(...) takes.
    """
    import torch
    samples = []
    timeouts = 0

    # Use a small alternating perturbation on joint 5 (wrist), which moves
    # the EE the least for a given joint angle change.
    target_axis = 5
    pattern = np.zeros(7)
    pattern[target_axis] = 1.0

    print(f'  [action] starting cartesian_impedance and probing joint {target_axis} '
          f'with ±{perturbation_rad*1000:.1f} mrad steps ...')
    # Stiff impedance so the joint actually moves promptly.
    Kx = torch.tensor([750., 750., 750., 15., 15., 15.])
    Kxd = torch.tensor([37., 37., 37., 2., 2., 2.])
    robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
    time.sleep(0.5)

    direction = 1
    for i in range(n):
        try:
            q0 = np.asarray(robot.get_joint_positions())
        except Exception as e:
            print(f'    [warn] get_joint_positions iter {i}: {e}')
            continue
        q1 = q0 + direction * perturbation_rad * pattern
        threshold = 0.5 * perturbation_rad
        t_send = time.monotonic()
        try:
            robot.update_desired_joint_positions(torch.from_numpy(q1).float())
        except Exception as e:
            print(f'    [warn] update_desired_joint_positions iter {i}: {e}')
            continue
        # Busy-poll observation
        t_observe = None
        timeout_s = 0.5
        while True:
            try:
                q_obs = np.asarray(robot.get_joint_positions())
            except Exception:
                break
            if abs(q_obs[target_axis] - q0[target_axis]) > threshold:
                t_observe = time.monotonic()
                break
            if time.monotonic() - t_send > timeout_s:
                break
        if t_observe is None:
            timeouts += 1
        else:
            samples.append(t_observe - t_send)
            print(f'    [{i:02d}] {(t_observe - t_send)*1000:6.1f} ms  '
                  f'Δq{target_axis}={(q_obs[target_axis] - q0[target_axis])*1000:.2f} mrad')
        # Hold the new target briefly, then return so robot doesn't drift
        time.sleep(0.2)
        try:
            robot.update_desired_joint_positions(torch.from_numpy(q0).float())
        except Exception:
            pass
        time.sleep(0.3)
        direction *= -1
    try:
        robot.terminate_current_policy()
    except Exception:
        pass

    if not samples:
        return {'samples_s': [], 'n': 0, 'timeouts': timeouts}
    samples.sort()
    return {
        'samples_s': samples,
        'n': len(samples),
        'median_s': samples[len(samples) // 2],
        'mean_s': mean(samples),
        'std_s': stdev(samples) if len(samples) > 1 else 0.0,
        'min_s': samples[0],
        'max_s': samples[-1],
        'timeouts': timeouts,
    }


@click.command()
@click.option('--robot_ip', default='192.168.1.12', help='NUC IP')
@click.option('--port', default=50051, type=int, help='polymetis arm gRPC port')
@click.option('--obs_n', default=300, type=int)
@click.option('--measure_action/--no_measure_action', default=False,
              help='Also measure action latency (small joint perturbation).')
@click.option('--action_n', default=10, type=int)
@click.option('--perturbation_rad', default=0.01, type=float,
              help='Joint perturbation size in radians (default 10 mrad).')
@click.option('--patch/--no_patch', default=True,
              help='Write measured numbers into install/latency_calibration.json')
def main(robot_ip, port, obs_n, measure_action, action_n, perturbation_rad, patch):
    print('=' * 60)
    print(f'Franka arm latency calibration (DIRECT polymetis :50051)')
    print(f'  target: {robot_ip}:{port}')
    print('=' * 60)

    try:
        from polymetis import RobotInterface
    except Exception as e:
        print(f'[ERROR] cannot import polymetis: {e}')
        sys.exit(1)

    robot = RobotInterface(ip_address=robot_ip, port=port)
    print(f'  connected; current joints (rad): '
          f'{np.asarray(robot.get_joint_positions()).round(3).tolist()}')

    obs = measure_obs_latency(robot, obs_n)
    print()
    print(f'  obs_latency over {obs["n"]} samples:')
    print(f'    network RTT/2 median   = {obs["rtt_half_median_s"]*1000:6.3f} ms '
          f'(p10/p90 = {obs["rtt_p10_s"]*1000:.3f} / {obs["rtt_p90_s"]*1000:.3f})')
    print(f'    cache-age median       = {obs["cache_age_median_s"]*1000:6.3f} ms '
          f'(p90 = {obs["cache_age_p90_s"]*1000:.3f})')
    print(f'    -> total obs_latency   = {obs["total_obs_latency_s"]*1000:6.3f} ms')
    print(f'    (estimated clock-skew NUC-pro4000 = {obs["estimated_clock_skew_s"]*1000:+.1f} ms)')

    act = None
    if measure_action:
        print()
        print('  ACTION measurement enabled — robot will move ~10 mrad on joint 5.')
        print('  Keep the e-stop in hand. Starting in 3 s ...')
        time.sleep(3.0)
        act = measure_action_latency(robot, action_n, perturbation_rad)
        print()
        if act['n'] == 0:
            print(f'  action_latency: NO SAMPLES (timeouts={act["timeouts"]}).')
        else:
            print(f'  action_latency over {act["n"]} samples (timeouts={act["timeouts"]}):')
            print(f'    median = {act["median_s"]*1000:6.3f} ms')
            print(f'    mean   = {act["mean_s"]*1000:6.3f} ms ± {act["std_s"]*1000:.3f} ms')
            print(f'    range  = {act["min_s"]*1000:.3f} ~ {act["max_s"]*1000:.3f} ms')

    if not patch:
        print()
        print('  --no_patch given; not writing JSON.')
        return

    from polymetis_franka_teleop.common.latency_config import patch_calibration
    updates = {
        'robot_obs_latency': float(round(obs['total_obs_latency_s'], 4)),
        '_calibration_dates': {'robot': date.today().isoformat()},
    }
    if act is not None and act['n'] > 0:
        updates['robot_action_latency'] = float(round(act['median_s'], 4))
    print()
    print('  Writing to install/latency_calibration.json:')
    for k, v in updates.items():
        print(f'    {k} = {v}')
    p = patch_calibration(updates)
    print(f'  patched: {p}')


if __name__ == '__main__':
    main()
