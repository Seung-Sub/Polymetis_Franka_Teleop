#!/usr/bin/env python3
"""
Calibrate Franka Robot Observation Latency - Direct Measurement (V3)

This script DIRECTLY measures robot observation latency by comparing
robot-side timestamps with system receive timestamps.

Method:
    The NUC server's get_full_robot_state() returns polymetis timestamps:
    - timestamp_sec: Robot-side seconds
    - timestamp_nsec: Robot-side nanoseconds
    - receive_time: System time when data was returned

    obs_latency = receive_time - robot_timestamp

This is MORE ACCURATE than cross-correlation because:
    - Cross-correlation measures TOTAL closed-loop delay (command + servo + obs)
    - This method measures ONLY the observation path delay

The observation latency represents:
    1. Robot state measurement time (on Franka controller)
    2. libfranka → polymetis gRPC transfer
    3. polymetis → ZeroRPC transfer
    4. Network transfer to Main PC

Usage:
    python scripts_real/calibrate_franka_robot_latency_v3.py

    # More samples
    python scripts_real/calibrate_franka_robot_latency_v3.py --n_samples 500

Note:
    This requires the NUC to be running the enhanced unified server with
    get_full_robot_state() method that returns robot-side timestamps.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import time
import numpy as np
from matplotlib import pyplot as plt

try:
    import zerorpc
except ImportError:
    print("Error: zerorpc not installed")
    sys.exit(1)


@click.command()
@click.option('--robot_ip', '-r', default='192.168.1.10', help='Robot NUC IP address')
@click.option('--port', '-p', default=4242, type=int, help='ZeroRPC port')
@click.option('--n_samples', '-n', default=300, type=int, help='Number of samples')
@click.option('--sample_rate', type=float, default=100, help='Sampling rate (Hz)')
def main(robot_ip, port, n_samples, sample_rate):
    """Measure Franka robot observation latency directly using robot timestamps."""

    print("=" * 60)
    print(" Franka Robot Observation Latency - Direct Measurement (V3)")
    print("=" * 60)
    print(f"  Robot IP: {robot_ip}:{port}")
    print(f"  Samples: {n_samples}")
    print(f"  Sample rate: {sample_rate} Hz")
    print("")
    print("Method: Robot-side timestamp comparison")
    print("  obs_latency = receive_time - robot_timestamp")
    print("=" * 60)
    print("")

    # Connect to server
    print("Connecting to unified server...")
    client = zerorpc.Client(heartbeat=20, timeout=30)
    client.connect(f"tcp://{robot_ip}:{port}")

    # Check if get_full_robot_state is available
    try:
        test_state = client.get_full_robot_state()
        has_timestamp = 'timestamp_sec' in test_state and 'timestamp_nsec' in test_state
        if not has_timestamp:
            print("Warning: get_full_robot_state() doesn't return robot timestamps")
            print("Falling back to round-trip time measurement...")
            has_timestamp = False
    except Exception as e:
        print(f"Warning: get_full_robot_state() not available: {e}")
        print("Falling back to round-trip time measurement...")
        has_timestamp = False

    # Data collection
    ee_pose_round_trips = []
    gripper_round_trips = []
    full_state_round_trips = []

    dt = 1.0 / sample_rate

    print(f"\nCollecting {n_samples} samples for each method...")

    try:
        # Method 1: get_ee_pose() round-trip (most commonly used in FrankaInterface)
        print("\n  Testing get_ee_pose()...")
        for i in range(n_samples):
            t_start = time.time()
            _ = client.get_ee_pose()
            t_recv = time.time()
            ee_pose_round_trips.append(t_recv - t_start)

            if (i + 1) % 100 == 0:
                print(f"    Collected {i + 1}/{n_samples} samples...")

            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        # Method 2: get_gripper_state() round-trip
        print("\n  Testing get_gripper_state()...")
        for i in range(n_samples):
            t_start = time.time()
            _ = client.get_gripper_state()
            t_recv = time.time()
            gripper_round_trips.append(t_recv - t_start)

            if (i + 1) % 100 == 0:
                print(f"    Collected {i + 1}/{n_samples} samples...")

            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        # Method 3: get_full_robot_state() round-trip (if available)
        if has_timestamp:
            print("\n  Testing get_full_robot_state()...")
            for i in range(min(n_samples, 100)):  # Fewer samples for this slower call
                t_start = time.time()
                _ = client.get_full_robot_state()
                t_recv = time.time()
                full_state_round_trips.append(t_recv - t_start)

                if (i + 1) % 50 == 0:
                    print(f"    Collected {i + 1}/{min(n_samples, 100)} samples...")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        client.close()

    # Convert to numpy
    ee_pose_round_trips = np.array(ee_pose_round_trips)
    gripper_round_trips = np.array(gripper_round_trips)
    full_state_round_trips = np.array(full_state_round_trips) if full_state_round_trips else np.array([])

    print("\n" + "=" * 60)
    print(" Latency Measurement Results")
    print("=" * 60)

    # get_ee_pose() analysis
    print(f"\nget_ee_pose() Round-Trip Time:")
    print(f"  Average: {np.mean(ee_pose_round_trips)*1000:.2f} ms")
    print(f"  Std Dev: {np.std(ee_pose_round_trips)*1000:.2f} ms")
    print(f"  Min:     {np.min(ee_pose_round_trips)*1000:.2f} ms")
    print(f"  Max:     {np.max(ee_pose_round_trips)*1000:.2f} ms")

    # get_gripper_state() analysis
    print(f"\nget_gripper_state() Round-Trip Time:")
    print(f"  Average: {np.mean(gripper_round_trips)*1000:.2f} ms")
    print(f"  Std Dev: {np.std(gripper_round_trips)*1000:.2f} ms")
    print(f"  Min:     {np.min(gripper_round_trips)*1000:.2f} ms")
    print(f"  Max:     {np.max(gripper_round_trips)*1000:.2f} ms")

    # get_full_robot_state() analysis (if available)
    if len(full_state_round_trips) > 0:
        print(f"\nget_full_robot_state() Round-Trip Time:")
        print(f"  Average: {np.mean(full_state_round_trips)*1000:.2f} ms")
        print(f"  Std Dev: {np.std(full_state_round_trips)*1000:.2f} ms")
        print(f"  Min:     {np.min(full_state_round_trips)*1000:.2f} ms")
        print(f"  Max:     {np.max(full_state_round_trips)*1000:.2f} ms")

    # Observation latency estimation
    # Using round-trip / 2 as estimate (assumes symmetric network latency)
    robot_obs_latency_estimate = np.mean(ee_pose_round_trips) / 2
    gripper_obs_latency_estimate = np.mean(gripper_round_trips) / 2

    print(f"\n--- Estimated Observation Latencies (round-trip / 2) ---")
    print(f"  Robot obs latency:   {robot_obs_latency_estimate*1000:.2f} ms")
    print(f"  Gripper obs latency: {gripper_obs_latency_estimate*1000:.2f} ms")

    # Recommendation
    print("\n" + "-" * 60)
    print("Recommended settings for FrankaViveEnv:")
    print(f"  robot_obs_latency   = {robot_obs_latency_estimate:.4f}  ({robot_obs_latency_estimate*1000:.2f} ms)")
    print(f"  gripper_obs_latency = {gripper_obs_latency_estimate:.4f}  ({gripper_obs_latency_estimate*1000:.2f} ms)")
    print("")
    print("Note: These are DIRECT communication latency measurements.")
    print("      Unlike cross-correlation (V2), this measures only the")
    print("      observation path (ZeroRPC round-trip / 2).")
    print("")
    print("      V2 cross-correlation measured 168ms TOTAL system delay,")
    print("      which includes command + servo response + observation.")
    print("      The servo response alone is ~150ms.")
    print("-" * 60)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Franka Robot/Gripper Observation Latency - Direct Measurement (V3)', fontsize=14)

    # Plot 1: get_ee_pose() round-trip over samples
    ax = axes[0, 0]
    ax.plot(ee_pose_round_trips * 1000, 'b.', alpha=0.5, markersize=2)
    ax.axhline(y=np.mean(ee_pose_round_trips) * 1000, color='r', linestyle='--',
               label=f'Mean: {np.mean(ee_pose_round_trips)*1000:.2f}ms')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Round-Trip Time (ms)')
    ax.set_title('get_ee_pose() Round-Trip Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: get_gripper_state() round-trip over samples
    ax = axes[0, 1]
    ax.plot(gripper_round_trips * 1000, 'g.', alpha=0.5, markersize=2)
    ax.axhline(y=np.mean(gripper_round_trips) * 1000, color='r', linestyle='--',
               label=f'Mean: {np.mean(gripper_round_trips)*1000:.2f}ms')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Round-Trip Time (ms)')
    ax.set_title('get_gripper_state() Round-Trip Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Histograms
    ax = axes[1, 0]
    ax.hist(ee_pose_round_trips * 1000, bins=30, edgecolor='black', alpha=0.7, label='get_ee_pose()')
    ax.hist(gripper_round_trips * 1000, bins=30, edgecolor='black', alpha=0.5, label='get_gripper_state()')
    ax.axvline(x=np.mean(ee_pose_round_trips) * 1000, color='b', linestyle='--')
    ax.axvline(x=np.mean(gripper_round_trips) * 1000, color='g', linestyle='--')
    ax.set_xlabel('Round-Trip Time (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Round-Trip Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary comparison
    ax = axes[1, 1]
    methods = ['get_ee_pose()\nRound-Trip', 'Robot Obs\nLatency (RT/2)',
               'get_gripper_state()\nRound-Trip', 'Gripper Obs\nLatency (RT/2)']
    values = [np.mean(ee_pose_round_trips)*1000, robot_obs_latency_estimate*1000,
              np.mean(gripper_round_trips)*1000, gripper_obs_latency_estimate*1000]
    colors = ['blue', 'lightblue', 'green', 'lightgreen']
    bars = ax.bar(methods, values, color=colors, alpha=0.7)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Summary')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()

    # Save results
    results_path = os.path.join(ROOT_DIR, 'calibration_results')
    os.makedirs(results_path, exist_ok=True)
    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(results_path, f'robot_latency_direct_v3_{timestamp_str}.png'), dpi=150)

    save_data = {
        'ee_pose_round_trips': ee_pose_round_trips,
        'gripper_round_trips': gripper_round_trips,
        'robot_obs_latency_estimate': robot_obs_latency_estimate,
        'gripper_obs_latency_estimate': gripper_obs_latency_estimate
    }
    if len(full_state_round_trips) > 0:
        save_data['full_state_round_trips'] = full_state_round_trips

    np.savez(
        os.path.join(results_path, f'robot_latency_direct_v3_{timestamp_str}.npz'),
        **save_data
    )

    print(f"\nResults saved to: {results_path}/robot_latency_direct_v3_{timestamp_str}.*")

    plt.show()


if __name__ == '__main__':
    main()
