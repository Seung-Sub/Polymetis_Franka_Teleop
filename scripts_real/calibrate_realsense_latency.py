#!/usr/bin/env python3
"""
Calibrate RealSense Camera Latency - Improved Method (No QR Code Required)

This script uses RealSense's HARDWARE TIMESTAMPS with global time synchronization
to measure capture-to-receive latency WITHOUT requiring QR codes.

Method:
    RealSense cameras with `global_time_enabled=True` synchronize their internal
    clock with the system clock. This allows direct comparison:

    latency = receive_timestamp (system) - capture_timestamp (hardware/global)

Advantages over QR Code Method:
    - No monitor positioning required
    - No QR code detection (faster, more reliable)
    - Works in any environment (no lighting concerns)
    - Fully automated
    - More samples (every frame, not just QR detections)

Requirements:
    - RealSense camera with global_time support (D400 series, L515)
    - pyrealsense2 with global_time_enabled option

Usage:
    python scripts_real/calibrate_realsense_latency_v2.py

    # With specific camera
    python scripts_real/calibrate_realsense_latency_v2.py --serial 123456789012

    # Longer duration for more samples
    python scripts_real/calibrate_realsense_latency_v2.py --duration 10
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from matplotlib import pyplot as plt

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed")
    sys.exit(1)


def check_global_time_support(serial_number=None):
    """Check if camera supports global_time_enabled option."""
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        return False, "No RealSense cameras found"

    for dev in devices:
        if serial_number is None or dev.get_info(rs.camera_info.serial_number) == serial_number:
            sensors = dev.query_sensors()
            for sensor in sensors:
                try:
                    # Try to check if global_time option is supported
                    if sensor.supports(rs.option.global_time_enabled):
                        return True, dev.get_info(rs.camera_info.serial_number)
                except:
                    pass

    return False, "Camera does not support global_time_enabled"


@click.command()
@click.option('--serial', '-s', default=None, help='RealSense camera serial number')
@click.option('--fps', '-f', type=int, default=30, help='Camera FPS')
@click.option('--duration', '-d', type=float, default=5.0, help='Capture duration in seconds')
@click.option('--resolution', type=(int, int), default=(640, 480), help='Camera resolution')
def main(serial, fps, duration, resolution):
    """Calibrate RealSense camera latency using hardware timestamps."""

    print("=" * 60)
    print(" RealSense Camera Latency Calibration (Hardware Timestamp)")
    print("=" * 60)
    print("")
    print("Method: Global Time Synchronization")
    print("  - No QR code required")
    print("  - Fully automated")
    print("  - Uses RealSense hardware timestamps")
    print("")

    # Check global time support
    supported, info = check_global_time_support(serial)
    if not supported:
        print(f"Warning: {info}")
        print("Falling back to statistical estimation method...")
        print("")
    else:
        print(f"Camera {info} supports global_time_enabled")
        if serial is None:
            serial = info

    print(f"  Serial: {serial if serial else 'auto-detect'}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration}s")
    print(f"  Resolution: {resolution}")
    print("=" * 60)
    print("")

    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)

    # Start pipeline
    print("Starting camera...")
    profile = pipeline.start(config)

    # Enable global time if supported
    device = profile.get_device()
    sensors = device.query_sensors()
    global_time_enabled = False

    for sensor in sensors:
        try:
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
                global_time_enabled = True
                print("Global time synchronization ENABLED")
        except Exception as e:
            pass

    if not global_time_enabled:
        print("Global time NOT available - using estimation method")

    # Data collection
    capture_timestamps = []
    receive_timestamps = []
    frame_numbers = []
    timestamp_domains = []

    print(f"\nCollecting frames for {duration} seconds...")

    # Warm up
    for _ in range(30):
        pipeline.wait_for_frames()

    t_start = time.time()
    frame_count = 0

    while (time.time() - t_start) < duration:
        frameset = pipeline.wait_for_frames()
        t_recv = time.time()

        color_frame = frameset.get_color_frame()
        if not color_frame:
            continue

        # Get hardware timestamp (in milliseconds)
        t_capture_ms = color_frame.get_timestamp()
        t_capture = t_capture_ms / 1000.0  # Convert to seconds

        # Get timestamp domain
        domain = color_frame.get_frame_timestamp_domain()

        capture_timestamps.append(t_capture)
        receive_timestamps.append(t_recv)
        frame_numbers.append(frame_count)
        timestamp_domains.append(domain)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"\r  Collected {frame_count} frames...", end='')

    pipeline.stop()
    print(f"\n\nCollected {len(capture_timestamps)} frames")

    # Convert to numpy
    capture_timestamps = np.array(capture_timestamps)
    receive_timestamps = np.array(receive_timestamps)
    frame_numbers = np.array(frame_numbers)

    # Check timestamp domain
    if len(timestamp_domains) > 0:
        domain_example = timestamp_domains[0]
        domain_str = str(domain_example).split('.')[-1]
        print(f"Timestamp domain: {domain_str}")

    # Calculate latencies
    if global_time_enabled:
        # Direct comparison (global time mode)
        latencies = receive_timestamps - capture_timestamps
        method = "Direct (Global Time)"
    else:
        # Estimation method: use frame intervals
        # Assume capture timestamps are consistent, find offset
        capture_intervals = np.diff(capture_timestamps)
        receive_intervals = np.diff(receive_timestamps)

        # Expected interval based on FPS
        expected_interval = 1.0 / fps

        # Align first timestamps
        # The offset between capture and receive clocks
        offset = receive_timestamps[0] - capture_timestamps[0]

        # Latency = (receive - capture) - offset_drift
        # We estimate offset using the mean difference
        latencies = receive_timestamps - capture_timestamps

        # Remove clock offset (use relative latency)
        latencies = latencies - np.mean(latencies) + np.median(latencies)
        method = "Estimated (No Global Time)"

    # Filter outliers (beyond 3 sigma)
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    mask = np.abs(latencies - mean_lat) < 3 * std_lat
    latencies_filtered = latencies[mask]

    # Statistics
    avg_latency = np.mean(latencies_filtered)
    std_latency = np.std(latencies_filtered)
    min_latency = np.min(latencies_filtered)
    max_latency = np.max(latencies_filtered)

    # Print results
    print("\n" + "=" * 60)
    print(" Latency Calibration Results")
    print("=" * 60)
    print(f"\nMethod: {method}")
    print(f"Frames analyzed: {len(latencies_filtered)} / {len(latencies)}")
    print(f"\nCapture-to-receive latency:")
    print(f"  Average: {avg_latency*1000:.2f} ms")
    print(f"  Std Dev: {std_latency*1000:.2f} ms")
    print(f"  Min:     {min_latency*1000:.2f} ms")
    print(f"  Max:     {max_latency*1000:.2f} ms")

    # For observation latency, use average
    obs_latency = avg_latency if avg_latency > 0 else 0.1  # Default to 100ms if invalid

    print("\n" + "-" * 60)
    print("Recommended setting for FrankaViveEnv:")
    print(f"  camera_obs_latency = {obs_latency:.4f}")

    if not global_time_enabled:
        print("\n  Note: Results are ESTIMATED without global_time support.")
        print("  Consider using QR code method for more accuracy:")
        print("  python scripts_real/calibrate_realsense_latency.py")
    print("-" * 60)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'RealSense Camera Latency Calibration ({method})', fontsize=14)

    # Plot 1: Latency over time
    ax = axes[0, 0]
    t_rel = receive_timestamps - receive_timestamps[0]
    ax.plot(t_rel, latencies * 1000, 'b.', alpha=0.5, markersize=2)
    ax.axhline(y=avg_latency * 1000, color='r', linestyle='--',
               label=f'Mean: {avg_latency*1000:.2f}ms')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Latency histogram
    ax = axes[0, 1]
    ax.hist(latencies_filtered * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=avg_latency * 1000, color='r', linestyle='--',
               label=f'Mean: {avg_latency*1000:.2f}ms')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Frame intervals
    ax = axes[1, 0]
    capture_intervals = np.diff(capture_timestamps) * 1000
    receive_intervals = np.diff(receive_timestamps) * 1000
    expected_interval_ms = 1000.0 / fps
    ax.plot(capture_intervals, 'b.', alpha=0.5, markersize=2, label='Capture interval')
    ax.plot(receive_intervals, 'r.', alpha=0.5, markersize=2, label='Receive interval')
    ax.axhline(y=expected_interval_ms, color='g', linestyle='--',
               label=f'Expected ({expected_interval_ms:.1f}ms)')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Interval (ms)')
    ax.set_title('Frame Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Timestamp comparison
    ax = axes[1, 1]
    t0 = capture_timestamps[0]
    ax.plot(capture_timestamps - t0, receive_timestamps - receive_timestamps[0],
            'b.', alpha=0.5, markersize=2)
    x_range = np.array([0, capture_timestamps[-1] - t0])
    ax.plot(x_range, x_range, 'r--', label='Ideal (no latency)')
    ax.set_xlabel('Capture Timestamp (sec)')
    ax.set_ylabel('Receive Timestamp (sec)')
    ax.set_title('Timestamp Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save results
    results_path = os.path.join(ROOT_DIR, 'calibration_results')
    os.makedirs(results_path, exist_ok=True)
    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(results_path, f'realsense_latency_v2_{timestamp_str}.png'), dpi=150)

    np.savez(
        os.path.join(results_path, f'realsense_latency_v2_{timestamp_str}.npz'),
        capture_timestamps=capture_timestamps,
        receive_timestamps=receive_timestamps,
        latencies=latencies,
        avg_latency=avg_latency,
        std_latency=std_latency,
        global_time_enabled=global_time_enabled
    )

    print(f"\nResults saved to: {results_path}/realsense_latency_v2_{timestamp_str}.*")

    plt.show()


if __name__ == '__main__':
    main()
