#!/usr/bin/env python3
"""
Calibrate Franka Gripper Latency - Improved Method (Non-blocking)

The original gripper calibration had an issue: gripper_goto/gripper_grasp
commands may be BLOCKING (waiting for motion to complete or start).

This improved version:
1. Uses a separate thread to send the gripper command (non-blocking measurement)
2. Measures precise timing by polling state at high frequency
3. Distinguishes between:
   - Command return latency (ZeroRPC roundtrip)
   - First movement latency (when gripper actually starts moving)
   - Total response time (when target is reached)

Method:
    For accurate communication latency:
    1. Start state polling in main thread
    2. Send command in separate thread
    3. Detect exact moment when gripper state changes
    4. Calculate: comm_latency = t_state_change - t_command_sent

Usage:
    python scripts_real/calibrate_franka_gripper_latency_v2.py

    # Faster sampling (requires good network)
    python scripts_real/calibrate_franka_gripper_latency_v2.py --sample_rate 200
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
    import gevent
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


class GripperCommandResult:
    """Container for command results from gevent greenlet."""
    def __init__(self):
        self.success = False
        self.t_send = None
        self.t_return = None
        self.rpc_latency = None
        self.error = None


def run_gripper_command(robot_ip, port, command, args, result):
    """Run gripper command in a separate greenlet with its own client."""
    # Create a separate client for this greenlet to avoid gevent conflicts
    client = zerorpc.Client(heartbeat=20, timeout=30)
    client.connect(f"tcp://{robot_ip}:{port}")

    result.t_send = time.time()
    try:
        if command == 'goto':
            client.gripper_goto(*args)
        elif command == 'grasp':
            client.gripper_grasp(*args)
        result.t_return = time.time()
        result.rpc_latency = result.t_return - result.t_send
        result.success = True
    except Exception as e:
        result.error = str(e)
        result.success = False
    finally:
        client.close()


@click.command()
@click.option('--robot_ip', '-r', default='192.168.1.10', help='Robot NUC IP address')
@click.option('--port', '-p', default=4242, type=int, help='ZeroRPC port')
@click.option('--n_cycles', '-n', default=3, type=int, help='Number of open/close cycles')
@click.option('--open_width', type=float, default=0.07, help='Open width in meters')
@click.option('--close_width', type=float, default=0.02, help='Close width in meters')
@click.option('--sample_rate', type=float, default=200, help='State sampling rate (Hz)')
def main(robot_ip, port, n_cycles, open_width, close_width, sample_rate):
    """Calibrate Franka gripper latency with non-blocking command."""

    print("=" * 60)
    print(" Franka Gripper Latency Calibration V2 (Non-blocking)")
    print("=" * 60)
    print(f"  Robot IP: {robot_ip}:{port}")
    print(f"  Cycles: {n_cycles}")
    print(f"  Open width: {open_width*1000:.1f} mm")
    print(f"  Close width: {close_width*1000:.1f} mm")
    print(f"  Sample rate: {sample_rate} Hz")
    print("")
    print("Method: Non-blocking command + high-frequency polling")
    print("")

    # Connect
    print("Connecting to unified server...")
    client = zerorpc.Client(heartbeat=20, timeout=30)
    client.connect(f"tcp://{robot_ip}:{port}")

    # Get initial state
    state = client.get_gripper_state()
    initial_width = state['width']
    print(f"Initial gripper width: {initial_width*1000:.1f} mm")

    dt = 1.0 / sample_rate
    movement_threshold = 0.0005  # 0.5mm movement detection threshold

    # Results storage
    rpc_latencies = []          # Time for ZeroRPC call to return
    first_movement_latencies = []  # Time from command send to first movement detected
    step_responses = []

    # Create a separate client for polling (main greenlet)
    poll_client = zerorpc.Client(heartbeat=20, timeout=30)
    poll_client.connect(f"tcp://{robot_ip}:{port}")

    try:
        # First, move to known starting position
        print("\nMoving to starting position...")
        client.gripper_goto(open_width, 0.05, 20.0)
        time.sleep(1.0)

        for cycle in range(n_cycles):
            print(f"\nCycle {cycle + 1}/{n_cycles}")

            # === CLOSE ===
            print(f"  Closing to {close_width*1000:.1f} mm...")

            # Get current state
            state = poll_client.get_gripper_state()
            start_width = state['width']

            # Prepare command result container
            cmd_result = GripperCommandResult()

            # Start command greenlet (uses its own client internally)
            cmd_greenlet = gevent.spawn(
                run_gripper_command,
                robot_ip, port, 'grasp',
                (0.1, 40.0, close_width),
                cmd_result
            )

            # Start polling immediately
            poll_times = []
            poll_widths = []
            poll_start = time.time()
            t_first_movement = None
            width = start_width

            while True:
                t_poll = time.time()

                # Get state using poll_client
                try:
                    state = poll_client.get_gripper_state()
                    width = state['width']
                    poll_times.append(t_poll)
                    poll_widths.append(width)

                    # Detect first movement
                    if t_first_movement is None:
                        if abs(width - start_width) > movement_threshold:
                            t_first_movement = t_poll
                except:
                    pass

                # Check if target reached
                if width <= close_width + 0.001:
                    break

                # Timeout
                if time.time() - poll_start > 5.0:
                    break

                gevent.sleep(dt)

            # Wait for command greenlet to finish
            cmd_greenlet.join()

            if cmd_result.success:
                rpc_latencies.append(cmd_result.rpc_latency)
                t_cmd_send = cmd_result.t_send

                if t_first_movement is not None:
                    first_mov_lat = t_first_movement - t_cmd_send
                    first_movement_latencies.append(first_mov_lat)
                    print(f"    RPC latency: {cmd_result.rpc_latency*1000:.1f} ms")
                    print(f"    First movement: {first_mov_lat*1000:.1f} ms")
                else:
                    print(f"    RPC latency: {cmd_result.rpc_latency*1000:.1f} ms")
                    print(f"    First movement: NOT DETECTED")

                step_responses.append({
                    'type': 'close',
                    'cycle': cycle,
                    't_cmd_send': t_cmd_send,
                    't_cmd_return': cmd_result.t_return,
                    't_first_movement': t_first_movement,
                    'poll_times': np.array(poll_times) - t_cmd_send if poll_times else np.array([]),
                    'poll_widths': np.array(poll_widths) if poll_widths else np.array([]),
                    'rpc_latency': cmd_result.rpc_latency,
                    'first_mov_latency': (t_first_movement - t_cmd_send) if t_first_movement else None
                })
            else:
                print(f"    Command failed: {cmd_result.error}")

            gevent.sleep(0.5)

            # === OPEN ===
            print(f"  Opening to {open_width*1000:.1f} mm...")

            state = poll_client.get_gripper_state()
            start_width = state['width']

            cmd_result = GripperCommandResult()
            cmd_greenlet = gevent.spawn(
                run_gripper_command,
                robot_ip, port, 'goto',
                (open_width, 0.1, 20.0),
                cmd_result
            )

            poll_times = []
            poll_widths = []
            poll_start = time.time()
            t_first_movement = None
            width = start_width

            while True:
                t_poll = time.time()
                try:
                    state = poll_client.get_gripper_state()
                    width = state['width']
                    poll_times.append(t_poll)
                    poll_widths.append(width)

                    if t_first_movement is None:
                        if abs(width - start_width) > movement_threshold:
                            t_first_movement = t_poll
                except:
                    pass

                if width >= open_width - 0.001:
                    break

                if time.time() - poll_start > 5.0:
                    break

                gevent.sleep(dt)

            cmd_greenlet.join()

            if cmd_result.success:
                rpc_latencies.append(cmd_result.rpc_latency)
                t_cmd_send = cmd_result.t_send

                if t_first_movement is not None:
                    first_mov_lat = t_first_movement - t_cmd_send
                    first_movement_latencies.append(first_mov_lat)
                    print(f"    RPC latency: {cmd_result.rpc_latency*1000:.1f} ms")
                    print(f"    First movement: {first_mov_lat*1000:.1f} ms")
                else:
                    print(f"    RPC latency: {cmd_result.rpc_latency*1000:.1f} ms")
                    print(f"    First movement: NOT DETECTED")

                step_responses.append({
                    'type': 'open',
                    'cycle': cycle,
                    't_cmd_send': t_cmd_send,
                    't_cmd_return': cmd_result.t_return,
                    't_first_movement': t_first_movement,
                    'poll_times': np.array(poll_times) - t_cmd_send if poll_times else np.array([]),
                    'poll_widths': np.array(poll_widths) if poll_widths else np.array([]),
                    'rpc_latency': cmd_result.rpc_latency,
                    'first_mov_latency': (t_first_movement - t_cmd_send) if t_first_movement else None
                })
            else:
                print(f"    Command failed: {cmd_result.error}")

            gevent.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        poll_client.close()
        client.close()

    # Analysis
    rpc_latencies = np.array(rpc_latencies)
    first_movement_latencies = np.array([l for l in first_movement_latencies if l is not None])

    print("\n" + "=" * 60)
    print(" Latency Calibration Results")
    print("=" * 60)

    if len(rpc_latencies) > 0:
        print(f"\nZeroRPC Call Latency (command send → command return):")
        print(f"  Average: {np.mean(rpc_latencies)*1000:.1f} ms")
        print(f"  Std Dev: {np.std(rpc_latencies)*1000:.1f} ms")
        print(f"  Min:     {np.min(rpc_latencies)*1000:.1f} ms")
        print(f"  Max:     {np.max(rpc_latencies)*1000:.1f} ms")

    if len(first_movement_latencies) > 0:
        print(f"\nFirst Movement Latency (command send → gripper moves):")
        print(f"  Average: {np.mean(first_movement_latencies)*1000:.1f} ms")
        print(f"  Std Dev: {np.std(first_movement_latencies)*1000:.1f} ms")
        print(f"  Min:     {np.min(first_movement_latencies)*1000:.1f} ms")
        print(f"  Max:     {np.max(first_movement_latencies)*1000:.1f} ms")

        # For observation latency, use the minimum of RPC and first movement
        # This represents the actual data delay
        obs_latency = min(np.mean(rpc_latencies), np.mean(first_movement_latencies))
    else:
        obs_latency = np.mean(rpc_latencies) if len(rpc_latencies) > 0 else 0.05

    print("\n" + "-" * 60)
    print("Recommended setting for FrankaViveEnv:")
    print(f"  gripper_obs_latency = {obs_latency:.4f}")
    print("")
    print("Note: This is the communication latency for gripper state.")
    print("      The gripper motion time is additional and handled separately.")
    print("-" * 60)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Franka Gripper Latency Calibration V2 (Non-blocking)', fontsize=14)

    # Plot 1: Step responses
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(step_responses)))
    for i, step in enumerate(step_responses):
        if len(step['poll_times']) > 0:
            ax.plot(step['poll_times'] * 1000, step['poll_widths'] * 1000,
                    color=colors[i], alpha=0.7,
                    label=f"{step['type'].capitalize()} {step['cycle']+1}")
            # Mark first movement
            if step['first_mov_latency'] is not None:
                ax.axvline(x=step['first_mov_latency']*1000, color=colors[i],
                           linestyle=':', alpha=0.5)
    ax.set_xlabel('Time since command (ms)')
    ax.set_ylabel('Width (mm)')
    ax.set_title('Step Responses (dotted lines = first movement)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: RPC latency histogram
    ax = axes[0, 1]
    if len(rpc_latencies) > 0:
        ax.hist(rpc_latencies * 1000, bins=15, edgecolor='black', alpha=0.7, color='blue')
        ax.axvline(x=np.mean(rpc_latencies) * 1000, color='r', linestyle='--',
                   label=f'Mean: {np.mean(rpc_latencies)*1000:.1f}ms')
        ax.set_xlabel('RPC Latency (ms)')
        ax.set_ylabel('Count')
        ax.set_title('ZeroRPC Call Latency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: First movement latency histogram
    ax = axes[1, 0]
    if len(first_movement_latencies) > 0:
        ax.hist(first_movement_latencies * 1000, bins=15, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(x=np.mean(first_movement_latencies) * 1000, color='r', linestyle='--',
                   label=f'Mean: {np.mean(first_movement_latencies)*1000:.1f}ms')
        ax.set_xlabel('First Movement Latency (ms)')
        ax.set_ylabel('Count')
        ax.set_title('First Movement Latency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Comparison
    ax = axes[1, 1]
    x = np.arange(len(step_responses))
    rpc_lats = [s['rpc_latency']*1000 if s['rpc_latency'] else 0 for s in step_responses]
    mov_lats = [s['first_mov_latency']*1000 if s['first_mov_latency'] else 0 for s in step_responses]

    width = 0.35
    ax.bar(x - width/2, rpc_lats, width, label='RPC Latency', color='blue', alpha=0.7)
    ax.bar(x + width/2, mov_lats, width, label='First Movement', color='green', alpha=0.7)
    ax.set_xlabel('Step Index')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('RPC vs First Movement Latency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    results_path = os.path.join(ROOT_DIR, 'calibration_results')
    os.makedirs(results_path, exist_ok=True)
    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(results_path, f'gripper_latency_v2_{timestamp_str}.png'), dpi=150)

    np.savez(
        os.path.join(results_path, f'gripper_latency_v2_{timestamp_str}.npz'),
        rpc_latencies=rpc_latencies,
        first_movement_latencies=first_movement_latencies,
        obs_latency=obs_latency
    )

    print(f"\nResults saved to: {results_path}/gripper_latency_v2_{timestamp_str}.*")

    plt.show()


if __name__ == '__main__':
    main()
