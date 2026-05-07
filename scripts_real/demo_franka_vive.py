"""
Demo script for Franka data collection with HTC Vive controller teleoperation.

Usage:
    python scripts_real/demo_franka_vive.py -o ./data/demo_session

    # Skip preflight check
    python scripts_real/demo_franka_vive.py -o ./data/demo_session --skip-preflight

    # Show all cameras
    python scripts_real/demo_franka_vive.py -o ./data/demo_session --show-all-cameras

Controls:
    Vive Controller:
        - Grip button: Clutch (hold to enable teleoperation)
        - Trigger: Toggle gripper (open/close)

    Keyboard:
        - 'c': Start recording episode
        - 's': Stop recording and save episode
        - Backspace: Drop current episode (then 'y' to confirm)
        - 'q': Quit

Architecture:
    - ViveTeleopProcess runs at 100Hz for responsive control
    - Robot controller reads directly from teleop buffer (teleop_mode)
    - Main loop runs at 10Hz for data recording only
"""

import sys
import os
import multiprocessing

# IMPORTANT: Must set spawn method before importing modules that use multiprocessing
# This is required for RealSense cameras to work properly in subprocess
multiprocessing.set_start_method('spawn', force=True)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)  # Insert at beginning to override installed packages
os.chdir(ROOT_DIR)

# diffusion_policy provides ReplayBuffer / cv2_util / pose_repr_util / rotation_transformer
# at runtime. Prefer the user's source checkout if it's at the canonical KIST path.
_DP_PATH = os.environ.get('DIFFUSION_POLICY_PATH',
                          os.path.expanduser('~/diffusion_policy'))
if os.path.isdir(_DP_PATH) and _DP_PATH not in sys.path:
    sys.path.insert(0, _DP_PATH)

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
from polymetis_franka_teleop.real_world.franka_vive_env import FrankaViveEnv
from polymetis_franka_teleop.common.precise_sleep import precise_wait
from polymetis_franka_teleop.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)


def run_preflight_check():
    """Run preflight check and return True if passed."""
    try:
        from preflight_check import run_preflight_check as preflight
        return preflight(check_robot=True, check_vive=True)
    except ImportError:
        print("Warning: preflight_check.py not found, skipping checks")
        return True


@click.command()
@click.option('--output', '-o', required=True, help='Output directory for data')
@click.option('--robot_ip', default='192.168.1.12', help='Robot NUC IP address (KIST default)')
@click.option('--robot_port', default=50051, help='Polymetis arm gRPC port (NUC)')
@click.option('--vive_host', default='127.0.0.1', help='Vive input server host')
@click.option('--vive_port', default=12345, help='Vive input server port')
@click.option('--camera_backend', default='zed', type=click.Choice(['zed', 'realsense']),
              help='Camera backend (default: zed for KIST)')
@click.option('--gripper_backend', default='art', type=click.Choice(['art', 'franka']),
              help='Gripper backend (default: art for Hyundai gripper)')
@click.option('--art_gripper_host', default='127.0.0.1', help='ART daemon TCP host')
@click.option('--art_gripper_port', default=50053, type=int, help='ART daemon TCP port')
@click.option('--camera_serials', '-c', multiple=True,
              help='Camera serial numbers (e.g., -c 33538770 -c 11667817 for ZED 2i + Mini)')
@click.option('--camera_resolution', default='1280x720',
              help='Camera resolution (WxH). ZED HD720 default.')
@click.option('--camera_fps', default=60, type=int, help='Camera FPS (ZED HD720 max=60)')
@click.option('--obs_resolution', default=224, type=int,
              help='Observation image resolution for policy (default: 224)')
@click.option('--vis_camera_idx', default=0, type=int, help='Camera index for visualization')
@click.option('--show_all_cameras', is_flag=True, default=False,
              help='Show all cameras in visualization')
@click.option('--frequency', default=10, type=int, help='Main loop frequency (Hz)')
@click.option('--teleop_frequency', default=100, type=int, help='Teleop process frequency (Hz)')
@click.option('--tcp_offset', default=None, type=float,
              help='TCP offset (auto: 0.1034m Franka Hand, 0.216m ART)')
@click.option('--skip_preflight', is_flag=True, default=False, help='Skip preflight check')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose output')
def main(output, robot_ip, robot_port, vive_host, vive_port,
         camera_backend, gripper_backend, art_gripper_host, art_gripper_port,
         camera_serials, camera_resolution, camera_fps, obs_resolution,
         vis_camera_idx, show_all_cameras, frequency, teleop_frequency,
         tcp_offset, skip_preflight, verbose):

    # Run preflight check
    if not skip_preflight:
        if not run_preflight_check():
            print("\nPreflight check failed. Use --skip-preflight to bypass.")
            sys.exit(1)
        print()  # Add blank line after preflight

    # Parse camera resolution
    try:
        res_parts = camera_resolution.lower().split('x')
        cam_width = int(res_parts[0])
        cam_height = int(res_parts[1])
    except (ValueError, IndexError):
        print(f"Invalid camera resolution: {camera_resolution}. Use format WxH (e.g., 640x480)")
        sys.exit(1)

    # Convert camera serials to list or None.
    # ZED uses int serials; RealSense uses str serials.
    if camera_serials:
        if camera_backend == 'zed':
            camera_serials = [int(s) for s in camera_serials]
        else:
            camera_serials = list(camera_serials)
    else:
        camera_serials = None

    n_cameras = len(camera_serials) if camera_serials else 2

    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            FrankaViveEnv(
                output_dir=output,
                robot_ip=robot_ip,
                robot_port=robot_port,
                gripper_port=robot_port,
                frequency=frequency,
                camera_backend=camera_backend,
                gripper_backend=gripper_backend,
                art_gripper_host=art_gripper_host,
                art_gripper_port=art_gripper_port,
                camera_serial_numbers=camera_serials,
                camera_resolution=(cam_width, cam_height),
                camera_fps=camera_fps,
                obs_image_resolution=(obs_resolution, obs_resolution),
                vive_host=vive_host,
                vive_port=vive_port,
                teleop_frequency=teleop_frequency,
                tcp_offset=tcp_offset,
                enable_multi_cam_vis=False,
                shm_manager=shm_manager,
                verbose=verbose,
            ) as env:

            cv2.setNumThreads(1)
            time.sleep(1.0)
            print('Ready!')
            print('')
            print('Controls:')
            print('  Vive Grip button: Hold to enable teleoperation (clutch)')
            print('  Vive Trigger: Toggle gripper open/close')
            print('  Vive Trackpad Press: Move to HOME position (works during recording)')
            print('  Vive Trackpad Touch (Grip OFF):')
            print('    - Touch Top (Y > 0.7): Rotate gripper clockwise')
            print('    - Touch Bottom (Y < -0.7): Rotate gripper counter-clockwise')
            print('')
            print('  Keyboard c: Start recording')
            print('  Keyboard s: Stop recording and save')
            print('  Keyboard h: Move to HOME position (keyboard fallback)')
            print('  Keyboard Backspace: Drop episode (then Y to confirm, N to cancel)')
            print('  Keyboard q: Quit')
            print('')
            print(f'Camera: {cam_width}x{cam_height} @ {camera_fps}fps')
            print(f'Obs resolution: {obs_resolution}x{obs_resolution}')
            print(f'Visualization: {"All cameras" if show_all_cameras else f"Camera {vis_camera_idx}"}')
            print('')

            # === UMI-compatible timing setup ===
            # t_start: loop start time (monotonic for precise timing)
            # t_cycle_end: when current cycle should end
            # t_sample: when to sample teleop input (slightly before cycle end)
            # t_command_target: when action will be executed (next cycle start)
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            command_latency = 1/100  # 10ms latency buffer (same as UMI)

            # Track clutch state for display
            clutch_was_engaged = False

            # Track home state
            home_command_sent = False  # Prevent sending multiple home commands

            # Track pending drop state (for two-key confirmation in OpenCV window)
            pending_drop = False
            pending_drop_time = 0

            while not stop:
                # === Calculate timing (UMI-style) ===
                t_cycle_end = t_start + (iter_idx + 1) * dt  # Current cycle end
                t_sample = t_cycle_end - command_latency     # Sample time
                t_command_target = t_cycle_end + dt          # Action execution time (next cycle)

                # Get observation (also accumulates obs for recording)
                obs = env.get_obs()

                # Handle keyboard input
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(
                            t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == KeyCode(char='h'):
                        # Allow home even during recording (action will be robot trajectory)
                        if not home_command_sent:
                            print('[HOME] Moving to home position via keyboard...')
                            env.move_home(wait=False)
                            home_command_sent = True
                            key_counter.clear()
                    elif key_stroke == Key.backspace:
                        if is_recording:
                            pending_drop = True
                            pending_drop_time = time.monotonic()
                            key_counter.clear()
                    elif key_stroke == KeyCode(char='y'):
                        if pending_drop:
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                            pending_drop = False
                            print('[DROP] Episode dropped!')
                    elif key_stroke == KeyCode(char='n'):
                        if pending_drop:
                            pending_drop = False
                            key_counter.clear()
                            print('[DROP] Cancelled')

                # Check pending drop timeout (5 seconds)
                if pending_drop and (time.monotonic() - pending_drop_time) > 5.0:
                    pending_drop = False
                    print('[DROP] Confirmation timeout - cancelled')

                # Check clutch state
                clutch_engaged = env.is_clutch_engaged()
                if clutch_engaged and not clutch_was_engaged:
                    if verbose:
                        print('[Teleop] Clutch ENGAGED - control active')
                elif not clutch_engaged and clutch_was_engaged:
                    if verbose:
                        print('[Teleop] Clutch RELEASED')
                clutch_was_engaged = clutch_engaged

                # Check for home request from Vive trackpad
                teleop_state = env.get_teleop_state()
                home_requested = bool(teleop_state.get('home_requested', 0))
                home_active = bool(teleop_state.get('home_active', 0))
                rotation_active = bool(teleop_state.get('rotation_active', 0))

                # Trigger home when trackpad is pressed
                if home_requested and not home_command_sent:
                    print('[HOME] Moving to home position via Vive trackpad...')
                    env.move_home(wait=False)
                    home_command_sent = True

                # Reset home_command_sent when home motion completes
                if not home_active and home_command_sent:
                    home_command_sent = False
                    print('[HOME] Home motion completed - Release grip, then press again to continue teleop')

                # Visualize
                if show_all_cameras and n_cameras > 1:
                    # Show all cameras in a grid
                    cam_images = []
                    for i in range(n_cameras):
                        img = obs[f'camera{i}_rgb'][-1, :, :, ::-1].copy()
                        cam_images.append(img)

                    # Stack horizontally (or in grid for more cameras)
                    if n_cameras == 2:
                        vis_img = np.hstack(cam_images)
                    else:
                        # For 3-4 cameras, use 2x2 grid
                        while len(cam_images) < 4:
                            cam_images.append(np.zeros_like(cam_images[0]))
                        row1 = np.hstack(cam_images[:2])
                        row2 = np.hstack(cam_images[2:4])
                        vis_img = np.vstack([row1, row2])
                else:
                    # Single camera view
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1, :, :, ::-1].copy()

                episode_id = env.replay_buffer.n_episodes

                # Build status text
                text = f'Episode: {episode_id}'
                if is_recording:
                    text += ', Recording!'

                # Add clutch status
                if clutch_engaged:
                    text += ' [CLUTCH]'

                # Add home status
                if home_active:
                    text += ' [HOME]'

                # Add rotation status
                if rotation_active:
                    text += ' [ROTATE]'

                cv2.putText(
                    vis_img,
                    text,
                    (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255, 255, 255)
                )

                # Add teleop info (teleop_state already retrieved above)
                gripper_state = teleop_state.get('gripper_state', 0.0)
                gripper_text = 'OPEN' if gripper_state < 0.5 else 'CLOSED'
                cv2.putText(
                    vis_img,
                    f'Gripper: {gripper_text}',
                    (10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    thickness=2,
                    color=(0, 255, 0) if gripper_state < 0.5 else (0, 0, 255)
                )

                # Show drop confirmation message
                if pending_drop:
                    remaining = 5.0 - (time.monotonic() - pending_drop_time)
                    drop_text = f'DROP EPISODE? Press Y to confirm, N to cancel ({remaining:.1f}s)'
                    # Draw background rectangle for visibility
                    cv2.rectangle(vis_img, (5, 80), (650, 115), (0, 0, 150), -1)
                    cv2.putText(
                        vis_img,
                        drop_text,
                        (10, 105),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        thickness=2,
                        color=(255, 255, 255)
                    )

                cv2.imshow('Franka Vive Demo', vis_img)
                cv2.pollKey()

                # === UMI-style timing ===
                # Wait until sample time before recording action
                precise_wait(t_sample)

                # Record action at t_sample (same timing as UMI's exec_actions)
                # Action is sampled from ViveTeleopProcess's ring buffer
                # Timestamp points to t_command_target (when action will be executed)
                if is_recording:
                    action_timestamp = t_command_target - time.monotonic() + time.time()
                    env.record_action(timestamp=action_timestamp)

                # Wait until cycle end
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == '__main__':
    main()
