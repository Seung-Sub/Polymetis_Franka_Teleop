#!/usr/bin/env python3
"""
Calibrate All System Latencies for Franka-Vive Data Collection.

This script provides a comprehensive latency calibration for:
1. Franka Robot (command → actual pose)
2. Franka Gripper (command → actual width)
3. RealSense Camera (capture → receive)

After calibration, it outputs recommended settings for FrankaViveEnv.

Usage:
    # Full calibration (all devices)
    python scripts_real/calibrate_all_latencies.py

    # Only robot latency
    python scripts_real/calibrate_all_latencies.py --robot-only

    # Only gripper latency
    python scripts_real/calibrate_all_latencies.py --gripper-only

    # Only camera latency
    python scripts_real/calibrate_all_latencies.py --camera-only

    # Skip specific calibration
    python scripts_real/calibrate_all_latencies.py --skip-camera

Prerequisites:
    - NUC running: run_unified (robot + gripper)
    - Vive running: run_steamvr, run_vive
    - Monitor visible to RealSense camera (for camera calibration)
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import time
import json
import numpy as np
from pathlib import Path


def check_vive_connection(host='127.0.0.1', port=12345, timeout=2.0):
    """Check if Vive input server is running."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except:
        return False


def check_robot_connection(robot_ip, port=4242, timeout=2.0):
    """Check if robot ZeroRPC server is running."""
    try:
        import zerorpc
        client = zerorpc.Client(heartbeat=20, timeout=timeout)
        client.connect(f"tcp://{robot_ip}:{port}")
        # Try a simple call
        client.get_ee_pose()
        client.close()
        return True
    except:
        return False


def check_realsense():
    """Check if RealSense cameras are connected."""
    try:
        from polymetis_franka_teleop.real_world.single_realsense import SingleRealsense
        serials = SingleRealsense.get_connected_devices_serial()
        return len(serials) > 0, serials
    except:
        return False, []


@click.command()
@click.option('--robot_ip', '-r', default='192.168.1.10', help='Robot NUC IP address')
@click.option('--robot_port', default=4242, type=int, help='ZeroRPC port')
@click.option('--vive_host', default='127.0.0.1', help='Vive input server host')
@click.option('--vive_port', default=12345, type=int, help='Vive input server port')
@click.option('--camera_serial', default=None, help='RealSense camera serial')
@click.option('--robot-only', is_flag=True, help='Only calibrate robot')
@click.option('--gripper-only', is_flag=True, help='Only calibrate gripper')
@click.option('--camera-only', is_flag=True, help='Only calibrate camera')
@click.option('--skip-robot', is_flag=True, help='Skip robot calibration')
@click.option('--skip-gripper', is_flag=True, help='Skip gripper calibration')
@click.option('--skip-camera', is_flag=True, help='Skip camera calibration')
@click.option('--output', '-o', default=None, help='Output config file path')
def main(robot_ip, robot_port, vive_host, vive_port, camera_serial,
         robot_only, gripper_only, camera_only,
         skip_robot, skip_gripper, skip_camera, output):
    """Comprehensive latency calibration for Franka-Vive system."""

    print("=" * 70)
    print(" Franka-Vive System Latency Calibration")
    print("=" * 70)
    print("")

    # Determine what to calibrate
    if robot_only:
        skip_gripper = skip_camera = True
    elif gripper_only:
        skip_robot = skip_camera = True
    elif camera_only:
        skip_robot = skip_gripper = True

    calibrate_robot = not skip_robot
    calibrate_gripper = not skip_gripper
    calibrate_camera = not skip_camera

    # Results storage
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'robot_ip': robot_ip,
        'latencies': {}
    }

    # ========== System Check ==========
    print("[1/4] System Check")
    print("-" * 40)

    # Check robot connection
    if calibrate_robot or calibrate_gripper:
        print(f"  Checking robot at {robot_ip}:{robot_port}...", end=' ')
        if check_robot_connection(robot_ip, robot_port):
            print("OK")
        else:
            print("FAILED")
            print(f"    Error: Cannot connect to robot. Is run_unified running on NUC?")
            if calibrate_robot:
                calibrate_robot = False
            if calibrate_gripper:
                calibrate_gripper = False

    # Check Vive connection (only needed for robot calibration)
    if calibrate_robot:
        print(f"  Checking Vive at {vive_host}:{vive_port}...", end=' ')
        if check_vive_connection(vive_host, vive_port):
            print("OK")
        else:
            print("FAILED")
            print(f"    Error: Cannot connect to Vive. Are run_steamvr and run_vive running?")
            calibrate_robot = False

    # Check RealSense cameras
    if calibrate_camera:
        print("  Checking RealSense cameras...", end=' ')
        has_camera, serials = check_realsense()
        if has_camera:
            print(f"OK ({len(serials)} camera(s): {', '.join(serials)})")
            if camera_serial is None:
                camera_serial = serials[0]
        else:
            print("FAILED")
            print("    Error: No RealSense cameras found.")
            calibrate_camera = False

    print("")

    if not any([calibrate_robot, calibrate_gripper, calibrate_camera]):
        print("No calibrations to perform. Please check your connections.")
        return

    # ========== Robot Latency Calibration ==========
    if calibrate_robot:
        print("[2/4] Robot Latency Calibration")
        print("-" * 40)
        print("  Hold Vive GRIP button and move the robot around.")
        print("  Release GRIP when done.")
        print("")

        robot_latency = calibrate_robot_latency(
            robot_ip, robot_port, vive_host, vive_port
        )

        if robot_latency is not None:
            results['latencies']['robot'] = robot_latency
            print(f"\n  Robot latency: {robot_latency:.4f} sec ({robot_latency*1000:.2f} ms)")
        else:
            print("\n  Robot calibration failed or skipped.")
        print("")
    else:
        print("[2/4] Robot Latency Calibration - SKIPPED")
        print("")

    # ========== Gripper Latency Calibration ==========
    if calibrate_gripper:
        print("[3/4] Gripper Latency Calibration")
        print("-" * 40)
        print("  The gripper will move automatically.")
        print("  Keep hands clear!")
        print("")

        input("  Press Enter to start gripper calibration...")

        gripper_latency = calibrate_gripper_latency(robot_ip, robot_port)

        if gripper_latency is not None:
            results['latencies']['gripper'] = gripper_latency
            print(f"\n  Gripper latency: {gripper_latency:.4f} sec ({gripper_latency*1000:.2f} ms)")
        else:
            print("\n  Gripper calibration failed or skipped.")
        print("")
    else:
        print("[3/4] Gripper Latency Calibration - SKIPPED")
        print("")

    # ========== Camera Latency Calibration ==========
    if calibrate_camera:
        print("[4/4] Camera Latency Calibration")
        print("-" * 40)
        print(f"  Using camera: {camera_serial}")
        print("  Position camera to see monitor.")
        print("  Press 'c' in camera window when QR code is visible.")
        print("")

        input("  Press Enter to start camera calibration...")

        camera_latency = calibrate_camera_latency(camera_serial)

        if camera_latency is not None:
            results['latencies']['camera'] = camera_latency
            print(f"\n  Camera latency: {camera_latency:.4f} sec ({camera_latency*1000:.2f} ms)")
        else:
            print("\n  Camera calibration failed or skipped.")
        print("")
    else:
        print("[4/4] Camera Latency Calibration - SKIPPED")
        print("")

    # ========== Summary ==========
    print("=" * 70)
    print(" Calibration Summary")
    print("=" * 70)
    print("")

    if 'robot' in results['latencies']:
        print(f"  Robot Latency:   {results['latencies']['robot']:.4f} sec "
              f"({results['latencies']['robot']*1000:.2f} ms)")
    if 'gripper' in results['latencies']:
        print(f"  Gripper Latency: {results['latencies']['gripper']:.4f} sec "
              f"({results['latencies']['gripper']*1000:.2f} ms)")
    if 'camera' in results['latencies']:
        print(f"  Camera Latency:  {results['latencies']['camera']:.4f} sec "
              f"({results['latencies']['camera']*1000:.2f} ms)")

    print("")
    print("-" * 70)
    print(" Recommended FrankaViveEnv Settings:")
    print("-" * 70)
    print("")
    print("  FrankaViveEnv(")
    print(f"      robot_ip='{robot_ip}',")

    if 'robot' in results['latencies']:
        print(f"      robot_obs_latency={results['latencies']['robot']:.4f},")
    else:
        print(f"      robot_obs_latency=0.004,  # default (not calibrated)")

    if 'gripper' in results['latencies']:
        print(f"      gripper_obs_latency={results['latencies']['gripper']:.4f},")
    else:
        print(f"      gripper_obs_latency=0.01,  # default (not calibrated)")

    if 'camera' in results['latencies']:
        print(f"      camera_obs_latency={results['latencies']['camera']:.4f},")
    else:
        print(f"      camera_obs_latency=0.1,  # default (not calibrated)")

    print("      # ... other parameters")
    print("  )")
    print("")

    # Save results
    results_path = Path(ROOT_DIR) / 'calibration_results'
    results_path.mkdir(exist_ok=True)

    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    config_file = results_path / f'latency_config_{timestamp_str}.json'

    with open(config_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {config_file}")

    # Also save to specified output if provided
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Also saved to: {output}")

    print("=" * 70)


def calibrate_robot_latency(robot_ip, robot_port, vive_host, vive_port):
    """Run robot latency calibration."""
    from multiprocessing.managers import SharedMemoryManager
    from scipy.spatial.transform import Rotation as R
    from polymetis_franka_teleop.real_world.franka_interpolation_controller import FrankaInterpolationController
    from polymetis_franka_teleop.real_world.vive_shared_memory import ViveSharedMemory
    from polymetis_franka_teleop.common.precise_sleep import precise_wait
    from polymetis_franka_teleop.common.latency_util import get_latency

    frequency = 30
    dt = 1 / frequency
    command_latency = dt / 2

    try:
        with SharedMemoryManager() as shm_manager:
            with FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                robot_port=robot_port,
                frequency=200,
                tcp_offset=0.1034,
                Kx_scale=1.0,
                Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),
                get_max_k=10000,
                verbose=False
            ) as controller, \
            ViveSharedMemory(
                shm_manager=shm_manager,
                host=vive_host,
                port=vive_port,
                frequency=200,
                verbose=False
            ) as vive:

                print("  Waiting for initialization...")
                time.sleep(1.0)
                print("  Ready! Hold GRIP and move the robot.")

                state = controller.get_state()
                target_pose = state['ActualTCPPose'].copy()

                t_start = time.time()
                t_target = []
                x_target = []

                clutch_active = False
                clutch_vr_start_pos = None
                clutch_vr_start_quat = None
                clutch_robot_start = None

                iter_idx = 0
                timeout = 60  # 60 second timeout

                while (time.time() - t_start) < timeout:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    precise_wait(t_sample, time_func=time.time)

                    vive_state = vive.get_state()
                    vive_pos = vive_state['position']
                    vive_quat = vive_state['quaternion']
                    grip_pressed = bool(vive_state['grip'])

                    if grip_pressed:
                        if not clutch_active:
                            clutch_active = True
                            clutch_vr_start_pos = vive_pos.copy()
                            clutch_vr_start_quat = vive_quat.copy()
                            state = controller.get_state()
                            clutch_robot_start = state['ActualTCPPose'].copy()
                            target_pose = clutch_robot_start.copy()
                            print("  [Recording...]")
                        else:
                            # Compute target pose
                            world_delta = vive_pos - clutch_vr_start_pos
                            r_start = R.from_quat(clutch_vr_start_quat)
                            local_pos = r_start.inv().apply(world_delta)
                            robot_pos = np.array([local_pos[1], local_pos[0], -local_pos[2]])

                            r_current = R.from_quat(vive_quat)
                            local_rot = r_start.inv() * r_current
                            rotvec = local_rot.as_rotvec()
                            angle = np.linalg.norm(rotvec)
                            if angle > 1e-8:
                                axis = rotvec / angle
                                robot_axis = np.array([axis[1], axis[0], -axis[2]])
                                robot_axis = robot_axis / np.linalg.norm(robot_axis)
                                robot_rot = R.from_rotvec(robot_axis * angle)
                            else:
                                robot_rot = R.identity()

                            target_pose[:3] = clutch_robot_start[:3] + robot_pos
                            r_start_robot = R.from_rotvec(clutch_robot_start[3:])
                            target_pose[3:] = (robot_rot * r_start_robot).as_rotvec()

                        t_target.append(t_command_target)
                        x_target.append(target_pose.copy())
                        controller.schedule_waypoint(target_pose, t_command_target)
                    else:
                        if clutch_active:
                            print("  [Stopped]")
                            break

                    precise_wait(t_cycle_end, time_func=time.time)
                    iter_idx += 1

                states = controller.get_all_state()

        if len(t_target) < 10:
            print("  Not enough data collected.")
            return None

        t_target = np.array(t_target)
        x_target = np.array(x_target)
        t_actual = states['robot_receive_timestamp']
        x_actual = states['ActualTCPPose']

        latencies = []
        for i in range(6):
            lat, _ = get_latency(
                x_target[..., i], t_target,
                x_actual[..., i], t_actual,
                force_positive=True
            )
            latencies.append(lat)

        return np.mean(latencies)

    except Exception as e:
        print(f"  Error: {e}")
        return None


def calibrate_gripper_latency(robot_ip, robot_port):
    """Run gripper latency calibration."""
    from multiprocessing.managers import SharedMemoryManager
    from polymetis_franka_teleop.real_world.franka_gripper_controller import FrankaGripperController
    from polymetis_franka_teleop.common.precise_sleep import precise_sleep
    from polymetis_franka_teleop.common.latency_util import get_latency

    duration = 10.0
    sample_dt = 1 / 100
    k = int(duration / sample_dt)
    sample_t = np.linspace(0, duration, k)
    width_trajectory = 0.02 * np.sin(2 * np.pi * sample_t / 2.5) + 0.04

    try:
        with SharedMemoryManager() as shm_manager:
            with FrankaGripperController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                gripper_port=robot_port,
                frequency=30,
                move_max_speed=0.15,
                get_max_k=int(k * 1.5),
                command_queue_size=int(k * 1.5),
                verbose=False
            ) as gripper:

                print("  Initializing gripper...")
                gripper.start_wait()
                time.sleep(0.5)

                gripper.schedule_waypoint(width_trajectory[0], time.time() + 1.0)
                precise_sleep(2.0)

                print("  Running sinusoidal motion test...")
                timestamps = time.time() + sample_t + 0.5

                for i in range(k):
                    gripper.schedule_waypoint(width_trajectory[i], timestamps[i])
                    if i % 100 == 0:
                        time.sleep(0.001)

                precise_sleep(duration + 2.0)
                states = gripper.get_all_state()

        latency, _ = get_latency(
            x_target=width_trajectory,
            t_target=timestamps,
            x_actual=states['gripper_width'],
            t_actual=states['gripper_receive_timestamp'],
            force_positive=True
        )

        return latency

    except Exception as e:
        print(f"  Error: {e}")
        return None


def calibrate_camera_latency(camera_serial):
    """Run camera latency calibration."""
    import cv2

    try:
        import qrcode
    except ImportError:
        print("  Error: qrcode package not installed. Run: pip install qrcode[pil]")
        return None

    from multiprocessing.managers import SharedMemoryManager
    from collections import deque
    from polymetis_franka_teleop.real_world.single_realsense import SingleRealsense

    try:
        with SharedMemoryManager() as shm_manager:
            with SingleRealsense(
                shm_manager=shm_manager,
                serial_number=camera_serial,
                resolution=(640, 480),
                capture_fps=60,
                get_max_k=300,
                verbose=False
            ) as camera:

                cv2.setNumThreads(1)
                detector = cv2.QRCodeDetector()

                qr_latency_deque = deque(maxlen=300)
                qr_det_queue = deque(maxlen=300)

                print("  Position camera to see QR code on monitor.")
                print("  Press 'c' when QR code is visible, 'q' to skip.")

                data = None
                while True:
                    data = camera.get(out=data)
                    cam_img = data['color'].copy()

                    code, corners, _ = detector.detectAndDecodeCurved(cam_img)

                    if len(code) > 0:
                        try:
                            ts_qr = float(code)
                            ts_recv = data['timestamp']
                            qr_det_queue.append(ts_recv - ts_qr)
                        except:
                            qr_det_queue.append(float('nan'))
                    else:
                        qr_det_queue.append(float('nan'))

                    if corners is not None:
                        color = (0, 255, 0) if len(code) > 0 else (0, 0, 255)
                        cv2.fillPoly(cam_img, corners.astype(np.int32), color)

                    # Generate QR
                    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H)
                    t_sample = time.time()
                    qr.add_data(str(t_sample))
                    qr.make(fit=True)
                    qr_img = np.array(qr.make_image()).astype(np.uint8) * 255
                    qr_img = np.repeat(qr_img[:, :, None], 3, axis=-1)
                    qr_img = cv2.resize(qr_img, (500, 500), cv2.INTER_NEAREST)

                    cv2.imshow('QR Code', qr_img)
                    t_show = time.time()
                    qr_latency_deque.append(t_show - t_sample)

                    cv2.imshow('Camera', cam_img)

                    key = cv2.pollKey()
                    if key == ord('c'):
                        break
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        return None

                # Process captured data
                data = camera.get(k=300)

        cv2.destroyAllWindows()

        qr_recv_map = {}
        for i in range(len(data['timestamp'])):
            code, _, _ = detector.detectAndDecodeCurved(data['color'][i])
            if len(code) > 0:
                try:
                    ts_qr = float(code)
                    if ts_qr not in qr_recv_map:
                        qr_recv_map[ts_qr] = data['timestamp'][i]
                except:
                    pass

        if len(qr_recv_map) < 5:
            print(f"  Only {len(qr_recv_map)} QR codes detected. Need at least 5.")
            return None

        avg_qr_latency = np.mean(qr_latency_deque) if len(qr_latency_deque) > 0 else 0
        t_offsets = [v - k - avg_qr_latency for k, v in qr_recv_map.items()]

        return np.mean(t_offsets)

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
