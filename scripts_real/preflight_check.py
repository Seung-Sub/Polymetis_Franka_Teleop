#!/usr/bin/env python3
"""
Pre-flight check script for Franka Vive Demo.

This script checks and cleans up any processes or resources that might
interfere with the demo before running.

Usage:
    python scripts_real/preflight_check.py

    # Or import and use programmatically:
    from scripts_real.preflight_check import run_preflight_check
    if run_preflight_check():
        # Safe to run demo
"""

import subprocess
import sys
import time
import os
import signal

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(msg, status='info'):
    """Print colored status message."""
    if status == 'ok':
        print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")
    elif status == 'warn':
        print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")
    elif status == 'error':
        print(f"  {Colors.RED}✗{Colors.RESET} {msg}")
    elif status == 'fix':
        print(f"  {Colors.BLUE}→{Colors.RESET} {msg}")
    else:
        print(f"  • {msg}")


def kill_process_by_pattern(pattern, signal_type=signal.SIGTERM, timeout=3):
    """Kill processes matching a pattern."""
    try:
        # Find PIDs matching pattern
        result = subprocess.run(
            ['pgrep', '-f', pattern],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            pids = [p for p in pids if p and p != str(os.getpid())]

            if pids:
                for pid in pids:
                    try:
                        os.kill(int(pid), signal_type)
                    except ProcessLookupError:
                        pass

                # Wait for processes to terminate
                time.sleep(timeout)

                # Force kill if still running
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

                return len(pids)
        return 0
    except Exception as e:
        return 0


def check_and_kill_demo_processes():
    """Check and kill any running demo processes."""
    print(f"\n{Colors.BOLD}[1/5] Checking for running demo processes...{Colors.RESET}")

    patterns = [
        'demo_franka_vive',
        'demo_real_umi',
        'SingleRealsense',
        'MultiRealsense',
        'FrankaInterpolationController',
        'FrankaGripperController',
        'ViveTeleopProcess',
        'ViveSharedMemory',
    ]

    total_killed = 0
    for pattern in patterns:
        killed = kill_process_by_pattern(pattern)
        if killed > 0:
            print_status(f"Killed {killed} process(es) matching '{pattern}'", 'fix')
            total_killed += killed

    if total_killed == 0:
        print_status("No conflicting demo processes found", 'ok')
    else:
        print_status(f"Cleaned up {total_killed} process(es) total", 'ok')
        time.sleep(1)  # Give time for cleanup

    return True


def check_and_kill_orphan_multiprocessing():
    """Check and kill orphaned multiprocessing spawned processes."""
    print(f"\n{Colors.BOLD}[2/5] Checking for orphaned multiprocessing...{Colors.RESET}")

    patterns = [
        'multiprocessing.spawn',
        'multiprocessing.resource_tracker',
    ]

    total_killed = 0
    for pattern in patterns:
        killed = kill_process_by_pattern(pattern, signal.SIGKILL, 1)
        if killed > 0:
            print_status(f"Killed {killed} orphaned '{pattern}' process(es)", 'fix')
            total_killed += killed

    if total_killed == 0:
        print_status("No orphaned multiprocessing processes found", 'ok')

    return True


def check_zed_cameras():
    """Check if ZED cameras are available (KIST default)."""
    print(f"\n{Colors.BOLD}[3/5] Checking ZED cameras...{Colors.RESET}")
    try:
        import pyzed.sl as sl
        devs = sl.Camera.get_device_list()
        if not devs:
            print_status("No ZED cameras detected!", 'error')
            return False
        print_status(f"Found {len(devs)} ZED camera(s):", 'ok')
        for d in devs:
            print_status(f"  sn={d.serial_number} {d.camera_model} ({d.camera_state})", 'info')
        avail = [d for d in devs if d.camera_state == sl.CAMERA_STATE.AVAILABLE]
        if not avail:
            print_status("All ZED devices are unavailable (in use elsewhere?)", 'warn')
            return False
        return True
    except ImportError:
        print_status("pyzed.sl not installed in this env — skip if using --camera_backend realsense", 'warn')
        return True


def check_realsense_cameras():
    """Check if RealSense cameras are available (alt backend)."""
    print(f"\n{Colors.BOLD}[3b] Checking RealSense cameras (optional)...{Colors.RESET}")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            print_status("No RealSense cameras detected (OK if using --camera_backend zed)", 'warn')
            return True
        print_status(f"Found {len(devs)} RealSense camera(s):", 'ok')
        for d in devs:
            print_status(f"  {d.get_info(rs.camera_info.name)}: {d.get_info(rs.camera_info.serial_number)}", 'info')
        return True
    except ImportError:
        print_status("pyrealsense2 not installed (OK if using --camera_backend zed)", 'warn')
        return True
    except Exception as e:
        print_status(f"Error checking RealSense: {e}", 'warn')
        return True


def check_vive_server():
    """Check if the ROS-free Vive input TCP server is up."""
    print(f"\n{Colors.BOLD}[4/5] Checking Vive input server...{Colors.RESET}")
    import socket
    host, port = '127.0.0.1', 12345
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        ok = sock.connect_ex((host, port)) == 0
        sock.close()
        if ok:
            print_status(f"Vive input listening at {host}:{port}", 'ok')
        else:
            print_status(f"Vive input NOT listening at {host}:{port}", 'warn')
            print_status("Start with: bash bin/start_vive_input.sh   (ROS-free)", 'info')
            print_status("(Make sure SteamVR GUI is open first.)", 'info')
        return True
    except Exception as e:
        print_status(f"Error checking Vive: {e}", 'warn')
        return True


def check_robot_connection(robot_ip: str = '192.168.1.12', robot_port: int = 50051):
    """Check polymetis arm gRPC reachability (no ZeroRPC indirection)."""
    print(f"\n{Colors.BOLD}[5/5] Checking polymetis arm at {robot_ip}:{robot_port}...{Colors.RESET}")
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        ok = sock.connect_ex((robot_ip, robot_port)) == 0
        sock.close()
        if ok:
            print_status(f"Polymetis arm reachable at {robot_ip}:{robot_port}", 'ok')
            return True
        print_status(f"Polymetis arm not reachable at {robot_ip}:{robot_port}", 'error')
        print_status("On NUC: sudo bash /usr/local/sbin/start_franka_arm.sh", 'info')
        print_status("Also: Franka Desk → Activate FCI", 'info')
        return False
    except Exception as e:
        print_status(f"Error checking polymetis arm: {e}", 'error')
        return False


def check_art_gripper_daemon(host: str = '127.0.0.1', port: int = 50053):
    """Check ART gripper TCP daemon (KIST extension)."""
    print(f"\n{Colors.BOLD}[+] Checking ART gripper daemon at {host}:{port}...{Colors.RESET}")
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        ok = sock.connect_ex((host, port)) == 0
        sock.close()
        if ok:
            print_status(f"ART gripper daemon reachable at {host}:{port}", 'ok')
        else:
            print_status(f"ART gripper daemon not reachable at {host}:{port}", 'warn')
            print_status("On pro4000: systemctl status art-gripper-daemon", 'info')
            print_status("Or recover: sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh", 'info')
        return True
    except Exception as e:
        print_status(f"Error checking ART daemon: {e}", 'warn')
        return True


def check_camera_640x480_60fps_support(serial_numbers=None):
    """Check which cameras support 640x480 @ 60fps."""
    print(f"\n{Colors.BOLD}[Bonus] Checking 640x480@60fps support...{Colors.RESET}")

    try:
        import pyrealsense2 as rs
        ctx = rs.context()

        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)

            if serial_numbers and serial not in serial_numbers:
                continue

            supports_60fps = False
            for sensor in dev.query_sensors():
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() == rs.stream.color:
                        vp = profile.as_video_stream_profile()
                        if vp.width() == 640 and vp.height() == 480 and vp.fps() == 60:
                            supports_60fps = True
                            break
                if supports_60fps:
                    break

            if supports_60fps:
                print_status(f"{name} ({serial}): 640x480@60fps SUPPORTED", 'ok')
            else:
                print_status(f"{name} ({serial}): 640x480@60fps NOT supported", 'warn')

        return True

    except Exception as e:
        print_status(f"Error checking camera modes: {e}", 'warn')
        return True


def run_preflight_check(check_robot=True, check_vive=True):
    """
    Run all pre-flight checks.

    Args:
        check_robot: Whether to check robot connection
        check_vive: Whether to check Vive server

    Returns:
        bool: True if all critical checks passed
    """
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Franka Vive Demo - Pre-flight Check{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

    all_passed = True

    # 1. Kill conflicting processes
    check_and_kill_demo_processes()

    # 2. Kill orphaned multiprocessing
    check_and_kill_orphan_multiprocessing()

    # 3. Check cameras — KIST default is ZED; RealSense is optional fallback
    if not check_zed_cameras():
        all_passed = False
    check_realsense_cameras()  # warn-only

    # 4. Check Vive server (optional)
    if check_vive:
        check_vive_server()

    # 5. Check robot connection (optional)
    # ART gripper daemon is non-fatal but useful to surface
    check_art_gripper_daemon()

    if check_robot:
        if not check_robot_connection():
            all_passed = False

    # (legacy 640x480@60fps support check removed — irrelevant for ZED workflow)

    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}  All critical checks passed! Ready to run demo.{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}  Some checks failed. Please resolve issues above.{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    return all_passed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pre-flight check for Franka Vive Demo')
    parser.add_argument('--skip-robot', action='store_true', help='Skip robot connection check')
    parser.add_argument('--skip-vive', action='store_true', help='Skip Vive server check')
    parser.add_argument('--clean-only', action='store_true', help='Only clean up processes')
    args = parser.parse_args()

    if args.clean_only:
        check_and_kill_demo_processes()
        check_and_kill_orphan_multiprocessing()
        sys.exit(0)

    success = run_preflight_check(
        check_robot=not args.skip_robot,
        check_vive=not args.skip_vive
    )

    sys.exit(0 if success else 1)
