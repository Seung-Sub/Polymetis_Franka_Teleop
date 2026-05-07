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
import glob


def _run_sudo(cmd_list, input_text: str = '', timeout: float = 30.0):
    """Run a command with elevated privileges. Tries:
    1. ``sudo -n`` (NOPASSWD) — silent, fast.
    2. ``sudo -S`` with $POLYMETIS_SUDO_PASSWORD — for unattended ops.
    Returns (returncode, stdout, stderr). rc=126 means we couldn't elevate.
    """
    # 1. passwordless sudo
    try:
        p = subprocess.run(['sudo', '-n'] + cmd_list,
                           input=input_text, capture_output=True,
                           text=True, timeout=timeout)
        if p.returncode == 0:
            return 0, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'sudo -n timed out'
    except FileNotFoundError:
        return 127, '', 'sudo not installed'
    # 2. password via env var
    pwd = os.environ.get('POLYMETIS_SUDO_PASSWORD')
    if pwd:
        try:
            p = subprocess.run(['sudo', '-S', '-p', ''] + cmd_list,
                               input=pwd + '\n' + input_text,
                               capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, '', 'sudo -S timed out'
    return 126, '', 'sudo unavailable (no NOPASSWD, no $POLYMETIS_SUDO_PASSWORD)'

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


def _release_zed_video_handles():
    """Find any process holding /dev/video* (ZED v4l2 nodes) and kill it.
    Returns the number of holders killed.

    A previous demo crash leaves the camera-process child of multiprocessing
    spawn alive and still attached to the v4l2 device. lsusb still shows the
    camera, but ZED SDK's ``Camera.get_device_list()`` returns it as missing
    until the file descriptor is released.
    """
    killed = 0
    try:
        out = subprocess.check_output(['lsof', '-t', '-w'] + glob.glob('/dev/video*'),
                                      text=True, stderr=subprocess.DEVNULL)
        pids = sorted({int(p) for p in out.split() if p.strip().isdigit()})
    except (subprocess.CalledProcessError, FileNotFoundError):
        pids = []
    self_pid = os.getpid()
    for pid in pids:
        if pid == self_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        time.sleep(1.5)  # let kernel release fds
    return killed


def _zed_usb_authorize_cycle():
    """Toggle USB ``authorized`` for every Stereolabs (vid 2b03) device, then
    re-enable. This kicks the ZED firmware back into a state where the SDK
    can probe it. Requires sudo. Returns True if at least one device was
    cycled."""
    script = (
        "for d in /sys/bus/usb/devices/*/; do "
        "  [ \"$(cat \"$d/idVendor\" 2>/dev/null)\" = \"2b03\" ] || continue; "
        "  echo 0 > \"$d/authorized\" 2>/dev/null; "
        "  sleep 1; "
        "  echo 1 > \"$d/authorized\" 2>/dev/null; "
        "  basename \"$d\"; "
        "done"
    )
    rc, out, _err = _run_sudo(['bash', '-c', script], timeout=20)
    return rc == 0 and bool(out.strip())


def check_zed_cameras(expected: int = 2):
    """ZED detection with two-stage auto-recovery: kill stale holders, then
    USB authorize cycle if SDK still can't see the expected count."""
    print(f"\n{Colors.BOLD}[3/5] Checking ZED cameras (expected {expected})...{Colors.RESET}")
    try:
        import pyzed.sl as sl
    except ImportError:
        print_status("pyzed.sl not installed — skip if using --camera_backend realsense", 'warn')
        return True

    devs = sl.Camera.get_device_list()
    if len(devs) < expected:
        # First: kill any process still holding /dev/video* (orphan from a
        # crashed demo).
        killed = _release_zed_video_handles()
        if killed:
            print_status(f"Killed {killed} stale process(es) holding /dev/video*", 'fix')
        time.sleep(1.0)
        devs = sl.Camera.get_device_list()
        # Second: if still short, USB authorize toggle (sudo) to unstick a
        # frozen firmware.
        if len(devs) < expected:
            print_status(f"Only {len(devs)} ZED detected — USB authorize cycle...", 'fix')
            if _zed_usb_authorize_cycle():
                time.sleep(8)  # ZED firmware re-init takes 5–8s
                devs = sl.Camera.get_device_list()
            else:
                print_status("USB authorize cycle skipped (need sudo or "
                             "$POLYMETIS_SUDO_PASSWORD)", 'warn')

    if not devs:
        print_status("No ZED cameras detected!", 'error')
        return False
    print_status(f"Found {len(devs)} ZED camera(s):", 'ok' if len(devs) >= expected else 'warn')
    for d in devs:
        print_status(f"  sn={d.serial_number} {d.camera_model} ({d.camera_state})", 'info')
    avail = [d for d in devs if d.camera_state == sl.CAMERA_STATE.AVAILABLE]
    if not avail:
        print_status("All ZED devices are unavailable (in use elsewhere?)", 'warn')
        return False
    if len(devs) < expected:
        print_status(f"Got {len(devs)}/{expected} expected — one camera may "
                     f"need to be physically replugged", 'warn')
        return False
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


def _is_listening(host: str, port: int, timeout: float = 1.5) -> bool:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        ok = s.connect_ex((host, port)) == 0
        s.close()
        return ok
    except Exception:
        return False


def _pgrep(pattern: str) -> bool:
    try:
        return subprocess.call(['pgrep', '-f', pattern],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0
    except FileNotFoundError:
        return False


def check_vive_server():
    """Vive input TCP server + SteamVR vrserver. Auto-starts both via
    ``bin/start_vive_stack.sh start`` if either is missing."""
    print(f"\n{Colors.BOLD}[4/5] Checking Vive input server...{Colors.RESET}")
    host, port = '127.0.0.1', 12345
    listening = _is_listening(host, port)
    vrserver = _pgrep('vrserver')
    if listening and vrserver:
        print_status(f"Vive input listening at {host}:{port} (vrserver alive)", 'ok')
        return True

    print_status(f"Vive stack incomplete (listening={listening} vrserver={vrserver}) "
                 f"— attempting auto-start...", 'fix')
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    starter = os.path.join(repo_root, 'bin', 'start_vive_stack.sh')
    if not os.path.isfile(starter):
        print_status(f"Auto-start script not found at {starter}", 'warn')
        print_status("Manual: bash bin/start_vive_stack.sh start", 'info')
        return True
    try:
        subprocess.run(['bash', starter, 'start'],
                       check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print_status("start_vive_stack.sh timed out (vrserver may need GUI/headset)", 'warn')
    # vrserver takes ~5s to come up
    for _ in range(15):
        if _is_listening(host, port) and _pgrep('vrserver'):
            print_status(f"Vive stack started ({host}:{port})", 'ok')
            return True
        time.sleep(1)
    print_status(f"Vive stack failed to start within 15s — controller unusable until fixed", 'warn')
    print_status("Check: ssh into pro4000 + bash bin/start_vive_stack.sh status", 'info')
    return True


def check_robot_connection(robot_ip: str = '192.168.1.12', robot_port: int = 50051):
    """Check polymetis arm: TCP reachable + can start/terminate a controller.
    The full round-trip catches stale-controller / watchdog states that pass
    a simple port-open test."""
    print(f"\n{Colors.BOLD}[5/5] Checking polymetis arm at {robot_ip}:{robot_port}...{Colors.RESET}")
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        ok = sock.connect_ex((robot_ip, robot_port)) == 0
        sock.close()
    except Exception as e:
        print_status(f"Error checking polymetis arm: {e}", 'error')
        return False
    if not ok:
        print_status(f"Polymetis arm not reachable at {robot_ip}:{robot_port}", 'error')
        print_status("On NUC: sudo bash /usr/local/sbin/start_franka_arm.sh", 'info')
        print_status("Also: Franka Desk → Activate FCI", 'info')
        return False

    # Live test: read-only joint+ee pose RPCs. Avoids leaving any controller
    # state on the NUC. The FrankaInterpolationController itself has watchdog
    # recovery for the start_cartesian_impedance race.
    try:
        from polymetis import RobotInterface
        robot = RobotInterface(ip_address=robot_ip, port=robot_port,
                               enforce_version=False)
        q = robot.get_joint_positions().numpy()
        ee_pos, ee_quat = robot.get_ee_pose()
        ee_pos = ee_pos.numpy()
        print_status(f"Polymetis arm healthy "
                     f"(q[0]={q[0]:.2f}, ee_xyz=[{ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f}])", 'ok')
        return True
    except ImportError:
        # polymetis client not in this env (zerorpc-only flow). TCP-reachable
        # is the best we can do.
        print_status(f"Polymetis arm reachable; client not importable here "
                     f"(ok for --polymetis_mode zerorpc)", 'ok')
        return True
    except Exception as e:
        print_status(f"Polymetis read probe failed: {type(e).__name__}: {e}", 'error')
        print_status("On NUC: sudo systemctl restart polymetis-server "
                     "(or: sudo bash /usr/local/sbin/start_franka_arm.sh)", 'info')
        return False


def _art_is_ready(host: str, port: int):
    """Probe ART gripper firmware-ready state. Returns (reachable, healthy,
    state_str). reachable=False on connection error; healthy means
    is_ready=True AND is_fault=False — both required for goto/grasp to work.

    A previous demo crash can leave the firmware with GS_READY=0 (commands
    silently dropped) OR GS_FAULT=1 (motor disabled until reset). Either way
    we run restart_gripper.sh."""
    try:
        from art_gripper_client import ArtGripperInterface
    except ImportError:
        # try pypath fallback that demo wrappers also use
        for cand in (os.environ.get('ART_GRIPPER_PYPATH'),
                     os.path.expanduser('~/Hyundai_motors_Gripper/python')):
            if cand and os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        try:
            from art_gripper_client import ArtGripperInterface
        except ImportError:
            return None, False, 'art_gripper_client unavailable'
    try:
        g = ArtGripperInterface(ip_address=host, port=port, auto_motor_on=False)
        s = g.get_state()
        healthy = bool(s.is_ready) and not bool(s.is_fault)
        return True, healthy, \
               f"width={s.width:.3f} ready={s.is_ready} grasped={s.is_grasped} fault={s.is_fault}"
    except Exception as e:
        return False, False, repr(e)


def _art_run_recovery():
    """Run the documented EtherCAT/daemon restart on pro4000. Needs sudo."""
    script = os.path.expanduser('~/Hyundai_motors_Gripper/scripts/restart_gripper.sh')
    if not os.path.isfile(script):
        return False, f'recovery script missing: {script}'
    rc, _out, err = _run_sudo(['bash', script], timeout=30)
    return rc == 0, err.strip() or 'unknown'


def check_art_gripper_daemon(host: str = '127.0.0.1', port: int = 50053):
    """ART daemon health: TCP reachability + firmware GS_READY. Auto-runs
    ``restart_gripper.sh`` (sudo) if the firmware is stuck."""
    print(f"\n{Colors.BOLD}[+] Checking ART gripper daemon at {host}:{port}...{Colors.RESET}")
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        reachable = sock.connect_ex((host, port)) == 0
        sock.close()
    except Exception as e:
        print_status(f"Error checking ART daemon: {e}", 'warn')
        return True  # non-fatal — user might be on Franka Hand backend
    if not reachable:
        print_status(f"ART daemon not reachable at {host}:{port}", 'warn')
        print_status("Start with: systemctl start art-gripper-daemon", 'info')
        return True

    # Drain stale CLOSE-WAIT TCP sessions on :50053 — leaked from previously-
    # crashed demos. Counted via ss; daemon is single-threaded and slow new
    # clients respond to PING when too many half-open sockets pile up.
    try:
        ssout = subprocess.check_output(
            ['ss', '-tn', f'sport = :{port}'],
            text=True, stderr=subprocess.DEVNULL)
        stale = sum(1 for ln in ssout.splitlines() if 'CLOSE-WAIT' in ln)
        if stale >= 2:
            print_status(f"Detected {stale} stale CLOSE-WAIT TCP sessions on "
                         f":{port} — restarting daemon to clear", 'fix')
            _run_sudo(['systemctl', 'restart', 'art-gripper-daemon'], timeout=15)
            time.sleep(3)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    ok, healthy, state = _art_is_ready(host, port)
    if ok and healthy:
        print_status(f"ART daemon ready ({state})", 'ok')
        return True
    if ok and not healthy:
        # Could be is_ready=False, is_fault=True, or both. restart_gripper.sh
        # does a kernel-level EtherCAT module reload + daemon restart that
        # clears all firmware state.
        print_status(f"ART firmware unhealthy ({state})", 'warn')
        print_status("Running restart_gripper.sh (sudo) to reset firmware...", 'fix')
        recovered, err = _art_run_recovery()
        if not recovered:
            print_status(f"Recovery failed: {err}", 'warn')
            print_status("Manual: sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh", 'info')
            return True
        time.sleep(3)
        ok2, healthy2, state2 = _art_is_ready(host, port)
        if healthy2:
            print_status(f"Recovery successful ({state2})", 'ok')
        else:
            print_status(f"Still unhealthy after recovery ({state2})", 'warn')
            print_status("Try power-cycling the gripper 24V rail.", 'info')
        return True
    print_status(f"ART daemon probe error: {state}", 'warn')
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


def run_preflight_check(check_robot=True, check_vive=True, expected_cameras=2):
    """
    Run all pre-flight checks.

    Args:
        check_robot: Whether to check robot connection (live test, not just port)
        check_vive: Whether to check Vive server (auto-starts if down)
        expected_cameras: How many ZED cameras the demo will use (auto-recovers
                          if SDK sees fewer)

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
    if not check_zed_cameras(expected=expected_cameras):
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
