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

import math
import subprocess
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np

# Out-of-process visualization: demo writes JPEG, feh polls it. Path under
# /tmp (tmpfs on Ubuntu = RAM-disk, ~5 ms per cv2.imwrite at 224x224).
VIS_JPEG_PATH = '/tmp/teleop_vis.jpg'
from polymetis_franka_teleop.real_world.franka_vive_env import FrankaViveEnv
from polymetis_franka_teleop.common.precise_sleep import precise_wait
# pynput-based KeystrokeCounter removed: its X RECORD listener thread shared
# the X11 connection with cv2.imshow and deadlocked Qt5's window mapping. We
# now read keys via cv2.waitKey() directly, which means the cv2 window must
# have focus to receive keystrokes.


def run_preflight_check(expected_cameras: int = 2):
    """Run preflight check and return True if passed."""
    try:
        from preflight_check import run_preflight_check as preflight
        return preflight(check_robot=True, check_vive=True,
                         expected_cameras=expected_cameras)
    except ImportError:
        print("Warning: preflight_check.py not found, skipping checks")
        return True


@click.command()
@click.option('--output', '-o', required=True, help='Output directory for data')
@click.option('--robot_ip', default='192.168.1.12', help='Robot NUC IP address (KIST default)')
@click.option('--robot_port', default=50051, help='Polymetis arm gRPC port (NUC)')
@click.option('--gripper_port', default=4242, type=int,
              help='Franka Hand polymetis service port (zerorpc :4242). Ignored for ART backend.')
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
# ---- teleop tuning ----
@click.option('--tuning_preset', default='normal',
              type=click.Choice(['coarse', 'normal', 'precise', 'custom']),
              help='Teleop feel preset (overrides scale/gain flags unless --tuning_preset custom)')
@click.option('--pos_scale', default=None, type=float,
              help='Vive→robot position scaling (custom only). 1.0=1:1, 0.5=damped/precise, 1.5=amplified')
@click.option('--rot_scale', default=None, type=float,
              help='Vive→robot rotation scaling (custom only)')
@click.option('--kx_scale', default=None, type=float,
              help='Cartesian impedance position-stiffness multiplier on UMI default [750]×3+[15]×3 (custom)')
@click.option('--kxd_scale', default=None, type=float,
              help='Cartesian impedance damping multiplier on UMI default [37]×3+[2]×3 (custom)')
@click.option('--velocity_clamp/--no_velocity_clamp', default=None,
              help='Enable Vive velocity clamping (default per preset)')
@click.option('--max_pos_velocity', default=None, type=float, help='m/s, used when --velocity_clamp')
@click.option('--max_rot_velocity', default=None, type=float, help='rad/s, used when --velocity_clamp')
# ---- ready pose / data-format ----
@click.option('--data_format', default='groot',
              type=click.Choice(['groot', 'diffusion']),
              help='Output format. Determines the ready pose: groot → DROID tilt '
                   '(matches GR00T-N1.7-DROID training distribution); diffusion → '
                   'standard Franka home (used by ACT / Diffusion Policy / UMI). '
                   'joint-7 (gripper yaw) is set by --gripper_backend.')
@click.option('--auto_home/--no_auto_home', default=True,
              help='Auto-move to ready pose at startup before the teleop loop begins')
# ---- gripper grip strength ----
@click.option('--grip_force', default=None, type=float,
              help='Grip force in Newtons (default: 60N ART, 30N Franka Hand). '
                   'Higher = firmer hold on smooth/thin objects, max 100N for ART.')
@click.option('--gripper_close_width', default=None, type=float,
              help='Width when fully closed, meters (default: 0.0 ART = full mechanical close, '
                   '0.005 Franka Hand to avoid libfranka exception)')
# ---- visualization ----
@click.option('--vis/--no_vis', default=True,
              help='Show OpenCV camera preview window. --no_vis runs headless and prints '
                   'status to terminal; useful when cv2.imshow deadlocks under heavy load.')
@click.option('--vis_throttle_n', default=1, type=int,
              help='Write vis JPEG every Nth main-loop iter. Default 1 = 10 Hz '
                   '(matches main_loop frequency). Increase to 2 or 3 if JPEG '
                   'encoding consumes too much CPU.')
@click.option('--skip_preflight', is_flag=True, default=False, help='Skip preflight check')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose output')
def main(output, robot_ip, robot_port, gripper_port, vive_host, vive_port,
         camera_backend, gripper_backend, art_gripper_host, art_gripper_port,
         camera_serials, camera_resolution, camera_fps, obs_resolution,
         vis_camera_idx, show_all_cameras, frequency, teleop_frequency,
         tcp_offset, tuning_preset, pos_scale, rot_scale, kx_scale, kxd_scale,
         velocity_clamp, max_pos_velocity, max_rot_velocity,
         data_format, auto_home, grip_force, gripper_close_width,
         vis, vis_throttle_n,
         skip_preflight, verbose):

    # ---- Resolve tuning preset ----
    # Each preset is (pos_scale, rot_scale, Kx_scale, Kxd_scale, vel_clamp, max_pos_v, max_rot_v)
    PRESETS = {
        # Default UMI feel; same Vive↔robot mapping, no velocity clamp.
        'normal':  dict(pos_scale=1.0, rot_scale=1.0,
                        kx_scale=1.0, kxd_scale=1.0,
                        velocity_clamp=False, max_pos_velocity=2.0, max_rot_velocity=2.5),
        # Slow / fine work — robot follows hand at half speed, stiffer tracking + soft cap.
        'precise': dict(pos_scale=0.5, rot_scale=0.5,
                        kx_scale=1.3, kxd_scale=1.3,
                        velocity_clamp=True,  max_pos_velocity=0.4, max_rot_velocity=1.0),
        # Coarse / large reach — amplified Vive motion, softer impedance for smoother arcs.
        'coarse':  dict(pos_scale=1.5, rot_scale=1.0,
                        kx_scale=0.8, kxd_scale=1.1,
                        velocity_clamp=False, max_pos_velocity=2.0, max_rot_velocity=2.5),
    }
    if tuning_preset != 'custom':
        p = PRESETS[tuning_preset]
        pos_scale       = p['pos_scale']      if pos_scale       is None else pos_scale
        rot_scale       = p['rot_scale']      if rot_scale       is None else rot_scale
        kx_scale        = p['kx_scale']       if kx_scale        is None else kx_scale
        kxd_scale       = p['kxd_scale']      if kxd_scale       is None else kxd_scale
        velocity_clamp  = p['velocity_clamp'] if velocity_clamp  is None else velocity_clamp
        max_pos_velocity= p['max_pos_velocity'] if max_pos_velocity is None else max_pos_velocity
        max_rot_velocity= p['max_rot_velocity'] if max_rot_velocity is None else max_rot_velocity
    else:
        # custom — fill any None with normal defaults
        n = PRESETS['normal']
        pos_scale       = n['pos_scale']       if pos_scale       is None else pos_scale
        rot_scale       = n['rot_scale']       if rot_scale       is None else rot_scale
        kx_scale        = n['kx_scale']        if kx_scale        is None else kx_scale
        kxd_scale       = n['kxd_scale']       if kxd_scale       is None else kxd_scale
        velocity_clamp  = n['velocity_clamp']  if velocity_clamp  is None else velocity_clamp
        max_pos_velocity= n['max_pos_velocity'] if max_pos_velocity is None else max_pos_velocity
        max_rot_velocity= n['max_rot_velocity'] if max_rot_velocity is None else max_rot_velocity

    print(f"[tuning] preset={tuning_preset} | pos_scale={pos_scale} rot_scale={rot_scale} | "
          f"Kx×={kx_scale} Kxd×={kxd_scale} | vel_clamp={velocity_clamp} "
          f"v_max={max_pos_velocity} m/s, ω_max={max_rot_velocity} rad/s")

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

    # Run preflight check (auto-recovers ZED handles, ART firmware, polymetis,
    # vrserver — see scripts_real/preflight_check.py)
    if not skip_preflight:
        if not run_preflight_check(expected_cameras=n_cameras):
            print("\nPreflight check failed. Use --skip-preflight to bypass.")
            sys.exit(1)
        print()  # Add blank line after preflight

    dt = 1 / frequency

    # NOTE: KeystrokeCounter (pynput) used to capture keys globally via the X
    # RECORD extension. That deadlocked cv2.imshow on the same X11 connection
    # — main thread blocked inside Qt5's window-mapping code waiting for the
    # X mutex pynput's listener thread held. We now read keys exclusively via
    # cv2.waitKey return code, which means **the cv2 window must have focus**
    # for keystrokes to register. Click the window title bar after it appears.
    with SharedMemoryManager() as shm_manager:
        with FrankaViveEnv(
                output_dir=output,
                robot_ip=robot_ip,
                robot_port=robot_port,
                gripper_port=gripper_port,
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
                # tuning knobs
                pos_scale=pos_scale,
                rot_scale=rot_scale,
                use_velocity_clamping=velocity_clamp,
                max_pos_velocity=max_pos_velocity,
                max_rot_velocity=max_rot_velocity,
                Kx_scale=kx_scale,
                Kxd_scale=kxd_scale,
                # ready pose / gripper strength
                data_format=data_format,
                auto_home_on_start=auto_home,
                grip_force=grip_force,
                gripper_close_width=gripper_close_width,
                enable_multi_cam_vis=False,
                shm_manager=shm_manager,
                verbose=verbose,
            ) as env:

            cv2.setNumThreads(1)
            # Visualization is handled out-of-process by feh — see VIS_JPEG_PATH
            # writes in the main loop. cv2.imshow inside the demo deadlocks
            # under multi-subprocess load on OpenCV 4.6 + Qt5; a separate
            # viewer process avoids X mutex contention entirely.
            feh_proc = None
            if vis:
                # Seed an initial image so the viewer doesn't error on a
                # missing file during the first 0.5 s before the demo's main
                # loop fires its first cv2.imwrite.
                seed = np.full((400, 800, 3), 32, dtype=np.uint8)
                cv2.putText(seed, 'starting up...', (40, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
                cv2.imwrite(VIS_JPEG_PATH, seed)
                # geometry includes "+x+y" — without explicit position, gnome-shell
                # places the window wherever it likes (was observed at the bottom-
                # right of a 4K screen, hidden behind maximized apps). Force
                # upper-left at (50, 50) and use 1280x800 so it's visible without
                # being huge.
                # Custom Python cv2 viewer in a fresh subprocess. cv2.imshow
                # in the demo's main process deadlocks with multi-subprocess
                # load (ZED grab, polymetis client, recorders all contend for
                # the X mutex via Qt5). A fresh interpreter with no other
                # multiprocessing renders fine.
                viewer_script = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'bin', 'cv2_viewer.py')
                try:
                    feh_proc = subprocess.Popen(
                        [sys.executable, viewer_script,
                         VIS_JPEG_PATH,
                         '--signal-pid', str(os.getpid()),
                         '--poll-ms', '30',
                         '--win-name', 'Franka Vive Demo'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(0.5)
                    if feh_proc.poll() is not None:
                        err = feh_proc.stderr.read().decode(errors='replace')[:500]
                        print(f'[vis] viewer exited immediately (rc={feh_proc.returncode}): {err}',
                              flush=True)
                        feh_proc = None
                    else:
                        print(f'[vis] cv2 viewer started (PID {feh_proc.pid}, '
                              f'~30 Hz refresh from {VIS_JPEG_PATH})')
                    # atexit cleanup so the viewer dies even on SIGINT/crash.
                    import atexit
                    def _kill_viewer(p=feh_proc):
                        try:
                            p.terminate(); p.wait(timeout=1.5)
                        except Exception:
                            try: p.kill()
                            except Exception: pass
                    atexit.register(_kill_viewer)
                except Exception as e:
                    print(f'[vis] viewer launch failed: {e}', flush=True)
                    feh_proc = None

            # Warmup: env.start_wait() returns on the first camera frame, but
            # env.get_obs() needs k frames in the ring buffer. Sleep long
            # enough that both cameras have well over k frames.
            #
            # Note: when --auto_home (default), the FrankaInterpolationController
            # already moved the arm to ready pose during its boot sequence
            # (see joints_init in env). No extra demo-side move needed.
            time.sleep(3.0)
            if auto_home:
                print(f'[startup] Ready pose reached (data_format={data_format}, '
                      f'gripper={gripper_backend}). Hold Vive Grip to begin teleop.')
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
            print(f'Visualization: {n_cameras} camera(s), grid 360x360 each + status panel')
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
            # After HOME completes, the controller waits for clutch 0->1
            # transition before accepting teleop targets. Mirror that here so
            # the UI can tell the user clutch is currently disarmed.
            waiting_for_clutch = False

            # Signal handlers — the cv2_viewer subprocess (separate Python
            # process) relays user keystrokes to us via Unix signals because
            # cv2 windows in a child process can't deliver waitKey() returns
            # to the demo. Map:
            #   SIGINT/SIGTERM -> 'q' (clean shutdown)
            #   SIGUSR1        -> 'c' (start recording episode)
            #   SIGUSR2        -> 's' (stop recording episode + save)
            #   SIGHUP         -> 'h' (move to home pose)
            import signal as _sig
            # Flags read in the main loop body; set from signal handlers.
            sig_record_request = [None]   # 'start' | 'stop' | None
            sig_home_request = [False]
            def _on_quit_signal(sig, frame):
                nonlocal stop  # noqa: F824
                stop = True
                print(f'\n[demo] received signal {sig}, requesting clean shutdown...',
                      flush=True)
            def _on_record_start(sig, frame):
                sig_record_request[0] = 'start'
            def _on_record_stop(sig, frame):
                sig_record_request[0] = 'stop'
            def _on_home(sig, frame):
                sig_home_request[0] = True
            try:
                _sig.signal(_sig.SIGINT,  _on_quit_signal)
                _sig.signal(_sig.SIGTERM, _on_quit_signal)
                _sig.signal(_sig.SIGUSR1, _on_record_start)
                _sig.signal(_sig.SIGUSR2, _on_record_stop)
                _sig.signal(_sig.SIGHUP,  _on_home)
            except ValueError:
                pass

            # Track pending drop state (for two-key confirmation in OpenCV window)
            pending_drop = False
            pending_drop_time = 0

            # Last key from cv2.waitKey (-1 = no key). Consumed at start of
            # each iter; refreshed at bottom of each iter after cv2.imshow.
            # The first cv2.imshow inside the loop will lazily create the
            # window; explicit namedWindow before the loop was observed to
            # hang in Qt5 initialization on this build.
            last_key = -1

            while not stop:
                # === Calculate timing (UMI-style) ===
                t_cycle_end = t_start + (iter_idx + 1) * dt  # Current cycle end
                t_sample = t_cycle_end - command_latency     # Sample time
                t_command_target = t_cycle_end + dt          # Action execution time (next cycle)

                # Get observation (also accumulates obs for recording)
                obs = env.get_obs()

                # Handle keyboard input — keys come from the cv2 window via
                # cv2.waitKey() at the bottom of the previous iter (stored in
                # ``last_key``). The user must click the cv2 window to give it
                # focus so cv2 receives keystrokes.
                if last_key == ord('q'):
                    stop = True
                elif last_key == ord('c'):
                    env.start_episode(
                        t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                    is_recording = True
                    print('Recording!')
                elif last_key == ord('s'):
                    env.end_episode()
                    is_recording = False
                    print('Stopped.')
                elif last_key == ord('h'):
                    if not home_command_sent:
                        print('[HOME] Moving to home position via keyboard...')
                        env.move_home(wait=False)
                        home_command_sent = True
                elif last_key in (8, 127):   # Backspace (8) / Delete (127)
                    if is_recording:
                        pending_drop = True
                        pending_drop_time = time.monotonic()
                elif last_key == ord('y'):
                    if pending_drop:
                        env.drop_episode()
                        is_recording = False
                        pending_drop = False
                        print('[DROP] Episode dropped!')
                elif last_key == ord('n'):
                    if pending_drop:
                        pending_drop = False
                        print('[DROP] Cancelled')
                last_key = -1  # consume; next iter reads fresh waitKey return

                # Signal-relayed keystrokes from cv2_viewer (separate process).
                # SIGUSR1 = 'c' record start, SIGUSR2 = 's' record stop,
                # SIGHUP = 'h' home. SIGINT already handled above (-> stop).
                if sig_record_request[0] == 'start':
                    sig_record_request[0] = None
                    if not is_recording:
                        env.start_episode(
                            t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        is_recording = True
                        print('Recording!')
                    else:
                        print('[record] already recording, ignored')
                elif sig_record_request[0] == 'stop':
                    sig_record_request[0] = None
                    if is_recording:
                        env.end_episode()
                        is_recording = False
                        print('Stopped.')
                    else:
                        print('[record] not recording, ignored')
                if sig_home_request[0]:
                    sig_home_request[0] = False
                    if not home_command_sent:
                        print('[HOME] Moving to home position via keyboard signal...')
                        env.move_home(wait=False)
                        home_command_sent = True

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

                # ---- Build visualization image. Always show ALL cameras when
                # there's more than one (user can still pick a single via
                # --no_show_all_cameras). Each camera frame is upscaled to
                # 360x360 so text overlay fits comfortably on the viewer.
                cam_imgs = []
                for i in range(n_cameras):
                    raw = obs[f'camera{i}_rgb'][-1, :, :, ::-1]  # last frame, BGR
                    up = cv2.resize(raw, (360, 360), interpolation=cv2.INTER_LINEAR)
                    cv2.putText(up, f'cam{i}', (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cam_imgs.append(up)
                if len(cam_imgs) == 1:
                    cam_grid = cam_imgs[0]
                elif len(cam_imgs) == 2:
                    cam_grid = np.hstack(cam_imgs)        # 720 x 360
                else:
                    while len(cam_imgs) < 4:
                        cam_imgs.append(np.zeros_like(cam_imgs[0]))
                    cam_grid = np.vstack([np.hstack(cam_imgs[:2]),
                                           np.hstack(cam_imgs[2:4])])  # 720 x 720

                # Status panel below the camera grid (height 180 px).
                grid_h, grid_w = cam_grid.shape[:2]
                panel = np.full((180, grid_w, 3), 24, dtype=np.uint8)
                vis_img = np.vstack([cam_grid, panel])

                # ---- Track waiting-for-clutch state.
                # The robot's controller sets wait_for_clutch_engage=True after
                # any HOME/recovery and requires the user to release-then-press
                # Grip again before teleop targets are accepted. We mirror it
                # here so the UI can tell the user whether teleop input will be
                # honored right now.
                if home_command_sent and not home_active:
                    # HOME just completed — controller is waiting for clutch
                    # 0->1 transition.
                    waiting_for_clutch = True
                if clutch_engaged and not clutch_was_engaged:
                    # User re-engaged grip — teleop is now active.
                    waiting_for_clutch = False

                episode_id = env.replay_buffer.n_episodes
                gripper_state = teleop_state.get('gripper_state', 0.0)
                gripper_text = 'CLOSED' if gripper_state >= 0.5 else 'OPEN'

                # Decide top-line teleop status with priority:
                #   HOME-active > waiting-for-clutch > recording > clutch state
                # Note: only ASCII characters here — cv2.FONT_HERSHEY_SIMPLEX
                # renders non-ASCII (em-dash, Korean, etc.) as garbled '?'.
                if home_active:
                    big_text = 'HOMING: robot returning to ready pose'
                    big_bg   = (0, 90, 200)   # orange
                elif waiting_for_clutch:
                    big_text = 'RELEASE then PRESS Vive Grip to resume teleop'
                    big_bg   = (0, 140, 200)  # amber
                elif clutch_engaged:
                    big_text = 'TELEOP ACTIVE: Vive Grip held'
                    big_bg   = (40, 130, 0)   # green
                elif is_recording:
                    big_text = 'RECORDING: press Grip to teleop'
                    big_bg   = (0, 0, 160)    # red-ish
                else:
                    big_text = 'IDLE: hold Vive Grip to start teleop'
                    big_bg   = (60, 60, 60)   # neutral grey

                # Big colored banner — full grid width, ~50 px tall
                cv2.rectangle(vis_img, (0, grid_h), (grid_w, grid_h + 50),
                              big_bg, -1)
                cv2.putText(vis_img, big_text, (10, grid_h + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

                # Second row: episode + recording state
                rec_str  = 'REC' if is_recording else 'idle'
                rec_color = (60, 60, 200) if is_recording else (180, 180, 180)
                cv2.putText(vis_img, f'Episode: {episode_id}  [{rec_str}]',
                            (10, grid_h + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)

                # Third row: gripper state (color-coded)
                grip_color = (0, 220, 0) if gripper_text == 'OPEN' else (0, 80, 230)
                cv2.putText(vis_img, f'Gripper: {gripper_text}',
                            (10, grid_h + 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, grip_color, 2)

                # Fourth row: joint-limit margins (j3/j5/j6 — the three joints
                # that historically trigger libfranka safety_controller). Show
                # in a single line, color-coded green if margin > 0.5 rad,
                # amber if > 0.2, red if smaller. Catalog #28.
                try:
                    jp = obs.get('robot0_joint_pos', None)
                    if jp is not None and len(jp.shape) >= 1:
                        last_q = jp[-1] if jp.ndim == 2 else jp
                        j3_a = abs(float(last_q[3]))
                        j5_a = abs(float(last_q[5]))
                        j6_a = abs(float(last_q[6]))
                        # margins to soft limit (slightly conservative vs 2.97/2.97/2.89)
                        m3 = 2.97 - j3_a
                        m5 = 2.97 - j5_a
                        m6 = 2.89 - j6_a
                        def _jcolor(m):
                            if m < 0.2: return (0, 0, 220)       # red
                            if m < 0.5: return (0, 140, 220)     # amber
                            return (60, 200, 60)                  # green
                        cv2.putText(vis_img,
                                    f'j3:{j3_a:.2f} ({m3:+.2f})  '
                                    f'j5:{j5_a:.2f} ({m5:+.2f})  '
                                    f'j6:{j6_a:.2f} ({m6:+.2f})',
                                    (10, grid_h + 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    _jcolor(min(m3, m5, m6)), 1)
                except Exception:
                    pass

                # Fifth row: pending-drop confirmation if active
                if pending_drop:
                    remaining = 5.0 - (time.monotonic() - pending_drop_time)
                    cv2.rectangle(vis_img, (0, grid_h + 140), (grid_w, grid_h + 178),
                                  (0, 0, 160), -1)
                    cv2.putText(vis_img,
                                f'DROP EPISODE? Y=confirm  N=cancel  ({remaining:.1f}s)',
                                (10, grid_h + 168),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # Compact help line
                    cv2.putText(vis_img,
                                'q=quit  c=record start  s=stop  Backspace=drop  Trackpad=HOME',
                                (10, grid_h + 165),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # ---- Write JPEG for the viewer subprocess to pick up.
                # vis_throttle_n=1 gives 10 Hz writes; the viewer polls at
                # ~30 Hz so user sees smooth updates without the demo's main
                # loop spending too long on JPEG compression.
                if vis and (iter_idx % vis_throttle_n == 0):
                    try:
                        cv2.imwrite(VIS_JPEG_PATH, vis_img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                    except Exception as e:
                        if iter_idx == 0:
                            print(f'[vis] JPEG write failed: {e}', flush=True)

                # ---- If the viewer subprocess died (user closed the window
                # or pressed q), tear the demo down too. The viewer also sends
                # SIGINT directly to our PID via --signal-pid; this poll covers
                # the case where the user clicked the X button instead.
                if feh_proc is not None and feh_proc.poll() is not None:
                    print(f'[vis] viewer exited (rc={feh_proc.returncode}) — '
                          f'shutting down demo', flush=True)
                    stop = True

                # status print every ~1 s, regardless of vis (so the user can
                # tail the log to monitor clutch/record/home/gripper state).
                if iter_idx % max(1, frequency) == 0:
                    ep = env.replay_buffer.n_episodes
                    rec_str = 'REC' if is_recording else '   '
                    cl_str = 'CLU' if clutch_engaged else '   '
                    h_str = 'HOM' if home_active else '   '
                    gw = teleop_state.get('gripper_state', 0.0)
                    g_str = 'CLOSE' if gw > 0.5 else 'OPEN '
                    print(f'[main] iter={iter_idx} ep={ep} {rec_str} {cl_str} {h_str} '
                          f'gripper={g_str}', flush=True)
                # cv2 keystrokes are unavailable in this viz mode (feh doesn't
                # forward keys), so keep last_key=-1. Use keyboard 'q' fallback
                # via SIGINT/SIGTERM (pkill -INT -f demo_franka_vive) and
                # Vive trackpad for HOME.
                last_key = -1

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

            # Main loop exited (stop=True). Tear down feh viewer if running.
            if feh_proc is not None:
                try:
                    feh_proc.terminate()
                    feh_proc.wait(timeout=2)
                except Exception:
                    try:
                        feh_proc.kill()
                    except Exception:
                        pass


if __name__ == '__main__':
    main()
