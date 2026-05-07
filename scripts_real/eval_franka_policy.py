"""
Eval script for deploying trained diffusion policy on Franka robot.

Usage:
    python scripts_real/eval_franka_policy.py \
        -i data/outputs/2026.01.27/06.02.47_train_diffusion_unet_hybrid_franka_vive_umi/checkpoints/latest.ckpt \
        -o data/eval_session

    # With specific cameras
    python scripts_real/eval_franka_policy.py \
        -i checkpoint.ckpt -o data/eval \
        -c 233722074266 -c f1480592

Controls:
    ================ Waiting for policy start ==============
    Press "C" to start policy execution
    Press "H" to move robot to HOME position
    Press "Q" to exit program

    ================ Policy running ==============
    Press "S" to stop policy and return to waiting mode
    Press "Q" to exit program (stops policy first)

    IMPORTANT: Keep your hand on the emergency stop button!
    The robot will move autonomously when policy is running.

Architecture:
    - Policy inference runs at 10Hz (main loop frequency)
    - Robot controller interpolates at 200Hz for smooth motion
    - Actions are scheduled with timestamps for precise execution
    - Observations are timestamp-aligned (camera as reference)
"""

import sys
import os
import subprocess
import signal

# IMPORTANT: Must set spawn method before importing modules that use multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
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
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


def cleanup_environment():
    """
    Clean up environment before starting eval.
    - Kill previous eval processes (only exact matches)
    - Kill ROS2 nodes that conflict (realsense, franka, etc.)
    - Kill camera processes that might be holding resources
    - Clean up stale ROS2/FastRTPS shared memory
    - Stop ROS2 daemon
    - Reset OpenCV windows

    NOTE: Only kills specific processes to avoid terminating unrelated sessions.
          Training processes (python.*train, pt_data, __KMP_*) are never touched.
    """
    print("="*60)
    print(" Environment Cleanup")
    print("="*60)

    # Get current process info to exclude
    current_pid = os.getpid()
    parent_pid = os.getppid()

    # --- Phase 1: Kill conflicting processes ---
    print("  [1/4] Killing conflicting processes...")

    # Processes to kill with their search method
    # Format: (pattern, use_exact_match)
    # use_exact_match=True: use pgrep -x (exact process name)
    # use_exact_match=False: use pgrep -f (full command line) but with specific patterns
    processes_to_kill = [
        # Previous eval processes - match full script path
        ('scripts_real/eval_franka_policy.py', False),
        # ROS2 RealSense camera nodes
        ('realsense2_camera_node', False),
        ('rs_launch', False),
        ('rs_launch_headeyes', False),
        # ROS2 component containers (used by realsense, etc.)
        ('component_container', True),
        ('component_container_mt', True),
        # ROS2 Franka nodes
        ('franka_ros2', False),
        ('joint_impedance_example_controller', False),
        # ROS2 visualization
        ('rviz2', True),
        ('rqt', True),
        # RealSense standalone tools
        ('realsense-viewer', True),
        ('rs-enumerate-devices', True),
        ('rs-capture', True),
        # ROS2 launch processes (only ros2 launch, not ros2 daemon)
        ('ros2 launch', False),
    ]

    killed = []
    for proc_pattern, exact_match in processes_to_kill:
        try:
            if exact_match:
                cmd = ['pgrep', '-x', proc_pattern]
            else:
                cmd = ['pgrep', '-f', proc_pattern]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    pid_int = int(pid) if pid else 0
                    # Skip self, parent, and invalid PIDs
                    if pid_int and pid_int != current_pid and pid_int != parent_pid:
                        try:
                            os.kill(pid_int, signal.SIGTERM)
                            killed.append(f"{proc_pattern}:{pid}")
                        except (ProcessLookupError, PermissionError):
                            pass
        except Exception:
            pass

    if killed:
        print(f"        Killed: {killed}")
        time.sleep(1.0)
    else:
        print("        No conflicting processes found.")

    # --- Phase 2: Stop ROS2 daemon ---
    print("  [2/4] Stopping ROS2 daemon...")
    try:
        result = subprocess.run(
            ['ros2', 'daemon', 'stop'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("        ROS2 daemon stopped.")
        else:
            print("        ROS2 daemon was not running.")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("        ROS2 daemon stop skipped (not available).")

    # --- Phase 3: Clean up stale shared memory ---
    print("  [3/4] Cleaning stale shared memory...")
    import glob as glob_module
    shm_cleaned = 0

    # Clean FastRTPS shared memory (stale ROS2 DDS segments)
    for pattern in ['fastrtps_*', 'sem.fastrtps_*']:
        for shm_file in glob_module.glob(f'/dev/shm/{pattern}'):
            try:
                os.unlink(shm_file)
                shm_cleaned += 1
            except (PermissionError, OSError):
                pass

    # Clean stale Python shared memory (psm_*) that don't belong to running processes
    for shm_file in glob_module.glob('/dev/shm/psm_*'):
        try:
            os.unlink(shm_file)
            shm_cleaned += 1
        except (PermissionError, OSError):
            pass

    # NOTE: Do NOT clean __KMP_* (active training) or u1000-Shm_* (SteamVR)
    if shm_cleaned > 0:
        print(f"        Cleaned {shm_cleaned} stale shared memory segments.")
    else:
        print("        No stale shared memory found.")

    # --- Phase 4: Reset OpenCV ---
    print("  [4/4] Resetting OpenCV...")
    cv2.destroyAllWindows()
    print("        OpenCV windows cleared.")

    print("="*60)
    print()

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from polymetis_franka_teleop.common.precise_sleep import precise_wait
from polymetis_franka_teleop.real_world.franka_policy_env import FrankaPolicyEnv
from polymetis_franka_teleop.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from polymetis_franka_teleop.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint (.ckpt file or directory)')
@click.option('--output', '-o', required=True, help='Directory to save evaluation recordings')
@click.option('--robot_ip', default='192.168.1.12', help='Robot NUC IP address (KIST default)')
@click.option('--robot_port', default=50051, help='NUC port (50051 polymetis-direct, default; 4242 ZeroRPC bridge)')
@click.option('--polymetis_mode', default='direct', type=click.Choice(['direct', 'zerorpc']),
              help='direct=raw polymetis :50051 (default); zerorpc=UMI/DROID bridge :4242 (opt-in)')
@click.option('--gripper_port', default=4242, help='Gripper ZeroRPC port (usually same as robot)')
@click.option('--camera_backend', default='zed', type=click.Choice(['zed', 'realsense']),
              help='Camera backend (default: zed)')
@click.option('--gripper_backend', default='art', type=click.Choice(['art', 'franka']),
              help='Gripper backend (default: art)')
@click.option('--art_gripper_host', default='127.0.0.1')
@click.option('--art_gripper_port', default=50053, type=int)
@click.option('--camera_serials', '-c', multiple=True,
              default=['233722074266', 'f1480592'],
              help='RealSense camera serial numbers')
@click.option('--camera_resolution', default='640x480',
              help='Camera resolution (WxH)')
@click.option('--camera_fps', default=60, type=int, help='Camera FPS')
@click.option('--vis_camera_idx', default=0, type=int, help='Camera index for visualization')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz')
@click.option('--steps_per_inference', '-si', default=6, type=int,
              help='Number of action steps to execute per inference')
@click.option('--max_duration', '-md', default=6000, type=float,
              help='Max duration for each episode in seconds')
@click.option('--tcp_offset', default=None, type=float, help='TCP offset (auto: 0.1034m Franka Hand, 0.216m ART)')
@click.option('--num_inference_steps', default=16, type=int,
              help='Number of DDIM inference steps (default: 16)')
@click.option('--record_episode', is_flag=True, default=False,
              help='Record episode data during policy execution')
@click.option('--auto_start', is_flag=True, default=False,
              help='Automatically start policy execution (skip waiting for C key)')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable verbose output')
def main(input, output, robot_ip, robot_port, polymetis_mode, gripper_port,
         camera_backend, gripper_backend, art_gripper_host, art_gripper_port,
         camera_serials, camera_resolution, camera_fps,
         vis_camera_idx, frequency, steps_per_inference,
         max_duration, tcp_offset, num_inference_steps,
         record_episode, auto_start, verbose):

    # === Environment cleanup ===
    cleanup_environment()

    # === Load checkpoint ===
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')

    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    # Print model info
    print(f"Model: {cfg._target_}")
    if hasattr(cfg.policy, 'obs_encoder') and hasattr(cfg.policy.obs_encoder, 'model_name'):
        print(f"Encoder: {cfg.policy.obs_encoder.model_name}")
    print(f"Dataset: {cfg.task.dataset.dataset_path}")

    # === Parse configuration ===
    dt = 1 / frequency

    # Get observation resolution from config
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    print(f"Observation resolution: {obs_res}")

    # Get pose representation from config
    obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print(f"Obs pose repr: {obs_pose_repr}")
    print(f"Action pose repr: {action_pose_repr}")

    # Parse camera resolution
    try:
        res_parts = camera_resolution.lower().split('x')
        cam_width = int(res_parts[0])
        cam_height = int(res_parts[1])
    except (ValueError, IndexError):
        print(f"Invalid camera resolution: {camera_resolution}. Use format WxH (e.g., 640x480)")
        sys.exit(1)

    # Convert camera serials to list
    if camera_serials:
        if camera_backend == 'zed':
            camera_serials = [int(s) for s in camera_serials]
        else:
            camera_serials = list(camera_serials)
    else:
        camera_serials = None

    print(f"Steps per inference: {steps_per_inference}")

    # === Setup environment and model ===
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            FrankaPolicyEnv(
                output_dir=output,
                robot_ip=robot_ip,
                robot_port=robot_port,
                polymetis_mode=polymetis_mode,
                gripper_port=gripper_port,
                frequency=frequency,
                camera_backend=camera_backend,
                gripper_backend=gripper_backend,
                art_gripper_host=art_gripper_host,
                art_gripper_port=art_gripper_port,
                camera_serial_numbers=camera_serials,
                camera_resolution=(cam_width, cam_height),
                camera_fps=camera_fps,
                obs_image_resolution=obs_res,
                obs_float32=True,
                # Observation horizons from config
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                # Latency settings (measured values)
                camera_obs_latency=0.015,   # Keep consistent with training data
                robot_obs_latency=0.001,    # Keep consistent with training data
                gripper_obs_latency=0.001,  # Keep consistent with training data
                robot_action_latency=0.055,   # Measured: 54.1ms ± 3.3ms (schedule_waypoint → arrival)
                gripper_action_latency=0.085, # v2.1 Measured: mean=84.5ms, range=63-100ms (direct command via ZeroRPC)
                # Robot params
                tcp_offset=tcp_offset,
                enable_multi_cam_vis=False,  # Disabled to avoid pickle issues with spawn
                shm_manager=shm_manager,
                verbose=verbose
            ) as env:

            cv2.setNumThreads(2)
            print("Waiting for system initialization...")
            time.sleep(1.0)

            # === Create model ===
            # Must be done after fork to prevent duplicating CUDA context
            print("Creating model...")
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            # Get policy (use EMA if available)
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = num_inference_steps
            print(f"Using {num_inference_steps} DDIM inference steps")

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            policy.eval().to(device)

            # === Warmup inference ===
            print("Warming up policy inference...")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_repr
                )
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10, f"Expected 10D action, got {action.shape[-1]}"
                env_action = get_real_umi_action(action, obs, action_pose_repr)
                assert env_action.shape[-1] == 7, f"Expected 7D env action, got {env_action.shape[-1]}"
                del result

            print('')
            print('='*60)
            print('Ready for policy deployment!')
            print('='*60)
            print('')
            print('Controls (Waiting mode):')
            print('  C: Start policy execution')
            print('  H: Move robot to HOME position')
            print('  Q: Quit program')
            print('')
            print('Controls (Running mode):')
            print('  S: Stop policy (return to waiting)')
            print('  Q: Quit program')
            print('')
            print('IMPORTANT: Keep your hand on the emergency stop button!')
            print('')

            # === Main loop ===
            first_episode = True
            while True:
                # ========= Waiting mode ==========
                if auto_start and first_episode:
                    print("Auto-start enabled, starting policy immediately...")
                    first_episode = False
                else:
                    print("Waiting for policy start... (C=start, H=home, Q=quit)")

                    while True:
                        # Get observation (keep system alive)
                        obs = env.get_obs()

                        # Visualize
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        if vis_img.dtype == np.float32:
                            vis_img = (vis_img * 255).astype(np.uint8)
                        vis_img = vis_img[:, :, ::-1].copy()  # RGB to BGR

                        episode_id = env.replay_buffer.n_episodes
                        text = f'Episode: {episode_id} | WAITING (C=start, H=home, Q=quit)'
                        cv2.putText(vis_img, text, (10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, thickness=2, color=(0, 255, 0))

                        cv2.imshow('Franka Policy Eval', vis_img)
                        cv2.pollKey()

                        # Handle key presses
                        press_events = key_counter.get_press_events()
                        start_policy = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='q'):
                                print("Exiting...")
                                env.end_episode()
                                cv2.destroyAllWindows()
                                return
                            elif key_stroke == KeyCode(char='c'):
                                start_policy = True
                                key_counter.clear()
                            elif key_stroke == KeyCode(char='h'):
                                print("\n>>> Moving to HOME position...")
                                env.move_home(wait_time=3.0)
                                print(">>> HOME position reached.\n")
                                key_counter.clear()

                        if start_policy:
                            break

                        time.sleep(0.05)  # Small delay in waiting mode

                # ========== Policy execution mode ==============
                print('')
                print('='*60)
                print('POLICY EXECUTION STARTED')
                print('='*60)
                print('Press S to stop, Q to quit')
                print('')

                try:
                    # Reset policy state
                    policy.reset()

                    # Start timing
                    start_delay = 1.0  # 1 second delay before starting
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay

                    # Optionally record episode
                    if record_episode:
                        env.start_episode(eval_t_start)

                    # Wait for start time (reduces latency on first inference)
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy running!")

                    iter_idx = 0
                    stop_episode = False

                    while not stop_episode:
                        # Calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # Get observation
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        if verbose:
                            print(f'Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')

                        # Run inference
                        with torch.no_grad():
                            t_inference_start = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs,
                                shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_repr
                            )
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            t_inference_end = time.time()

                            # Calculate inference stats
                            inference_latency = (t_inference_end - t_inference_start) * 1000  # ms
                            actual_hz = 1.0 / (t_inference_end - t_inference_start) if (t_inference_end - t_inference_start) > 0 else 0

                            # Gripper action analysis
                            gripper_vals = action[:, 6]  # gripper is 7th column (index 6)
                            gripper_min, gripper_max, gripper_mean = gripper_vals.min(), gripper_vals.max(), gripper_vals.mean()
                            # Threshold for OPEN/CLOSE decision: (open_width + close_width) / 2
                            gripper_threshold = (0.075 + 0.005) / 2  # = 0.04
                            n_open = np.sum(gripper_vals > gripper_threshold)
                            n_close = len(gripper_vals) - n_open
                            decision = 'OPEN' if gripper_mean > gripper_threshold else 'CLOSE'

                            # Combined status line (always show)
                            t_elapsed = time.time() - eval_t_start
                            print(f'[{t_elapsed:5.1f}s] Infer: {inference_latency:4.0f}ms ({actual_hz:4.1f}Hz) | '
                                  f'Actions: {len(action)} steps | '
                                  f'Gripper: {gripper_mean:.3f} → {decision} (O:{n_open}/C:{n_close})')

                            # Step-by-step gripper values (model's raw output)
                            # Show each step's gripper value with visual indicator
                            step_str = "  Steps: "
                            for i, gv in enumerate(gripper_vals):
                                marker = '●' if gv > gripper_threshold else '○'  # ● = OPEN, ○ = CLOSE
                                step_str += f"{gv:.3f}{marker} "
                                if (i + 1) % 8 == 0 and i < len(gripper_vals) - 1:
                                    step_str += "\n         "
                            print(step_str)

                            # Show trend: first vs last few steps
                            first_3_mean = gripper_vals[:3].mean() if len(gripper_vals) >= 3 else gripper_vals[0]
                            last_3_mean = gripper_vals[-3:].mean() if len(gripper_vals) >= 3 else gripper_vals[-1]
                            trend = "→CLOSING" if last_3_mean < first_3_mean - 0.005 else ("→OPENING" if last_3_mean > first_3_mean + 0.005 else "→STABLE")
                            print(f"  Trend: first3={first_3_mean:.3f}, last3={last_3_mean:.3f} {trend}")

                        # Calculate action timestamps
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64) * dt
                            + obs_timestamps[-1]
                        )

                        # Filter only future actions
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)

                        if np.sum(is_new) == 0:
                            # Exceeded time budget, still execute something
                            this_actions = action[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + next_step_idx * dt
                            if verbose:
                                print(f'Over budget: {action_timestamp - curr_time:.3f}s')
                            this_timestamps = np.array([action_timestamp])
                        else:
                            this_actions = action[is_new]
                            this_timestamps = action_timestamps[is_new]

                        # Execute actions
                        env.exec_actions(
                            actions=this_actions,
                            timestamps=this_timestamps,
                            compensate_latency=True
                        )
                        if verbose:
                            print(f"Submitted {len(this_actions)} action steps")

                        # Visualize
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        if vis_img.dtype == np.float32:
                            vis_img = (vis_img * 255).astype(np.uint8)
                        vis_img = vis_img[:, :, ::-1].copy()

                        elapsed = time.monotonic() - t_start
                        text = f'RUNNING | Time: {elapsed:.1f}s | Press S to stop'
                        cv2.putText(vis_img, text, (10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, thickness=2, color=(0, 0, 255))

                        cv2.imshow('Franka Policy Eval', vis_img)
                        cv2.pollKey()

                        # Check for stop command
                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                print('Stop requested.')
                                stop_episode = True
                            elif key_stroke == KeyCode(char='q'):
                                print('Quit requested.')
                                stop_episode = True
                                if record_episode:
                                    env.end_episode()
                                cv2.destroyAllWindows()
                                return

                        # Check max duration
                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print(f"Max duration ({max_duration}s) reached.")
                            stop_episode = True

                        # Wait for cycle end
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                    # End episode
                    if record_episode:
                        env.end_episode()
                    print("Episode stopped.")

                except KeyboardInterrupt:
                    print("Interrupted!")
                    if record_episode:
                        env.end_episode()

                print("")


if __name__ == '__main__':
    main()
