"""LIVE Vive teleop test — user holds the controller, robot mirrors.

Records DURATION_S (default 180 s, env override LIVE_DURATION) of
Vive teleoperation. No interactive prompt — auto-starts after 5 s
countdown.

Pattern:
    t=0..5s        countdown (wake Vive controller, get ready)
    t=5..(5+D)     RECORDING — engage grip and move; trigger toggles ART
    t=(5+D)        auto stop_episode + save + GR00T-LeRobot convert + verify
"""
import os, sys, time, subprocess, shutil
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DP = os.path.expanduser('~/diffusion_policy')
if os.path.isdir(_DP) and _DP not in sys.path:
    sys.path.insert(0, _DP)

from multiprocessing.managers import SharedMemoryManager
import numpy as np
from polymetis_franka_teleop.real_world.franka_vive_env import FrankaViveEnv
from polymetis_franka_teleop.common.precise_sleep import precise_wait


def _list_zed_serials():
    import pyzed.sl as sl
    return [int(d.serial_number) for d in sl.Camera.get_device_list()
            if d.camera_state == sl.CAMERA_STATE.AVAILABLE]


def main():
    output = os.path.expanduser('~/Polymetis_Franka_Teleop/data/_live_test')
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    duration = float(os.environ.get('LIVE_DURATION', '180'))
    cams = _list_zed_serials()
    print(f'=== LIVE Vive teleop test (cameras={cams}, duration={duration}s) ===')
    print()
    print('  SAFETY: keep e-stop in your hand!')
    print('  Pattern:')
    print(f'    [0..5s]            countdown — wake Vive controller (move it)')
    print(f'    [5..{5+duration:.0f}s]  RECORDING — grip ON to mirror robot, trigger to toggle ART')
    print(f'    end                auto-save + convert + verify')
    print()
    for i in range(5, 0, -1):
        print(f'  starting in {i}s ...')
        time.sleep(1)

    with SharedMemoryManager() as shm:
        with FrankaViveEnv(
            output_dir=output,
            robot_ip='192.168.1.12', robot_port=50051, polymetis_mode='direct',
            frequency=10,
            camera_backend='zed', gripper_backend='art',
            art_gripper_host='127.0.0.1', art_gripper_port=50053,
            camera_serial_numbers=cams or None,
            camera_resolution=(1280, 720), camera_fps=60,
            obs_image_resolution=(224, 224),
            vive_host='127.0.0.1', vive_port=12345,
            teleop_frequency=100,
            tcp_offset=None,            # auto: 0.216 (ART)
            enable_multi_cam_vis=False,
            shm_manager=shm,
            verbose=False,
        ) as env:
            time.sleep(2.0)
            print('[live] env READY')

            t_start = time.monotonic()
            iter_idx = 0
            dt = 1.0 / 10
            command_latency = 1/100

            ep_start = t_start + 2 * dt - time.monotonic() + time.time()
            env.start_episode(ep_start)
            print('[live] start_episode')

            t_record_end = time.monotonic() + duration
            while time.monotonic() < t_record_end:
                t = time.monotonic() - t_start
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                obs = env.get_obs()

                # Live readout every 5s
                if iter_idx % 50 == 0:
                    eef = obs['robot0_eef_pos'][-1]
                    gw = obs['robot0_gripper_width'][-1, 0]
                    elapsed = t
                    remaining = max(0, duration - elapsed)
                    print(f'  t={elapsed:5.1f}s ({remaining:.0f}s left)  '
                          f'eef={eef.round(3).tolist()}  gw={gw:.3f}')

                precise_wait(t_sample)

                action_ts = t_command_target - time.monotonic() + time.time()
                env.record_action(timestamp=action_ts)

                precise_wait(t_cycle_end)
                iter_idx += 1

            env.end_episode()
            print('[live] end_episode')

    # convert to GR00T-LeRobot v2
    print()
    print('=== auto-convert to GR00T LeRobot v2 ===')
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([
        sys.executable, os.path.join(here, 'convert_to_gr00t_lerobot.py'),
        '-i', output, '-o', output + '_gr00t',
        '-t', 'live test motion',
        '--gripper_max_width', '0.100', '--fps', '10',
    ])

    # quick verify
    import zarr
    rb = zarr.open(os.path.join(output, 'replay_buffer.zarr'), mode='r')
    n_steps = int(rb['meta/episode_ends'][-1])
    eef = rb['data/robot0_eef_pos'][:]
    gw = rb['data/robot0_gripper_width'][:].reshape(-1)
    print()
    print(f'--- recorded ---')
    print(f'  steps        : {n_steps}')
    print(f'  eef pos range: x={eef[:,0].min():.3f}..{eef[:,0].max():.3f}  '
          f'y={eef[:,1].min():.3f}..{eef[:,1].max():.3f}  '
          f'z={eef[:,2].min():.3f}..{eef[:,2].max():.3f}')
    print(f'  eef pos delta: {(eef.max(axis=0)-eef.min(axis=0)).round(3).tolist()}')
    print(f'  gripper width: {gw.min():.3f}..{gw.max():.3f}  (Δ={gw.max()-gw.min():.3f})')

    moved = (eef.max(axis=0) - eef.min(axis=0)).max() > 0.01
    grip_used = (gw.max() - gw.min()) > 0.03

    print()
    print('=== LIVE TEST RESULT ===')
    print(f'  arm motion detected   : {moved}     (any axis > 1 cm)')
    print(f'  gripper toggle detected: {grip_used} (Δw > 3 cm)')


if __name__ == '__main__':
    main()
