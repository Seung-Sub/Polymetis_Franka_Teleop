"""Integrity-check a recorded episode dir."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser('~/diffusion_policy'))

import zarr, numpy as np
import click


@click.command()
@click.argument('out_dir')
def main(out_dir):
    rb = zarr.open(os.path.join(out_dir, 'replay_buffer.zarr'), mode='r')
    ends = rb['meta/episode_ends'][:]
    n = int(ends[-1])
    ts = rb['data/timestamp'][:]

    print('=== Zarr ===')
    print(f'  episodes : {len(ends)}')
    print(f'  steps    : {n}')
    print(f'  ts span  : {ts[0]:.3f} → {ts[-1]:.3f} ({ts[-1]-ts[0]:.2f}s)')
    dts = np.diff(ts)
    print(f'  ts dt    : mean={np.mean(dts)*1000:.2f}ms  std={np.std(dts)*1000:.2f}ms  (target 100ms)')
    is_mono = bool(np.all(dts > 0))
    print(f'  monotonic: {is_mono}')

    for k in sorted(rb['data'].array_keys()):
        a = rb[f'data/{k}'][:]
        nan_count = int(np.sum(np.isnan(a.astype(np.float64))))
        print(f'  data/{k:30s}: {str(a.shape):20s} {str(a.dtype):10s} nans={nan_count}')

    print()
    print('=== Videos ===')
    import cv2
    video_dir = os.path.join(out_dir, 'videos')
    for ep in sorted(os.listdir(video_dir)):
        for cam in sorted(os.listdir(os.path.join(video_dir, ep))):
            p = os.path.join(video_dir, ep, cam)
            v = cv2.VideoCapture(p)
            nf = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = v.get(cv2.CAP_PROP_FPS)
            w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, frame = v.read()
            v.release()
            sz = os.path.getsize(p)
            print(f'  {ep}/{cam}: {w}x{h}@{fps:.0f}fps  frames={nf}  size={sz/1024:.0f}KB  '
                  f'decode={"OK" if ret else "FAIL"}')

    print()
    print('=== GR00T LeRobot dataset (if exists) ===')
    gr = out_dir.rstrip('/') + '_gr00t'
    if os.path.isdir(gr):
        import json
        info = json.load(open(os.path.join(gr, 'meta/info.json')))
        modality = json.load(open(os.path.join(gr, 'meta/modality.json')))
        print(f'  total_episodes: {info["total_episodes"]}')
        print(f'  total_frames  : {info["total_frames"]}')
        print(f'  fps           : {info["fps"]}')
        print(f'  state keys    : {list(modality["state"].keys())}')
        print(f'  action keys   : {list(modality["action"].keys())}')
        print(f'  video keys    : {list(modality["video"].keys())}')
        # Verify a parquet
        import pandas as pd
        df = pd.read_parquet(os.path.join(gr, 'data/chunk-000/episode_000000.parquet'))
        s = np.stack(df['observation.state'].values)
        a = np.stack(df['action'].values)
        print(f'  parquet state : {s.shape}  range x={s[:,0].min():.3f}..{s[:,0].max():.3f}')
        print(f'  parquet action: {a.shape}')
    else:
        print('  (not converted yet)')


if __name__ == '__main__':
    main()
