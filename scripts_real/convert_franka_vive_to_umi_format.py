"""
Franka+Vive 데이터를 UmiDataset 형식으로 변환하는 스크립트

demo_franka_vive.py로 수집한 데이터를 UmiDataset이 읽을 수 있는 형식으로 변환합니다.

변환 과정:
1. replay_buffer.zarr (low-dim) + videos/ (MP4) 읽기
2. 비디오에서 이미지 추출하여 zarr에 통합
3. 카메라 키 이름 변환: camera_0 → camera0_rgb
4. zarr.zip으로 압축

Usage:
    python scripts_real/convert_franka_vive_to_umi_format.py \
        --input data/gripper_test \
        --output data/gripper_test/dataset.zarr.zip \
        --resolution 224,224
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import zarr
import numpy as np
import cv2
from tqdm import tqdm
from threadpoolctl import threadpool_limits
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
register_codecs()


@click.command()
@click.option('--input', '-i', required=True, help='Input directory (from demo_franka_vive.py)')
@click.option('--output', '-o', required=True, help='Output zarr.zip path')
@click.option('--resolution', '-r', default='640,480', help='Output image resolution (width,height). Default 640x480 matches reference config.')
def main(input, output, resolution):
    """Convert Franka+Vive data to UMI dataset format."""

    # Parse resolution
    width, height = map(int, resolution.split(','))
    out_resolution = (width, height)

    print(f"=== Franka+Vive → UMI Format Conversion ===")
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Resolution: {out_resolution}")

    # Check input
    in_zarr_path = os.path.join(input, 'replay_buffer.zarr')
    in_video_dir = os.path.join(input, 'videos')

    if not os.path.isdir(in_zarr_path):
        raise ValueError(f"replay_buffer.zarr not found at {in_zarr_path}")
    if not os.path.isdir(in_video_dir):
        raise ValueError(f"videos/ not found at {in_video_dir}")

    # Count cameras
    episode_dirs = [d for d in os.listdir(in_video_dir) if os.path.isdir(os.path.join(in_video_dir, d))]
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {in_video_dir}")

    first_episode_dir = os.path.join(in_video_dir, sorted(episode_dirs)[0])
    video_files = [f for f in os.listdir(first_episode_dir) if f.endswith('.mp4')]
    n_cameras = len(video_files)
    print(f"Found {n_cameras} cameras")

    # Setup image keys with UMI naming convention
    image_keys = [f'camera{i}_rgb' for i in range(n_cameras)]
    out_resolutions = {key: out_resolution for key in image_keys}

    # Load low-dim data first
    in_replay_buffer = ReplayBuffer.create_from_path(in_zarr_path, mode='r')
    lowdim_keys = ['action', 'robot0_eef_pos', 'robot0_eef_rot_axis_angle',
                   'robot0_gripper_width', 'robot0_joint_pos', 'robot0_joint_vel', 'timestamp']
    lowdim_keys = [k for k in lowdim_keys if k in in_replay_buffer.keys()]

    print(f"Low-dim keys: {lowdim_keys}")
    print(f"Image keys: {image_keys}")

    # Use real_data_to_replay_buffer but with UMI key naming
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        # Create temporary replay buffer with images
        temp_store = zarr.MemoryStore()

        # We need to modify the image key naming
        # real_data_to_replay_buffer uses 'camera_0', 'camera_1' but UMI expects 'camera0_rgb', 'camera1_rgb'

        # First load with original naming
        original_image_keys = [f'camera_{i}' for i in range(n_cameras)]
        original_resolutions = {key: out_resolution for key in original_image_keys}

        print("\nExtracting images from videos...")
        temp_buffer = real_data_to_replay_buffer(
            dataset_path=input,
            out_store=temp_store,
            out_resolutions=original_resolutions,
            lowdim_keys=lowdim_keys,
            image_keys=original_image_keys
        )

    # Now rename camera keys and save to zip
    print("\nRenaming camera keys and saving to zip...")

    with zarr.ZipStore(output, mode='w') as zip_store:
        out_root = zarr.group(zip_store)
        out_data = out_root.create_group('data')
        out_meta = out_root.create_group('meta')

        # Copy episode_ends
        out_meta.create_dataset('episode_ends', data=temp_buffer.episode_ends[:])

        # Copy low-dim data
        for key in lowdim_keys:
            if key in temp_buffer.keys():
                arr = temp_buffer[key][:]
                out_data.create_dataset(key, data=arr.astype(np.float32))
                print(f"  {key}: {arr.shape}")

        # Copy and rename image data
        for i in range(n_cameras):
            old_key = f'camera_{i}'
            new_key = f'camera{i}_rgb'
            if old_key in temp_buffer.keys():
                arr = temp_buffer[old_key][:]
                # Use Jpeg2k compression for images
                compressor = Jpeg2k(level=50)
                out_data.create_dataset(
                    new_key,
                    data=arr,
                    chunks=(1, height, width, 3),
                    compressor=compressor
                )
                print(f"  {old_key} → {new_key}: {arr.shape}")

    print(f"\n=== Conversion Complete ===")
    print(f"Output saved to: {output}")

    # Verify
    print("\nVerifying output...")
    with zarr.ZipStore(output, mode='r') as zip_store:
        root = zarr.group(zip_store)
        print("Keys in output:")
        for key in root['data'].keys():
            arr = root['data'][key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == '__main__':
    main()
