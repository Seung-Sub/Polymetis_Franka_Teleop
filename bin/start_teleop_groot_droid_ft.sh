#!/usr/bin/env bash
# Wrapper: data collection for GR00T-DROID fine-tuning.
#
# Difference vs start_teleop.sh (the Diffusion-Policy / UMI default):
#   * --frequency 15      (matches DROID dataset's 15 Hz cadence; demo_data/
#                          droid_sample is 15 fps. Models post-trained on
#                          DROID expect 1/15 s spacing in the action chunks.)
#   * --data_format groot (recorder writes meta/data_format='groot' so the
#                          converter can validate the pipeline at conversion
#                          time. Also drives env.compute_ready_pose to use
#                          the DROID base pose [0, -pi/5, 0, -4pi/5, 0, 3pi/5, 0]
#                          which the GR00T-DROID checkpoint already saw
#                          during pretraining.)
#
# Other defaults mirror start_teleop.sh (ART gripper + 2× ZED + Vive teleop;
# 60 fps native VGA cameras; --teleop_frequency 100).
#
# Output is fully convertible to LeRobot v2 with:
#   python scripts_real/convert_to_gr00t_lerobot.py \
#       -i <output_dir> -o <output_dir>_gr00t
#   # gripper_max_width, fps, episode_tasks all auto-loaded from zarr meta
#
# Usage:
#   bash bin/start_teleop_groot_droid_ft.sh <output_dir> [extra demo_franka_vive flags]
set -euo pipefail

OUTPUT="${1:?usage: start_teleop_groot_droid_ft.sh <output_dir> [extra args...]}"
shift || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate groot-client

export DIFFUSION_POLICY_PATH="${DIFFUSION_POLICY_PATH:-${HOME}/diffusion_policy}"
export ART_GRIPPER_PYPATH="${ART_GRIPPER_PYPATH:-${HOME}/Hyundai_motors_Gripper/python}"

exec python scripts_real/demo_franka_vive.py \
    --output "$OUTPUT" \
    --robot_ip 192.168.1.12 \
    --camera_backend zed --gripper_backend art \
    --camera_serials 33538770 --camera_serials 11667817 \
    --camera_resolution 672x376 --camera_fps 60 \
    --frequency 15 --teleop_frequency 100 \
    --data_format groot \
    -v "$@"
