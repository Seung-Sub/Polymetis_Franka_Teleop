#!/usr/bin/env bash
# Run every relevant latency calibrator for the selected backends and patch
# install/latency_calibration.json in one go.
#
# Usage:
#   bash bin/calibrate_all.sh                              # auto-defaults (zed + art + arm-obs)
#   bash bin/calibrate_all.sh --camera zed --gripper art   # explicit
#   bash bin/calibrate_all.sh --measure_action             # also run arm action measurement (robot moves!)
#   bash bin/calibrate_all.sh --gripper franka             # Franka Hand (requires NUC franka_hand server)
#   bash bin/calibrate_all.sh --camera realsense           # RealSense (requires pyrealsense2)
#   bash bin/calibrate_all.sh --skip_arm                   # do not run arm calibrator
#
# Pre-req: NUC polymetis arm running (:50051). For Franka Hand also start
# :50052; for ART start systemd ``art-gripper-daemon`` on pro4000.
set -uo pipefail

CAMERA="zed"
GRIPPER="art"
ROBOT_IP="${ROBOT_IP:-192.168.1.12}"
MEASURE_ACTION=0
SKIP_ARM=0
SKIP_CAMERA=0
SKIP_GRIPPER=0
ZED_SERIALS=(33538770 11667817)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera)            CAMERA="$2"; shift 2 ;;
        --gripper)           GRIPPER="$2"; shift 2 ;;
        --robot_ip)          ROBOT_IP="$2"; shift 2 ;;
        --zed_serial)        ZED_SERIALS=("$2"); shift 2 ;;
        --measure_action)    MEASURE_ACTION=1; shift ;;
        --skip_arm)          SKIP_ARM=1; shift ;;
        --skip_camera)       SKIP_CAMERA=1; shift ;;
        --skip_gripper)      SKIP_GRIPPER=1; shift ;;
        -h|--help)
            sed -n '1,20p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Activate the env that has all the Python deps (polymetis, pyzed, ...).
if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate groot-client
fi
export CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/anaconda3/envs/groot-client}"

echo "============================================================"
echo " latency calibration sweep"
echo "============================================================"
echo "  camera  = ${CAMERA}"
echo "  gripper = ${GRIPPER}"
echo "  arm     = polymetis :50051 @ ${ROBOT_IP}"
echo "  action  = $([ "${MEASURE_ACTION}" -eq 1 ] && echo "ON (robot moves)" || echo "off")"
echo

# 1) Arm
if [[ "${SKIP_ARM}" -eq 0 ]]; then
    echo "---- arm (polymetis direct :50051) ----"
    if [[ "${MEASURE_ACTION}" -eq 1 ]]; then
        python scripts_real/calibrate_franka_arm_direct.py \
            --robot_ip "${ROBOT_IP}" --measure_action
    else
        python scripts_real/calibrate_franka_arm_direct.py \
            --robot_ip "${ROBOT_IP}"
    fi
    echo
fi

# 2) Camera
if [[ "${SKIP_CAMERA}" -eq 0 ]]; then
    case "${CAMERA}" in
        zed)
            echo "---- camera (ZED) ----"
            zed_args=()
            for s in "${ZED_SERIALS[@]}"; do
                zed_args+=(--serial "${s}")
            done
            python scripts_real/calibrate_zed_latency.py "${zed_args[@]}"
            ;;
        realsense)
            echo "---- camera (RealSense) ----"
            python scripts_real/calibrate_realsense_latency.py
            ;;
        *)
            echo "[skip] unknown camera backend: ${CAMERA}" ;;
    esac
    echo
fi

# 3) Gripper
if [[ "${SKIP_GRIPPER}" -eq 0 ]]; then
    case "${GRIPPER}" in
        art)
            echo "---- gripper (ART, TCP :50053) ----"
            python scripts_real/calibrate_art_gripper_latency.py
            ;;
        franka)
            echo "---- gripper (Franka Hand, gRPC :50052) ----"
            python scripts_real/calibrate_franka_gripper_latency.py \
                --robot_ip "${ROBOT_IP}"
            ;;
        *)
            echo "[skip] unknown gripper backend: ${GRIPPER}" ;;
    esac
    echo
fi

echo "============================================================"
echo " calibration sweep complete. Current JSON state:"
python -c "
import json, pathlib
p = pathlib.Path('install/latency_calibration.json')
cfg = json.loads(p.read_text())
for k, v in cfg.items():
    if not k.startswith('_'):
        print(f'  {k} = {v}')
print('  dates:')
for k, v in cfg.get('_calibration_dates', {}).items():
    print(f'    {k:18s} {v}')
"
