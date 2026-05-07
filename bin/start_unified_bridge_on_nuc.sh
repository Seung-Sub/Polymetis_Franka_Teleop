#!/usr/bin/env bash
# Start the unified ZeroRPC bridge on the NUC (port 4242).
#
# Architecture:
#   pro4000 client  ──► NUC ZeroRPC :4242  ──►  polymetis :50051 (loopback)
#                                          ──►  franka_hand :50052 (optional)
#
# This is the community-canonical Polymetis remote-teleop pattern (UMI, DROID,
# R2D2). The NUC runs the polymetis Python client locally, exposing simplified
# RPC methods. Each remote teleop tick = ONE network call, instead of 4 polymetis
# gRPC roundtrips.
#
# Pre-req on NUC:
#   - polymetis arm server already up via /usr/local/sbin/start_franka_arm.sh
#   - conda env `polymetis-local` (has polymetis + zerorpc)
#
# This script ssh's into the NUC and launches the bridge in the background.
# It assumes:
#   - SSH access to NUC user `kist` (default: pwd "kist")
#   - sshpass installed
# Override via env: NUC_USER, NUC_HOST, NUC_PWD, BRIDGE_PORT, BRIDGE_FLAGS
set -euo pipefail

NUC_USER="${NUC_USER:-kist}"
NUC_HOST="${NUC_HOST:-192.168.1.12}"
NUC_PWD="${NUC_PWD:-kist}"
BRIDGE_PORT="${BRIDGE_PORT:-4242}"
# For ART gripper workflow: --no_gripper. For Franka Hand: leave empty.
BRIDGE_FLAGS="${BRIDGE_FLAGS:---no_gripper}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/scripts_real/launch_franka_unified_server.py"
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }

if ! command -v sshpass >/dev/null 2>&1; then
    echo "[bridge] sshpass not found — please run the NUC steps manually." >&2
    echo "" >&2
    echo "On NUC:" >&2
    echo "    scp $SRC $NUC_USER@$NUC_HOST:/tmp/" >&2
    echo "    source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local" >&2
    echo "    python /tmp/launch_franka_unified_server.py $BRIDGE_FLAGS" >&2
    exit 1
fi

echo "[bridge] uploading $(basename "$SRC") to $NUC_USER@$NUC_HOST:/tmp/"
sshpass -p "$NUC_PWD" scp -o StrictHostKeyChecking=no "$SRC" \
    "$NUC_USER@$NUC_HOST:/tmp/launch_franka_unified_server.py"

echo "[bridge] killing any stale bridge on NUC ..."
sshpass -p "$NUC_PWD" ssh -o StrictHostKeyChecking=no "$NUC_USER@$NUC_HOST" \
    "pkill -f launch_franka_unified_server || true" || true
sleep 1

echo "[bridge] launching bridge on NUC (port $BRIDGE_PORT, flags: $BRIDGE_FLAGS) ..."
sshpass -p "$NUC_PWD" ssh -o StrictHostKeyChecking=no "$NUC_USER@$NUC_HOST" "
    nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local && python /tmp/launch_franka_unified_server.py --port $BRIDGE_PORT $BRIDGE_FLAGS' \\
        > /tmp/unified_bridge.log 2>&1 &
    echo \"  PID=\$!\"
"

echo "[bridge] waiting for :$BRIDGE_PORT to listen ..."
for _ in $(seq 1 15); do
    if sshpass -p "$NUC_PWD" ssh -o StrictHostKeyChecking=no "$NUC_USER@$NUC_HOST" \
        "ss -tln | grep -q ':$BRIDGE_PORT '"; then
        echo "[bridge] OK — listening on $NUC_HOST:$BRIDGE_PORT"
        echo "[bridge] log on NUC: /tmp/unified_bridge.log"
        exit 0
    fi
    sleep 1
done

echo "[bridge] FAIL — bridge did not come up. tail of NUC log:" >&2
sshpass -p "$NUC_PWD" ssh -o StrictHostKeyChecking=no "$NUC_USER@$NUC_HOST" "tail -20 /tmp/unified_bridge.log"
exit 1
