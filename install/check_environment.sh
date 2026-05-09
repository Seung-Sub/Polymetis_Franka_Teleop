#!/usr/bin/env bash
# Verify the Polymetis_Franka_Teleop environment is ready to run.
# Run this on pro4000 BEFORE the first teleop session.
set -uo pipefail

ok()  { echo "  [OK]  $*"; }
warn(){ echo "  [WARN] $*"; }
fail(){ echo "  [FAIL] $*"; }

echo "=== Polymetis_Franka_Teleop environment check ==="

# 1. conda env
if [[ -d "${HOME}/anaconda3/envs/groot-client" || -d "${HOME}/miniconda3/envs/groot-client" ]]; then
    ok "conda env 'groot-client' present"
else
    warn "conda env 'groot-client' not found — adapt scripts to your env"
fi

# 2. polymetis client (pro4000 side, talks to NUC :50051 directly)
${CONDA_PREFIX:-${HOME}/anaconda3/envs/groot-client}/bin/python -c "import polymetis" 2>/dev/null \
    && ok "polymetis client" \
    || fail "polymetis client missing — install fairo-polymetis"

# 3. zerorpc (only used by --gripper_backend franka to reach the Franka Hand service :4242)
${CONDA_PREFIX:-${HOME}/anaconda3/envs/groot-client}/bin/python -c "import zerorpc" 2>/dev/null \
    && ok "zerorpc (needed only for Franka Hand gripper backend)" \
    || warn "zerorpc missing — required only for --gripper_backend franka"

# 4. ZED SDK + pyzed
${CONDA_PREFIX:-${HOME}/anaconda3/envs/groot-client}/bin/python -c "import pyzed.sl" 2>/dev/null \
    && ok "pyzed.sl" \
    || fail "pyzed not installed — install ZED SDK + pyzed wrappers"

# 5. ART gripper client
${CONDA_PREFIX:-${HOME}/anaconda3/envs/groot-client}/bin/python -c "import art_gripper_client" 2>/dev/null \
    && ok "art_gripper_client (pip)" \
    || {
        if [[ -d "${ART_GRIPPER_PYPATH:-${HOME}/Hyundai_motors_Gripper/python}" ]]; then
            warn "art_gripper_client not pip-installed — will fall back to ART_GRIPPER_PYPATH"
        else
            fail "art_gripper_client not found AND ${HOME}/Hyundai_motors_Gripper/python missing"
        fi
    }

# 6. diffusion_policy (Zarr ReplayBuffer + cv2_util etc.)
${CONDA_PREFIX:-${HOME}/anaconda3/envs/groot-client}/bin/python -c "import diffusion_policy" 2>/dev/null \
    && ok "diffusion_policy importable" \
    || {
        if [[ -d "${DIFFUSION_POLICY_PATH:-${HOME}/diffusion_policy}" ]]; then
            warn "diffusion_policy not on PYTHONPATH — bin/start_*.sh inject it via DIFFUSION_POLICY_PATH"
        else
            fail "diffusion_policy not found — clone https://github.com/columbia-ai-robotics/diffusion_policy"
        fi
    }

# 7. ART daemon (pro4000 systemd)
if systemctl is-active art-gripper-daemon >/dev/null 2>&1; then
    ok "art-gripper-daemon active"
else
    warn "art-gripper-daemon not active — needed for --gripper_backend art"
fi
ss -tln 2>/dev/null | grep -q ':50053 ' && ok "ART daemon :50053 listening" \
    || warn "ART daemon :50053 not listening"

# 8. SteamVR + vive_input
pgrep -x vrserver >/dev/null 2>&1 && ok "SteamVR (vrserver) running" \
    || warn "SteamVR vrserver not running — bin/start_vive_stack.sh start"
ss -tln 2>/dev/null | grep -q ':12345 ' && ok "vive_input :12345 listening" \
    || warn "vive_input :12345 not listening"

# 9. NUC reachability
ping -c 1 -W 1 192.168.1.12 >/dev/null 2>&1 && ok "NUC 192.168.1.12 reachable" \
    || fail "NUC 192.168.1.12 not pingable"

# 10. NUC services
nc -z -w 2 192.168.1.12 50051 2>/dev/null && ok "NUC polymetis arm :50051 reachable" \
    || warn "NUC polymetis arm :50051 not reachable — sudo bash /usr/local/sbin/start_franka_arm.sh on NUC"
nc -z -w 2 192.168.1.12 4242 2>/dev/null && ok "NUC ZeroRPC bridge :4242 reachable" \
    || warn "NUC ZeroRPC bridge :4242 not reachable — bash bin/start_unified_bridge_on_nuc.sh"

echo
echo "Done. Resolve any [FAIL] lines before running demo_franka_vive.py."
