"""Best-effort real-time tuning for multiprocessing children on pro4000.

Why this exists: pro4000 runs ~8 Python multiprocessing children (Vive, two
ZED workers, ART gripper, Franka interp controller, main loop, cv2 viewer,
multiprocessing.resource_tracker) on a 14-core PREEMPT_DYNAMIC kernel.
Default Linux scheduling is free to put the timing-critical Polymetis
client and ART gripper controller on cores that also handle the NUC-
subnet NIC IRQ, causing softirq RX/TX to delay our gRPC update_desired_*
calls. NUC's libfranka then misses the 1 ms FCI window and emits a
``communication_constraints_violation`` reflex. Pinning these children
to dedicated cores (away from the NIC IRQ cores configured by
install/pro4000/sbin/franka_client_rt_apply.sh) eliminates that path.

We try three layers, each best-effort:
  1. CPU affinity (sched_setaffinity) -- always works.
  2. SCHED_RR with priority -- requires CAP_SYS_NICE / root.
  3. nice value -- negative requires elevated /etc/security/limits.

Layer 1 alone is enough to break the IRQ-vs-Python-compute collision.
Layers 2/3 are gravy when available.
"""
import os


def apply_realtime(cores=None, sched_priority=20, nice_value=-10, name=""):
    """Apply CPU pin + (optional) RT scheduling + (optional) nice.

    Args:
        cores: iterable of CPU indices to pin to (e.g. {6, 7}). If None,
            affinity is left untouched.
        sched_priority: SCHED_RR priority (1..99). Higher = more urgent.
            20 is a moderate value, won't starve normal kernel work.
        nice_value: fallback nice value when SCHED_RR fails. -10 is
            two steps above default 0 toward higher priority.
        name: prefix for log messages so users can tell which child
            applied which knobs.

    Returns: dict {affinity, sched, nice} of which knobs succeeded.
    """
    result = {'affinity': False, 'sched': None, 'nice': None}

    if cores is not None:
        cores = set(int(c) for c in cores)
        try:
            os.sched_setaffinity(0, cores)
            result['affinity'] = True
            print(f"[rt {name}] CPU affinity set to {sorted(cores)}")
        except (PermissionError, OSError) as e:
            print(f"[rt {name}] WARN: sched_setaffinity failed: {e}")

    try:
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(sched_priority))
        result['sched'] = f"SCHED_RR/{sched_priority}"
        print(f"[rt {name}] SCHED_RR priority {sched_priority}")
    except (PermissionError, OSError) as e:
        # No CAP_SYS_NICE -> fall back to nice
        try:
            new_nice = os.nice(nice_value)
            result['nice'] = new_nice
            print(f"[rt {name}] SCHED_RR unavailable ({e}); set nice -> {new_nice}")
        except (PermissionError, OSError) as e2:
            print(f"[rt {name}] no priority elevation possible "
                  f"(SCHED_RR: {e}, nice: {e2}). Falling back to default scheduling.")

    return result


# Default core assignment for pro4000's Intel Core Ultra 5 245K (14 cores,
# no HT). NIC IRQ for enp130s0 (NUC subnet) is pinned to cores 0,1 by
# install/pro4000/sbin/franka_client_rt_apply.sh. Cores 2-5 are general
# pool. Cores 6-13 are reserved for our Python compute.
PRO4000_CORE_MAP = {
    'franka_interp':   {6, 7},     # 100 Hz Polymetis client + IK
    'art_gripper':     {8, 9},     # 60 Hz TCP polling + ring buffer
    'zed_camera_a':    {10},       # exterior ZED 2i
    'zed_camera_b':    {11},       # wrist ZED Mini
    'vive_teleop':     {12},       # 100 Hz Vive read + clutch
    'main_loop':       {13},       # 15 Hz demo main + accumulators
    # General pool 2-5 left for cv2_viewer, resource_tracker, helpers.
}
