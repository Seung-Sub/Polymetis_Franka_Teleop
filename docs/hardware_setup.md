# Hardware setup

```
                 wifi/ext  enxb0386cf0edc0  →  161.122.114.90  (pro4000 lab IP)
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
            │           pro4000 (kist-eval)      │
            │  • SteamVR + vive_input :12345     │
            │  • this repo (groot-client env)    │
            │  • art_gripper_daemon :50053       │
            │  • GR00T server (separately)       │
            │                                    │
            └─┬─────────────┬──────────────────┬─┘
              │             │                  │
   USB        │  EtherCAT   │   wired          │  USB
              │             │                  │
        ZED 2i + Mini    Hyundai gripper      NUC ─── Franka FCI
        (33538770,        (NETX 90-RE/ECS)   192.168.1.12
         11667817)         enxb0386cf13036    polymetis :50051
                                              franka_hand :50052
                                              unified ZeroRPC :4242
```

## Networking

| What | Where | Name | IP/Port |
|---|---|---|---|
| pro4000 ext | enxb0386cf0edc0 | KIST campus | 161.122.114.90 |
| pro4000 NUC subnet | enp130s0 | private | 192.168.1.20 |
| pro4000 Hyundai EtherCAT | enxb0386cf13036 | private | 161.122.115.1 |
| NUC | (Franka box) | kist-NUC13ANHi7 | 192.168.1.12 |
| Polymetis arm gRPC | NUC | franka_panda_client | :50051 |
| Polymetis franka_hand gRPC | NUC | franka_hand_client | :50052 |
| ZeroRPC unified | NUC | franka_unified | :4242 |
| ART daemon | pro4000 | art_gripper_daemon | :50053 |
| Vive input | pro4000 | vive_input (vive_ws) | :12345 (TCP) / :12346 (UDP haptic) |

## SSH cheat sheet

```
ssh kist@161.122.114.90       # pro4000   pwd: ' '   (one space)
ssh kist@192.168.1.12         # NUC      (only reachable from pro4000) pwd: kist
```

## Boot-time auto-services on pro4000

```
art-gripper-daemon.service         active   /usr/local/bin/art_gripper_daemon
ethercat.service                   active   IgH master, /dev/EtherCAT0
franka-rt-tune.service             active   RT priority + DMA latency for libfranka
franka-dma-latency.service         active   /usr/local/sbin/franka_dma_latency.py
```

If a fault latches the ART gripper (motor crashes, hot un-plug, etc.):

```
sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh
# stops daemon → restarts ethercat kmod → starts daemon → ping verify
```

If THAT can't recover (firmware-level latched fault), 24 V power cycle the
gripper, daemon auto-reattaches.

## NUC services (not auto-started)

```
sudo bash /usr/local/sbin/start_franka_arm.sh        # polymetis arm :50051
sudo bash /usr/local/sbin/start_franka_gripper.sh    # franka_hand :50052 (skip if --gripper_backend art)
```

Franka Desk web UI must be open and FCI Activated; e-stop must be
released.

## Cameras

```
ZED 2i      sn 33538770     exterior, tripod ~1 m from base
ZED Mini    sn 11667817     wrist, mounted on Franka flange
```

Both used **LEFT eye only** (matches DROID + GR00T training convention).
`pyzed.sl` is installed in the `groot-client` conda env. Verify visible:

```bash
python -c "import pyzed.sl as sl; \
  print([f'sn={d.serial_number} {d.camera_state}' for d in sl.Camera.get_device_list()])"
```

## Vive

The repo's `vive_shared_memory.py` connects to `vive_input` over TCP `:12345`.
Bring up SteamVR, ensure the controller is paired and tracked, then start the
`vive_input` binary (no ROS needed):

```
~/vive_ws/install/vive_ros2/lib/vive_ros2/vive_input        # raw sender
```

## End-effector dynamics (Franka Desk)

Calibrated for ART + ZED Mini wrist load:

| Field | Value |
|---|---|
| Mass | 1.05 kg |
| Flange→CoM | (0, 0.010, 0.100) m |
| Flange→TCP | (0, 0, 0.216, 0, 0, 0) — finger tip |
| Inertia diag | (0.004, 0.004, 0.001) |

(Re-measure with `~/Isaac-GR00T/scripts/kist/measure_ee_load.py` after any
EE hardware swap.)
