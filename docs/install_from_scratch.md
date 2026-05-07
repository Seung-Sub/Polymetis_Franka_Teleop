# Polymetis_Franka_Teleop — Install from scratch

KIST 환경 (NUC + pro4000 + Franka Panda + ZED + ART gripper + Vive) 처음부터 끝까지 재현 가이드.

이 문서 + `git clone` 만 있으면 동일 환경 재구성 가능. 일상 운용은 [`README.md`](../README.md) 의 *Bring up the stack* 섹션 참조.

> **이 문서의 범위**: 이 워크스페이스 (`Polymetis_Franka_Teleop`) 가 동작하기 위한 모든 의존성 + 설정.
> Polymetis/Franka 자체의 처음 셋업 (PREEMPT_RT 커널, libfranka, RT IRQ 핀 등 NUC 시스템 레벨)
> 은 자매 문서 [`Isaac-GR00T/INSTALL_FROM_SCRATCH.md`](../../Isaac-GR00T/INSTALL_FROM_SCRATCH.md)
> §3–§9 참고 (KIST 에서 GR00T + 본 repo 둘 다 같은 NUC 셋업 공유).

---

## 0. 요약 — 시스템 토폴로지

```
[Franka Panda + ART gripper]
   ↑ FCI 1 kHz Ethernet (172.16.0.2)
   │
[NUC i7-1360P] (Ubuntu 22.04 PREEMPT_RT)
   ├ libfranka realtime client    cores 6,7 핀 (P-core 격리)
   ├ polymetis_server :50051      gRPC server, 1 kHz inner loop
   ├ franka_hand_client :50052    (Franka Hand 사용 시)
   └ launch_franka_unified_server.py :4242  (선택 — UMI/DROID 호환 ZeroRPC bridge)
   │
   │ 1 Gbps 유선 / RTT < 1 ms
   ↓
[pro4000 RTX 4000 Blackwell] (Ubuntu 22.04, kist-eval)
   ├ Polymetis_Franka_Teleop      ← 이 repo
   │   ├ FrankaInterpolationController (mp.Process, 100 Hz)
   │   ├ ViveTeleopProcess (mp.Process, 100 Hz)
   │   ├ MultiZed (mp.Process×N, 60 fps capture)
   │   ├ FrankaGripperController / ArtGripperController (mp.Process, 30 Hz)
   │   └ Main loop (10 Hz, Zarr + H264 MP4 기록)
   ├ Hyundai_motors_Gripper       ← 자매 repo (ART daemon + python client)
   │   └ /usr/local/bin/art_gripper_daemon (systemd, EtherCAT :50053)
   ├ SteamVR + vive_input         ← Vive 트래킹 + TCP 서버 :12345
   └ ZED 2i + ZED Mini (USB)
```

전체 파이프라인 hz / 통신 / 핀 결정 근거는 [`docs/pipeline.md`](pipeline.md) 참고.

---

## 1. NUC 시스템 셋업 (한 번)

GR00T 작업 때 이미 셋업했다면 그대로 재사용. 처음 한다면 `Isaac-GR00T/INSTALL_FROM_SCRATCH.md` §3–§6 단계별 따라가기:

| 단계 | 내용 | KIST 핵심 결정 |
|---|---|---|
| §3 | Franka Desk 사전 설정 | FCI 활성화, end-effector 질량/CoM 입력 |
| §4 | NUC OS — Ubuntu 22.04 + **PREEMPT_RT 커널** | `uname -a` 에 `PREEMPT_RT` 포함 |
| §4-2 | GRUB `isolcpus=domain,managed_irq,6-7 nohz_full=6,7 rcu_nocbs=6,7` | P-core 6,7 RT 격리 |
| §4-3 | libfranka 빌드 + 설치 | Franka 공식 가이드 |
| §5 | fairo (Polymetis) clone + build, `polymetis-local` conda env | NUC 내부에서만 |
| §6 | RT 영구 튜닝 (`franka_pin_helper.sh`, NIC IRQ E-core 핀, RPS/XPS, Turbo OFF) | **이게 watchdog/jitter 안정성의 결정타** |
| §6-3 | systemd 자동 핀: `franka-rt-tune`, `franka-dma-latency` | 부팅 시 auto-apply |

**RT 튜닝 필수 이유** (KIST 실측 기반):
- `isolcpus=6,7` 만으로는 부족 — `franka_pin_helper.sh` 로 launch 후 명시 `taskset` 필수
- NIC IRQ default 가 P-core 0–5 분산 → GUI 와 경쟁 → success_rate 0.79 → reflex 발생
- → NIC IRQ 를 E-core 12–15 로 명시 핀
- RPS/XPS off (default) → RX/TX softirq 가 임의 코어로 → E-core mask 명시
- Turbo Boost ON → P-core 주파수 변동 → RT jitter → OFF 권장

이 작업이 안 되면 우리 repo 의 `polymetis-direct` 100 Hz 도 watchdog trip 가능.

---

## 2. NUC — 본 repo 가 사용하는 부분

```bash
# /usr/local/sbin/start_franka_arm.sh 가 이미 깔려 있어야 함 (GR00T 셋업)
ls /usr/local/sbin/start_franka_arm.sh   # OK 있어야

# Franka Hand 쓸 때만:
ls /usr/local/sbin/start_franka_gripper.sh

# ZeroRPC bridge 쓰려면 (선택):
# pro4000 의 bin/start_unified_bridge_on_nuc.sh 가 자동 deploy + launch
```

ART gripper 만 쓰는 KIST 워크플로에서는 NUC 에 **arm 서버만** 띄움. franka_hand 는 안 띄워도 됨.

---

## 3. pro4000 — repo 클론 + conda env

```bash
# 1) 자매 repo (필수)
cd $HOME
git clone <YOUR_GIT>/Hyundai_motors_Gripper       # ART gripper daemon + python client
git clone <YOUR_GIT>/Polymetis_Franka_Teleop      # 본 repo
git clone https://github.com/columbia-ai-robotics/diffusion_policy   # ReplayBuffer 등 의존

# 2) ART daemon 설치 + systemd 등록 (Hyundai_motors_Gripper README 참고)
cd ~/Hyundai_motors_Gripper
sudo bash scripts/install_etherlab.sh    # 첫 셋업만
sudo bash scripts/install_daemon.sh --system
systemctl enable --now art-gripper-daemon

# 3) ZED SDK + pyzed (본 repo 가 사용)
#    https://www.stereolabs.com/developers/release/  → Linux x86_64 .run
#    설치 후 python -c 'import pyzed.sl' 동작 확인

# 4) 본 repo conda env (groot-client 재사용 가능)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate groot-client    # GR00T 작업 때 만든 env (polymetis client + pyzed + zerorpc)

# 5) 본 repo pip install + ART client 의존
cd ~/Polymetis_Franka_Teleop
pip install -e .
pip install -e ~/Hyundai_motors_Gripper/python   # art_gripper_client 모듈

# 6) 환경 점검
bash install/check_environment.sh   # OK / WARN / FAIL 출력
```

**`groot-client` env 가 이미 있어야 하는 패키지** (GR00T INSTALL_FROM_SCRATCH §9 와 동일):
- `polymetis` (pro4000 측 클라이언트 — direct mode 시 사용)
- `pyzed.sl`
- `zerorpc` (zerorpc bridge mode 시 사용)
- `pynput`, `av`, `zarr<3`, `atomics`, `pyarrow`, `dill`
- `diffusion_policy` (~/diffusion_policy 의 PYTHONPATH 또는 pip install -e)

---

## 4. Vive 입력 셋업 (이 repo 만의 추가)

```bash
# SteamVR (Steam GUI 에서 SteamVR install)
ls ~/.steam/debian-installation/steamapps/common/SteamVR/bin/linux64/vrserver

# vive_input C++ 바이너리 (ROS-free, OpenVR + nlohmann/json)
#   원본 source: ~/Isaac-GR00T/vive_input/  (KIST 기존 보유)
#   이미 빌드돼 있어야 함:
ls ~/vive_ws/build/vive_ros2/vive_input  # 또는 ~/Isaac-GR00T/vive_input/build/vive_input

# HTC Vive 컨트롤러 + base station 2개 + (HMD 또는 트래커 dummy)
lsusb | grep -i htc       # HTC 디바이스 인식 확인
sudo chmod +rw /dev/hidraw*   # 첫 셋업 시 1회

# 부팅 (`bin/start_vive_stack.sh`):
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start   # vrserver --keepalive + vive_input :12345
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh status
```

`vrserver --keepalive` 는 헤드리스 (GUI 불필요) 라 SSH 환경에서도 띄울 수 있음.

---

## 5. 첫 부팅 시퀀스 (모든 컴포넌트 가동)

```bash
# 0. Franka Desk 웹UI 에서 FCI Activate, e-stop 풀기, joint locks 해제

# 1. NUC arm 서버
ssh kist@192.168.1.12
sudo bash /usr/local/sbin/start_franka_arm.sh
# "Connected." 메시지까지 기다림

# 2. (Franka Hand 만 쓸 때) NUC gripper 서버
ssh kist@192.168.1.12
sudo bash /usr/local/sbin/start_franka_gripper.sh

# 3. (선택, zerorpc 모드) NUC unified bridge
ssh kist@161.122.114.90      # pro4000
bash ~/Polymetis_Franka_Teleop/bin/start_unified_bridge_on_nuc.sh   # 자동 ssh + deploy + launch

# 4. pro4000 — ART gripper daemon (자동 부팅, 평소 손댈 일 없음)
systemctl status art-gripper-daemon

# 5. pro4000 — Vive
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start

# 6. pro4000 — preflight
cd ~/Polymetis_Franka_Teleop && bash install/check_environment.sh

# 7. pro4000 — 데이터 수집
bash bin/start_teleop.sh ~/Polymetis_Franka_Teleop/data/$(date +%Y%m%d_%H%M%S)
```

---

## 6. KIST 운용 노트 — GR00T 워크스페이스와의 공존

본 repo 와 `Isaac-GR00T` 워크스페이스는 **NUC 측 polymetis 서버 + ART daemon + ZED 카메라 + RT 튜닝을 공유**합니다. 동시에 띄우지는 못하지만 (서로 카메라/그리퍼 점유), 다음 패턴으로 번갈아 사용 가능:

| 작업 | 사용 워크스페이스 | 부팅 명령 |
|---|---|---|
| GR00T 추론 (Vive 미사용) | `Isaac-GR00T` | `start_franka_arm.sh` + `start_groot_server_*.sh` + `start_groot_client_art.sh` |
| Vive teleop 데이터 수집 | `Polymetis_Franka_Teleop` | `start_franka_arm.sh` + `start_vive_stack.sh` + `bin/start_teleop.sh` |
| Diffusion Policy eval | `Polymetis_Franka_Teleop` | `start_franka_arm.sh` + `bin/start_eval.sh <ckpt>` |

NUC 측 `start_franka_arm.sh` 는 둘 다 동일하게 사용. 카메라/그리퍼는 한 번에 한 워크스페이스만 점유.

---

## 7. 알려진 한계 + 향후 개선

| 항목 | 현재 | 개선 여지 |
|---|---|---|
| NUC realtime watchdog (1s) | `direct` 100 Hz / `zerorpc` 100 Hz 안정 | NUC RT 튜닝 강화로 200 Hz 도전 가능 (UMI 사례) |
| ZED Mini USB 2회 hot-plug 후 재인식 | 사용자 수동 재연결 필요 | udev rule 정리 (TODO) |
| SteamVR auto-start | 부팅 후 매번 `start_vive_stack.sh start` 수동 | systemd-user 단위 등록 가능 (선택) |

---

## 참고

- [Polymetis docs](https://facebookresearch.github.io/fairo/polymetis/overview.html)
- [libfranka FCI](https://frankarobotics.github.io/docs/libfranka.html)
- [Stanford UMI](https://github.com/real-stanford/universal_manipulation_interface)
- 자매 repo (KIST 동일 NUC 사용): `Isaac-GR00T/INSTALL_FROM_SCRATCH.md`, `Isaac-GR00T/KIST_USAGE.md`
- 자매 repo (ART gripper 자체 셋업): `Hyundai_motors_Gripper/README.md`
