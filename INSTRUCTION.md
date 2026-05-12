# Polymetis_Franka_Teleop — 운영 매뉴얼 (2026-05-12 최종본)

> 본 파일은 `before_instruction_Polymetis_Franka_Teleop.md`의 최종 개정판입니다. 두 머신(**pro4000** + **NUC**)에서 매일 보면서 그대로 따라가도록 작성. 초기 install은 본 문서가 아닌 `docs/install_from_scratch.md`에 위임합니다.
>
> 본문의 명령어는 모두 **현재 main 브랜치 (pro4000:`6f6190c`)** 의 스크립트 인자와 일치합니다.

---

## 0. 머신/역할 한 줄 정리

| 머신 | 역할 | SSH alias |
|---|---|---|
| **NUC (`192.168.1.12`)** | PREEMPT_RT polymetis arm server `:50051` (libfranka 1 kHz 루프). 외부 접근은 LAN 전용. | `kist@192.168.1.12` (pw `kist`) |
| **pro4000 (`kist-eval`)** | Teleop 클라이언트 / cv2 viewer / ZED·RealSense 캡쳐 / ART gripper `:50053` (systemd) / Vive `vrserver`+`vive_input :12345` / 데이터 변환. | `ssh kist_pro4000` |
| **kist_a6000_ss (RTX 6000 Ada ×10)** | GR00T / DP fine-tune 전용. 학습만 함 (추론 X). 데이터는 `rsync`로 받아옴. | `ssh kist_a6000_ss` |
| ~~3090 host~~ | KIST에서 더 이상 사용 안 함. 코드는 호환성 위해 남아있지만 운영에선 무시. | — |

---

## 1. 초기 환경 세팅 (처음 1회만)

### 1.1 하드웨어 연결 체크리스트

| 항목 | 확인 방법 | 정상 시 결과 |
|---|---|---|
| Franka arm 전원 + 외부 e-stop 해제 | 베이스 LED | 흰색/파랑 |
| Franka Desk (`https://172.16.0.2/desk/`) | 브라우저 (NUC가 직접 또는 NUC LAN 안에서 접속) | 조인트 unlock + **FCI Activate** |
| NUC ↔ pro4000 LAN | `ping 192.168.1.12` from pro4000 | < 1 ms |
| ZED 2i + ZED Mini USB | `lsusb \| grep -i stereolabs` on pro4000 | **4줄** (각 카메라 UVC + HID, 총 4개) |
| ART gripper EtherCAT | pro4000: `systemctl is-active ethercat art-gripper-daemon` | `active active` |
| Vive Link Box + base station | pro4000 `lsusb \| grep -ci 'htc\|valve'` | ≥ 1 |

USB 카메라가 4개 미만이면 케이블 재연결 (USB-3 와 USB-2 양쪽 모두). 자세한 절차는 `docs/hardware_setup.md`.

### 1.2 소프트웨어 설치

> ⚠️ 본 매뉴얼은 **이미 설치된 환경에서의 사용법**만 다룹니다. 처음 설치하는 경우:
>
> - **pro4000**: `docs/install_from_scratch.md` Phase A→J (conda env `groot-client`, ART gripper deps, vive_input 빌드 등)
> - **NUC**: `install/nuc/` 안의 RT 커널 + polymetis 빌드 스크립트
> - **kist_a6000_ss (학습 머신)**: 본 레포의 `INSTALL_FROM_SCRATCH.md` + `scripts/kist/install_groot_env.sh` (Stage 1→3: torch 2.7.1+cu128 → flash-attn → 29 deps + `pip install -e . --no-deps`)
>
> **HF / Git 로그인**: GR00T-N1.7-DROID 체크포인트는 비-gated 공개 모델이므로 로그인 불필요. Git push에는 PAT 필요 (`git config`로 저장).

---

## 2. 일일 운영 — 데이터 수집 시작 전 (5단계)

> 한 번이라도 데모를 띄웠다면 항상 **§2.1 cleanup_pipeline** 부터 시작. 깨끗한 부팅 직후라도 무해 (idempotent).

### 2.1 cleanup_pipeline.sh — 전체 파이프라인 자동 진단/복구

```bash
# pro4000:
bash ~/Polymetis_Franka_Teleop/bin/cleanup_pipeline.sh
```

이 한 줄이 자동으로 처리하는 것 (9 sections):

| # | 영역 | 동작 |
|---|---|---|
| 0 | active demo 체크 | 진행 중이면 중단 (`--force`로 강제) |
| 1 | pro4000 잔존 프로세스 | `demo_franka_vive`, `cv2_viewer`, `single_zed`, `vive_teleop_process`, `franka_interpolation_controller` 등을 SIGTERM → SIGKILL |
| 2 | `/dev/shm` 누수 | `wnsm_*`, `psm_*`, `sem.mp-*`, `u${UID}-Shm_*` 제거 (UMI SharedMemoryRingBuffer 잔재). SteamVR / Valve 세그먼트는 보존 |
| 3 | ZED SDK 락 | `/tmp/.zed*`, `/tmp/zed_*`, `/dev/shm/zed_*`, `/var/lib/zed/.cam_lock_*` 제거 |
| 4 | cv2_viewer 임시 JPG | `/tmp/teleop_vis*.jpg`, `/tmp/franka_vive_*.jpg` 제거 |
| 5 | ART gripper + ethercat | systemd inactive면 `sudo systemctl restart` 후 `:50053` 재확인 |
| 6 | NUC polymetis `:50051` | down이면 → NUC orphan kill (`run_server`, `launch_robot.py`) → `nohup sudo bash /usr/local/sbin/start_franka_arm.sh &` (detached) → 30 s까지 poll |
| 7 | 카메라 USB 가시성 | `lsusb` + `rs-enumerate-devices` + `/dev/video*` holder 리포트 |
| 8 | Vive `:12345` | `vrserver` alive 여부만 보고 (수동 액션 안내) |
| 9 | 요약 | 모두 OK면 `Pipeline READY` (exit 0), 아니면 exit 1 |

**Flags:**
- `--no-nuc`: NUC SSH 섹션 통째로 skip (LAN 끊겼을 때)
- `--no-arm-restart`: NUC `:50051` 자동 재시작 skip (down이라도 진단만)
- `--no-gripper-restart`: art-gripper-daemon 자동 재시작 skip
- `--force`: active session 있어도 강제 진행
- `--quiet`: section 배너 숨김

**예상 시간**: 정상 상태 ~2 s, NUC arm 재시작 필요시 ~30-40 s.

> 이전 버전 `cleanup_pro4000.sh` (commit f478d7d)는 service 재시작 없이 local cleanup만. 보수적 진단용으로 남겨둠.

### 2.2 Franka Desk (브라우저)

`https://172.16.0.2/desk/` 접속 →
1. 조인트 unlock (안전망 해제 위 자물쇠 클릭)
2. FCI **Activate**
3. 외부 e-stop 해제 확인

> NUC 또는 같은 LAN 내 PC에서 접속 가능. cleanup_pipeline이 NUC arm을 띄우려면 FCI가 먼저 Activate 돼있어야 함.

### 2.3 NUC arm 서버 — cleanup_pipeline이 이미 처리

`§2.1`에서 `:50051` UP이면 추가 작업 없음. 만약 cleanup_pipeline이 `polymetis :50051 still down`을 출력했다면:

```bash
# NUC 모니터에서 직접 (또는 ssh kist@192.168.1.12):
sudo bash /usr/local/sbin/start_franka_arm.sh
# 다음 두 줄이 나오면 OK:
#   [INFO] Connected.
#   [arm pinner] cores 6,7 핀 적용 완료
# → 이 터미널 그대로 둠 (Ctrl+C ❌)
```

### 2.4 Vive stack bring-up

```bash
# pro4000:
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start   # 또는 인자 생략
```

3 단계가 자동으로:
1. `/dev/hidraw*` 권한 (chmod +rw, sudo pw `kist`)
2. `vrserver --keepalive` (StreamVR GUI 없이 headless)
3. `vive_input` 바이너리 (TCP :12345 + UDP :12346)

마지막에 status가 출력됨. Vive 컨트롤러는 손으로 한 번 흔들어 sleep을 풀어줘야 `vive_input` Summary에 `Controllers=1` 이상이 나옴.

**보조 명령:**
```bash
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh status   # 현재 상태
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh stop     # 전체 종료
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh restart  # 재시작
```

### 2.5 preflight_full.sh — 마지막 종합 진단

```bash
# pro4000:
bash ~/Polymetis_Franka_Teleop/bin/preflight_full.sh
```

6 phase 자동 점검 + AUTO_FIX:
1. 잔존 Python 프로세스
2. ART daemon TCP OP_PING (단순 systemd 상태가 아니라 실제 응답까지 확인)
3. ZED 4/4 USB + pyzed가 2개를 잡는지
4. Vive `vrserver` + `vive_input` Summary에 controllers ≥ 1
5. NUC `:50051` TCP 연결
6. `ulimit -r ≥ 50` + `franka-client-rt-tune` service

마지막에 `All checks passed -- ready to launch demo.` 가 나오면 데이터 수집 시작 가능.

**환경 변수**:
- `AUTO_FIX=no` → 진단만, 자동 복구 안 함
- `NUC_USER/HOST/PWD` → NUC SSH 정보 override

---

## 3. Teleop & 데이터 수집

### 3.1 권장: GR00T-DROID 수집 모드 (15 Hz)

```bash
# pro4000 — 새 터미널 2:
bash ~/Polymetis_Franka_Teleop/bin/start_teleop_groot_droid_ft.sh \
     ~/Polymetis_Franka_Teleop/data/session_$(date +%Y%m%d_%H%M%S) \
     --task "Pick up the red block and place it in the bowl"
```

`start_teleop_groot_droid_ft.sh`가 내부적으로 고정해주는 것:
- `--frequency 15`  (DROID 15 Hz cadence)
- `--data_format groot`  (recorder가 `meta/data_format='groot'`로 기록 + DROID ready-pose 적용)
- `--camera_backend zed --gripper_backend art`
- `--camera_serials 33538770 11667817` (exterior + wrist)
- `--camera_resolution 672x376 --camera_fps 60`  (VGA, native 60fps)
- `--teleop_frequency 100`
- conda env `groot-client` 자동 activate

뒤에 붙는 `--task "..."` 등은 그대로 `demo_franka_vive.py`로 전달됨.

### 3.2 (대안) 자동 preflight + 로그 tail 한 번에 — `run_test_session_groot_ft.sh`

```bash
bash ~/Polymetis_Franka_Teleop/bin/run_test_session_groot_ft.sh \
     --task "Pick up the red block and place it in the bowl"
```
- `data/groot_ft_<TS>/`로 자동 출력
- preflight_full.sh 자동 실행 후 데모 setsid 분리
- `tail -f /tmp/teleop_groot_ft.log`까지 자동
- 데모가 끝나면 tail도 자동 종료, 잔존 프로세스 경고

### 3.3 (대안) Diffusion-Policy / UMI 수집 모드 (10 Hz)

```bash
bash ~/Polymetis_Franka_Teleop/bin/start_teleop.sh \
     ~/Polymetis_Franka_Teleop/data/session_$(date +%Y%m%d_%H%M%S)
```
- `--frequency 10 --data_format umi` 고정
- DROID와 별개의 ready-pose

### 3.4 demo_franka_vive.py 주요 인자 (필요 시 직접 호출)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--output / -o` | (필수) | 출력 디렉토리 |
| `--frequency` | 10 | 메인 loop Hz. GR00T-DROID=15, DP=10 |
| `--teleop_frequency` | 100 | Vive → robot 보간 frequency |
| `--data_format` | `groot` | `groot` / `umi` / `diffusion` — ready-pose 선택 |
| `--camera_backend` | `zed` | `zed` 또는 `realsense` |
| `--gripper_backend` | `art` | `art` (Hyundai EtherCAT) / `franka` (Franka Hand) |
| `--camera_serials / -c` | (반복) | 사용할 카메라 serial. 첫 번째가 ext1, 두 번째가 wrist로 매핑 |
| `--camera_resolution` | `1280x720` | ZED는 `672x376` (VGA, 60fps) 권장 |
| `--camera_fps` | 60 | ZED HD720 최대 60 |
| `--task` | None | episode-level 자연어 instruction. **GR00T 변환을 하려면 이 인자 필수** (없으면 placeholder) |
| `--task_id` | None | multi-task scene에서 task 식별자 |
| `--auto_home / --no_auto_home` | `--auto_home` | 시작 시 ready-pose로 자동 이동 |
| `--skip_preflight` | off | preflight_full.sh 자동 호출 skip |
| `--vis / --no_vis` | `--vis` | cv2 visualization 창 |
| `--verbose / -v` | off | 상세 로그 |
| `--tcp_offset` | auto | Franka Hand=0.1034m, ART=0.216m 자동 |
| `--tuning_preset` | `normal` | velocity/Kx 프리셋. `docs/teleop_tuning.md` 참고 |
| `--pos_scale / --rot_scale` | preset | Vive→robot 스케일 직접 지정 |

전체 list는 `python scripts_real/demo_franka_vive.py --help`.

### 3.5 cv2 viewer 키바인딩 (데이터 수집 흐름)

| 키 | 동작 |
|---|---|
| (대기 화면) **`[startup] Ready pose reached`** | 시작 준비 완료 |
| `c` | 다음 episode 시작 (recorder ON) |
| `s` | 현재 episode 종료 (정상 저장) |
| `Backspace` + `y` | 현재 episode **drop** (저장 안 함) |
| `d` | drop_arm 토글 (compliant 상태) |
| Vive Trigger | gripper 토글 |
| Vive Grip (홀드) | clutch 활성 — robot이 컨트롤러 따라옴 |
| `q` 두 번 | 깔끔하게 데모 종료 |

비상 종료가 필요하면 다른 터미널에서:
```bash
pkill -INT -f demo_franka_vive
```

### 3.6 모니터링 (별도 터미널, optional)

```bash
bash ~/Polymetis_Franka_Teleop/bin/monitor_session.sh
# 5초마다: ZED fps / Franka loop overrun / ArtGripper overrun
#         / Recovery / IK STUCK / NUC libfranka success rate
```

`run_test_session_groot_ft.sh`의 기본 로그(`/tmp/teleop_groot_ft.log`)를 watch. 다른 로그 위치 쓸 땐 인자로 전달.

### 3.7 1 세션 검증 체크리스트 (수집 직후)

```bash
SESSION=~/Polymetis_Franka_Teleop/data/session_<TS>
ls $SESSION/replay_buffer.zarr $SESSION/videos/0/0.mp4 $SESSION/videos/0/1.mp4
```

| 확인 항목 | 목표 |
|---|---|
| `FrankaInterp overruns` | **0** |
| `ArtGripper overruns` | **0** |
| `Recovery #N` count | 0–1 |
| `IK STUCK` count | 0 |
| `replay_buffer.zarr` 생성 | ✓ |
| episode당 `videos/<ep>/0.mp4`, `1.mp4` | ✓ (ext1 + wrist) |
| `meta.json` `data_format` | `groot` (또는 의도한 값) |
| `meta.json` `language_instruction` | `--task`에 넣은 값 (placeholder X) |

---

## 4. 데이터 변환 — 3 컨버터

> 모든 컨버터는 **conda env `groot-client`** 에서 실행. raw 세션 1개에 대해 같은 입력으로 3개 다른 출력을 만들 수 있음.

### 4.1 GR00T-DROID-FT 변환 (LeRobot v2.1, 17D, 15 Hz, relative actions)

```bash
SESSION=~/Polymetis_Franka_Teleop/data/session_20260512_HHMMSS

python ~/Polymetis_Franka_Teleop/scripts_real/convert_to_gr00t_droid.py \
    --input-session  $SESSION \
    --output-dataset ${SESSION}_gr00t \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
# 마지막: "[convert] OK -- N episodes, M frames"
#         embodiment-tag suggestion: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--input-session` | (필수) | raw 세션 디렉토리 (`replay_buffer.zarr` 있어야 함) |
| `--output-dataset` | (필수) | LeRobot v2.1 출력 디렉토리 |
| `--embodiment-tag` | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | GR00T embodiment slug |
| `--allow-placeholder` | off | `--task` 없이 수집한 세션(`<task placeholder>`)도 변환 허용 (학습은 무의미) |
| `--max-episodes` | -1 | N개만 변환 (디버깅) |
| `--verbose` | off | 진행 상세 |

출력 구조:
```
${SESSION}_gr00t/
├── meta/{info,episodes,tasks,modality,stats}.json + episodes.jsonl + tasks.jsonl
├── data/chunk-000/episode_NNNNNN.parquet
└── videos/chunk-000/observation.images.{exterior_1_left,exterior_2_left,wrist_left}/episode_NNNNNN.mp4
```

> 비디오는 **심볼릭 링크** (raw 세션의 mp4 그대로). raw 세션을 지우면 안 됨.

### 4.2 Diffusion Policy 변환 (robomimic HDF5)

```bash
python ~/Polymetis_Franka_Teleop/scripts_real/convert_to_diffusion_policy.py \
    --input-session  $SESSION \
    --output-dataset ${SESSION}_dp
# 마지막: "[convert] OK -- N episodes, M frames, action 8D (joint+gripper)"
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--input-session` | (필수) | raw 세션 |
| `--output-dataset` | (필수) | DP HDF5 출력 |
| `--max-episodes` | -1 | N개만 |
| `--verbose` | off | 진행 상세 |

출력: `${SESSION}_dp/dataset.hdf5` (robomimic-style: `demo_<i>/{obs,actions,...}`).

### 4.3 ACT / 일반 LeRobot 변환 (8D joint+gripper)

```bash
python ~/Polymetis_Franka_Teleop/scripts_real/convert_to_lerobot.py \
    -i $SESSION \
    -o ${SESSION}_act \
    --task "Pick up the red block and place it in the bowl" \
    --state_format joint --gripper_repr normalized
# 마지막: "[convert] OK -- N episodes, M frames, 8D state/action"
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--input / -i` | (필수) | raw 세션 |
| `--output / -o` | (필수) | LeRobot v2 출력 |
| `--task / -t` | None | 세션 `meta.json`의 instruction 덮어쓰기 |
| `--state_format` | `joint` | `joint`(7+1=8D) / `ee`(7+1) / `ee_quat`(8) |
| `--gripper_repr` | `normalized` | `normalized`(0~1) / `raw_m`(미터) / `raw_ticks` |
| `--gripper_max_width` | auto | 정규화 분모 (m). 미지정 시 세션 메타에서 자동 |
| `--fps` | auto | 출력 fps. 미지정 시 세션 fps 그대로 |
| `--video_keys` | (반복) | 출력에 포함할 카메라 키 (기본: 전부) |
| `--copy_videos / --symlink_videos` | `--copy_videos` | mp4 복사 vs 심링크 |

### 4.4 변환 검증 — round_trip_test (필수, push 전)

```bash
python ~/Polymetis_Franka_Teleop/scripts_real/tools/round_trip_test.py \
    --converter both \
    --input-session $SESSION
# 모든 셀이 "0.0e+00 OK" → bitwise match, exit 0
```

| 인자 | 설명 |
|---|---|
| `--converter` | `gr00t` / `dp` / `both` |
| `--input-session` | raw 세션 |
| `--threshold` | float (default 1e-6). 이보다 큰 차이가 있으면 FAIL |

원리: raw 세션을 직접 읽어 컨버터의 공식으로 다시 계산 → 컨버터가 실제 쓴 값과 element-wise 비교. 회귀 검출용. CI에서도 gating으로 사용.

### 4.5 데이터셋 헬스 체크

```bash
python ~/Polymetis_Franka_Teleop/scripts_real/tools/check_dataset_health.py \
    --dataset ${SESSION}_gr00t
# stats sanity, video FFmpeg, episode parquet schema, modality cross-check
```

---

## 5. 학습 (kist_a6000_ss)

### 5.1 데이터 전송

```bash
# pro4000 → 로컬 (또는 직접 pro4000 → a6000_ss):
rsync -avh --progress \
      ~/Polymetis_Franka_Teleop/data/session_<TS>_gr00t \
      knykist@local-or-jump:~/datasets/franka/
# 그 다음 a6000_ss로:
rsync -avh --progress \
      ~/datasets/franka/session_<TS>_gr00t \
      kist_a6000_ss:/data2/seungsub/datasets/
```

대용량은 항상 `/data2/seungsub/` (5.1 TB, persistent mount) 밑에. `/root` overlay는 container REPLACE 시 휘발성.

### 5.2 GR00T fine-tune (200-step smoke)

```bash
# kist_a6000_ss:
source /root/miniconda3/etc/profile.d/conda.sh
conda activate groot-finetune
cd /root/projects/Isaac-GR00T

export CUDA_VISIBLE_DEVICES=9         # 비어있는 GPU 1장
export NUM_GPUS=1
export MAX_STEPS=200
export SAVE_STEPS=100
export GLOBAL_BATCH_SIZE=4
export DATALOADER_NUM_WORKERS=2
export USE_WANDB=0
export SHARD_SIZE=512
export NUM_SHARDS_PER_EPOCH=10
export EPISODE_SAMPLING_RATE=1.0
export PYTHONHTTPSVERIFY=0            # corp SSL chain bypass
export HF_HUB_DISABLE_TELEMETRY=1

bash examples/finetune.sh \
    --base-model-path /root/checkpoints/GR00T-N1.7-DROID \
    --dataset-path    /data2/seungsub/datasets/session_<TS>_gr00t \
    --embodiment-tag  OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --output-dir      /data2/seungsub/experiments/franka_smoke_groot \
    --save-only-model 2>&1 | tee /tmp/groot_smoke.log
```

핵심 env 변수:
- `MAX_STEPS` — 200=smoke, 본 학습은 보통 10k+
- `SAVE_STEPS` — 체크포인트 빈도
- `GLOBAL_BATCH_SIZE` — 단일 GPU 4가 안전한 출발점 (RTX 6000 Ada 45 GB 기준)
- `DATALOADER_NUM_WORKERS` — 보통 2-4
- `EPISODE_SAMPLING_RATE` — < 1.0이면 매 epoch 일부 에피만 샘플
- `CUDA_VISIBLE_DEVICES` — `nvidia-smi`로 free GPU 먼저 확인 후 지정

### 5.3 Diffusion-Policy fine-tune (smoke 2 epoch)

```bash
# kist_a6000_ss, dp-finetune env:
conda activate dp-finetune
cd /root/diffusion_policy
python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task.dataset_path=/data2/seungsub/datasets/session_<TS>_dp/dataset.hdf5 \
    training.num_epochs=2 \
    training.device=cuda:9
```

> DP env에 `pytorch3d` 필요 (loader의 `RotationTransformer` axis_angle→rot6d 때문). 미설치 시 `pip install fvcore iopath` 먼저 → `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`.

### 5.4 GR00T inference server (배포 시)

```bash
# pro4000 또는 a6000_ss:
conda activate groot
python gr00t/eval/run_gr00t_server.py \
    --model-path /data2/seungsub/experiments/franka_smoke_groot/checkpoint-100 \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

---

## 6. Eval (학습된 정책으로 실제 로봇 제어)

### 6.1 Diffusion Policy / ACT eval (`.ckpt` 직접 load)

```bash
# pro4000:
bash ~/Polymetis_Franka_Teleop/bin/start_eval.sh \
     <path/to/checkpoint.ckpt> \
     ~/Polymetis_Franka_Teleop/data/eval_$(date +%Y%m%d_%H%M%S)
```

내부적으로 `scripts_real/eval_franka_policy.py`를 호출. 주요 인자 (필요 시 직접):

| 인자 | 기본 | 설명 |
|---|---|---|
| `--input / -i` | (필수) | `.ckpt` 또는 디렉토리 |
| `--output / -o` | (필수) | eval 녹화 저장 디렉토리 |
| `--frequency / -f` | 10 | inference Hz |
| `--steps_per_inference / -si` | 6 | DP receding-horizon 길이 |
| `--num_inference_steps` | 16 | diffusion denoise step |
| `--max_duration / -md` | 6000 | 최대 episode 길이 (초) |
| `--record_episode` | off | 매 episode 영상 + zarr 녹화 |
| `--auto_start` | off | Enter 안 누르고 자동 시작 |

### 6.2 GR00T policy eval (server 기반)

GR00T는 별도 inference server 띄우고 (위 §5.4) DROID 클라이언트가 `:5000` (또는 지정 포트)에 step 단위 inference 요청. `examples/DROID/main_gr00t.py` 참고.

---

## 7. 종료 sequence

```bash
# pro4000 데모 터미널: q 두 번 (자연 종료)
# 혹시 잔존이 있으면:
bash ~/Polymetis_Franka_Teleop/bin/cleanup_pipeline.sh --no-arm-restart --no-gripper-restart

# Vive stack 내림 (다음 세션이 한참 후일 때):
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh stop

# NUC arm 그대로 두기 — 다음 세션을 위해 살려둠
# 끄려면 NUC 터미널에서 Ctrl+C (start_franka_arm.sh)
```

---

## 8. 문제 해결 — 증상 → 처치 카탈로그

> 대부분의 증상은 **`cleanup_pipeline.sh` 한 번**으로 해결됨. 그래도 안 되는 케이스만 아래 참조.

### 8.1 카메라 / ZED

| 증상 | 원인 | 조치 |
|---|---|---|
| `ZED 2i missing` / `device not detected` | USB 케이블 헐겁거나 power 부족 | (1) `cleanup_pipeline.sh` → `lsusb` 4개 확인. 부족하면 (2) USB-3 케이블 재연결 (특히 ZED 2i 메인 cable). (3) 다른 USB 3.0 포트 (mini-PC의 일부 포트는 power throttle) |
| `pyzed sees only 1/2 cameras` | 잔존 Python 자식이 `/dev/video*` 잡고 있음 | `cleanup_pipeline.sh`로 해결 안 되면 `lsof /dev/video0 /dev/video1` → 직접 kill |
| `address already in use (SHM)` | 이전 demo 비정상 종료로 SHM 누수 (UMI SharedMemoryRingBuffer) | `cleanup_pipeline.sh` (sect 2가 자동 제거. 1차 검증에서 401개 잔재 확인됨) |
| `lock file exists /tmp/.zed*` | ZED SDK 비정상 종료 | `cleanup_pipeline.sh` sect 3 |
| `60 fps 안 나오고 30 fps 안팎` | 해상도 `1280x720` 잡고 60fps 요청 | ZED HD720은 60fps 가능하지만 USB-3 bandwidth tight. `672x376 @ 60fps`(VGA)로 떨어뜨리기 — `start_teleop_groot_droid_ft.sh`는 이미 VGA 기본 |

### 8.2 ART gripper / EtherCAT

| 증상 | 원인 | 조치 |
|---|---|---|
| `:50053 connection refused` | art-gripper-daemon 죽음 | `cleanup_pipeline.sh` sect 5 자동 재시작 |
| `ART daemon hung / TCP unresponsive` | mutex hold by zombie client | `preflight_full.sh`가 OP_PING으로 detect + `restart_gripper.sh` 자동. 수동: `sudo bash ~/Hyundai_motors_Gripper/scripts/restart_gripper.sh` |
| EtherCAT slave 0 not detected | `/opt/etherlab/bin/ethercat slaves` 비어있음 | (1) `systemctl restart ethercat` (2) 그래도 안 되면 24V 그리퍼 전원 cycle |
| `Overruns > 0 on ArtGripperController` | CPU 부하 / RT priority 부족 | `ulimit -r`가 50 이상인지 확인 (preflight sect 6). 부족하면 데스크탑 로그아웃 → 재로그인. `systemctl is-active franka-client-rt-tune` 확인 |

### 8.3 NUC / Franka arm

| 증상 | 원인 | 조치 |
|---|---|---|
| `:50051 unreachable` | polymetis 죽음 | `cleanup_pipeline.sh` sect 6 자동 재시작 (orphan kill + nohup start_franka_arm.sh). 그래도 안 되면 NUC SSH 직접 |
| `Connecting to Franka Emika ...` 에서 멈춤 | FCI 비활성 또는 e-stop | Franka Desk 다시 → unlock + FCI Activate |
| `Recovery #N` 자주 발생 | velocity clamp 너무 작거나 collision 빈번 | `--tuning_preset gentle` / `--max_pos_velocity` 낮추기. `docs/teleop_tuning.md` |
| `IK STUCK` count > 0 | 워크스페이스 한계 도달 | Vive grip을 잠시 떼고 robot이 자연스러운 자세로 복귀하게 둠 |
| `motion aborted by reflex` (NUC log) | collision_behavior threshold 초과 | 토크가 40 Nm 이상이면 발생. 워크스페이스/물체 확인 |

### 8.4 Vive / SteamVR

| 증상 | 원인 | 조치 |
|---|---|---|
| `vive_input :12345 NOT reachable` | vrserver 죽음 | `bash bin/start_vive_stack.sh restart` |
| `Controllers=0` in vive_input log | 컨트롤러 sleep 또는 line-of-sight 안 좋음 | (1) 컨트롤러 흔들기 (2) base station 전원 + LED 확인 (3) lighthouse 라인 시야 확보 (4) `start_vive_stack.sh restart` |
| 컨트롤러는 잡히는데 grip이 robot에 안 먹힘 | clutch off | Vive grip 버튼을 **누르고 있어야** clutch 활성 — 토글이 아님 |
| `vrserver: STEAMVR not found` | StreamVR 경로 다름 | `STEAMVR=<경로> bash start_vive_stack.sh start`로 override |

### 8.5 데이터 변환 / 학습

| 증상 | 원인 | 조치 |
|---|---|---|
| 변환 시 `task placeholder` warning | 수집 시 `--task` 안 넣음 | (1) `--task "..."` 붙여 재수집이 최선. (2) 강제 변환은 `--allow-placeholder` 이지만 학습은 무의미 |
| `convert_to_gr00t_droid` 가 episode 0개 | `replay_buffer.zarr`에 0 epi 또는 schema 불일치 | `python tools/check_dataset_health.py --dataset <raw>` 로 진단 |
| `round_trip_test FAIL` | converter 공식 ↔ tool 공식 divergence (회귀) | 신규 컨버터 코드를 push 안 했는지 확인. `git log scripts_real/convert_to_*.py` |
| GR00T fine-tune `flash-attn ImportError` | env가 잘못 / torch ABI 불일치 | a6000_ss는 cu128 + torch 2.7.1 고정. flash-attn은 GitHub prebuilt wheel: `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` |
| `SSL CERTIFICATE_VERIFY_FAILED` (HF download) | corp proxy 자체-서명 chain | `export PYTHONHTTPSVERIFY=0` + `~/.config/pip/pip.conf` trusted-host (smoke 스크립트에 이미 반영) |
| `wheel_stub BackendUnavailable for tensorrt-cu12` | TRT가 pip.conf trusted-host 우회 | 학습엔 TRT 불필요. `scripts/kist/install_groot_env.sh`처럼 tensorrt-cu12 빼고 install |

### 8.6 RT / 성능

| 증상 | 원인 | 조치 |
|---|---|---|
| `ulimit -r = 0` | PAM RT limit 미적용 (보통 데스크탑 로그인 1회 후 SSH session) | 데스크탑 로그아웃 → 재로그인. 또는 `ssh kist@localhost`로 새 session |
| `franka-client-rt-tune inactive` | RT IRQ pinning systemd 안 떠있음 | `sudo systemctl enable --now franka-client-rt-tune` |
| pro4000 load avg > 50 | 좀비 Python / cv2 viewer 다수 | `cleanup_pipeline.sh` → `top` 으로 잔재 확인 |

---

## 9. 빠른 참조 — 한 세션 한 화면

```bash
# === pro4000 (메인 터미널) ===
bash ~/Polymetis_Franka_Teleop/bin/cleanup_pipeline.sh
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh start
bash ~/Polymetis_Franka_Teleop/bin/preflight_full.sh

# === pro4000 (데모 터미널) ===
bash ~/Polymetis_Franka_Teleop/bin/start_teleop_groot_droid_ft.sh \
     ~/Polymetis_Franka_Teleop/data/session_$(date +%Y%m%d_%H%M%S) \
     --task "Pick up the red block and place it in the bowl"

# === pro4000 (모니터 터미널, 선택) ===
bash ~/Polymetis_Franka_Teleop/bin/monitor_session.sh

# === 수집 후 변환 (groot-client env) ===
SESSION=~/Polymetis_Franka_Teleop/data/session_<TS>
python ~/Polymetis_Franka_Teleop/scripts_real/convert_to_gr00t_droid.py \
    --input-session $SESSION --output-dataset ${SESSION}_gr00t
python ~/Polymetis_Franka_Teleop/scripts_real/tools/round_trip_test.py \
    --converter gr00t --input-session $SESSION

# === 학습 (kist_a6000_ss) ===
ssh kist_a6000_ss
# (env activate + bash examples/finetune.sh ... 위 §5.2)

# === 종료 (다음 세션이 한참 후일 때) ===
bash ~/Polymetis_Franka_Teleop/bin/start_vive_stack.sh stop
```

---

## 10. 변경 이력 (vs `before_instruction_*`)

| 변경 | 이유 |
|---|---|
| `cleanup_all.sh` → `cleanup_pipeline.sh` | 전체 파이프라인(pro4000+NUC+gripper+camera+Vive) 자동 진단/복구. orphan kill을 :50051 down 일 때만 실행, NUC 재시작은 nohup detach |
| `convert_to_gr00t_lerobot.py` → `convert_to_gr00t_droid.py` | LeRobot v2.1 + DROID embodiment 정식 schema (Phase 2-1 final). 컬럼명 / parquet layout / modality.json 모두 GR00T checkpoint와 일치 |
| `convert_to_lerobot.py` 추가 | ACT / 일반 LeRobot용 8D 컨버터 (Phase 2-3) |
| `convert_franka_vive_to_umi_format.py` → `convert_to_diffusion_policy.py` | robomimic HDF5 정식 (Phase 2-4). UMI zarr.zip은 deprecate |
| `tools/round_trip_test.py` 추가 | 두 컨버터 회귀 일괄 검증, CI gating용 |
| `tools/check_dataset_health.py` 추가 | stats / video / schema 헬스 자동 검증 |
| 3090 host 흔적 제거 | KIST 운영에서 더 이상 사용 안 함. 학습은 kist_a6000_ss |
| `--task` 인자 강조 | language_instruction이 GR00T 변환에 필수 — 수집 시점에 안 넣으면 학습이 무의미 |

---

운영 중 변경되는 사항(스크립트 인자 추가/이름 변경 등)은 본 파일을 수정해 같이 커밋해주세요. 가장 중요한 진실의 원천은 항상 `--help`와 git log입니다.
