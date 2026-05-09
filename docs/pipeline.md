# 개선된 Franka 로봇 워크스페이스 — 전체 구조 및 설계 분석

> 분석 대상 전체 파일:
> `umi/real_world/`, `umi/shared_memory/`, `umi/common/`,
> `scripts_real/demo_franka_vive.py`, `scripts_real/eval_franka_policy.py`,
> `umi/real_world/franka_vive_env.py`, `umi/real_world/franka_policy_env.py`,
> `umi/real_world/real_inference_util.py`, `umi/real_world/vive_teleop_process.py`,
> `umi/real_world/franka_interpolation_controller.py`,
> `umi/real_world/franka_gripper_controller.py`,
> `umi/real_world/multi_realsense.py`, `umi/real_world/single_realsense.py`,
> `umi/real_world/vive_shared_memory.py`, `umi/real_world/video_recorder.py`,
> `umi/real_world/image_transform.py`, `umi/common/latency_util.py`,
> `umi/common/precise_sleep.py`, `umi/common/timestamp_accumulator.py`

---

## 1. 시스템 전체 흐름 (2단계 파이프라인)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 데이터 수집 (Teleoperation)                         │
│                                                             │
│  python scripts_real/demo_franka_vive.py                    │
│       -o ./data/oneleg_insert -v --show_all_cameras         │
│                                                             │
│  HTC Vive Controller (100Hz raw)                            │
│  → ViveSharedMemory (200Hz 수신)                            │
│  → ViveTeleopProcess (100Hz 계산, 별도 프로세스)             │
│  → FrankaInterpolationController (200Hz, teleop_mode=True)  │
│  → Main Loop (10Hz, 데이터 기록)                             │
│  → Zarr Replay Buffer + H264 MP4 저장                        │
└─────────────────────────────────────────────────────────────┘
              ↓ (학습 후)
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: 정책 실행 (Policy Deployment)                       │
│                                                             │
│  python scripts_real/eval_franka_policy.py                  │
│       -i checkpoint.ckpt -o data/eval                       │
│                                                             │
│  FrankaPolicyEnv (10Hz obs, 타임스탬프 정렬)                 │
│  → Policy 추론 (DDIM 16스텝, GPU)                            │
│  → exec_actions() (타임스탬프 기반 스케줄)                   │
│  → FrankaInterpolationController (200Hz 보간, normal mode)   │
│  → 로봇 실행                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 전체 컴포넌트 구조 및 주파수

```
SharedMemoryManager (프로세스 간 공유 메모리 통합 관리자)
│
├── ViveSharedMemory          [200Hz]  Vive 컨트롤러 raw 입력 수신 (TCP 소켓)
│       └── ring_buffer  ──────────────────────► ViveTeleopProcess
│
├── ViveTeleopProcess         [100Hz]  텔레오프 계산 전담 (별도 프로세스)
│       ├── action_ring_buffer  ─────────────► Main Process (10Hz 기록)
│       ├── robot_ring_buffer   ─────────────► FrankaInterpolationController
│       └── gripper_ring_buffer ─────────────► FrankaGripperController
│
├── FrankaInterpolationController [200Hz] 로봇 팔 제어 (별도 프로세스)
│       ├── [teleop_mode=True]  → robot_ring_buffer 직접 읽기 (ViveTeleop 바이패스)
│       └── [teleop_mode=False] → SharedMemoryQueue (schedule_waypoint 큐)
│               └── PoseTrajectoryInterpolator → 부드러운 궤적 생성
│
├── FrankaGripperController   [30Hz]   그리퍼 제어 (별도 프로세스, 이산 v2.0)
│       ├── [teleop_mode=True]  → gripper_ring_buffer 직접 읽기
│       └── [teleop_mode=False] → SharedMemoryQueue (schedule_waypoint 큐)
│               └── 상태 전환 시에만 명령 발송 (중복 방지)
│
└── MultiRealsense            [60fps]  RealSense 카메라 (별도 프로세스, 카메라 당 1개)
        ├── ImageTransform    (obs용: resize + BGR→RGB + float32, Picklable)
        ├── VideoRecorder     (H264 MP4, CRF=18, av/PyAV 기반 실시간 스트리밍)
        └── SharedMemoryRingBuffer → Main Process (get_obs에서 읽기)
```

### Polymetis 서버 (NUC)

```
NUC (192.168.1.12)
├── polymetis arm 서버 :50051 (gRPC)
│    └── update_desired_ee_pose() / update_desired_joint_positions()
│        ← FrankaInterpolationController (direct gRPC client)에서 호출
└── polymetis Franka Hand 서버 :4242 (zerorpc — --gripper_backend franka 인 경우에만 띄움)
     └── gripper.goto() / grasp() ← FrankaGripperController에서 호출
```

---

## 3. 공유 메모리 인프라 (`umi/shared_memory/`)

### 3.1 SharedMemoryRingBuffer — Lock-free FILO 순환 버퍼

```
설계 원칙: 항상 최신 데이터 (FILO: First In, Last Out)
           쓰기와 읽기가 동시에 이루어져도 Lock 불필요

버퍼 크기 자동 계산:
  buffer_size = ceil(put_freq * get_time_budget * safety_margin) + get_max_k
  예) 100Hz 쓰기, 0.2s 예산, 1.5배 여유, k=32 → buffer_size = 30 + 32 = 62

API:
  ring_buffer.put(data)          # 새 데이터 기록 (원자적 카운터로 동기화)
  data = ring_buffer.get()       # 최신 1개 읽기
  data = ring_buffer.get_last_k(k) # 최신 k개 읽기
```

**SharedAtomicCounter** (64-bit, ACQUIRE/RELEASE 메모리 순서)로 read/write 인덱스를 동기화하여 Python GIL 없이도 안전한 멀티프로세스 접근을 보장.

### 3.2 SharedMemoryQueue — Lock-free FIFO 큐

```
설계 원칙: 명령 전달용 (FIFO: 순서 보장)
           로봇 컨트롤러에 schedule_waypoint 명령 전달 시 사용

API:
  queue.put(data)          # 큐에 추가 (Full 시 예외)
  data = queue.get()       # 1개 꺼내기
  data = queue.get_all()   # 전체 꺼내기 (플러시)
```

### 3.3 SharedNDArray

NumPy 배열을 SharedMemory 위에 직접 매핑. `.get()`으로 현재 메모리뷰 반환.
프로세스 간 복사 없이 배열 공유 가능.

---

## 4. ViveSharedMemory — HTC Vive 입력 처리 (`umi/real_world/vive_shared_memory.py`)

### 동작 방식

```
[vive_input 서버] ─ TCP(12345) ─► ViveSharedMemory[200Hz]
                                    ├── JSON 파싱
                                    ├── 좌표 변환 (VR World → Robot Base)
                                    └── SharedMemoryRingBuffer에 기록

[햅틱 피드백] ◄─ UDP(12346) ──── send_haptic()
```

### 좌표계 변환 (VR World → Robot Base Frame)

```
tx_robot_vive:
  Robot X+ = Vive Y+  (사용자 쪽으로 당김 = 로봇 전진)
  Robot Y+ = Vive X+  (오른쪽 이동       = 로봇 우측)
  Robot Z+ = -Vive Z+ (위로 올림          = 로봇 상승)
```

### 저장 상태 (RingBuffer 스키마)

| 필드 | 타입 | 설명 |
|---|---|---|
| `position` | float32 (3,) | xyz 위치 |
| `quaternion` | float32 (4,) | xyzw 회전 |
| `buttons.grip` | uint8 | 클러치 버튼 (텔레오프 ON/OFF) |
| `buttons.trigger` | uint8 | 그리퍼 토글 버튼 |
| `buttons.trackpad` | uint8 | 트랙패드 누름 (HOME) |
| `analog.trigger_value` | float32 | 트리거 아날로그 값 (0.0~1.0) |
| `analog.trackpad_x/y` | float32 | 트랙패드 터치 위치 |
| `connected` | uint8 | 연결 상태 |

---

## 5. ViveTeleopProcess — 100Hz 텔레오프 계산 (`umi/real_world/vive_teleop_process.py`)

별도 프로세스로 실행되며 ViveSharedMemory에서 데이터를 읽어 3개의 RingBuffer에 결과를 출력.

### 5.1 출력 채널 (3개 SharedMemoryRingBuffer)

| 버퍼 | 소비자 | 내용 |
|---|---|---|
| `action_ring_buffer` | Main Process (10Hz 기록) | target_pose, gripper_target_width, clutch_active, home_active, home_requested, rotation_active |
| `robot_ring_buffer` | FrankaInterpolationController | target_pose (6,), clutch_active, rotation_active |
| `gripper_ring_buffer` | FrankaGripperController | gripper_state (0/1), gripper_command, teleop_timestamp |

### 5.2 Grip 버튼 (클러치) 상태 머신

```
[클러치 OFF] → 위치 홀드 + Trackpad 회전 활성
    │
    │ grip 버튼 누름 (0→1 전환)
    │
    ▼
[포즈 오프셋 계산]  ← 핵심! 초기화 시 한 번만 수행
  robot_pose  = FrankaInterpolationController 현재 위치 (fresh 읽기)
  vr_pose     = Vive 현재 위치
  offset = robot_pose - vr_pose
  이후: target_pose = vr_pose + offset (모든 VR 명령에 offset 적용)
    │
    ▼
[클러치 ON] → VR 컨트롤러 추적 + 속도 클램핑
    │
    │ grip 버튼 뗌 (1→0 전환)
    │
    └─► [클러치 OFF] → 위치 홀드
```

**설계 의도:** Clutch 0→1 전환 시 VR-로봇 위치 갭이 있으면 순간 점프가 발생하므로,
이를 방지하기 위해 전환 시점의 포즈 오프셋을 계산하여 이후 모든 명령에 적용.

### 5.3 속도 클램핑 (선택적)

```python
use_velocity_clamping=True 시:
  max_pos_velocity = 2.0 m/s  (Franka 하드웨어 한계 ~1.7m/s)
  max_rot_velocity = 2.5 rad/s

  dt = 1/100 (10ms)
  delta_pos = vr_pos - prev_vr_pos
  if |delta_pos| / dt > max_pos_velocity:
      delta_pos = clamp(delta_pos)
```

### 5.4 트리거 그리퍼 토글

```
trigger_value (0.0~1.0 아날로그):
  ≥ 0.5 → CLOSE 명령 (awaiting_trigger_release = True)
  < 0.3 → awaiting_trigger_release = False (다음 토글 허용)

그리퍼 상태: OPEN(0) ↔ CLOSE(1) 토글 방식
gripper_target_width:
  OPEN  상태: 0.075m
  CLOSE 상태: 0.005m
```

### 5.5 Trackpad 회전 (Z축 추가 자유도)

```
클러치 OFF + trackpad 터치:
  Y > +0.7 → EE Z축 시계방향 회전 (+0.5°/update = +50°/sec)
  Y < -0.7 → EE Z축 반시계방향 회전 (-0.5°/update)
  회전 한계: ±90° (초기 자세 기준)

trackpad 누름(press) → HOME 이동 요청 신호
```

### 5.6 이상치 감지

```python
validate_position_change(delta_pos, threshold=0.05m):
  if |delta_pos| > threshold:
      skip this frame (VR 트래킹 손실 방지)
```

---

## 6. FrankaInterpolationController — 100Hz 로봇 팔 제어 (`polymetis_franka_teleop/real_world/franka_interpolation_controller.py`)

별도 프로세스로 실행. polymetis Python 클라이언트(`RobotInterface`)를 통해 NUC `:50051` gRPC 서버에 직접 연결 (UMI/DROID 시절의 ZeroRPC bridge 거치지 않음).

### 6.1 Cartesian Impedance Control 파라미터

```python
Kx  = [750, 750, 750, 15, 15, 15]   # 스프링 게인 (위치 x,y,z / 회전 rx,ry,rz)
Kxd = [37, 37, 37, 2, 2, 2]         # 댐핑 게인 (속도)

tcp_offset = 0.1034m    # Franka Hand TCP 오프셋 (컨트롤러 → 그리퍼 끝단)
```

### 6.2 HOME 포지션

```python
FRANKA_HOME_JOINTS = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
# 단위: radian (≈ [0, -45°, 0, -135°, 0, 90°, 45°])
```

### 6.3 명령 채널 (SharedMemoryQueue)

```
SERVOL (pose, duration)           # 즉시 목표 자세로 이동
SCHEDULE_WAYPOINT (pose, t)       # 절대 시간 기준 웨이포인트 예약
MOVE_HOME                         # HOME 포지션으로 이동
STOP                              # 제어 루프 중단
```

### 6.4 Normal Mode vs Teleop Mode

```
[Normal Mode] teleop_mode=False
  FrankaInterpolationController 루프 (200Hz):
    1. SharedMemoryQueue에서 SCHEDULE_WAYPOINT 명령 읽기
    2. PoseTrajectoryInterpolator에 waypoint 추가
    3. interpolated_pose = interpolator(t_now)    ← 매 5ms마다
    4. robot.update_desired_ee_pose(interpolated_pose)
    5. 상태를 RingBuffer에 기록

[Teleop Mode] teleop_mode=True
  FrankaInterpolationController 루프 (200Hz):
    1. SharedMemoryQueue 명령 무시 (SCHEDULE_WAYPOINT 차단)
    2. robot_ring_buffer.get() 직접 읽기 (100Hz ViveTeleopProcess 출력)
    3. Clutch 상태 확인:
       - 0→1 전환: 새 포즈 오프셋 계산 (네트워크 지연 보정)
       - 1 (ON):   offset 적용된 target_pose 추적
       - 1→0 전환: current_pose로 부드럽게 전환
    4. robot.update_desired_ee_pose(target_pose)   ← 200Hz 내부 루프
```

### 6.5 PoseTrajectoryInterpolator

```
복수의 (time, pose) 웨이포인트로부터 임의 시간 t에서의 포즈 보간:
  위치: 선형 보간
  회전: SLERP (Spherical Linear Interpolation)

과거 웨이포인트: 자동 정리 (메모리 누수 방지)
미래 웨이포인트 초과 시: 마지막 포즈 유지
```

### 6.6 에러 복구

```
libfranka 예외 (충돌 감지 등) 발생 시:
  1. 컨트롤러 중단 감지
  2. polymetis automaticErrorRecovery 호출
  3. Impedance 컨트롤러 재시작 + wait_until_controller_ready
  4. reset_ik_state (catalog #26) — IK seed 재초기화
  5. HOME 포지션으로 복귀 (auto-HOME 에스컬레이션, catalog #27)
  6. Teleop 모드: Clutch 재결합 대기
```

### 6.7 상태 출력 (RingBuffer 스키마)

| 필드 | 타입 | 설명 |
|---|---|---|
| `ActualTCPPose` | float64 (6,) | 현재 EEF 포즈 [x,y,z,rx,ry,rz] (axis-angle) |
| `ActualQ` | float64 (7,) | 현재 관절 각도 |
| `ActualQd` | float64 (7,) | 현재 관절 속도 |
| `robot_timestamp` | float64 | 측정 시각 (wall clock) |

---

## 7. FrankaGripperController — 30Hz 그리퍼 제어 v2.0 (`umi/real_world/franka_gripper_controller.py`)

### 7.1 이산 제어 설계 (v2.0)

```
v1.0 (연속): 연속 너비 명령 → 보간 → 복잡한 지연
v2.0 (이산): OPEN/CLOSE 전환 시에만 명령 발송 → 지연 최소화

이유:
  - 데이터 수집 시 binary 상태(OPEN/CLOSE)만 기록됨
  - 정책 출력도 threshold 기반 이진 판단
  - 보간은 불필요한 지연을 추가함
  - 동일한 명령을 반복하면 libfranka 예외 발생 가능
```

### 7.2 명령 파라미터

```python
gripper_open_width  = 0.075m   # 열린 상태 너비 (실제 관측 최대값에 맞춤, ≠ 0.08m)
gripper_close_width = 0.005m   # 닫힌 상태 너비 (0.0m 사용 시 libfranka 예외 발생)
DEFAULT_SPEED = 0.2 m/s
DEFAULT_FORCE = 50.0 N         # 파지 힘 (최대 70N)
MAX_WIDTH = 0.08m              # 하드웨어 물리 한계
```

### 7.3 Normal Mode schedule_waypoint 흐름

```
exec_actions() 호출 시 (eval_franka_policy.py에서):
  1. 각 액션 스텝의 gripper_value > threshold → OPEN / ≤ threshold → CLOSE 판단
  2. 이전 상태와 비교하여 전환 감지
  3. 전환이 있는 첫 번째 스텝 시간에 schedule_waypoint(width, target_time) 호출
  4. 전환 없으면 → 명령 없음 (현재 상태 유지)

FrankaGripperController 루프 (30Hz):
  1. queue에서 pending 명령 확인
  2. target_time 도달 확인
  3. 현재 그리퍼가 이미 동작 중이면 대기
  4. goto(width) 또는 grasp() 실행
```

### 7.4 상태 출력 (RingBuffer 스키마)

| 필드 | 타입 | 설명 |
|---|---|---|
| `gripper_width` | float64 | 현재 그리퍼 너비 (m) |
| `gripper_timestamp` | float64 | 측정 시각 |
| `is_moving` | uint8 | 동작 중 여부 |

---

## 8. MultiRealsense / SingleRealsense — 카메라 관리 (`umi/real_world/multi_realsense.py`, `single_realsense.py`)

### 8.1 SingleRealsense — 카메라당 독립 프로세스

```
SingleRealsense(mp.Process) 루프:
  1. rs.Pipeline 초기화 (resolution, fps 설정)
  2. 카메라 내재 파라미터 저장 (fx, fy, ppx, ppy)
  3. 캡처 루프:
     a. pipeline.wait_for_frames() ← RealSense SDK 블로킹 호출
     b. align.process()             ← 모든 스트림을 color 프레임으로 정렬
     c. ImageTransform 적용         ← resize + 채널 변환 + float32
     d. VideoRecorder.write_frame() ← 기록 중이면 인코딩
     e. RingBuffer.put(frame, timestamp) ← 공유 메모리에 저장
```

### 8.2 주요 파라미터

```python
resolution      = (640, 480)   # 기본 캡처 해상도 (WxH)
capture_fps     = 60           # 캡처 주파수 (RealSense 하드웨어)
put_downsample  = False        # 다운샘플링 없이 모든 프레임 저장
get_max_k       = 60           # 한 번에 읽을 수 있는 최대 프레임 수
receive_latency = 0.015        # 하드웨어 타임스탬프 오프셋 보정 (15ms)
```

### 8.3 명령 채널

```
SET_COLOR_OPTION  → 노출/게인/화이트밸런스 설정
START_RECORDING   → video_path, start_time 지정하여 MP4 기록 시작
STOP_RECORDING    → 기록 중단 및 파일 flush
RESTART_PUT       → 링 버퍼 타이밍 리셋 (에피소드 시작 시)
```

### 8.4 VideoRecorder — H264 스트리밍 인코더

```python
VideoRecorder.create_h264(
    fps=camera_fps,
    codec='h264',
    input_pix_fmt='bgr24',   # RealSense 기본 출력 포맷
    output_pix_fmt='yuv420p',
    crf=18,                  # 품질 (낮을수록 고품질, 0~51)
    thread_type='FRAME',
    thread_count=1
)
```

**인코딩 파이프라인:** `bgr24 → yuv420p → H264 → MP4 (av/PyAV)`

Frame 타이밍 처리:
- 드롭된 프레임: 이전 프레임 반복하여 타임스탬프 연속성 유지
- `no_repeat=False`: 기본값, 프레임 반복 허용

### 8.5 ImageTransform — Picklable 이미지 변환

```python
# spawn 방식 multiprocessing에서 closure 불가 → 클래스로 구현
ImageTransform(
    input_res=(640, 480),
    output_res=(256, 256),   # 정책 obs 해상도
    bgr_to_rgb=True,         # RealSense BGR → RGB
    float32=True             # uint8 [0,255] → float32 [0,1]
)
```

---

## 9. 정밀 타이밍 (`umi/common/precise_sleep.py`)

### Hybrid Sleep + Spin 전략

```python
def precise_wait(t_end, slack_time=0.001):
    """절대 시각 t_end까지 정확히 대기"""
    # Phase 1: OS sleep (CPU 절약)
    sleep_time = t_end - time.monotonic() - slack_time
    if sleep_time > 0:
        time.sleep(sleep_time)  # 대략 0.5~2ms 오차

    # Phase 2: Spin (마지막 1ms 정밀 대기)
    while time.monotonic() < t_end:
        pass  # 적극적 대기 (CPU 사용하지만 ±10μs 정확도)
```

**적용 위치:**
- `demo_franka_vive.py`: `precise_wait(t_sample)`, `precise_wait(t_cycle_end)`
- `eval_franka_policy.py`: `precise_wait(t_cycle_end - frame_latency)`

---

## 10. 관측 동기화 (`umi/common/timestamp_accumulator.py`)

### TimestampActionAccumulator

```
목적: 비정기적으로 들어오는 액션을 규칙적인 그리드로 정렬
입력: start_time + dt로 정의된 시간 그리드
      비정기적 액션 타임스탬프

알고리즘: get_accumulate_timestamp_idxs()
  각 시간 윈도우 [start + k*dt, start + (k+1)*dt) 에서
  가장 먼저 들어온 타임스탬프를 선택
  누락된 윈도우는 직전 데이터로 채움 (repeat)
  결과: global_idxs (규칙적 그리드 상의 인덱스 배열)
```

### ObsAccumulator

```
목적: 연속 시간에 걸친 관측 데이터 누적
API:
  put(data, timestamps)    # 새 데이터 추가
  timestamps[key]          # 각 데이터의 타임스탬프
  data[key]                # 누적된 데이터
```

---

## 11. 레이턴시 측정 및 보정 체계 (`umi/common/latency_util.py`)

### 측정 방법

```python
get_latency(
    target_signal,      # 명령 신호
    target_timestamps,
    actual_signal,      # 실제 측정 신호
    actual_timestamps
):
    # Cross-correlation으로 두 신호 간 시간 지연 계산
    # 반환: latency (초), 신뢰도 정보
```

### 측정된 레이턴시 값 (scripts_real/calibrate_*.py로 측정)

| 컴포넌트 | 레이턴시 | 측정 방법 |
|---|---|---|
| 카메라 관측 | **15ms** | HW 타임스탬프 기반 V3 측정 |
| 로봇 관측 | **1ms** | 왕복 시간의 절반 |
| 그리퍼 관측 | **1ms** | 왕복 시간의 절반 |
| 로봇 액션 실행 | **55ms ± 3.3ms** | schedule_waypoint → 실제 도달 측정 (calibrate_franka_arm_direct.py) |
| 그리퍼 액션 실행 (ART) | 측정값 | floor-offset 측정 (calibrate_art_gripper_latency.py) |
| 그리퍼 액션 실행 (Franka Hand) | 측정값 | 직접 명령 측정 — 기존 V3 fallback 85ms (63–100ms range) |

### 보정 적용 위치 (`exec_actions()` 내)

```python
# 정책이 t 시점에 실행되기를 원하는 액션을
# (t - latency) 시점에 미리 스케줄하여 도착 시간 = t가 되도록 보정

robot.schedule_waypoint(
    pose=action[:6],
    target_time=timestamps[i] - robot_action_latency  # 55ms 앞서 발송
)
gripper.schedule_waypoint(
    width=gripper_width,
    target_time=timestamps[i] - gripper_action_latency  # 85ms 앞서 발송
)
```

---

## 12. 데이터 수집 상세 (`demo_franka_vive.py` + `FrankaViveEnv`)

### 12.1 UMI-스타일 타이밍 설계

```python
# 10Hz 메인 루프의 정밀 타이밍
t_start = time.monotonic()
while not stop:
    t_cycle_end     = t_start + (iter_idx + 1) * dt    # 현재 사이클 종료
    t_sample        = t_cycle_end - command_latency     # 샘플 시각 (10ms 여유)
    t_command_target = t_cycle_end + dt                 # 액션 실행 예정 시각

    obs = env.get_obs()

    # 정확한 샘플 시각까지 대기
    precise_wait(t_sample)

    if is_recording:
        # 미래 타임스탬프로 액션 기록
        # → 학습 시 obs[t] → action[t+1] 관계 보장
        action_timestamp = t_command_target - time.monotonic() + time.time()
        env.record_action(timestamp=action_timestamp)

    precise_wait(t_cycle_end)
    iter_idx += 1
```

| 타임스탬프 | 의미 |
|---|---|
| `t_cycle_end` | 현재 사이클 종료 (관측 기준) |
| `t_sample` | ViveTeleop 입력 샘플 시각 |
| `t_command_target` | 액션이 실제 실행될 미래 시각 |

### 12.2 FrankaViveEnv.get_obs() — 타임스탬프 정렬

```
기준 시계: 카메라(align_camera_idx=0) 최신 타임스탬프 = t_ref

카메라 obs (nearest-neighbor):
  camera_obs_timestamps = [t_ref - dt, t_ref]   (obs_horizon=2)
  → 각 카메라 버퍼에서 가장 가까운 프레임 인덱스 선택

로봇 포즈 obs (연속 보간):
  robot_obs_timestamps = [t_ref - dt, t_ref]
  → PoseInterpolator (위치: 선형, 회전: SLERP) 로 보간

그리퍼 폭 obs (선형 보간):
  gripper_obs_timestamps = [t_ref - dt, t_ref]
  → interp1d (scipy) 선형 보간
```

**결과:** 모든 관측이 동일한 가상의 타임스탬프를 가짐
→ 이미지와 로봇 상태 간 수십 ms 불일치 완전 해소

### 12.3 record_action() — 액션 소스 결정

```
clutch_active=True, home_active=False:
  → target_pose = ViveTeleopProcess의 action_ring_buffer에서 읽은 target_pose
     (사용자가 실제로 이동시키는 명령값)

clutch_active=False 또는 home_active=True:
  → target_pose = _last_robot_obs_pose (get_obs()에서 기록한 최신 관측 포즈)
     (로봇이 현재 있는 위치 = 정지 상태 or HOME 자율 이동)
  설계 의도: 클러치를 누르지 않는 동안 action = obs 보장 → 연속 궤적 유지

gripper_target_width:
  → 항상 ViveTeleopProcess의 gripper_target_width 사용
  (클러치와 독립: 그리퍼는 트리거로만 제어)
```

### 12.4 에피소드 저장 (`end_episode()`)

```
1. camera.stop_recording()  → MP4 파일 flush 및 닫기
2. action_timestamps 기반으로 유효 스텝 수(n_steps) 결정
   (obs와 action 타임스탬프 중 최솟값까지만 저장)
3. obs_accumulator에서 로봇 포즈/조인트/그리퍼 보간
4. Zarr replay_buffer.add_episode(episode) 저장
```

**저장 에피소드 구조:**

```python
episode = {
    'timestamp':                 np.array (N,),       float64  wall clock
    'action':                    np.array (N, 7),     float32  [x,y,z,rx,ry,rz,gripper_width]
    'robot0_eef_pos':            np.array (N, 3),     float64
    'robot0_eef_rot_axis_angle': np.array (N, 3),     float64  axis-angle
    'robot0_joint_pos':          np.array (N, 7),     float64
    'robot0_joint_vel':          np.array (N, 7),     float64
    'robot0_gripper_width':      np.array (N, 1),     float64
}
```

### 12.5 Zarr Replay Buffer 파일 구조

```
replay_buffer.zarr/
├── data/
│   ├── action/                  (총N, 7)    float32
│   ├── timestamp/               (총N,)      float64
│   ├── robot0_eef_pos/          (총N, 3)    float64
│   ├── robot0_eef_rot_axis_angle/(총N, 3)   float64
│   ├── robot0_joint_pos/        (총N, 7)    float64
│   ├── robot0_joint_vel/        (총N, 7)    float64
│   └── robot0_gripper_width/    (총N, 1)    float64
└── meta/
    └── episode_ends/            에피소드 경계 인덱스 배열

videos/
└── {episode_id}/
    ├── 0.mp4   (camera0, H264, CRF=18, 60fps)
    └── 1.mp4   (camera1, H264, CRF=18, 60fps)
```

### 12.6 키/Vive 컨트롤 맵

| 입력 | 동작 |
|---|---|
| Vive Grip (hold) | 클러치 → 텔레오프 활성화 |
| Vive Trigger (≥0.5) | 그리퍼 토글 (OPEN↔CLOSE) |
| Vive Trackpad Press | HOME 이동 요청 |
| Vive Trackpad Y > 0.7 | EE Z축 시계방향 회전 |
| Vive Trackpad Y < -0.7 | EE Z축 반시계방향 회전 |
| `c` | 에피소드 기록 시작 |
| `s` | 에피소드 기록 종료 및 저장 |
| `Backspace` → `y` (5초 내) | 에피소드 드롭 |
| `h` | HOME 이동 (키보드 폴백) |
| `q` | 종료 |

---

## 13. 정책 실행 상세 (`eval_franka_policy.py` + `FrankaPolicyEnv`)

### 13.1 실행 전 환경 정리 (`cleanup_environment()`)

```
[1/4] 충돌 프로세스 SIGTERM:
      eval_franka_policy.py (이전 실행), realsense2_camera_node,
      franka_ros2, rviz2, ros2 launch 등
      (훈련 프로세스 __KMP_*, pt_data는 절대 종료 안 함)

[2/4] ROS2 데몬 중지: ros2 daemon stop

[3/4] 스테일 공유 메모리 삭제:
      /dev/shm/fastrtps_*   (ROS2 DDS 잔재)
      /dev/shm/psm_*        (Python SharedMemory 잔재)
      (u1000-Shm_* SteamVR는 유지)

[4/4] cv2.destroyAllWindows()
```

### 13.2 체크포인트 로드 및 설정 자동 파싱

```python
payload = torch.load(ckpt_path, map_location='cpu', pickle_module=dill)
cfg = payload['cfg']

# 체크포인트 설정에서 자동 추출 (하드코딩 없음)
obs_res          = get_real_obs_resolution(cfg.task.shape_meta)
obs_pose_repr    = cfg.task.pose_repr.obs_pose_repr     # e.g., 'abs' or 'rel'
action_pose_repr = cfg.task.pose_repr.action_pose_repr  # e.g., 'abs' or 'rel'
camera_obs_horizon = cfg.task.shape_meta.obs.camera0_rgb.horizon
robot_obs_horizon  = cfg.task.shape_meta.obs.robot0_eef_pos.horizon
```

### 13.3 CUDA 컨텍스트 안전 초기화

```python
# 반드시 FrankaPolicyEnv (fork) 이후에 모델 생성
# → fork 전 CUDA 컨텍스트 생성 시 자식 프로세스에서 복사되어 충돌
with FrankaPolicyEnv(...) as env:
    # 환경 시작 (모든 서브프로세스 spawn) 완료 후
    policy = workspace.model
    policy.eval().to(device)  # ← 안전
```

### 13.4 Warmup 추론

```python
# GPU 알고리즘 캐시 초기화 + shape 검증
obs = env.get_obs()
result = policy.predict_action(obs_dict)
action = result['action_pred'][0].detach().to('cpu').numpy()
assert action.shape[-1] == 10   # 10D 정책 출력 확인
env_action = get_real_umi_action(action, obs, action_pose_repr)
assert env_action.shape[-1] == 7  # 7D 환경 액션 확인
```

### 13.5 메인 추론 루프

```python
while not stop_episode:
    # 1. 타이밍 계산
    t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

    # 2. 관측 수집 (타임스탬프 정렬)
    obs = env.get_obs()
    obs_timestamps = obs['timestamp']

    # 3. 관측 전처리
    obs_dict_np = get_real_umi_obs_dict(obs, shape_meta, obs_pose_repr)

    # 4. GPU 추론 (DDIM 16스텝)
    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
    result = policy.predict_action(obs_dict)
    raw_action = result['action_pred'][0].detach().cpu().numpy()  # (N, 10)

    # 5. 액션 후처리 (상대 → 절대 포즈 역변환)
    action = get_real_umi_action(raw_action, obs, action_pose_repr)  # (N, 7)

    # 6. 타임스탬프 생성 (obs 기준)
    action_timestamps = obs_timestamps[-1] + np.arange(len(action)) * dt

    # 7. 미래 액션만 필터링
    is_new = action_timestamps > (curr_time + 0.01)
    if np.sum(is_new) == 0:
        # 시간 초과 시: 마지막 액션 1개만 실행
        this_actions = action[[-1]]

    # 8. 액션 실행 (레이턴시 보정 포함)
    env.exec_actions(this_actions, this_timestamps, compensate_latency=True)

    # 9. 다음 사이클까지 정밀 대기
    precise_wait(t_cycle_end - frame_latency)
    iter_idx += steps_per_inference
```

### 13.6 포즈 표현 변환 (`real_inference_util.py`)

#### get_real_umi_obs_dict() — 관측 전처리

```
입력 (env_obs):
  robot0_eef_pos            (T, 3)  axis-angle 포즈의 위치 부분
  robot0_eef_rot_axis_angle (T, 3)  axis-angle 포즈의 회전 부분
  robot0_gripper_width      (T, 1)  그리퍼 너비
  camera0_rgb               (T, H, W, C)  이미지

처리:
  1. 이미지: THWC → TCHW, float32 [0,1]
  2. 포즈: axis-angle → 4×4 행렬 (pose_to_mat)
          → 상대 포즈 변환 (현재 포즈 기준 상대 표현)
          → 9D 회전 표현으로 변환 (mat_to_pose10d)
          → obs_dict의 'robot0_eef_pos', 'robot0_eef_rot_axis_angle' 업데이트

출력:
  robot0_eef_pos            (T, 3)   상대 위치
  robot0_eef_rot_axis_angle (T, 9)   9D 회전 (3×3 행렬 펼친 것)
  robot0_gripper_width      (T, 1)   그대로
  camera0_rgb               (T, C, H, W)  TCHW float32
```

#### get_real_umi_action() — 액션 후처리

```
입력: raw_action (N, 10)
  [0:3]  위치 (3D)
  [3:9]  9D 회전 행렬 (3×3)
  [9:10] 그리퍼 너비

처리:
  1. [0:9] → pose10d_to_mat → 4×4 행렬
  2. 상대 → 절대 포즈 역변환
     (현재 관측 포즈를 base로 사용)
  3. mat_to_pose → axis-angle 포즈 (6D)
  4. 그리퍼 너비 연결

출력: env_action (N, 7)
  [0:6]  절대 TCP 포즈 [x,y,z,rx,ry,rz]
  [6]    그리퍼 목표 너비 (m)
```

#### 9D 회전 표현 (pose10d)

```
구성: position (3) + rotation_matrix_flattened (9) = 12D
     → 실제 학습에서 pos(3) + rot9d(9) = 10D로 사용

장점:
  - 6D 회전(그람-슈미트)보다 명시적 (행렬 직접 표현)
  - 쿼터니언(4D)보다 모호성 없음 (이중 덮개 문제 없음)
  - SO(3)에서의 연속성 보장
```

### 13.7 FrankaPolicyEnv.exec_actions() — 그리퍼 전이 감지

```python
# 그리퍼: 전환 시점에만 1번 명령 (v2.0 이산 제어)
gripper_threshold = (gripper_open_width + gripper_close_width) / 2  # = 0.04m

gripper_actions = new_actions[:, 6]
gripper_is_open = gripper_actions > gripper_threshold

# 이전 상태와 비교하여 첫 번째 전환 지점 탐색
transition_idx = None
for i in range(len(gripper_is_open)):
    prev = self._last_gripper_is_open if i == 0 else gripper_is_open[i-1]
    if gripper_is_open[i] != prev:
        transition_idx = i
        break

if transition_idx is not None:
    target_width = open_width if gripper_is_open[transition_idx] else close_width
    gripper.schedule_waypoint(target_width, timestamps[transition_idx] - 85ms)
    self._last_gripper_is_open = gripper_is_open[transition_idx]
```

### 13.8 그리퍼 상태 디버그 출력 (매 추론 시)

```
[5.2s] Infer: 245ms (4.1Hz) | Actions: 6 steps | Gripper: 0.032 → CLOSE (O:1/C:5)
  Steps: 0.071● 0.041● 0.028○ 0.021○ 0.018○ 0.016○
  Trend: first3=0.047, last3=0.018 →CLOSING
```

---

## 14. 프로세스 시작 순서 및 Context Manager

### 데이터 수집 시 (`FrankaViveEnv.start()`)

```python
1. vive.start(wait=True)           # Vive 수신 먼저 (ring_buffer 생성)
2. teleop.start(wait=True)         # ViveTeleopProcess (vive ring_buffer 참조)
3. robot.start(wait=False)         # FrankaInterpolationController
4. gripper.start(wait=False)       # FrankaGripperController
5. time.sleep(0.5)                 # robot/gripper 초기화 대기
6. camera.start(wait=False)        # MultiRealsense (카메라는 마지막)
7. [wait=True 시] start_wait():
   └── time.sleep(2.0)             # 카메라 버퍼 채움 대기 (60fps × 2s = 120 프레임)
```

### 정책 실행 시 (`FrankaPolicyEnv.start()`)

```python
1. robot.start(wait=False)         # ViveTeleopProcess 없음
2. gripper.start(wait=False)
3. time.sleep(0.5)
4. camera.start(wait=False)
5. [wait=True 시] start_wait():
   └── time.sleep(2.0)
```

### Context Manager 패턴

```python
# __enter__ → start(), __exit__ → stop()
with FrankaViveEnv(...) as env:
    ...
# 종료 시: end_episode() → camera/gripper/robot/teleop/vive 역순 정지
```

---

## 15. Diffusion Policy 설정 파일 (`diffusion_policy/config/task/`)

### franka_vive_image.yaml

```yaml
# 이미지를 직접 HDF5로 저장한 데이터셋 사용
dataset_class: FrankaViveImageDataset
image_resolution: [480, 640]  # HxW
action_dim: 10                # pos(3) + rotation_6d(6) + gripper(1)
obs_keys:
  - agentview_image
  - robot0_eye_in_hand_image
  - robot0_eef_pos
  - robot0_eef_quat
  - robot0_gripper_qpos
```

### franka_vive_umi.yaml

```yaml
# convert_franka_vive_to_umi_format.py로 변환된 UMI 형식 사용
dataset_class: UmiDataset   # 표준 UMI 데이터셋 파이프라인
image_resolution: [480, 640]
action_dim: 10              # pos(3) + rotation_6d(6) + gripper(1)
pose_repr:
  obs_pose_repr: 'abs'      # 절대 포즈 표현
  action_pose_repr: 'abs'
```

---

## 16. 구버전과의 비교 분석

### 16.1 데이터 수집 (`live_record_total.py` vs `demo_franka_vive.py` + `FrankaViveEnv`)

| 항목 | 구버전 (`live_record_total.py`) | 신버전 (`demo_franka_vive.py`) |
|---|---|---|
| **아키텍처** | 단일 ROS2 노드 + 1 스레드 | SharedMemory 기반 멀티프로세스 |
| **텔레오프 입력** | 없음 (외부 시스템 제어) | Vive 100Hz 전용 프로세스 |
| **카메라 드라이버** | ROS2 토픽 구독 (ROS 의존) | RealSense SDK 직접 (ROS 불필요) |
| **제어 주파수** | 30Hz (느슨한 sleep) | 10Hz (precise_wait, 정밀) |
| **관측 동기화** | 없음 (최신값 단순 결합) | 카메라 기준 타임스탬프 보간 정렬 |
| **텔레오프 주파수** | 해당 없음 | 100Hz (ViveTeleopProcess) |
| **로봇 제어 주파수** | 해당 없음 | 200Hz (FrankaInterpolationController) |
| **그리퍼 상태** | delta 기반 이진화 | 연속 폭 + 이산 명령 분리 저장 |
| **저장 포맷** | HDF5 + CSV (비표준) | Zarr + H264 MP4 (UMI 표준) |
| **action 정의** | `action = qpos` (관측 = 액션) | 실제 텔레오프 목표 포즈 |
| **스레드 안전성** | Lock 없음 | SharedMemoryRingBuffer (Lock-free) |
| **GIL 영향** | 받음 (단일 프로세스) | 없음 (멀티프로세스) |
| **에피소드 드롭** | 없음 | `Backspace + y` (5초 타임아웃) |
| **HOME 기능** | 없음 | Vive 트랙패드 or `h` 키 |
| **프리플라이트 체크** | 없음 | `preflight_check.py` 자동 실행 |
| **비디오 기록** | 별도 저장 없음 | H264 MP4 (CRF=18 고화질) |

### 16.2 정책 실행 (`real_eval_fr3_te_unet_pap.py` vs `eval_franka_policy.py` + `FrankaPolicyEnv`)

| 항목 | 구버전 | 신버전 |
|---|---|---|
| **아키텍처** | 단일 ROS2 노드 + 1 추론 스레드 | SharedMemory 멀티프로세스 |
| **제어 주파수** | 목표 30Hz (루프 외 sleep 버그) | 정밀 10Hz (precise_wait) |
| **관측 동기화** | 없음 | 카메라 기준 타임스탬프 보간 |
| **액션 평활화** | Temporal Ensembling (직접 구현) | 200Hz 보간 컨트롤러 (위임) |
| **레이턴시 보정** | `latency_steps=3` 경험값 하드코딩 | 측정값 기반 (55ms, 85ms) |
| **그리퍼 제어** | `send_goal_async()` 매 스텝 | `schedule_waypoint()` 전환 시에만 |
| **포즈 표현** | 6D 회전 (그람-슈미트 직접 구현) | 9D 회전 행렬 (pose10d) |
| **상대 포즈** | 없음 (절대 포즈만) | 설정 가능 (`abs` 또는 `rel`) |
| **수집-실행 포맷 일치** | 불일치 가능 (HDF5 vs ROS2) | 완전 일치 (동일한 util 함수) |
| **GPU 선택** | `cuda:1` 하드코딩 | `cuda:0` or 자동 감지 |
| **환경 정리** | 없음 | `cleanup_environment()` 4단계 자동 |
| **스레드 안전성** | Lock 없음 | SharedMemoryRingBuffer |
| **GIL 영향** | 받음 | 없음 (멀티프로세스) |
| **Warmup 추론** | 없음 | shape 검증 포함 Warmup |
| **DDIM steps** | 16 하드코딩 | CLI `--num_inference_steps` |
| **steps_per_inference** | 없음 (매 스텝 추론) | 6 (CLI 설정 가능) |
| **관측 horizon** | 하드코딩 n_obs_steps=2 | cfg.task.shape_meta에서 자동 파싱 |

### 16.3 가장 중요한 구조적 개선 4가지

#### 개선 1: Temporal Ensembling 제거 → 200Hz 보간 컨트롤러로 대체

```
[구버전] Temporal Ensembler (직접 구현)
  - 최대 n_action_steps 개의 시퀀스를 누적
  - exp(-k * age) 가중치로 블렌딩
  - t+3 스텝 그리퍼 선행 조회 (경험적 보정)
  - 복잡하고 k, latency_steps 수동 튜닝 필요
  - 시간 불일치 여전히 존재

[신버전] 타임스탬프 스케줄링 + 200Hz 보간
  - policy → exec_actions(actions, timestamps)
  - FrankaInterpolationController가 timestamps에 맞춰 200Hz로 보간 실행
  - 레이턴시는 측정된 55ms로 명시적 보정
  - PoseTrajectoryInterpolator가 부드러운 궤적 보장
```

#### 개선 2: 관측 동기화

```
[구버전]
  latest_images['left'] (가장 최근 ROS 콜백 프레임)
  + ee_curr_pose (가장 최근 PoseStamped 콜백)
  → 두 값의 타임스탬프 불일치 (수십 ms)

[신버전]
  카메라 기준 타임스탬프 t_ref 결정
  → 로봇 포즈를 t_ref 시점으로 SLERP 보간
  → 그리퍼 폭을 t_ref 시점으로 선형 보간
  → 완전 동기화된 관측
```

#### 개선 3: 데이터 수집-실행 파이프라인 일관성

```
[구버전]
  수집: HDF5에 qpos를 action으로 저장 (관측 ≈ 명령, 부정확)
  실행: 6D 회전 출력 (그람-슈미트)
  → 학습 데이터 포맷과 실행 포맷이 다를 수 있음

[신버전]
  수집: Vive target_pose를 action으로 저장 (실제 명령값)
        record_action() → action_accumulator
  실행: get_real_umi_obs_dict() / get_real_umi_action()
        동일한 함수가 학습/실행 모두에 사용
  → 학습 데이터 포맷 = 실행 포맷 (완전 일치)
```

#### 개선 4: 멀티프로세스 격리 (Python GIL 완전 회피)

```
[구버전]
  단일 ROS2 프로세스 내:
  - 이미지 콜백 스레드
  - 조인트 상태 콜백 스레드
  - 추론 스레드
  - 메인 제어 루프
  → Python GIL로 인한 지연, Lock 없는 공유 상태

[신버전]
  독립 OS 프로세스:
  - ViveSharedMemory 프로세스 (200Hz)
  - ViveTeleopProcess 프로세스 (100Hz)
  - FrankaInterpolationController 프로세스 (200Hz)
  - FrankaGripperController 프로세스 (30Hz)
  - SingleRealsense 프로세스 × n_cameras (60fps each)
  - Main 프로세스 (10Hz)
  → GIL 영향 없음, SharedMemoryRingBuffer로 Lock-free 통신
```

---

## 17. 남은 한계 및 고려사항

| 항목 | 내용 |
|---|---|
| **MultiCameraVisualizer 비활성화** | `spawn` 방식과 picklable 문제로 `enable_multi_cam_vis=False` 고정 |
| **Trackpad 회전 한계** | ±90° 범위 제한 (초기 자세 기준) — 더 넓은 작업 공간에서 부족할 수 있음 |
| **그리퍼 전이 초기화** | `_last_gripper_is_open` 첫 초기화 시 실제 상태와 불일치 가능성 |
| **steps_per_inference=6** | 추론이 늦으면 일부 스텝 건너뜀 (시간 초과 시 마지막 액션 1개만 실행) |
| **gripper_close_width=0.005m** | 0.0m 사용 시 libfranka 예외 — 5mm 여유 필요 |
| **포즈 표현 설정 의존성** | 학습 시 `obs_pose_repr`/`action_pose_repr`와 실행 시 설정이 반드시 일치해야 함 |
| **ROS2 비의존** | RealSense SDK / ZED SDK 직접 임포트, polymetis는 :50051 gRPC 직접 — ROS2 노드와 충돌 주의 |
