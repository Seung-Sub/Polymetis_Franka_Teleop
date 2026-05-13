"""KIST에서 DROID/main_gr00t.py를 돌릴 때 사용하는 최소형 RobotEnv.

상위 NUC에 DROID 전체를 설치하지 않고, NUC가 띄운 Polymetis 서버에 직접
gRPC client로 붙어 동작한다. 카메라(ZED 2i + ZED Mini)는 3090에 USB로
연결되어 있으므로 pyzed로 직접 캡처한다.

main_gr00t.py가 사용하는 RobotEnv API의 부분집합만 구현한다:
  - get_observation() -> dict("image": {...}, "robot_state": {...})
  - step(action) — joint_position(7) + gripper(1) = 8D 액션
  - reset(randomize=False)
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import numpy as np
import pyzed.sl as sl
import torch
from scipy.spatial.transform import Rotation as R

from polymetis import GripperInterface, RobotInterface


def _open_zed(serial: int) -> Optional[sl.Camera]:
    """주어진 ZED 시리얼 카메라를 VGA@60 으로 연다 (KIST 학습 데이터 정합). 실패 시 None."""
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.VGA
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.set_from_serial_number(int(serial))
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        warnings.warn(f"ZED open failed sn={serial}: {err}")
        cam.close()
        return None
    return cam


class RobotEnv:
    """DROID RobotEnv API의 최소 호환 구현. main_gr00t.py와 호환."""

    def __init__(
        self,
        action_space: str = "joint_position",
        gripper_action_space: str = "position",
        polymetis_ip: str = None,
        polymetis_port: int = 50051,
        gripper_port: int = 50052,
        camera_serials: dict | None = None,
        gripper_speed: float = 0.05,
        gripper_force: float = 10.0,
    ):
        assert action_space == "joint_position", \
            "현재 wrapper는 joint_position 만 지원 (DROID main_gr00t.py 기본값)"
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        # Franka Hand 안전 기본값. DROID 코드는 speed=0.05, force=0.1(정규화) 였으나
        # Polymetis franka_hand_client의 force 인자는 Newton 단위 → 10 N 정도로 보정.
        # ART Gripper 사용 시 force 기본값을 자동 상향 (max 100 N 까지 사용 가능).
        # ART_GRIPPER_FORCE / ART_GRIPPER_SPEED env var 로 task 별 override 가능.
        self.gripper_speed = gripper_speed
        if os.environ.get("KIST_GRIPPER", "franka_hand").lower() == "art_gripper":
            self.gripper_force = float(os.environ.get("ART_GRIPPER_FORCE", "30"))
            self.gripper_speed = float(os.environ.get("ART_GRIPPER_SPEED", "0.15"))
        else:
            self.gripper_force = gripper_force

        self.polymetis_ip = polymetis_ip or os.environ.get("POLYMETIS_IP", "192.168.1.12")

        # ----- Polymetis client (NUC) -----
        print(f"[KIST] Polymetis arm 서버 접속: {self.polymetis_ip}:{polymetis_port}")
        self._robot = RobotInterface(ip_address=self.polymetis_ip, port=polymetis_port)

        # ----- Gripper client — env var KIST_GRIPPER 로 전환 -----
        #   "franka_hand" (default)  : Polymetis GripperInterface (NUC franka_hand_client, 50052)
        #   "art_gripper"            : ART Gripper standalone daemon TCP (localhost:50053)
        # 인터페이스는 polymetis 모양으로 통일 (.metadata.max_width, .get_state(), .goto()).
        self._gripper_kind = os.environ.get("KIST_GRIPPER", "franka_hand").lower()
        if self._gripper_kind == "art_gripper":
            try:
                # Lazy import — only require the standalone client when selected.
                # Try the pip-installed package first (recommended). Fall back
                # to a sys.path injection so users can run the source repo
                # without `pip install`.
                try:
                    from art_gripper_client import ArtGripperInterface  # type: ignore
                except ImportError:
                    _art_path = os.environ.get(
                        "ART_GRIPPER_PYPATH",
                        os.path.expanduser("~/Hyundai_motors_Gripper/python"),
                    )
                    if _art_path not in os.sys.path:
                        os.sys.path.insert(0, _art_path)
                    from art_gripper_client import ArtGripperInterface  # type: ignore

                art_host = os.environ.get("ART_GRIPPER_HOST", "127.0.0.1")
                art_port = int(os.environ.get("ART_GRIPPER_PORT", "50053"))
                print(f"[KIST] ART Gripper daemon 접속: {art_host}:{art_port}")
                self._gripper = ArtGripperInterface(ip_address=art_host, port=art_port)
                self._max_gripper_width = float(self._gripper.metadata.max_width)
                self._has_gripper = True
                print(f"[KIST] ART Gripper 연결 OK, max_width={self._max_gripper_width:.3f} m")
            except Exception as e:
                print(f"[KIST] ART Gripper 연결 실패 — gripper-less 모드로 진행: {e}")
                self._gripper = None
                self._max_gripper_width = 0.08
                self._has_gripper = False
        else:
            try:
                print(f"[KIST] Polymetis gripper 서버 접속: {self.polymetis_ip}:{gripper_port}")
                self._gripper = GripperInterface(ip_address=self.polymetis_ip, port=gripper_port)
                try:
                    w = float(self._gripper.metadata.max_width)
                    self._max_gripper_width = w if w > 0 else 0.08
                except Exception:
                    self._max_gripper_width = 0.08
                self._has_gripper = True
                print(f"[KIST] Franka Hand 연결 OK, max_width={self._max_gripper_width:.3f} m")
            except Exception as e:
                print(f"[KIST] Franka Hand 연결 실패 — gripper-less 모드로 진행: {e}")
                self._gripper = None
                self._max_gripper_width = 0.08
                self._has_gripper = False

        # Home pose 변형 — 환경 변수 KIST_HOME_VARIANT 로 전환:
        #   default / "droid"          : DROID 학습 데이터 그대로 (joint 7 = 0°). state 분포 일치.
        #                                 단 Franka Hand 가 45° 비틀려 보이고 wrist 카메라 이미지도
        #                                 회전될 수 있음.
        #   "franka_aligned"           : DROID home 의 joint 7 만 45°(π/4) 로 — Franka Hand 정자세.
        #                                 단 state(eef_9d rot6d, joint_position[6])가 45° OOD shift.
        # 다른 6개 joint 는 둘 다 동일.
        _home_variant = os.environ.get("KIST_HOME_VARIANT", "droid").lower()
        if _home_variant == "franka_aligned":
            self._home = torch.tensor([0.0, -np.pi/5, 0.0, -4*np.pi/5, 0.0, 3*np.pi/5, np.pi/4])
            print(f"[KIST] home variant = 'franka_aligned' (joint 7 = 45°, Franka Hand 정자세)")
        else:
            self._home = torch.tensor([0.0, -np.pi/5, 0.0, -4*np.pi/5, 0.0, 3*np.pi/5, 0.0])
            print(f"[KIST] home variant = 'droid' (joint 7 = 0°, DROID 학습 자세 그대로)")

        # ----- Cameras (3090 USB) -----
        # camera_serials: {"left": <int|None>, "right": <int|None>, "wrist": <int>}
        camera_serials = camera_serials or {}
        self._cam_serials = camera_serials
        self._cams: dict[int, sl.Camera] = {}
        for label, sn in camera_serials.items():
            if sn is None or sn in self._cams:
                continue
            cam = _open_zed(sn)
            if cam is not None:
                self._cams[int(sn)] = cam
        print(f"[KIST] 열린 카메라: {list(self._cams.keys())}")

        # ----- 첫 reset (home pose + start_joint_impedance) -----
        # DROID upstream RobotEnv는 __init__에서 do_reset=True로 자동 reset.
        # main_gr00t.py가 첫 rollout 전에 reset() 안 부르므로 여기서 호출 필수
        # (없으면 step()이 'Unable to update desired joint positions' 에러로 실패).
        print("[KIST] 첫 reset — home pose + start_joint_impedance ...")
        self.reset(randomize=False)
        print("[KIST] 첫 reset 완료. step() 호출 가능.")

    # ---- gym.Env compat (DROID main_gr00t.py 가 호출하는 메서드만) ----

    def reset(self, randomize: bool = False):
        """그리퍼 열기 + home joint 이동.

        이전 세션의 controller가 살아있으면 move_to_joint_positions가 조용히
        리턴만 하고 실제로 안 움직이는 버그가 있어서, 먼저 강제로 종료한다.
        """
        try:
            self._robot.terminate_current_policy()
            time.sleep(0.3)
        except Exception:
            pass
        if self._has_gripper:
            self._gripper.goto(
                width=self._max_gripper_width,
                speed=self.gripper_speed,
                force=self.gripper_force,
                blocking=True,
            )
        self._robot.move_to_joint_positions(self._home, time_to_go=4.0)
        # streaming joint control 모드 시작
        self._robot.start_joint_impedance()

    def get_observation(self) -> dict:
        # --- robot state (Polymetis gRPC 호출 1회) ---
        state = self._robot.get_robot_state()
        joint_positions = np.asarray(state.joint_positions, dtype=np.float64)
        try:
            ee_pose = self._robot.get_ee_pose()
            pos = np.asarray(ee_pose[0], dtype=np.float64)
            quat = np.asarray(ee_pose[1], dtype=np.float64)
            euler = R.from_quat(quat).as_euler("XYZ")
            cartesian_position = np.concatenate([pos, euler])
        except Exception:
            cartesian_position = np.zeros(6, dtype=np.float64)

        if self._has_gripper:
            try:
                g_state = self._gripper.get_state()
                gripper_position = 1.0 - (g_state.width / self._max_gripper_width)
            except Exception:
                gripper_position = 0.0
        else:
            gripper_position = 0.0

        # --- camera frames (BGRA) ---
        images: dict[str, np.ndarray] = {}
        rt = sl.RuntimeParameters()
        for sn, cam in self._cams.items():
            mat = sl.Mat()
            if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                images[f"{sn}_left"] = mat.get_data()  # H x W x 4 (BGRA)

        return {
            "image": images,
            "robot_state": {
                "cartesian_position": cartesian_position.tolist(),
                "joint_positions": joint_positions.tolist(),
                "gripper_position": float(gripper_position),
            },
        }

    def step(self, action: np.ndarray):
        """8D 액션: action[:7]=joint_position(rad), action[7]=gripper(0=open, 1=close)."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action.shape[0] == 8, f"action must be 8D got {action.shape}"
        joint_target = torch.tensor(action[:7], dtype=torch.float32)
        try:
            self._robot.update_desired_joint_positions(joint_target)
        except Exception:
            # streaming control이 안 떠있으면 blocking move로 폴백
            self._robot.move_to_joint_positions(joint_target, time_to_go=0.1)

        if self._has_gripper:
            cmd = float(action[7])
            target_width = max(0.0, min(self._max_gripper_width,
                                        self._max_gripper_width * (1.0 - cmd)))
            self._gripper.goto(
                width=target_width,
                speed=self.gripper_speed,
                force=self.gripper_force,
                blocking=False,
            )

    # ---- main_gr00t.py 가 사용하는 _extract_observation의 키 호환 보조 ----

    def close(self):
        for cam in self._cams.values():
            cam.close()
        try:
            self._robot.terminate_current_policy()
        except Exception:
            pass
