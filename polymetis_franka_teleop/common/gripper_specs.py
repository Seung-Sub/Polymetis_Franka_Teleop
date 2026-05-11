"""Per-gripper hardware specification registry.

A single source of truth for gripper-specific constants (width envelope,
default force, TCP offset, control port, controller class). Add a new
gripper backend in three places:

    1. Implement a controller subclass (see art_gripper_controller.py /
       franka_gripper_controller.py for the contract).
    2. Add an entry to ``GRIPPER_SPECS`` below.
    3. Register the controller class in ``get_controller_class``.

After that the new backend is selectable via ``--gripper_backend <name>``
from the CLI; no further env / wrapper changes needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type


@dataclass(frozen=True)
class GripperSpec:
    """Hardware envelope + control wiring for one gripper model."""

    name: str
    open_width: float       # m  — full-open jaw separation
    close_width: float      # m  — full-close jaw separation (libfranka exception below this for franka_hand)
    default_force: float    # N  — preferred grasp force
    tcp_offset: float       # m  — Z offset from flange to TCP (used to align cartesian targets)
    host: str               # control endpoint hostname
    port: int               # control endpoint port
    protocol: str           # 'tcp' (ART daemon) | 'grpc' (fairo polymetis)

    @property
    def width_range(self) -> float:
        return self.open_width - self.close_width


# ============================================================
# Registry — add a row when wiring up a new gripper backend.
# ============================================================
GRIPPER_SPECS: dict[str, GripperSpec] = {
    # Hyundai Motors ART (KIST default).
    # Talks to art-gripper-daemon over raw TCP on pro4000 :50053.
    # close_width=0.0 m: ART firmware accepts full mechanical close, allowing
    # the jaws to pinch thin objects (paper, cards) that wouldn't trigger
    # contact at a 5 mm gap.
    'art': GripperSpec(
        name='art',
        open_width=0.095,
        close_width=0.0,
        default_force=60.0,
        tcp_offset=0.216,
        host='127.0.0.1',
        port=50053,
        protocol='tcp',
    ),

    # Franka Hand (Franka factory gripper).
    # Talks to fairo-polymetis GripperServerLauncher over gRPC on NUC :50052
    # (port from /home/kist/fairo/polymetis/polymetis/conf/launch_gripper.yaml).
    # close_width=0.005 m: libfranka raises on width=0.0; 5 mm safety margin.
    'franka': GripperSpec(
        name='franka',
        open_width=0.075,
        close_width=0.005,
        default_force=30.0,
        tcp_offset=0.1034,
        host='192.168.1.12',
        port=50052,
        protocol='grpc',
    ),
}


def get_spec(backend: str) -> GripperSpec:
    """Return the registered spec, or raise ValueError listing valid names."""
    try:
        return GRIPPER_SPECS[backend]
    except KeyError:
        valid = ', '.join(sorted(GRIPPER_SPECS))
        raise ValueError(
            f'Unknown gripper_backend={backend!r}. Valid: {valid}.'
        )


def get_controller_class(backend: str) -> Type:
    """Lazy import of the controller class — avoids hard import of one
    backend's deps (e.g. polymetis for franka) when the other is in use.
    """
    spec = get_spec(backend)
    if spec.name == 'art':
        from polymetis_franka_teleop.real_world.art_gripper_controller import (
            ArtGripperController,
        )
        return ArtGripperController
    if spec.name == 'franka':
        from polymetis_franka_teleop.real_world.franka_gripper_controller import (
            FrankaGripperController,
        )
        return FrankaGripperController
    raise ValueError(f'No controller class registered for backend={spec.name!r}')
