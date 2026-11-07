"""Shared, simulator-independent robot and task data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
RGBArray: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray((self.x, self.y, self.z), dtype=np.float64)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Pose3D:
    position: Vec3
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraInfo:
    K: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    T_base_cam: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    width: int
    height: int


@dataclass
class RobotState:
    joint_pos: dict[str, float]
    joint_vel: dict[str, float]
    base_pose: Pose2D
    ee_pose: Pose3D
    gripper_opening: float
    gripper_holding: bool | None = None
    base_pose_world: Pose3D | None = None


@dataclass
class Observation:
    t: float
    rgb: RGBArray
    depth: NDArray[np.float32] | None
    cam_info: CameraInfo
    robot: RobotState


@dataclass(frozen=True)
class ScenarioSpec:
    name: str = "default"
    objects: tuple[dict[str, object], ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth:
    object_poses: dict[str, Pose3D] = field(default_factory=dict)
    contacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApplyResult:
    accepted: bool
    reason: str | None = None


class Action:
    """Base type for backend-independent actions."""


@dataclass(frozen=True)
class EEPoseTarget(Action):
    pose: Pose3D
    gripper: float | None
    duration: float


@dataclass(frozen=True)
class EEDelta(Action):
    dpos: Vec3
    drot: Vec3
    gripper: float


@dataclass(frozen=True)
class JointTarget(Action):
    q: dict[str, float]


@dataclass(frozen=True)
class BaseGoal(Action):
    pose: Pose2D


@dataclass(frozen=True)
class GripperCmd(Action):
    opening: float
    force: float | None = None


@dataclass
class ActionChunk:
    actions: list[Action]
    dt: float
    source: str


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass
class Mask:
    data: NDArray[np.bool_]


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    label: str
    pose: Pose3D
    dimensions: Vec3
    confidence: float


@dataclass
class SceneState:
    objects: list[ObjectState] = field(default_factory=list)


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    target: str
    destination: str
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Clarification:
    question: str
    options: tuple[str, ...] = ()


@dataclass
class WorldState:
    scene: SceneState
    robot: RobotState


@dataclass(frozen=True)
class PredictionError:
    position_error: float
    orientation_error: float


@dataclass(frozen=True)
class Outcome:
    success_probability: float
    collision_probability: float
    drop_probability: float = 0.0
    duration: float = 0.0


@dataclass(frozen=True)
class RiskReport:
    vetoed: bool
    reasons: tuple[str, ...] = ()
    p_success: float = 0.0
    p_collision: float = 0.0
    p_drop: float = 0.0


@dataclass(frozen=True)
class Decision:
    selected_index: int | None
    rationale: str


@dataclass(frozen=True)
class VerifyResult:
    success: bool
    reason: str = ""
