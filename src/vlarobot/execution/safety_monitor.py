"""Hard-constraint action validation shared by backend clients and simulators."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from vlarobot.core.types import (
    Action,
    ApplyResult,
    BaseGoal,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    JointTarget,
    Pose2D,
    Pose3D,
    RobotState,
    Vec3,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyLimits:
    joint_position: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_velocity: dict[str, float] = field(default_factory=dict)
    workspace_min: tuple[float, float, float] = (0.0, -1.0, 0.0)
    workspace_max: tuple[float, float, float] = (1.5, 1.0, 1.8)
    gripper_opening: tuple[float, float] = (0.0, 0.045)
    gripper_force_max: float = 100.0
    command_horizon: float = 1.0

    def __post_init__(self) -> None:
        for name, bounds in self.joint_position.items():
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"Invalid position limits for joint {name!r}.")
        for name, velocity in self.joint_velocity.items():
            if not math.isfinite(velocity) or velocity <= 0.0:
                raise ValueError(f"Invalid velocity limit for joint {name!r}.")
        if (
            len(self.workspace_min) != 3
            or len(self.workspace_max) != 3
            or not self._finite(*self.workspace_min, *self.workspace_max)
            or any(
                lower >= upper
                for lower, upper in zip(self.workspace_min, self.workspace_max, strict=True)
            )
        ):
            raise ValueError("Invalid workspace bounds.")
        if (
            len(self.gripper_opening) != 2
            or not self._finite(*self.gripper_opening)
            or self.gripper_opening[0] > self.gripper_opening[1]
        ):
            raise ValueError("Invalid gripper opening limits.")
        if not math.isfinite(self.gripper_force_max) or self.gripper_force_max < 0.0:
            raise ValueError("Invalid gripper force limit.")
        if not math.isfinite(self.command_horizon) or self.command_horizon <= 0.0:
            raise ValueError("command_horizon must be positive and finite.")

    @staticmethod
    def _finite(*values: float) -> bool:
        return all(math.isfinite(value) for value in values)


class SafetyMonitor:
    """Reject malformed or out-of-bounds commands before they reach a backend."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    def validate(self, action: Action, state: RobotState) -> ApplyResult:
        """Return rejection details unless all applicable hard constraints pass."""
        if isinstance(action, JointTarget):
            return self._validate_joint_target(action, state)
        if isinstance(action, EEPoseTarget):
            if not isinstance(action.pose, Pose3D) or not isinstance(action.pose.position, Vec3):
                return self._reject("end-effector target pose is malformed")
            position = action.pose.position
            orientation = action.pose.orientation
            if (
                not isinstance(orientation, tuple)
                or len(orientation) != 4
                or not self._finite_pose(position.x, position.y, position.z, *orientation)
            ):
                return self._reject("end-effector pose must contain finite values")
            orientation_norm = math.sqrt(sum(value * value for value in orientation))
            if not math.isclose(orientation_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                return self._reject("end-effector orientation must be a unit quaternion")
            workspace = self._validate_workspace((position.x, position.y, position.z))
            if not workspace.accepted:
                return workspace
            if not self._is_finite_number(action.duration) or action.duration <= 0.0:
                return self._reject("end-effector target duration must be positive and finite")
            if action.gripper is not None:
                result = self._validate_gripper(action.gripper)
                if not result.accepted:
                    return result
            return ApplyResult(True)
        if isinstance(action, EEDelta):
            if not isinstance(action.dpos, Vec3) or not isinstance(action.drot, Vec3):
                return self._reject("end-effector delta is malformed")
            values = (
                action.dpos.x,
                action.dpos.y,
                action.dpos.z,
                action.drot.x,
                action.drot.y,
                action.drot.z,
                action.gripper,
            )
            if not self._finite_pose(*values):
                return self._reject("end-effector delta must contain finite values")
            position = state.ee_pose.position
            workspace = self._validate_workspace(
                (position.x + action.dpos.x, position.y + action.dpos.y, position.z + action.dpos.z)
            )
            return workspace if not workspace.accepted else self._validate_gripper(action.gripper)
        if isinstance(action, BaseGoal):
            if not isinstance(action.pose, Pose2D):
                return self._reject("base goal is malformed")
            values = (action.pose.x, action.pose.y, action.pose.yaw)
            if not self._finite_pose(*values):
                return self._reject("base goal must contain finite values")
            return ApplyResult(True)
        if isinstance(action, GripperCmd):
            if action.force is not None and (
                not self._is_finite_number(action.force)
                or action.force < 0.0
                or action.force > self.limits.gripper_force_max
            ):
                return self._reject("gripper force is outside configured limits")
            return self._validate_gripper(action.opening)
        return self._reject(f"unsupported action type: {type(action).__name__}")

    def _validate_joint_target(self, action: JointTarget, state: RobotState) -> ApplyResult:
        if not isinstance(action.q, dict) or not action.q:
            return self._reject("joint target must include at least one joint")
        for name, target in action.q.items():
            if not isinstance(name, str) or not self._is_finite_number(target):
                return self._reject(f"joint target for {name} must be finite")
            limits = self.limits.joint_position.get(name)
            if limits is None:
                return self._reject(f"joint {name} has no configured position limits")
            if target < limits[0] or target > limits[1]:
                return self._reject(f"joint target for {name} violates position limits")
            current = state.joint_pos.get(name)
            if current is None or not self._is_finite_number(current):
                return self._reject(f"current position for {name} is unavailable")
            max_velocity = self.limits.joint_velocity.get(name)
            if (
                max_velocity is None
                or not self._is_finite_number(max_velocity)
                or max_velocity <= 0.0
            ):
                return self._reject(f"joint {name} has no valid velocity limit")
            implied_velocity = abs(target - current) / self.limits.command_horizon
            if implied_velocity > max_velocity:
                return self._reject(f"joint target for {name} violates velocity limits")
        return ApplyResult(True)

    def _validate_workspace(self, position: tuple[float, float, float]) -> ApplyResult:
        if not self._finite_pose(*position):
            return self._reject("position must contain finite values")
        for axis, value, lower, upper in zip(
            ("x", "y", "z"),
            position,
            self.limits.workspace_min,
            self.limits.workspace_max,
            strict=True,
        ):
            if value < lower or value > upper:
                return self._reject(
                    f"position {axis}={value} is outside workspace [{lower}, {upper}]"
                )
        return ApplyResult(True)

    def _validate_gripper(self, opening: float) -> ApplyResult:
        if not self._is_finite_number(opening):
            return self._reject("gripper opening must be finite")
        lower, upper = self.limits.gripper_opening
        if opening < lower or opening > upper:
            return self._reject("gripper opening is outside configured limits")
        return ApplyResult(True)

    @staticmethod
    def _finite_pose(*values: float) -> bool:
        return all(SafetyMonitor._is_finite_number(value) for value in values)

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    @staticmethod
    def _reject(reason: str) -> ApplyResult:
        _LOGGER.warning("Safety validation rejected action: %s", reason)
        return ApplyResult(False, reason)
