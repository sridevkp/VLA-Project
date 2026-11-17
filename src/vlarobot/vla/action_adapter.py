"""Decode physical 7-D VLA actions, solve bounded IK, and retain safety veto power."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vlarobot.core.types import (
    ApplyResult,
    EEPoseTarget,
    GripperCmd,
    JointTarget,
    Observation,
    Pose3D,
    RobotState,
    Vec3,
)
from vlarobot.execution.safety_monitor import SafetyMonitor


@dataclass(frozen=True)
class PoseSolution:
    joints: dict[str, float]
    position_error_m: float
    orientation_error_rad: float


class PoseSolver(Protocol):
    def solve(self, pose: Pose3D, state: RobotState) -> PoseSolution: ...


class ActionBackend(Protocol):
    def apply(self, action: JointTarget | GripperCmd) -> ApplyResult: ...

    def step(self, n_steps: int = 1) -> Observation: ...


def as_quaternion(values: np.ndarray) -> tuple[float, float, float, float]:
    """Check finite xyzw components and make the fixed-width message tuple explicit."""
    if np.shape(values) != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite xyzw components")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions, with left-multiplied base-frame deltas."""
    av, bv = a[:3], b[:3]
    return np.r_[a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - av @ bv]


def rotation_delta_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector in radians to an xyzw unit quaternion."""
    angle = float(np.linalg.norm(rotvec))
    if angle < np.finfo(float).eps:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.r_[rotvec * (math.sin(angle / 2) / angle), math.cos(angle / 2)]


def pose_delta(source: Pose3D, target: Pose3D) -> np.ndarray:
    """Commanded target minus observation, with base-frame rotation-vector delta (6 values)."""
    source_q = np.asarray(source.orientation, dtype=float)
    target_q = np.asarray(target.orientation, dtype=float)
    inverse = np.r_[-source_q[:3], source_q[3]]
    delta = quaternion_product(target_q, inverse)
    delta /= np.linalg.norm(delta)
    if delta[3] < 0:
        delta = -delta
    sine = float(np.linalg.norm(delta[:3]))
    rotvec = np.zeros(3) if sine < 1e-12 else delta[:3] * (2 * math.atan2(sine, delta[3]) / sine)
    return np.r_[target.position.as_array() - source.position.as_array(), rotvec]


@dataclass(frozen=True)
class AdaptedAction:
    joint_target: JointTarget
    gripper: GripperCmd
    target_pose: Pose3D
    position_error_m: float
    orientation_error_rad: float


class ActionAdapter:
    """The caller supplies physical unnormalized values; invalid outputs are never clipped."""

    def __init__(
        self,
        solver: PoseSolver,
        safety_monitor: SafetyMonitor,
        *,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> None:
        self.solver = solver
        self.safety_monitor = safety_monitor
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        if not all(
            math.isfinite(v) and v > 0 for v in (position_tolerance_m, orientation_tolerance_rad)
        ):
            raise ValueError("IK tolerances must be positive and finite")

    def decode(self, values: np.ndarray, state: RobotState) -> AdaptedAction:
        """Return validated joint/gripper targets; raise before any command on bad output."""
        values = np.asarray(values)
        if values.dtype.kind not in "fiu":
            raise ValueError("VLA action must contain numerical values, not booleans or text")
        values = values.astype(float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("VLA action must be seven finite physical values")
        # Restore exact configured endpoints after float32 serialization (45 mm
        # becomes 0.045000001788...). This is not general output clipping: the
        # next representable float32 value outside the bound is still rejected.
        for endpoint in self.safety_monitor.limits.gripper_opening:
            if values[6] == float(np.float32(endpoint)):
                values[6] = endpoint
        gripper = GripperCmd(float(values[6]))
        verdict = self.safety_monitor.validate(gripper, state)
        if not verdict.accepted:
            raise ValueError(verdict.reason)
        q = quaternion_product(
            rotation_delta_quaternion(values[3:6]), np.asarray(state.ee_pose.orientation)
        )
        q /= np.linalg.norm(q)
        position = state.ee_pose.position.as_array() + values[:3]
        target = Pose3D(Vec3(*map(float, position)), as_quaternion(q))
        # Workspace validation uses the existing safety monitor, before solving IK.
        verdict = self.safety_monitor.validate(
            EEPoseTarget(target, gripper.opening, self.safety_monitor.limits.command_horizon), state
        )
        if not verdict.accepted:
            raise ValueError(verdict.reason)
        solution = self.solver.solve(target, state)
        if (
            not math.isfinite(solution.position_error_m)
            or not math.isfinite(solution.orientation_error_rad)
            or solution.position_error_m < 0
            or solution.orientation_error_rad < 0
            or solution.position_error_m > self.position_tolerance_m
            or solution.orientation_error_rad > self.orientation_tolerance_rad
        ):
            raise ValueError("VLA IK pose residual exceeds configured tolerances")
        joints = JointTarget(solution.joints)
        verdict = self.safety_monitor.validate(joints, state)
        if not verdict.accepted:
            raise ValueError(verdict.reason)
        return AdaptedAction(
            joints, gripper, target, solution.position_error_m, solution.orientation_error_rad
        )

    def apply(self, backend: ActionBackend, values: np.ndarray, state: RobotState) -> ApplyResult:
        """Apply only a wholly validated proposal; backend safety can still veto either command."""
        try:
            proposal = self.decode(values, state)
        except (ValueError, RuntimeError) as error:
            return ApplyResult(False, str(error))
        for command in (proposal.joint_target, proposal.gripper):
            verdict = self.safety_monitor.validate(command, state)
            if not verdict.accepted:
                return verdict
            result = backend.apply(command)
            if not result.accepted:
                return result
        return ApplyResult(True)

    def execute_chunk(
        self,
        backend: ActionBackend,
        actions: np.ndarray,
        observation: Observation,
        *,
        steps_per_action: int,
        max_actions: int,
    ) -> tuple[ApplyResult, int]:
        """Execute at explicit lockstep cadence; veto stops the chunk without later commands."""
        actions = np.asarray(actions)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or not 0 < len(actions) <= max_actions
            or actions.dtype.kind not in "fiu"
            or not np.all(np.isfinite(actions))
            or steps_per_action <= 0
        ):
            return ApplyResult(False, "malformed VLA action chunk or lockstep cadence"), 0
        executed = 0
        for action in actions:
            result = self.apply(backend, action, observation.robot)
            if not result.accepted:
                return result, executed
            observation = backend.step(steps_per_action)
            executed += 1
        return ApplyResult(True), executed
