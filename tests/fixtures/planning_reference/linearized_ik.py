"""Deterministic per-chunk Jacobian proposal with exact FK residual rejection."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np

from vlarobot.core.types import JointTarget, Pose3D, RobotState
from vlarobot.world_model.rollout import quaternion_product


def pose_error(target: Pose3D, actual: Pose3D) -> np.ndarray:
    """Base-frame translation and shortest base-left orientation rotation vector."""
    a, b = np.asarray(target.orientation), np.asarray(actual.orientation)
    q = quaternion_product(a, np.r_[-b[:3], b[3]])
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    rotation = q[:3] * (2 * math.atan2(length, q[3]) / length if length else 2.0)
    return np.r_[target.position.as_array() - actual.position.as_array(), rotation]


class LinearizedIK:
    """Planning approximation only. No backend import or replacement of execution IK.

    Reset once per candidate chunk makes predictions independent of earlier candidates.
    Refresh after leaving the local joint trust region or a failed exact-FK check.
    """

    def __init__(
        self,
        forward: Callable[[dict[str, float]], Pose3D],
        *,
        bounds: Mapping[str, tuple[float, float]],
        max_joint_step_rad: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        difference_step_rad: float,
        damping: float,
    ) -> None:
        values = (
            max_joint_step_rad,
            position_tolerance_m,
            orientation_tolerance_rad,
            difference_step_rad,
            damping,
        )
        if not bounds or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("explicit finite positive numerical/safety settings required")
        if any(
            not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1]
            for pair in bounds.values()
        ):
            raise ValueError("ordered finite joint bounds required")
        self.forward, self.names, self.bounds = forward, tuple(bounds), dict(bounds)
        self.trust, self.position_tol, self.rotation_tol, self.eps, self.damping = values
        self.reference: dict[str, float] | None = None
        self.jacobian: np.ndarray | None = None
        self.refreshes = 0

    def reset(self, state: RobotState) -> None:
        self.reference = None
        self.jacobian = None
        self.refreshes = 0
        self._refresh(state.joint_pos)

    def _refresh(self, joints: dict[str, float]) -> None:
        if any(n not in joints or not math.isfinite(joints[n]) for n in self.names):
            raise ValueError("finite complete joint seed required")
        columns = []
        for name in self.names:
            left, right = dict(joints), dict(joints)
            left[name] -= self.eps
            right[name] += self.eps
            columns.append(pose_error(self.forward(right), self.forward(left)) / (2 * self.eps))
        self.jacobian = np.column_stack(columns)
        self.reference = dict(joints)
        self.refreshes += 1

    def solve(self, target: Pose3D, state: RobotState) -> JointTarget | None:
        current = dict(state.joint_pos)
        if self.reference is None or any(
            abs(current[n] - self.reference[n]) > self.trust for n in self.names
        ):
            self._refresh(current)
        for attempt in range(2):
            error = pose_error(target, self.forward(current))
            assert self.jacobian is not None
            j = self.jacobian
            delta = j.T @ np.linalg.solve(j @ j.T + self.damping * np.eye(6), error)
            q = dict(current)
            for i, name in enumerate(self.names):
                low, high = self.bounds[name]
                q[name] = float(
                    np.clip(current[name] + np.clip(delta[i], -self.trust, self.trust), low, high)
                )
            residual = pose_error(target, self.forward(q))
            if (
                np.isfinite(residual).all()
                and np.linalg.norm(residual[:3]) <= self.position_tol
                and np.linalg.norm(residual[3:]) <= self.rotation_tol
            ):
                return JointTarget({n: q[n] for n in self.names})
            if attempt == 0:
                self._refresh(current)
        return None
