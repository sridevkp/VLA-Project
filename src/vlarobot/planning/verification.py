"""Numeric placement/HOME verification and one-retry outcome, with explicit frozen limits."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from vlarobot.core.types import Pose3D, TaskSpec, VerifyResult, WorldState
from vlarobot.risk.gate import RecoveryState
from vlarobot.world_model.rollout import rotate


class PlacementVerifier:
    """Use observed poses only; a predicted placement or HOME alone cannot complete a task."""

    def __init__(
        self,
        *,
        destination_base: Pose3D,
        rest_center_base: Pose3D,
        up_axis_base: tuple[float, float, float],
        home_joints: Mapping[str, float],
        distance_tolerance_m: float,
        height_tolerance_m: float,
        tilt_limit_rad: float,
        home_tolerance_rad: float,
    ) -> None:
        values = (distance_tolerance_m, height_tolerance_m, tilt_limit_rad, home_tolerance_rad)
        if any(not math.isfinite(v) or v <= 0 for v in values) or not home_joints:
            raise ValueError(
                "explicit positive frozen verification limits and HOME joints required"
            )
        up = np.asarray(up_axis_base)
        if not np.isfinite(up).all() or not np.isclose(up @ up, 1):
            raise ValueError("observed base-frame world-up unit vector required")
        self.destination, self.rest, self.up = destination_base, rest_center_base, up
        self.home = dict(home_joints)
        self.distance_tol, self.height_tol, self.tilt_tol, self.home_tol = values
        self.recovery = RecoveryState()

    def check(self, task: TaskSpec, predicted: WorldState, observed: WorldState) -> VerifyResult:
        matches = [o for o in observed.scene.objects if o.object_id == task.target]
        if len(matches) != 1:
            return VerifyResult(False, "target pose missing or ambiguous")
        obj = matches[0]
        position = obj.pose.position.as_array()
        q = np.asarray(obj.pose.orientation)
        if not np.isfinite(np.r_[position, q]).all() or not np.isclose(q @ q, 1):
            return VerifyResult(False, "target pose nonfinite or malformed")
        if any(
            n not in observed.robot.joint_pos or not math.isfinite(observed.robot.joint_pos[n])
            for n in self.home
        ):
            return VerifyResult(False, "HOME joint feedback unavailable")
        distance = float(np.linalg.norm(position - self.destination.position.as_array()))
        height = abs(float((position - self.rest.position.as_array()) @ self.up))
        tilt = float(np.arccos(np.clip(rotate(q, np.array([0.0, 0.0, 1.0])) @ self.up, -1, 1)))
        home = max(abs(observed.robot.joint_pos[n] - v) for n, v in self.home.items())
        success = (
            distance <= self.distance_tol
            and height <= self.height_tol
            and tilt < self.tilt_tol
            and home <= self.home_tol
        )
        return VerifyResult(
            success,
            f"distance_m={distance:.9g}; height_error_m={height:.9g}; "
            f"tilt_rad={tilt:.9g}; home_max_rad={home:.9g}",
        )

    def next_action(self, result: VerifyResult) -> str:
        return self.recovery.verify(result.success)
