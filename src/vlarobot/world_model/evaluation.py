"""Quantity-separated continuous trajectory errors, never probabilities or task success."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vlarobot.core.types import WorldState
from vlarobot.world_model.analytic import ObjectWorldModel
from vlarobot.world_model.rollout import Trajectory


def prediction_errors(
    predicted: Trajectory, observed: Sequence[WorldState]
) -> dict[str, float | int]:
    """Require aligned complete horizons and identities; report SI errors per physical quantity."""
    if len(predicted.states) != len(observed) or len(observed) < 2:
        raise ValueError("aligned nonempty trajectory observations required")
    robot, joints, objects, orientation = [], [], [], []
    correction = ObjectWorldModel()
    for a, b in zip(predicted.states[1:], observed[1:], strict=True):
        delta = a.robot.ee_pose.position.as_array() - b.robot.ee_pose.position.as_array()
        robot.append(float(np.linalg.norm(delta)))
        if set(a.robot.joint_pos) != set(b.robot.joint_pos) or not a.robot.joint_pos:
            raise ValueError("aligned robot joint identities required")
        joints.extend(a.robot.joint_pos[n] - b.robot.joint_pos[n] for n in a.robot.joint_pos)
        error = correction.correct(a, b)
        objects.append(error.position_error)
        orientation.append(error.orientation_error)
    if not np.isfinite(np.r_[robot, joints, objects, orientation]).all():
        raise ValueError("nonfinite observed trajectory errors")
    return {
        "prediction_steps": len(observed) - 1,
        "robot_position_rmse_m": float(np.sqrt(np.mean(np.square(robot)))),
        "robot_position_mae_m": float(np.mean(np.abs(robot))),
        "joint_rmse_rad": float(np.sqrt(np.mean(np.square(joints)))),
        "joint_mae_rad": float(np.mean(np.abs(joints))),
        "object_position_rmse_m": float(np.sqrt(np.mean(np.square(objects)))),
        "object_orientation_rmse_rad": float(np.sqrt(np.mean(np.square(orientation)))),
    }
