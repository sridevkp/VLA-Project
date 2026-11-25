"""Deterministic velocity-limited arm tracking surrogate, with injected registered FK."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

from vlarobot.core.types import JointTarget, Pose3D, RobotState


class VelocityLimitedKinematics:
    """Nominal actuator tracking only; contact, grasp and object motion are not predicted."""

    def __init__(
        self, velocity_rad_s: dict[str, float], forward: Callable[[dict[str, float]], Pose3D]
    ) -> None:
        if not velocity_rad_s or not all(
            math.isfinite(v) and v > 0 for v in velocity_rad_s.values()
        ):
            raise ValueError("explicit positive commanded joint speeds required")
        self.velocity = dict(velocity_rad_s)
        self.forward = forward

    def step(self, state: RobotState, target: JointTarget, dt_s: float) -> RobotState:
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("explicit positive simulation step required")
        result = copy.deepcopy(state)
        result.joint_vel = {name: 0.0 for name in state.joint_pos}
        for name, value in target.q.items():
            if (
                name not in self.velocity
                or name not in state.joint_pos
                or not math.isfinite(value)
                or not math.isfinite(state.joint_pos[name])
            ):
                raise ValueError("unknown or nonfinite surrogate joint target/state")
            delta = value - state.joint_pos[name]
            bound = self.velocity[name] * dt_s
            move = max(-bound, min(bound, delta))
            result.joint_pos[name] += move
            result.joint_vel[name] = move / dt_s
        result.ee_pose = self.forward(result.joint_pos)
        return result
