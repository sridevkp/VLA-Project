"""Brain-side scripted CAN pick-and-place policy used for M1 regression checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlarobot.backends.webots.config import (
    APPROACH_H,
    ARM_JOINTS,
    GRASP_Z,
    PLACE_BASE,
    PLACE_CLEAR,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import GripperCmd, GroundTruth, JointTarget, ScenarioSpec


@dataclass(frozen=True)
class ScriptedCanResult:
    seed: int
    success: bool
    rejections: int
    final_target_distance: float
    final_state: str


class ScriptedCanPolicy:
    """Drive the CAN object through the original waypoint sequence in joint space."""

    def __init__(self, kinematics: Kinematics | None = None) -> None:
        self.kinematics = kinematics or Kinematics()

    def execute(self, backend, seed: int) -> ScriptedCanResult:
        observation = backend.reset(ScenarioSpec(), seed)
        current_q = self._full_joint_vector(observation.robot.joint_pos)
        rejections = 0
        final_state = "INIT"

        can_pose = self._can_pose(backend.get_ground_truth())
        grasp_tip = can_pose + np.array((0.0, 0.0, GRASP_Z), dtype=float)
        pregrasp = grasp_tip + np.array((0.0, 0.0, APPROACH_H), dtype=float)
        place_tip = np.asarray(PLACE_BASE, dtype=float) + np.array(
            (0.0, 0.0, GRASP_Z + PLACE_CLEAR),
            dtype=float,
        )
        place_above = place_tip + np.array((0.0, 0.0, APPROACH_H), dtype=float)

        for name, target in (
            ("pregrasp", pregrasp),
            ("grasp", grasp_tip),
            ("lift", pregrasp),
            ("transit", place_above),
            ("place_descend", place_tip),
            ("release", place_tip),
            ("retreat", place_above),
            ("home", np.asarray((0.0, 0.0, 0.0), dtype=float)),
        ):
            next_q = self._move_to(backend, current_q, target)
            if next_q is None:
                return ScriptedCanResult(
                    seed=seed,
                    success=False,
                    rejections=rejections + 1,
                    final_target_distance=float("inf"),
                    final_state=final_state,
                )
            current_q = next_q
            final_state = name

            if name == "grasp":
                response = backend.apply(GripperCmd(0.0, force=50.0))
                if not response.accepted:
                    rejections += 1
                    return ScriptedCanResult(
                        seed=seed,
                        success=False,
                        rejections=rejections,
                        final_target_distance=float("inf"),
                        final_state=final_state,
                    )
                backend.step(1)
            elif name == "release":
                response = backend.apply(GripperCmd(0.045, force=50.0))
                if not response.accepted:
                    rejections += 1
                    return ScriptedCanResult(
                        seed=seed,
                        success=False,
                        rejections=rejections,
                        final_target_distance=float("inf"),
                        final_state=final_state,
                    )
                backend.step(1)

        final_ground_truth = backend.get_ground_truth()
        final_can = self._can_pose(final_ground_truth)
        final_distance = float(np.linalg.norm(final_can - np.asarray(PLACE_BASE, dtype=float)))
        return ScriptedCanResult(
            seed=seed,
            success=final_distance < 0.10,
            rejections=rejections,
            final_target_distance=final_distance,
            final_state=final_state,
        )

    def _move_to(self, backend, current_q: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        try:
            q, _, _ = self.kinematics.solve(target, current_q, restarts=True)
        except RuntimeError:
            return None

        step_count = max(2, int(np.ceil(np.linalg.norm(q - current_q) / 0.15)))
        observation = None
        for alpha in np.linspace(0.0, 1.0, step_count + 1)[1:]:
            waypoint = (1.0 - alpha) * current_q + alpha * q
            joint_target = {name: float(waypoint[self.kinematics.idx[name]]) for name in ARM_JOINTS}
            response = backend.apply(JointTarget(joint_target))
            if not response.accepted:
                return None
            observation = backend.step(1)

        return self._full_joint_vector(observation.robot.joint_pos)

    def _full_joint_vector(self, state: dict[str, float]) -> np.ndarray:
        vector = np.zeros(len(self.kinematics.chain.links), dtype=float)
        for name, value in state.items():
            if name in self.kinematics.idx:
                vector[self.kinematics.idx[name]] = float(value)
        return vector

    @staticmethod
    def _can_pose(ground_truth: GroundTruth) -> np.ndarray:
        can = ground_truth.object_poses.get("CAN")
        if can is None:
            return np.asarray((0.0, 0.0, 0.0), dtype=float)
        pose = can.position
        return np.asarray((pose.x, pose.y, pose.z), dtype=float)
