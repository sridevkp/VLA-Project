"""Initial candidate utility fit using demonstrated and measured TRAIN decisions."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.webots.config import CAN_RADIUS_M
from vlarobot.core.types import (
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.skill_demonstrations import partition_demonstrations
from vlarobot.execution.skill_selection import ScoringInput, UtilityExample, fit_utility
from vlarobot.execution.skills import Skill, SkillAction
from vlarobot.execution.worldmind_pipeline import _inverse_pose
from vlarobot.world_model.rollout import Trajectory, compose

_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _example(
    group: str,
    skill: str,
    target: Vec3 | None,
    tool: Vec3,
    can: Vec3,
    opening: float,
    holding: bool | None,
    utility: float,
    source: str,
) -> UtilityExample:
    state = WorldState(
        SceneState([ObjectState("CAN", "can", Pose3D(can, _IDENTITY), Vec3(0, 0, 0), 1.0)]),
        RobotState({}, {}, Pose2D(0, 0, 0), Pose3D(tool, _IDENTITY), opening, holding),
    )
    candidate = ScoringInput(
        SkillAction(Skill(skill), target), Trajectory((state,), 0.0, (), (), ()), ()
    )
    return UtilityExample(group, state, candidate, utility, source)


def train_empirical(dataset: Path, output: Path, measured: list[Path]) -> dict[str, Any]:
    """Fit a no-context utility baseline before candidate forecasts are available."""
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if data["split"] != "train":
        raise ValueError("TRAIN input required")
    partition = partition_demonstrations(data["decisions"])
    fit_groups = set(partition["fit_groups"])
    examples = []
    for row in data["decisions"]:
        before = row["observed_before"]
        target = row["target_base_m"]
        examples.append(
            _example(
                row["scene_group"],
                row["skill"],
                Vec3(*target) if target is not None else None,
                Vec3(*before["tool_base_m"]),
                Vec3(*before["oracle_can_base_m"]),
                before["gripper_opening_m"],
                before["gripper_holding"],
                1.0,
                "demonstration",
            )
        )
    outcomes = []
    for path in measured:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["split"] != "train" or not report["isolated"]:
            raise ValueError(f"isolated TRAIN evidence required: {path}")
        group = report["scene"].rsplit("-s", 1)[0]
        for row in report["rows"]:
            if row["measurement_source"] != "simulator-observed":
                raise ValueError(f"measured outcome required: {path}")
            if row["skill"] == Skill.APPROACH and "tip_can_xy_after_m" in row:
                utility = CAN_RADIUS_M / (CAN_RADIUS_M + row["tip_can_xy_after_m"])
            elif row.get("empty_grasp_verified") is True:
                utility = 0.0
            else:
                continue
            robot = row["robot_before"]
            base = robot["base_pose_world"]
            transform = _inverse_pose(Pose3D(Vec3(**base["position"]), tuple(base["orientation"])))
            can_world = Pose3D(Vec3(**row["can_before_world_m"]), _IDENTITY)
            target = row["target_base_m"]
            outcomes.append(
                _example(
                    group,
                    row["skill"],
                    Vec3(**target) if target is not None else None,
                    Vec3(**row["tool_before_base_m"]),
                    compose(transform, can_world).position,
                    robot["gripper_opening"],
                    robot["gripper_holding"],
                    utility,
                    "simulator-observed",
                )
            )
    if not any(row.scene_group in fit_groups for row in outcomes):
        raise ValueError("measured alternative-action outcomes required in fit groups")
    model = fit_utility(
        examples + outcomes, train_groups=fit_groups, prediction_context=False, ridge=1.0
    )
    result = {
        "identity": "v4-initial-empirical-nominal-no-context",
        "partition": partition,
        "demonstration_fit_n": sum(row.scene_group in fit_groups for row in examples),
        "measured_fit_n": sum(row.scene_group in fit_groups for row in outcomes),
        "observed_reward": (
            "CAN radius / (CAN radius + final planar error), or 0 for verified empty close"
        ),
        "demonstration_preference_label": 1.0,
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "coefficients": model.coefficients.tolist(),
        "intercept": model.intercept,
        "prediction_context": False,
        "fit_demonstration_mean_score": float(
            np.mean(
                [
                    model.score(row.state, row.candidate.action, None)
                    for row in examples
                    if row.scene_group in fit_groups
                ]
            )
        ),
        "measured_scores": [
            {
                "group": row.scene_group,
                "skill": row.candidate.action.skill.value,
                "fitting_record": row.scene_group in fit_groups,
                "utility": row.utility,
                "score": model.score(row.state, row.candidate.action, None),
            }
            for row in outcomes
        ],
        "privileged_input": "oracle CAN position in the robot base frame",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
