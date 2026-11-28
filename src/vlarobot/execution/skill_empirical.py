"""Initial candidate utility fit using demonstrated and measured TRAIN decisions."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.webots.config import ARM_JOINTS, CAN_RADIUS_M, GRIPPER_OPEN, SETTINGS
from vlarobot.backends.worldmind_bindings import (
    _corrected_forward,
    _full_joint_vector,
    corrected_kinematics,
    runtime_safety_limits,
)
from vlarobot.core.types import (
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.execution.demo_collection import write_json
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.skill_demonstrations import partition_demonstrations
from vlarobot.execution.skill_selection import (
    ScoringInput,
    UtilityExample,
    fit_utility,
    predict_candidates,
)
from vlarobot.execution.skills import Skill, SkillAction
from vlarobot.execution.worldmind_pipeline import _inverse_pose
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate, Trajectory, compose

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


def _forecast(row: UtilityExample, robot: dict) -> UtilityExample:
    state = replace(
        row.state,
        robot=replace(
            row.state.robot,
            joint_pos=robot["joint_pos"],
            joint_vel=robot["joint_vel"],
        ),
    )
    monitor = SafetyMonitor(runtime_safety_limits())
    kin = corrected_kinematics()
    kin.set_seed(0)

    def inverse(pose: Pose3D, current: RobotState) -> JointTarget | None:
        try:
            q, error, tilt = kin.solve(
                pose.position.as_array(), _full_joint_vector(current.joint_pos), restarts=True
            )
        except RuntimeError:
            return None
        if error > SETTINGS.ik_tol or tilt > SETTINGS.max_tilt_deg:
            return None
        return JointTarget({name: float(q[kin.idx[name]]) for name in ARM_JOINTS})

    can_z = state.scene.objects[0].pose.position.z
    surrogate = AnalyticSurrogate(
        VelocityLimitedKinematics(monitor.limits.joint_velocity, _corrected_forward),
        inverse=inverse,
        gripper_speed_m_s=SETTINGS.finger_speed,
        gravity_m_s2=9.81,
        support_z_m={"CAN": can_z},
        validate=lambda a, r: monitor.validate(a, r).accepted,
    )
    candidate = predict_candidates(
        state, (row.candidate.action,), surrogate, horizon_s=1.0, open_position_m=GRIPPER_OPEN
    )[0]
    return replace(row, state=state, candidate=candidate)


def _matched_models(rows: list[UtilityExample], fit_groups: set[str]) -> dict:
    matched = [
        row
        for row in rows
        if len(row.candidate.trajectory.states) > 1
        and set(row.candidate.trajectory.hard_reasons) <= {"unverified-arm-sweep"}
    ]
    if not any(row.scene_group in fit_groups for row in matched):
        return {"status": "no complete forecasts in fit groups"}
    context = fit_utility(matched, train_groups=fit_groups, prediction_context=True, ridge=1.0)
    baseline = fit_utility(matched, train_groups=fit_groups, prediction_context=False, ridge=1.0)
    futures = [row.candidate.trajectory.states[-1] for row in matched]
    shift = int(np.random.default_rng(0).integers(1, len(futures))) if len(futures) > 1 else 0
    shuffled = np.roll(np.arange(len(futures)), shift)
    records = []
    for index, row in enumerate(matched):
        state, action = row.state, row.candidate.action
        records.append(
            {
                "scene_group": row.scene_group,
                "skill": action.skill.value,
                "target_base_m": None
                if action.target_base is None
                else action.target_base.as_array().tolist(),
                "measured_utility": row.utility,
                "fit_record": row.scene_group in fit_groups,
                "strict_prediction_reasons": row.candidate.trajectory.hard_reasons,
                "score_no_context": baseline.score(state, action, None),
                "score_predicted_context": context.score(state, action, futures[index]),
                "score_shuffled_context": context.score(state, action, futures[shuffled[index]]),
                "score_controlled_no_change": context.score(state, action, state),
            }
        )
    return {
        "context_model": {
            "mean": context.mean.tolist(),
            "scale": context.scale.tolist(),
            "coefficients": context.coefficients.tolist(),
            "intercept": context.intercept,
        },
        "matched_no_context_model": {
            "mean": baseline.mean.tolist(),
            "scale": baseline.scale.tolist(),
            "coefficients": baseline.coefficients.tolist(),
            "intercept": baseline.intercept,
        },
        "records": records,
        "forecast_excluded": [
            {
                "skill": row.candidate.action.skill.value,
                "reasons": row.candidate.trajectory.hard_reasons,
            }
            for row in rows
            if set(row.candidate.trajectory.hard_reasons) - {"unverified-arm-sweep"}
        ],
    }


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
    forecasts = []
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
            forecasts.append(_forecast(outcomes[-1], robot))
    if not any(row.scene_group in fit_groups for row in outcomes):
        raise ValueError("measured alternative-action outcomes required in fit groups")
    model = fit_utility(
        examples + outcomes, train_groups=fit_groups, prediction_context=False, ridge=1.0
    )
    result = {
        "identity": "v4-initial-empirical-nominal-no-context",
        "partition": partition,
        "measured_only_forecast_comparison": _matched_models(forecasts, fit_groups),
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
