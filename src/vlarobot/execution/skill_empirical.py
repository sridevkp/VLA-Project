"""Initial candidate utility fit using demonstrated and measured TRAIN decisions."""

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    CAN_RADIUS_M,
    CLOSE_ALIGN_TOL,
    GRIPPER_OPEN,
    LIFT_MIN,
    PLACE_TOLERANCE_M,
    PLACE_WORLD,
    SETTINGS,
)
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
_RESET_MATCH_TOLERANCE = 1e-6


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
    return replace(row, state=state, candidate=forecast_candidate(state, row.candidate.action))


def forecast_candidate(state: WorldState, action: SkillAction) -> ScoringInput:
    """Generate the same analytic context in empirical fitting and live decisions."""
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
    return predict_candidates(
        state, (action,), surrogate, horizon_s=1.0, open_position_m=GRIPPER_OPEN
    )[0]


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
    pairs = defaultdict(list)
    for index, row in enumerate(matched):
        state, action = row.state, row.candidate.action
        records.append(
            {
                "scene_group": row.scene_group,
                "pair_id": row.pair_id,
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
        if row.pair_id is not None and action.skill is Skill.APPROACH:
            pairs[row.pair_id].append(index)
    comparisons = []
    for pair_id, indices in sorted(pairs.items()):
        if len(indices) != 2:
            continue
        a, b = (matched[i] for i in indices)
        first, second = a.state, b.state
        joints = first.robot.joint_pos
        if (
            joints.keys() != second.robot.joint_pos.keys()
            or any(
                abs(value - second.robot.joint_pos[name]) > _RESET_MATCH_TOLERANCE
                for name, value in joints.items()
            )
            or not np.allclose(
                first.scene.objects[0].pose.position.as_array(),
                second.scene.objects[0].pose.position.as_array(),
                atol=_RESET_MATCH_TOLERANCE,
                rtol=0,
            )
        ):
            continue
        scores = {
            "no_context": [records[i]["score_no_context"] for i in indices],
            "predicted_context": [records[i]["score_predicted_context"] for i in indices],
            "shuffled_within_pair": [
                context.score(matched[i].state, matched[i].candidate.action, futures[j])
                for i, j in (indices, indices[::-1])
            ],
            "controlled_no_change": [records[i]["score_controlled_no_change"] for i in indices],
        }
        comparisons.append(
            {
                "pair_id": pair_id,
                "fit_record": a.scene_group in fit_groups,
                "candidate_count": 2,
                "skill_count": 1,
                "observed_utilities": [a.utility, b.utility],
                "choices": {
                    mode: {
                        "chosen_alternative": int(np.argmax(values)),
                        "chosen_observed_utility": (a, b)[int(np.argmax(values))].utility,
                    }
                    for mode, values in scores.items()
                },
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
        "matched_reset_replay": comparisons,
        "reset_match_tolerance_rad_and_m": _RESET_MATCH_TOLERANCE,
        "forecast_excluded": [
            {
                "skill": row.candidate.action.skill.value,
                "reasons": row.candidate.trajectory.hard_reasons,
            }
            for row in rows
            if set(row.candidate.trajectory.hard_reasons) - {"unverified-arm-sweep"}
        ],
    }


def measured_utility(row: dict) -> float | None:
    """Label measured progress within each action's physical outcome channel."""
    if not row.get("settled") or not row.get("isolated"):
        return None
    skill = Skill(row["skill"])
    if row.get("empty_grasp_verified") is True:
        return 0.0
    if skill is Skill.APPROACH:
        after = row["tip_can_xy_after_m"]
        before = row.get("tip_can_xy_before_m")
        accuracy = CAN_RADIUS_M / (CAN_RADIUS_M + after)
        if before is None:
            robot = row["robot_before"]
            base = robot["base_pose_world"]
            transform = _inverse_pose(Pose3D(Vec3(**base["position"]), tuple(base["orientation"])))
            can = compose(transform, Pose3D(Vec3(**row["can_before_world_m"]), _IDENTITY))
            tool = Vec3(**row["tool_before_base_m"]).as_array()
            before = float(np.linalg.norm((can.position.as_array() - tool)[:2]))
        return accuracy * float(np.clip((before - after) / CAN_RADIUS_M, 0, 1))
    if "tip_can_xy_before_m" not in row:
        return None
    if skill is Skill.DESCEND:
        target = Vec3(**row["target_base_m"]).as_array()
        before = Vec3(**row["tool_before_base_m"]).as_array()
        after = Vec3(**row["tool_after_base_m"]).as_array()
        can_before = Vec3(**row["can_before_world_m"]).as_array()
        can_after = Vec3(**row["can_after_world_m"]).as_array()
        orientation = row["can_orientation_after_xyzw"]
        tilt = np.degrees(
            np.arccos(np.clip(1 - 2 * (orientation[0] ** 2 + orientation[1] ** 2), -1, 1))
        )
        if tilt >= SETTINGS.can_tilt_limit_deg:
            return 0.0
        progress = np.linalg.norm(before - target) - np.linalg.norm(after - target)
        disturbance = np.linalg.norm(can_after - can_before)
        return float(np.clip((progress - disturbance) / CAN_RADIUS_M, 0, 1))
    if skill is Skill.CLOSE:
        retained_opening = 0.002 < row["gripper_after_m"] < GRIPPER_OPEN - 0.002
        return float(retained_opening and row["tip_can_xy_after_m"] <= CLOSE_ALIGN_TOL)
    if skill is Skill.OPEN:
        return float(
            np.clip((row["gripper_after_m"] - row["gripper_before_m"]) / GRIPPER_OPEN, 0, 1)
        )
    if skill is Skill.LIFT:
        return float(np.clip(row["can_rise_m"] / LIFT_MIN, 0, 1))
    if skill in (Skill.TRANSFER, Skill.LOWER, Skill.RELEASE, Skill.RETREAT):
        before = Vec3(**row["can_before_world_m"]).as_array()
        after = Vec3(**row["can_after_world_m"]).as_array()
        goal = np.asarray(PLACE_WORLD)
        old_error, new_error = np.linalg.norm(before - goal), np.linalg.norm(after - goal)
        orientation = row["can_orientation_after_xyzw"]
        tilt = np.degrees(
            np.arccos(np.clip(1 - 2 * (orientation[0] ** 2 + orientation[1] ** 2), -1, 1))
        )
        if tilt >= SETTINGS.can_tilt_limit_deg:
            return 0.0
        if skill in (Skill.RELEASE, Skill.RETREAT):
            return float(
                new_error <= PLACE_TOLERANCE_M and row["gripper_after_m"] >= GRIPPER_OPEN - 0.002
            )
        return float(np.clip((old_error - new_error) / CAN_RADIUS_M, 0, 1))
    return None


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
        if report["split"] != "train":
            raise ValueError(f"isolated TRAIN evidence required: {path}")
        group = report["scene"].rsplit("-s", 1)[0]
        for row in report["rows"]:
            if row["measurement_source"] != "simulator-observed":
                raise ValueError(f"measured outcome required: {path}")
            utility = measured_utility(row)
            if utility is None:
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
            measured = replace(outcomes[-1], pair_id=report["scene"])
            forecasts.append(_forecast(measured, robot))
    if not any(row.scene_group in fit_groups for row in outcomes):
        raise ValueError("measured alternative-action outcomes required in fit groups")
    model = fit_utility(
        examples + outcomes, train_groups=fit_groups, prediction_context=False, ridge=1.0
    )
    result = {
        "identity": "v4-initial-empirical-nominal-no-context",
        "measured_label_version": "v4-skill-progress-with-descent-v2",
        "partition": partition,
        "measured_only_forecast_comparison": _matched_models(forecasts, fit_groups),
        "demonstration_fit_n": sum(row.scene_group in fit_groups for row in examples),
        "measured_label_counts_by_skill": {
            skill.value: sum(row.candidate.action.skill is skill for row in outcomes)
            for skill in Skill
        },
        "measured_fit_n": sum(row.scene_group in fit_groups for row in outcomes),
        "observed_reward": (
            "Measured alignment, descent, aperture, lift and placement progress; empty close is 0"
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
