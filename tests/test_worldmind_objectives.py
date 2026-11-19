"""Synthetic tests for the inert Eq. 3.8 objective PROPOSALS.

No proposal is registered as a scored component, evaluator, weight or threshold. These tests
verify only the pure functions, decision-time availability, fail-closed behaviour and the explicit
engineering-choice labelling.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from vlarobot.core.types import (
    ActionChunk,
    EEDelta,
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval import worldmind_objectives as objectives
from vlarobot.eval.worldmind_objectives import (
    AMENDMENT_STATUS,
    COMPONENT_NAMES,
    PROPOSAL_STATUS,
    PROPOSED_OBJECTIVES,
    HardVetoPrecedesUtility,
    MissingObjectiveInput,
    combine,
    goal_error,
    predicted_risk,
    prediction_consistency,
    proposal_records,
    soft_geometric_risk,
    tool_envelope_clearances,
    trajectory_cost,
)
from vlarobot.world_model.rollout import Trajectory

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _state(x: float, *, object_id: str = "CAN", object_x: float | None = None) -> WorldState:
    obj_x = x if object_x is None else object_x
    robot = RobotState(
        joint_pos={"arm_1_joint": 0.0},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(x, 0.0, 0.5), IDENTITY),
        gripper_opening=0.045,
    )
    scene = SceneState(
        [
            ObjectState(
                object_id,
                "can",
                Pose3D(Vec3(obj_x, 0.0, 0.5), IDENTITY),
                Vec3(0.08, 0.08, 0.12),
                1.0,
            )
        ]
    )
    return WorldState(scene, robot)


def _trajectory(*states: WorldState) -> Trajectory:
    return Trajectory(states, 0.032, (), (), ())


def _delta_chunk(dx: float, dy: float) -> ActionChunk:
    return ActionChunk([EEDelta(Vec3(dx, dy, 0.0), Vec3(0.0, 0.0, 0.0), 0.045)], 0.032, "test")


def test_all_proposals_are_explicitly_non_authoritative():
    assert set(PROPOSED_OBJECTIVES) == set(COMPONENT_NAMES)
    records = proposal_records()
    assert len(records) == 4
    for record in records:
        expected = PROPOSAL_STATUS if record["name"] in {"Jg", "Jl"} else AMENDMENT_STATUS
        assert record["status"] == expected
        assert record["authoritative"] is False
        assert record["ranking_direction"] == "minimize"
        assert record["formula"] and record["units"] and record["normalization"]
        assert "owner" in record["weight_selection"].lower()
        assert "veto" in record["safety_precedence"].lower()
        if expected == AMENDMENT_STATUS:
            assert "approval" in expected


def test_goal_error_is_normalized_symmetric_and_fails_closed():
    goal = Pose3D(Vec3(0.5, 0.0, 0.5), IDENTITY)
    same = goal_error(goal, goal, position_scale_m=0.1, orientation_scale_rad=0.5)
    assert same == pytest.approx(0.0)
    shifted = Pose3D(Vec3(0.6, 0.0, 0.5), IDENTITY)
    value = goal_error(shifted, goal, position_scale_m=0.1, orientation_scale_rad=0.5)
    assert value == pytest.approx(1.0)
    with pytest.raises(MissingObjectiveInput, match="scale"):
        goal_error(goal, goal, position_scale_m=0.0, orientation_scale_rad=0.5)
    with pytest.raises(MissingObjectiveInput):
        goal_error(goal, None, position_scale_m=0.1, orientation_scale_rad=0.5)  # type: ignore[arg-type]


def test_trajectory_cost_uses_control_time_not_wall_clock():
    positions = [Vec3(0.0, 0.0, 0.0), Vec3(0.3, 0.0, 0.0), Vec3(0.3, 0.4, 0.0)]
    value = trajectory_cost(
        positions, dt_s=0.032, path_scale_m=0.1, time_scale_s=1.0, time_weight=0.5
    )
    assert value == pytest.approx(7.0 + 0.5 * 2 * 0.032)
    with pytest.raises(MissingObjectiveInput, match="at least two"):
        trajectory_cost(positions[:1], dt_s=0.032, path_scale_m=0.1, time_scale_s=1.0)
    with pytest.raises(MissingObjectiveInput):
        trajectory_cost(positions, dt_s=0.0, path_scale_m=0.1, time_scale_s=1.0)


def test_soft_risk_uses_uncertified_tool_clearance_and_fails_closed():
    current = _state(0.0, object_x=0.5)
    chunk = _delta_chunk(0.2, 0.0)
    predicted = _state(0.19, object_x=0.5)
    trajectory = _trajectory(current, predicted)
    clearances = tool_envelope_clearances(
        current, chunk, trajectory, tool_radius_m=0.02, scene_complete=True
    )
    assert len(clearances) == 1
    assert clearances[0] == pytest.approx((0.5 - 0.04) - 0.19 - 0.02)
    assert soft_geometric_risk(
        current,
        chunk,
        trajectory,
        tool_radius_m=0.02,
        margin_m=0.1,
        scene_complete=True,
    ) == pytest.approx(0.0)

    assert predicted_risk([0.2, 0.2], margin_m=0.1) == pytest.approx(0.0)
    assert predicted_risk([0.0, -0.05], margin_m=0.1) == pytest.approx(1.0)
    assert predicted_risk([0.05, 0.15], margin_m=0.1) == pytest.approx(0.25)

    with pytest.raises(MissingObjectiveInput):
        predicted_risk([], margin_m=0.1)
    with pytest.raises(MissingObjectiveInput, match="scene_complete"):
        soft_geometric_risk(
            current,
            chunk,
            trajectory,
            tool_radius_m=0.02,
            margin_m=0.1,
            scene_complete=False,
        )
    with pytest.raises(MissingObjectiveInput, match="per candidate action"):
        soft_geometric_risk(
            current,
            chunk,
            _trajectory(current),
            tool_radius_m=0.02,
            margin_m=0.1,
            scene_complete=True,
        )
    empty = replace(current, scene=SceneState([]))
    with pytest.raises(MissingObjectiveInput, match="observed object"):
        soft_geometric_risk(
            empty,
            chunk,
            _trajectory(empty, replace(predicted, scene=SceneState([]))),
            tool_radius_m=0.02,
            margin_m=0.1,
            scene_complete=True,
        )


def test_prediction_consistency_is_decision_time_only_and_fails_closed():
    current = _state(0.0, object_x=0.5)
    chunk = _delta_chunk(0.2, 0.01)
    predicted = _state(0.19, object_x=0.5)
    trajectory = _trajectory(current, predicted)
    scales = {
        "position_scale_m": 0.1,
        "orientation_scale_rad": 0.5,
        "joint_scale_rad": 0.1,
        "gripper_scale_m": 0.05,
    }
    value = prediction_consistency(current, chunk, trajectory, **scales)
    assert value == pytest.approx((math.hypot(0.01, 0.01) / 0.1) / 3.0)

    joint_chunk = ActionChunk([JointTarget({"arm_1_joint": 0.2})], 0.032, "test")
    joint_state = replace(
        predicted, robot=replace(predicted.robot, joint_pos={"arm_1_joint": 0.15})
    )
    assert prediction_consistency(
        current, joint_chunk, _trajectory(current, joint_state), **scales
    ) == pytest.approx(0.5)

    with pytest.raises(MissingObjectiveInput, match="per candidate action"):
        prediction_consistency(current, chunk, _trajectory(current), **scales)
    with pytest.raises(MissingObjectiveInput, match="lacks joints"):
        prediction_consistency(
            current,
            ActionChunk([JointTarget({"missing": 0.0})], 0.032, "test"),
            trajectory,
            **scales,
        )
    with pytest.raises(MissingObjectiveInput, match="scale"):
        prediction_consistency(current, chunk, trajectory, **{**scales, "position_scale_m": 0.0})


def test_all_components_are_available_from_decision_time_inputs():
    current = _state(0.0, object_x=0.5)
    chunk = _delta_chunk(0.2, 0.01)
    predicted = _state(0.19, object_x=0.5)
    trajectory = _trajectory(current, predicted)
    goal = Pose3D(Vec3(0.19, 0.01, 0.5), IDENTITY)
    components = {
        "Jg": goal_error(
            predicted.robot.ee_pose,
            goal,
            position_scale_m=0.1,
            orientation_scale_rad=0.5,
        ),
        "Jl": trajectory_cost(
            [state.robot.ee_pose.position for state in trajectory.states],
            dt_s=trajectory.dt_s,
            path_scale_m=0.1,
            time_scale_s=1.0,
        ),
        "Jr": soft_geometric_risk(
            current,
            chunk,
            trajectory,
            tool_radius_m=0.02,
            margin_m=0.1,
            scene_complete=True,
        ),
        "Ju": prediction_consistency(
            current,
            chunk,
            trajectory,
            position_scale_m=0.1,
            orientation_scale_rad=0.5,
            joint_scale_rad=0.1,
            gripper_scale_m=0.05,
        ),
    }
    assert set(components) == set(COMPONENT_NAMES)
    assert all(math.isfinite(value) for value in components.values())
    weights = {name: 1.0 for name in COMPONENT_NAMES}
    assert math.isfinite(combine(components, weights, maximize=False))


def test_future_observations_cannot_leak_into_candidate_ranking():
    current = _state(0.0, object_x=0.5)
    chunk = _delta_chunk(0.2, 0.01)
    predicted = _state(0.19, object_x=0.5)
    trajectory = _trajectory(current, predicted)
    components = {
        "Jg": goal_error(
            predicted.robot.ee_pose,
            predicted.robot.ee_pose,
            position_scale_m=0.1,
            orientation_scale_rad=0.5,
        ),
        "Jl": trajectory_cost(
            [state.robot.ee_pose.position for state in trajectory.states],
            dt_s=trajectory.dt_s,
            path_scale_m=0.1,
            time_scale_s=1.0,
        ),
        "Jr": soft_geometric_risk(
            current,
            chunk,
            trajectory,
            tool_radius_m=0.02,
            margin_m=0.1,
            scene_complete=True,
        ),
        "Ju": prediction_consistency(
            current,
            chunk,
            trajectory,
            position_scale_m=0.1,
            orientation_scale_rad=0.5,
            joint_scale_rad=0.1,
            gripper_scale_m=0.05,
        ),
    }
    weights = {name: 1.0 for name in COMPONENT_NAMES}
    before = combine(components, weights, maximize=False)

    # These "future" observations are deliberately not passed to any component.
    future = _state(0.9, object_x=0.9)
    future.scene.objects[0] = replace(
        future.scene.objects[0], pose=Pose3D(Vec3(-9.0, -9.0, -9.0), IDENTITY)
    )
    future.robot.ee_pose = Pose3D(Vec3(99.0, 99.0, 99.0), IDENTITY)
    after = combine(components, weights, maximize=False)
    assert after == pytest.approx(before)

    signature = inspect.signature(prediction_consistency)
    assert "corrected" not in signature.parameters
    assert "future" not in signature.parameters
    assert not hasattr(objectives, "ObjectWorldModel")


def test_combine_respects_safety_precedence_and_ranking_direction():
    components = {"Jg": 0.5, "Jl": 0.25, "Jr": 0.0, "Ju": 0.125}
    weights = {"Jg": 1.0, "Jl": 1.0, "Jr": 1.0, "Ju": 1.0}
    maximize = combine(components, weights, maximize=True)
    minimize = combine(components, weights, maximize=False)
    assert maximize == pytest.approx(0.875)
    assert minimize == pytest.approx(-0.875)
    with pytest.raises(HardVetoPrecedesUtility):
        combine(components, weights, maximize=True, hard_veto=True)
    with pytest.raises(MissingObjectiveInput):
        combine(components, {**weights, "Ju": -1.0}, maximize=True)
    with pytest.raises(MissingObjectiveInput):
        combine({**components, "Jg": float("nan")}, weights, maximize=True)


def test_proposal_artifact_is_not_owner_approved_and_not_scorable():
    path = Path(__file__).resolve().parents[1] / "configs" / "report" / "objectives.proposal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["review_status"] == "engineering_proposal_not_approved"
    assert payload["scored_execution_enabled"] is False
    assert set(payload["components"]) == set(COMPONENT_NAMES)
    assert all(item["authoritative"] is False for item in payload["components"].values())
    assert payload["components"]["Jr"]["status"] == AMENDMENT_STATUS
    assert payload["components"]["Ju"]["status"] == AMENDMENT_STATUS
    assert "uncertified" in payload["components"]["Jr"]["formula"].lower()
    assert "command-following" in payload["components"]["Ju"]["units"].lower()
    assert payload["decision_time_amendment"]["owner_approval_required"] is True


def test_no_proposal_component_is_registered_in_scored_bindings():
    from vlarobot.backends.worldmind_bindings import production_worldmind_bindings

    registry = production_worldmind_bindings()
    assert registry.components == {}
    assert registry.calibration_evaluators == {}
