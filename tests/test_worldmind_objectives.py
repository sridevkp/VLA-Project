"""Synthetic tests for the inert Eq. 3.8 objective PROPOSALS.

No proposal is registered as a scored component, evaluator, weight or threshold. These tests
verify only the pure functions and the explicit engineering-choice labelling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlarobot.core.types import (
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_objectives import (
    COMPONENT_NAMES,
    PROPOSAL_STATUS,
    PROPOSED_OBJECTIVES,
    HardVetoPrecedesUtility,
    MissingObjectiveInput,
    combine,
    goal_error,
    predicted_risk,
    prediction_penalty,
    proposal_records,
    trajectory_cost,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _state(x: float, *, object_id: str = "CAN") -> WorldState:
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
                Pose3D(Vec3(x, 0.0, 0.5), IDENTITY),
                Vec3(0.08, 0.08, 0.12),
                1.0,
            )
        ]
    )
    return WorldState(scene, robot)


def test_all_proposals_are_explicitly_non_authoritative():
    assert set(PROPOSED_OBJECTIVES) == set(COMPONENT_NAMES)
    records = proposal_records()
    assert len(records) == 4
    for record in records:
        assert record["status"] == PROPOSAL_STATUS
        assert record["authoritative"] is False
        assert record["ranking_direction"] == "minimize"
        assert record["formula"] and record["units"] and record["normalization"]
        assert "owner" in record["weight_selection"].lower()
        assert "veto" in record["safety_precedence"].lower()


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


def test_predicted_risk_is_bounded_and_rejects_unmeasured_clearance():
    assert predicted_risk([0.2, 0.2], margin_m=0.1) == pytest.approx(0.0)
    assert predicted_risk([0.0, -0.05], margin_m=0.1) == pytest.approx(1.0)
    assert predicted_risk([0.05, 0.15], margin_m=0.1) == pytest.approx(0.25)
    with pytest.raises(MissingObjectiveInput):
        predicted_risk([], margin_m=0.1)


def test_prediction_penalty_is_deterministic_and_requires_alignment():
    predicted = [_state(0.0), _state(0.01)]
    corrected = [_state(0.0), _state(0.0)]
    first = prediction_penalty(
        predicted, corrected, position_scale_m=0.1, orientation_scale_rad=0.5
    )
    second = prediction_penalty(
        predicted, corrected, position_scale_m=0.1, orientation_scale_rad=0.5
    )
    assert first == pytest.approx(second)
    assert first > 0.0
    with pytest.raises(MissingObjectiveInput, match="aligned"):
        prediction_penalty(
            predicted, corrected[:1], position_scale_m=0.1, orientation_scale_rad=0.5
        )


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


def test_no_proposal_component_is_registered_in_scored_bindings():
    from vlarobot.backends.worldmind_bindings import production_worldmind_bindings

    registry = production_worldmind_bindings()
    assert registry.components == {}
    assert registry.calibration_evaluators == {}
