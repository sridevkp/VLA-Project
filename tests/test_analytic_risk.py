"""Geometry catches between-endpoint collisions and hard veto cannot be offset by utility."""

import copy

import numpy as np
import pytest

from vlarobot.core.types import (
    ActionChunk,
    EEDelta,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.risk.gate import CandidateAssessment, RecoveryState, RiskGate
from vlarobot.world_model.analytic import (
    CartesianPredictor,
    ObjectWorldModel,
    segment_box_clearance,
)


def world():
    robot = RobotState({}, {}, Pose2D(0, 0, 0), Pose3D(Vec3(0, 0, 0.5), (0, 0, 0, 1)), 0.045)
    obstacle = ObjectState(
        "box", "obstacle", Pose3D(Vec3(0.5, 0, 0.5), (0, 0, 0, 1)), Vec3(0.001, 0.2, 0.2), 1
    )
    return WorldState(SceneState([obstacle]), robot)


def test_analytic_sweep_catches_thin_obstacle_between_safe_endpoints():
    state = world()
    predictor = CartesianPredictor(
        tool_radius_m=0.02,
        clearance_margin_m=0.01,
        workspace_min_m=(-1, -1, 0),
        workspace_max_m=(2, 2, 2),
        ik_check=lambda s, c: True,
    )
    chunk = ActionChunk([EEDelta(Vec3(1, 0, 0), Vec3(0, 0, 0), 0.045)], 0.032, "vla")
    result = predictor.predict(state, chunk, scene_complete=True)
    assert result.min_clearance_m == pytest.approx(-0.02)
    assert "tool-clearance" in result.hard_reasons
    assert not result.calibrated_probabilities
    assert state.robot.ee_pose.position.x == 0
    missing = CartesianPredictor(
        tool_radius_m=0.02,
        clearance_margin_m=0.01,
        workspace_min_m=(-1, -1, 0),
        workspace_max_m=(2, 2, 2),
    )
    report = missing.predict(state, chunk, scene_complete=False)
    assert {"incomplete-scene", "unverified-ik-or-arm-sweep"} <= set(report.hard_reasons)


def test_segment_box_clearance_matches_dense_numerical_reference():
    rng = np.random.default_rng(42)
    for _ in range(12):
        start, end = rng.uniform(-2, 2, (2, 3))
        lower, upper = np.array([-0.2, -0.3, -0.4]), np.array([0.4, 0.2, 0.1])
        exact = segment_box_clearance(start, end, lower, upper, 0.02)
        points = start + (end - start) * np.linspace(0, 1, 10001)[:, None]
        distance = np.linalg.norm(np.maximum(np.maximum(lower - points, points - upper), 0), axis=1)
        assert exact == pytest.approx(distance.min() - 0.02, abs=0.0005)


def test_world_snapshots_do_not_alias_and_correction_reports_real_errors():
    model = ObjectWorldModel()
    predicted = world()
    model.update(predicted.scene, predicted.robot)
    snapshot = model.snapshot()
    snapshot.scene.objects.clear()
    assert len(model.snapshot().scene.objects) == 1
    truth = copy.deepcopy(predicted)
    obj = truth.scene.objects[0]
    truth.scene.objects[0] = ObjectState(
        obj.object_id, obj.label, Pose3D(Vec3(0.53, 0.04, 0.5), (0, 0, 0, -1)), obj.dimensions, 1
    )
    error = model.correct(predicted, truth)
    assert error.position_error == pytest.approx(0.05)
    assert error.orientation_error == pytest.approx(0)


def test_risk_veto_missing_nan_and_failures_precede_arbitrary_utility():
    gate = RiskGate(weights=(1, 1, 1, 1), maximize=True, required_constraints=("geometry", "ik"))

    def candidate(checks, score):
        return CandidateAssessment(checks, (score, 0, 0, 0))

    result = gate.select(
        [
            candidate((("geometry", False), ("ik", True)), 1000),
            candidate((("geometry", True),), 2000),
            candidate((("geometry", True), ("ik", None)), 3000),
            candidate((("geometry", True), ("ik", True)), 1),
        ]
    )
    assert result.selected_index == 3
    assert "J=1" in result.rationale and "missing:ik" in result.rationale
    bad = CandidateAssessment((("geometry", True), ("ik", True)), (float("nan"), 0, 0, 0))
    assert gate.select([bad]).selected_index is None
    safe = candidate((("geometry", True), ("ik", True)), 1)
    assert gate.select([safe, safe]).selected_index == 0


def test_no_safe_recovery_and_verification_retry_are_bounded():
    state = RecoveryState()
    assert [state.no_safe() for _ in range(4)] == [
        "reobserve",
        "alternate_candidate_family",
        "abort",
        "abort",
    ]
    assert [state.verify(False) for _ in range(3)] == ["retry_once", "abort", "abort"]


def test_clearance_rejects_invalid_geometry():
    from vlarobot.world_model.analytic import segment_box_clearance

    zero = np.zeros(3)
    for radius in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            segment_box_clearance(zero, zero, zero, np.ones(3), radius)
    with pytest.raises(ValueError):
        segment_box_clearance(np.zeros(2), zero, zero, np.ones(3), 0.1)
    with pytest.raises(ValueError):
        segment_box_clearance(zero, zero, np.ones(3), zero, 0.1)


def test_world_correction_rejects_ambiguous_object_identities():
    model = ObjectWorldModel()
    predicted = world()
    observed = copy.deepcopy(predicted)
    observed.scene.objects.append(copy.deepcopy(observed.scene.objects[0]))
    with pytest.raises(ValueError, match="matched object identities"):
        model.correct(predicted, observed)
