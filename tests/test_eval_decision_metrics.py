"""Inert aggregation and pairing tests; all states are synthetic fixtures."""

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
from vlarobot.eval.decision_evidence import DecisionEvidence, PairingStatus, PhaseTimings
from vlarobot.eval.decision_metrics import (
    candidate_disposition_counts,
    pair_executed_prediction,
    risk_rejection_aggregate,
    risk_rejection_rate,
    summarize_phase_latencies,
)
from vlarobot.world_model.rollout import Trajectory

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "decision_observability"
    / "decision-evidence-example.json"
)


def _fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _decision() -> DecisionEvidence:
    return DecisionEvidence.from_dict(_fixture_payload())


def _state(z: float) -> WorldState:
    robot = RobotState(
        {"joint": z},
        {},
        Pose2D(0.0, 0.0, 0.0),
        Pose3D(Vec3(0.0, 0.0, z), (0.0, 0.0, 0.0, 1.0)),
        0.02,
        True,
    )
    obj = ObjectState(
        "can", "can", Pose3D(Vec3(0.0, 0.0, 0.4), (0, 0, 0, 1)), Vec3(0.1, 0.1, 0.1), 1.0
    )
    return WorldState(SceneState([obj]), robot)


def _prediction(states: tuple[WorldState, ...]) -> Trajectory:
    return Trajectory(states, 0.032, (), (), ())


def test_risk_rejection_rate_uses_explicit_evaluated_denominator() -> None:
    decision = _decision()
    rate = risk_rejection_rate(decision.candidates)
    assert (rate.rejected, rate.evaluated, rate.invalid_or_unevaluable) == (2, 4, 1)
    assert rate.rate == pytest.approx(0.5)
    assert "invalid_or_unevaluable" in rate.denominator_definition
    counts = candidate_disposition_counts([decision])
    assert counts["evaluated"] == 4
    assert counts["invalid_or_unevaluable"] == 1


def test_empty_evaluated_denominator_is_undefined_not_zero() -> None:
    decision = _decision()
    unevaluable = [
        c for c in decision.candidates if c.disposition.value == "invalid_or_unevaluable"
    ]
    rate = risk_rejection_rate(unevaluable)
    assert rate.rate is None
    assert rate.undefined_reason == "no evaluated candidates"


def test_aggregate_counts_undefined_decisions_and_reports_macro_separately() -> None:
    decision = _decision()
    aggregate = risk_rejection_aggregate([decision, decision])
    assert aggregate.pooled.rate == pytest.approx(0.5)
    assert aggregate.macro is not None
    assert aggregate.macro.rate == pytest.approx(0.5)
    assert aggregate.decisions_with_defined_rate == 2
    assert aggregate.decisions_with_undefined_rate == 0


def test_latency_summary_uses_deterministic_linear_percentiles() -> None:
    payload = dict(_fixture_payload()["timings"])
    payload["candidate_generation_s"] = 1.0
    first = PhaseTimings.from_dict(payload)
    payload["candidate_generation_s"] = 3.0
    second = PhaseTimings.from_dict(payload)
    summary = summarize_phase_latencies([first, second])
    assert summary["candidate_generation_s"].p50_s == pytest.approx(2.0)
    assert summary["candidate_generation_s"].p95_s == pytest.approx(2.9)
    assert summary["model_inference_s"].samples == 0
    assert summary["model_inference_s"].p50_s is None


def test_prediction_pairing_rejects_rejected_candidate_identity() -> None:
    states = tuple(_state(float(index)) for index in range(3))
    with pytest.raises(ValueError, match="rejected candidates"):
        pair_executed_prediction(
            decision_id="decision-0",
            executed_candidate_id="candidate-01",
            prediction_candidate_id="candidate-00",
            prediction=_prediction(states),
            observed=list(states),
            prediction_timestamp_s=0.0,
            observed_timestamp_s=0.0,
            observed_timestamps_s=(0.0, 0.032, 0.064),
        )


def test_prediction_pairing_full_alignment_reports_si_errors() -> None:
    states = tuple(_state(float(index)) for index in range(3))
    paired = pair_executed_prediction(
        decision_id="decision-0",
        executed_candidate_id="candidate-01",
        prediction_candidate_id="candidate-01",
        prediction=_prediction(states),
        observed=list(states),
        prediction_timestamp_s=0.0,
        observed_timestamp_s=0.0,
        observed_timestamps_s=(0.0, 0.032, 0.064),
    )
    assert paired.status is PairingStatus.PAIRED
    assert paired.valid_paired_steps == 2
    assert paired.error_metrics_si["robot_position_rmse_m"] == pytest.approx(0.0)
    assert paired.error_units["joint_rmse_rad"] == "rad"


def test_prediction_pairing_missing_observations_are_not_invented() -> None:
    states = tuple(_state(float(index)) for index in range(3))
    missing = pair_executed_prediction(
        decision_id="decision-0",
        executed_candidate_id="candidate-01",
        prediction_candidate_id="candidate-01",
        prediction=_prediction(states),
        observed=list(states[:2]),
        prediction_timestamp_s=0.0,
        observed_timestamp_s=0.0,
        observed_timestamps_s=(0.0, 0.032),
    )
    assert missing.status is PairingStatus.MISSING_OBSERVATIONS
    assert missing.error_metrics_si == {}
    assert missing.missing_observations == (2,)
    truncated = pair_executed_prediction(
        decision_id="decision-0",
        executed_candidate_id="candidate-01",
        prediction_candidate_id="candidate-01",
        prediction=_prediction(states),
        observed=list(states[:2]),
        prediction_timestamp_s=0.0,
        observed_timestamp_s=0.0,
        observed_timestamps_s=(0.0, 0.032),
        allow_truncated=True,
    )
    assert truncated.status is PairingStatus.TRUNCATED
    assert truncated.truncated_trajectory
    assert truncated.valid_paired_steps == 1
    assert "robot_position_rmse_m" in truncated.error_metrics_si


def test_prediction_pairing_abort_boundary_prevents_scoring() -> None:
    states = tuple(_state(float(index)) for index in range(3))
    paired = pair_executed_prediction(
        decision_id="decision-0",
        executed_candidate_id="candidate-01",
        prediction_candidate_id="candidate-01",
        prediction=_prediction(states),
        observed=list(states),
        prediction_timestamp_s=0.0,
        observed_timestamp_s=0.0,
        observed_timestamps_s=(0.0, 0.032, 0.064),
        abort_boundary="no-safe-set",
    )
    assert paired.status is PairingStatus.ABORT_BOUNDARY
    assert paired.error_metrics_si == {}
