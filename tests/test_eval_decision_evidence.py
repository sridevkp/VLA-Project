"""Inert schema tests for additive decision evidence; no run or execution path is used."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlarobot.eval.decision_evidence import (
    CostComponents,
    DecisionEvidence,
    MissingCostReason,
    PhaseTimings,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "decision_observability"
    / "decision-evidence-example.json"
)


def _fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_synthetic_fixture_round_trips_and_serializes_without_nan() -> None:
    payload = _fixture_payload()
    evidence = DecisionEvidence.from_dict(payload)
    assert evidence.candidate_count == 5
    assert evidence.selected_candidate_id == "candidate-04"
    assert evidence.candidates[3].costs is not None
    assert evidence.candidates[3].costs.Jg is None
    assert evidence.candidates[3].costs.missing_reason is MissingCostReason.ASSESSMENT_ERROR
    encoded = evidence.to_dict()
    assert encoded["candidates"][0]["disposition"] == "hard_vetoed"
    json.dumps(encoded, allow_nan=False, sort_keys=True)


def test_missing_cost_components_require_an_explicit_reason() -> None:
    with pytest.raises(ValueError, match="explicit reason"):
        CostComponents(
            Jg=None,
            Jl=0.1,
            Jr=0.1,
            Ju=0.1,
            total_j=None,
            weights=(1.0, 1.0, 1.0, 1.0),
            maximize=False,
            missing_reason=None,
        )


def test_selected_candidate_must_match_decision_level_identifier() -> None:
    payload = _fixture_payload()
    payload["selected_candidate_id"] = None
    with pytest.raises(ValueError, match="selected candidate without"):
        DecisionEvidence.from_dict(payload)


def test_no_safe_set_cannot_carry_a_selected_candidate() -> None:
    payload = _fixture_payload()
    payload["no_safe_set"] = True
    with pytest.raises(ValueError, match="no-safe-set"):
        DecisionEvidence.from_dict(payload)


def test_unavailable_timings_require_a_reason_for_every_phase() -> None:
    with pytest.raises(ValueError, match="every phase"):
        PhaseTimings.unavailable({"candidate_generation_s": "not instrumented"})
    timings = PhaseTimings.from_dict(_fixture_payload()["timings"])
    assert timings.candidate_generation_s == pytest.approx(0.01)
    assert timings.model_inference_s is None
    assert "model_inference_s" in timings.missing_reasons
