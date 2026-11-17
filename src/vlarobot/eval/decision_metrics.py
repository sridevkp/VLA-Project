"""Pure aggregations over additive decision evidence; no scoring or execution path is changed."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

import numpy as np

from vlarobot.core.types import WorldState
from vlarobot.eval.decision_evidence import (
    EVALUATED_DISPOSITIONS,
    CandidateDisposition,
    CandidateEvidence,
    DecisionEvidence,
    PairingStatus,
    PhaseTimings,
    PredictionPairingEvidence,
    is_evaluated,
)
from vlarobot.world_model.evaluation import prediction_errors
from vlarobot.world_model.rollout import Trajectory

RISK_REJECTION_DENOMINATOR = (
    "evaluated candidates = hard_vetoed + safe_unselected + selected; "
    "invalid_or_unevaluable candidates are excluded and reported separately"
)
PHASE_FIELDS = (
    "candidate_generation_s",
    "surrogate_prediction_s",
    "hard_safety_checks_s",
    "riskgate_ranking_s",
    "selection_s",
    "planning_end_to_end_s",
    "model_inference_s",
    "verification_s",
    "recovery_s",
)
ERROR_UNITS = {
    "prediction_steps": "steps",
    "robot_position_rmse_m": "m",
    "robot_position_mae_m": "m",
    "joint_rmse_rad": "rad",
    "joint_mae_rad": "rad",
    "object_position_rmse_m": "m",
    "object_orientation_rmse_rad": "rad",
}


@dataclass(frozen=True)
class RiskRejectionRate:
    """A rate with its exact numerator, denominator, and excluded invalid count."""

    rejected: int
    evaluated: int
    invalid_or_unevaluable: int
    rate: float | None
    denominator_definition: str = RISK_REJECTION_DENOMINATOR
    undefined_reason: str | None = None


def risk_rejection_rate(candidates: Sequence[CandidateEvidence]) -> RiskRejectionRate:
    """Return hard vetoes divided by evaluated candidates, or an explicit undefined rate."""
    rejected = sum(c.disposition is CandidateDisposition.HARD_VETOED for c in candidates)
    evaluated = sum(is_evaluated(c.disposition) for c in candidates)
    invalid = sum(c.disposition is CandidateDisposition.INVALID_OR_UNEVALUABLE for c in candidates)
    rate = rejected / evaluated if evaluated else None
    return RiskRejectionRate(
        rejected=rejected,
        evaluated=evaluated,
        invalid_or_unevaluable=invalid,
        rate=rate,
        undefined_reason=None if evaluated else "no evaluated candidates",
    )


@dataclass(frozen=True)
class RiskRejectionAggregate:
    """Pooled and optional macro rates; undefined decisions are counted, never dropped silently."""

    pooled: RiskRejectionRate
    macro: RiskRejectionRate | None
    decisions_with_defined_rate: int
    decisions_with_undefined_rate: int


def risk_rejection_aggregate(
    decisions: Sequence[DecisionEvidence], *, include_macro: bool = True
) -> RiskRejectionAggregate:
    """Aggregate candidate-level rates over decisions without changing the denominator rule."""
    per_decision = [risk_rejection_rate(decision.candidates) for decision in decisions]
    pooled_candidates = [candidate for decision in decisions for candidate in decision.candidates]
    defined = [row.rate for row in per_decision if row.rate is not None]
    undefined = len(per_decision) - len(defined)
    macro: RiskRejectionRate | None = None
    if include_macro:
        macro_values = [float(value) for value in defined]
        macro = RiskRejectionRate(
            rejected=len(macro_values),
            evaluated=len(per_decision),
            invalid_or_unevaluable=sum(row.invalid_or_unevaluable for row in per_decision),
            rate=float(np.mean(macro_values)) if macro_values else None,
            denominator_definition=(
                "unweighted mean of defined per-decision rates; "
                "decisions with zero evaluated candidates are reported but excluded"
            ),
            undefined_reason=None if macro_values else "no decision has evaluated candidates",
        )
    return RiskRejectionAggregate(
        pooled=risk_rejection_rate(pooled_candidates),
        macro=macro,
        decisions_with_defined_rate=len(defined),
        decisions_with_undefined_rate=undefined,
    )


def candidate_disposition_counts(decisions: Sequence[DecisionEvidence]) -> dict[str, int]:
    """Count all disjoint classes plus the evaluated union and invalid/unevaluable class."""
    counter: Counter[str] = Counter(
        candidate.disposition.value for decision in decisions for candidate in decision.candidates
    )
    result = {
        disposition.value: int(counter[disposition.value]) for disposition in CandidateDisposition
    }
    result["evaluated"] = sum(int(counter[name]) for name in EVALUATED_DISPOSITIONS)
    return result


@dataclass(frozen=True)
class PhaseLatencySummary:
    """Deterministic linear-interpolated p50/p95 over explicit phase samples."""

    phase: str
    samples: int
    p50_s: float | None
    p95_s: float | None


def summarize_phase_latencies(timings: Sequence[PhaseTimings]) -> dict[str, PhaseLatencySummary]:
    """Summarize each phase independently; inclusive phases are not added together."""
    summary: dict[str, PhaseLatencySummary] = {}
    for phase in PHASE_FIELDS:
        samples = [getattr(row, phase) for row in timings]
        values = [float(value) for value in samples if value is not None]
        if any(value is not None and (not isfinite(value) or value < 0.0) for value in samples):
            raise ValueError(f"{phase} contains nonfinite or negative timing")
        if values:
            p50, p95 = (float(value) for value in np.percentile(values, [50.0, 95.0]))
        else:
            p50 = p95 = None
        summary[phase] = PhaseLatencySummary(phase, len(values), p50, p95)
    return summary


def pair_executed_prediction(
    *,
    decision_id: str,
    executed_candidate_id: str,
    prediction_candidate_id: str,
    prediction: Trajectory,
    observed: Sequence[WorldState],
    prediction_timestamp_s: float,
    observed_timestamp_s: float,
    observed_timestamps_s: Sequence[float],
    observed_reference_frame: Literal["base"] = "base",
    prediction_dt_s: float | None = None,
    recovery_boundary: str | None = None,
    abort_boundary: str | None = None,
    allow_truncated: bool = False,
) -> PredictionPairingEvidence:
    """Pair only the executed candidate with aligned observations and explicit boundaries."""
    if executed_candidate_id != prediction_candidate_id:
        raise ValueError("rejected candidates cannot be scored as executed predictions")
    if observed_reference_frame != "base":
        raise ValueError("observed_reference_frame must be explicitly confirmed as 'base'")
    if len(observed_timestamps_s) != len(observed):
        raise ValueError("observed states and timestamps must have equal length")
    if any(not isfinite(float(value)) or float(value) < 0.0 for value in observed_timestamps_s):
        raise ValueError("observed timestamps must be finite and nonnegative")
    if observed_timestamps_s and float(observed_timestamp_s) != float(observed_timestamps_s[0]):
        raise ValueError("observed_timestamp_s must match the first observed timestamp")
    if any(
        observed_timestamps_s[index] <= observed_timestamps_s[index - 1]
        for index in range(1, len(observed_timestamps_s))
    ):
        raise ValueError("observed timestamps must be strictly increasing")
    dt = prediction.dt_s if prediction_dt_s is None else prediction_dt_s
    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("prediction_dt_s must be positive and finite")
    horizon = max(0, len(prediction.states) - 1)
    common = min(len(prediction.states), len(observed))
    valid_steps = max(0, common - 1)
    truncated = len(prediction.states) != len(observed)
    missing = (
        tuple(range(common, len(prediction.states)))
        if len(observed) < len(prediction.states)
        else ()
    )

    def evidence(
        status: PairingStatus, metrics: dict[str, float] | None = None
    ) -> PredictionPairingEvidence:
        return PredictionPairingEvidence(
            decision_id=decision_id,
            executed_candidate_id=executed_candidate_id,
            prediction_candidate_id=prediction_candidate_id,
            prediction_timestamp_s=prediction_timestamp_s,
            prediction_horizon_steps=horizon,
            prediction_dt_s=dt,
            observed_timestamp_s=observed_timestamp_s,
            observed_reference_frame=observed_reference_frame,
            valid_paired_steps=valid_steps,
            status=status,
            error_metrics_si={} if metrics is None else metrics,
            error_units={}
            if metrics is None
            else {name: ERROR_UNITS.get(name, "unspecified") for name in metrics},
            missing_observations=missing,
            truncated_trajectory=truncated,
            recovery_boundary=recovery_boundary,
            abort_boundary=abort_boundary,
            reason_codes=("truncated_horizon",) if truncated else (),
        )

    if abort_boundary:
        return evidence(PairingStatus.ABORT_BOUNDARY)
    if recovery_boundary and truncated and not allow_truncated:
        return evidence(PairingStatus.RECOVERY_BOUNDARY)
    if common < 2:
        return evidence(PairingStatus.UNDERSPECIFIED)
    if truncated and not allow_truncated:
        status = (
            PairingStatus.MISSING_OBSERVATIONS
            if len(observed) < len(prediction.states)
            else PairingStatus.TRUNCATED
        )
        return evidence(status)
    aligned_prediction = (
        prediction
        if not truncated
        else replace(
            prediction,
            states=tuple(prediction.states[:common]),
            hard_reasons=(*prediction.hard_reasons, "observation_truncated_for_scoring"),
        )
    )
    metrics = prediction_errors(aligned_prediction, list(observed[:common]))
    status = PairingStatus.TRUNCATED if truncated else PairingStatus.PAIRED
    if recovery_boundary and status is PairingStatus.PAIRED:
        return evidence(PairingStatus.PAIRED, dict(metrics))
    return evidence(status, dict(metrics))
