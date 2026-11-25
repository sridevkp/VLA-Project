"""Additive, fail-closed evidence types for risk-gate decision observability.

These records are sidecar-only. They are not inputs to candidate generation, surrogate
prediction, RiskGate ranking, SafetyMonitor, recovery, execution, or the frozen recorder.
Missing values stay ``None`` with an explicit reason; no cost, timing, or pairing value is
invented to complete a record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from math import isfinite
from typing import Any, Literal, TypeVar

SCHEMA_VERSION: Literal[1] = 1
COMPONENT_NAMES = ("Jg", "Jl", "Jr", "Ju")
PHASE_NAMES = (
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
EVALUATED_DISPOSITIONS = frozenset({"selected", "safe_unselected", "hard_vetoed"})


class CandidateDisposition(StrEnum):
    """Disjoint classes; ``evaluated`` is the union of the first three classes."""

    SELECTED = "selected"
    SAFE_UNSELECTED = "safe_unselected"
    HARD_VETOED = "hard_vetoed"
    INVALID_OR_UNEVALUABLE = "invalid_or_unevaluable"


class HardCheckResult(StrEnum):
    """Normalized result of a named hard check; ``UNKNOWN`` means the gate saw ``None``."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_RUN = "not_run"
    DUPLICATE = "duplicate"


class MissingCostReason(StrEnum):
    """Why a report cost component or total is absent, never a substitute value."""

    NOT_APPLICABLE_ARM = "not_applicable_arm"
    NOT_COMPUTED = "not_computed"
    NOT_OBSERVED = "not_observed"
    NONFINITE_COMPONENT = "nonfinite_component"
    NONFINITE_TOTAL = "nonfinite_total"
    ASSESSMENT_ERROR = "assessment_error"


class PairingStatus(StrEnum):
    PAIRED = "paired"
    MISSING_OBSERVATIONS = "missing_observations"
    TRUNCATED = "truncated"
    RECOVERY_BOUNDARY = "recovery_boundary"
    ABORT_BOUNDARY = "abort_boundary"
    EXECUTED_CANDIDATE_MISMATCH = "executed_candidate_mismatch"
    UNDERSPECIFIED = "underspecified"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def evidence_to_dict(evidence: Any) -> dict[str, Any]:
    """Return a JSON-ready mapping with enum values and no dataclass instances."""
    payload = _jsonable(evidence)
    if not isinstance(payload, dict):
        raise TypeError("evidence must be a dataclass instance")
    return payload


EnumT = TypeVar("EnumT", bound=Enum)


def _enum(enum_type: type[EnumT], value: Any) -> EnumT:
    return value if isinstance(value, enum_type) else enum_type(value)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_finite_nonnegative(value: float | None, field_name: str) -> None:
    if value is not None and (not isfinite(value) or value < 0.0):
        raise ValueError(f"{field_name} must be finite and nonnegative")


@dataclass(frozen=True)
class CostComponents:
    """Eq. 3.8 components, total, weights, and direction with explicit absence provenance."""

    Jg: float | None
    Jl: float | None
    Jr: float | None
    Ju: float | None
    total_j: float | None
    weights: tuple[float, float, float, float]
    maximize: bool
    missing_reason: MissingCostReason | None = None

    def __post_init__(self) -> None:
        values = (self.Jg, self.Jl, self.Jr, self.Ju)
        if len(self.weights) != 4 or any(not isfinite(weight) for weight in self.weights):
            raise ValueError("four finite Eq. 3.8 weights required")
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("nonfinite cost components must be absent with a reason")
        missing = any(value is None for value in values)
        if missing and self.missing_reason is None:
            raise ValueError("missing cost components require an explicit reason")
        if not missing and self.missing_reason is not None:
            raise ValueError("complete cost components cannot carry a missing reason")
        if missing != (self.total_j is None):
            raise ValueError("total_j must be present exactly when all components are present")
        if self.total_j is not None and not isfinite(self.total_j):
            raise ValueError("total_j must be finite")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CostComponents:
        reason = payload.get("missing_reason")
        return cls(
            Jg=payload["Jg"],
            Jl=payload["Jl"],
            Jr=payload["Jr"],
            Ju=payload["Ju"],
            total_j=payload["total_j"],
            weights=tuple(payload["weights"]),
            maximize=bool(payload["maximize"]),
            missing_reason=None if reason is None else _enum(MissingCostReason, reason),
        )


def make_cost_components(
    values: Sequence[float | None],
    *,
    weights: tuple[float, float, float, float],
    maximize: bool,
    missing_reason: MissingCostReason | None = None,
) -> CostComponents:
    """Build cost evidence, converting nonfinite values to absence with a reason."""
    if len(values) != 4:
        raise ValueError("exactly four Eq. 3.8 components required")
    reason = missing_reason
    cleaned: list[float | None] = []
    for value in values:
        if value is None:
            reason = reason or MissingCostReason.NOT_COMPUTED
            cleaned.append(None)
        elif isfinite(value):
            cleaned.append(float(value))
        else:
            reason = reason or MissingCostReason.NONFINITE_COMPONENT
            cleaned.append(None)
    if reason is None:
        complete = [float(value) for value in cleaned if value is not None]
        total = sum(weight * value for weight, value in zip(weights, complete, strict=True))
        if not isfinite(total):
            reason = MissingCostReason.NONFINITE_TOTAL
            total = None
    else:
        total = None
    return CostComponents(
        Jg=cleaned[0],
        Jl=cleaned[1],
        Jr=cleaned[2],
        Ju=cleaned[3],
        total_j=total,
        weights=weights,
        maximize=maximize,
        missing_reason=reason,
    )


@dataclass(frozen=True)
class CandidateEvidence:
    """One candidate's disjoint disposition, hard-check evidence, and legitimate costs."""

    episode_id: str
    decision_id: str
    candidate_id: str
    candidate_index: int
    sampling_seed: int | None
    proposer_source: str
    candidate_count: int
    disposition: CandidateDisposition
    hard_check_result: HardCheckResult
    hard_veto_reasons: tuple[str, ...] = ()
    unevaluable_reasons: tuple[str, ...] = ()
    hard_checks: tuple[tuple[str, HardCheckResult], ...] = ()
    safe: bool | None = None
    rankable: bool = False
    costs: CostComponents | None = None
    selected_candidate_id: str | None = None
    no_safe_set: bool = False
    recovery_transition: str | None = None
    prediction_available: bool = False
    prediction_horizon_steps: int | None = None
    prediction_hard_reasons: tuple[str, ...] = ()
    trajectory_state_count: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.episode_id, "episode_id")
        _require_text(self.decision_id, "decision_id")
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.proposer_source, "proposer_source")
        if self.candidate_index < 0 or self.candidate_count < 0:
            raise ValueError("candidate index and count must be nonnegative")
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise ValueError("sampling_seed must be nonnegative when present")
        if all(isinstance(reason, str) and reason for reason in self.hard_veto_reasons) is False:
            raise ValueError("hard_veto_reasons must contain non-empty strings")
        if all(isinstance(reason, str) and reason for reason in self.unevaluable_reasons) is False:
            raise ValueError("unevaluable_reasons must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name, _ in self.hard_checks):
            raise ValueError("hard_checks must contain named results")
        if self.disposition in (
            CandidateDisposition.SELECTED,
            CandidateDisposition.SAFE_UNSELECTED,
        ):
            if self.safe is not True or self.rankable is not True:
                raise ValueError("safe/rankable candidates must be safe and rankable")
        elif self.disposition is CandidateDisposition.HARD_VETOED:
            if self.safe is not False or self.rankable:
                raise ValueError("hard-vetoed candidates must be unsafe and unrankable")
        elif self.rankable:
            raise ValueError("invalid/unevaluable candidates cannot be rankable")
        if self.disposition is CandidateDisposition.SELECTED:
            if self.selected_candidate_id != self.candidate_id:
                raise ValueError("selected candidate must carry its own ID")
        elif self.selected_candidate_id == self.candidate_id:
            raise ValueError("unselected candidate cannot carry its own ID as selected")
        if self.no_safe_set and self.selected_candidate_id is not None:
            raise ValueError("no-safe-set evidence cannot name a selected candidate")
        if self.prediction_horizon_steps is not None and self.prediction_horizon_steps < 0:
            raise ValueError("prediction_horizon_steps must be nonnegative")
        if self.trajectory_state_count is not None and self.trajectory_state_count < 0:
            raise ValueError("trajectory_state_count must be nonnegative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateEvidence:
        costs = payload.get("costs")
        return cls(
            episode_id=payload["episode_id"],
            decision_id=payload["decision_id"],
            candidate_id=payload["candidate_id"],
            candidate_index=int(payload["candidate_index"]),
            sampling_seed=payload.get("sampling_seed"),
            proposer_source=payload["proposer_source"],
            candidate_count=int(payload["candidate_count"]),
            disposition=_enum(CandidateDisposition, payload["disposition"]),
            hard_check_result=_enum(HardCheckResult, payload["hard_check_result"]),
            hard_veto_reasons=tuple(payload.get("hard_veto_reasons", ())),
            unevaluable_reasons=tuple(payload.get("unevaluable_reasons", ())),
            hard_checks=tuple(
                (name, _enum(HardCheckResult, result))
                for name, result in payload.get("hard_checks", ())
            ),
            safe=payload.get("safe"),
            rankable=bool(payload.get("rankable", False)),
            costs=None if costs is None else CostComponents.from_dict(costs),
            selected_candidate_id=payload.get("selected_candidate_id"),
            no_safe_set=bool(payload.get("no_safe_set", False)),
            recovery_transition=payload.get("recovery_transition"),
            prediction_available=bool(payload.get("prediction_available", False)),
            prediction_horizon_steps=payload.get("prediction_horizon_steps"),
            prediction_hard_reasons=tuple(payload.get("prediction_hard_reasons", ())),
            trajectory_state_count=payload.get("trajectory_state_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


def is_evaluated(disposition: CandidateDisposition) -> bool:
    """Return whether a candidate entered the hard-check/evaluation denominator."""
    return disposition.value in EVALUATED_DISPOSITIONS


@dataclass(frozen=True)
class PhaseTimings:
    """Whole-phase ``time.perf_counter_ns`` durations; components are not additive."""

    candidate_generation_s: float | None = None
    surrogate_prediction_s: float | None = None
    hard_safety_checks_s: float | None = None
    riskgate_ranking_s: float | None = None
    selection_s: float | None = None
    planning_end_to_end_s: float | None = None
    model_inference_s: float | None = None
    verification_s: float | None = None
    recovery_s: float | None = None
    missing_reasons: Mapping[str, str] = field(default_factory=dict)
    clock: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"

    def __post_init__(self) -> None:
        for name in PHASE_NAMES:
            _require_finite_nonnegative(getattr(self, name), name)
        unknown = set(self.missing_reasons) - set(PHASE_NAMES)
        if unknown:
            raise ValueError(f"unknown phase names in missing_reasons: {sorted(unknown)}")
        for name, reason in self.missing_reasons.items():
            _require_text(reason, f"missing_reasons[{name}]")
        for name in PHASE_NAMES:
            if getattr(self, name) is None and name not in self.missing_reasons:
                raise ValueError(f"timing {name} is absent without a reason")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhaseTimings:
        return cls(
            **{name: payload.get(name) for name in PHASE_NAMES},
            missing_reasons=dict(payload.get("missing_reasons", {})),
            clock=payload.get("clock", "time.perf_counter_ns"),
        )

    @classmethod
    def unavailable(cls, missing_reasons: Mapping[str, str]) -> PhaseTimings:
        """Explicit all-missing timing evidence for an adapter with timing disabled."""
        reasons = dict(missing_reasons)
        if set(reasons) != set(PHASE_NAMES):
            raise ValueError("unavailable timings require a reason for every phase")
        return cls(
            candidate_generation_s=None,
            surrogate_prediction_s=None,
            hard_safety_checks_s=None,
            riskgate_ranking_s=None,
            selection_s=None,
            planning_end_to_end_s=None,
            model_inference_s=None,
            verification_s=None,
            recovery_s=None,
            missing_reasons=reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


@dataclass(frozen=True)
class PredictionPairingEvidence:
    """Executed-candidate-only prediction/observation pairing with explicit boundaries."""

    decision_id: str
    executed_candidate_id: str
    prediction_candidate_id: str
    prediction_timestamp_s: float
    prediction_horizon_steps: int
    prediction_dt_s: float
    observed_timestamp_s: float
    observed_reference_frame: Literal["base"]
    valid_paired_steps: int
    status: PairingStatus
    error_metrics_si: Mapping[str, float] = field(default_factory=dict)
    error_units: Mapping[str, str] = field(default_factory=dict)
    missing_observations: tuple[int, ...] = ()
    truncated_trajectory: bool = False
    recovery_boundary: str | None = None
    abort_boundary: str | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id")
        _require_text(self.executed_candidate_id, "executed_candidate_id")
        _require_text(self.prediction_candidate_id, "prediction_candidate_id")
        if self.executed_candidate_id != self.prediction_candidate_id:
            raise ValueError("prediction pairing requires the executed candidate")
        if self.observed_reference_frame != "base":
            raise ValueError("observed_reference_frame must be explicitly confirmed as 'base'")
        for name in ("prediction_timestamp_s", "observed_timestamp_s", "prediction_dt_s"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.prediction_horizon_steps < 0 or self.valid_paired_steps < 0:
            raise ValueError("horizon and valid step counts must be nonnegative")
        if self.valid_paired_steps > self.prediction_horizon_steps:
            raise ValueError("valid_paired_steps cannot exceed the prediction horizon")
        if any(not isfinite(value) for value in self.error_metrics_si.values()):
            raise ValueError("prediction error metrics must be finite")
        if any(step < 0 for step in self.missing_observations):
            raise ValueError("missing observation indices must be nonnegative")
        if self.status not in (PairingStatus.PAIRED, PairingStatus.TRUNCATED) and (
            self.error_metrics_si
        ):
            raise ValueError("only paired or explicitly truncated evidence may carry errors")
        if self.status is PairingStatus.TRUNCATED and not self.truncated_trajectory:
            raise ValueError("truncated status requires truncated_trajectory")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PredictionPairingEvidence:
        return cls(
            decision_id=payload["decision_id"],
            executed_candidate_id=payload["executed_candidate_id"],
            prediction_candidate_id=payload["prediction_candidate_id"],
            prediction_timestamp_s=float(payload["prediction_timestamp_s"]),
            prediction_horizon_steps=int(payload["prediction_horizon_steps"]),
            prediction_dt_s=float(payload["prediction_dt_s"]),
            observed_timestamp_s=float(payload["observed_timestamp_s"]),
            observed_reference_frame=payload["observed_reference_frame"],
            valid_paired_steps=int(payload["valid_paired_steps"]),
            status=_enum(PairingStatus, payload["status"]),
            error_metrics_si=dict(payload.get("error_metrics_si", {})),
            error_units=dict(payload.get("error_units", {})),
            missing_observations=tuple(payload.get("missing_observations", ())),
            truncated_trajectory=bool(payload.get("truncated_trajectory", False)),
            recovery_boundary=payload.get("recovery_boundary"),
            abort_boundary=payload.get("abort_boundary"),
            reason_codes=tuple(payload.get("reason_codes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


CHUNK_EXECUTION_TERMINATION_REASONS = (
    "chunk_complete",
    "safety_rejected",
    "backend_apply_rejected",
    "execution_exception",
)


@dataclass(frozen=True)
class AppliedActionEvidence:
    """One reviewed action attempt inside a diagnostic full-chunk execution."""

    decision_id: str
    action_index: int
    action_type: str
    action: Mapping[str, Any]
    applied: bool
    pre_apply_accepted: bool | None
    pre_apply_reason: str | None
    apply_accepted: bool | None
    apply_reason: str | None
    backend_stepped: bool
    observation_t_s: float
    action_t_s: float | None

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id")
        _require_text(self.action_type, "action_type")
        if isinstance(self.action_index, bool) or self.action_index < 0:
            raise ValueError("action_index must be a nonnegative integer")
        if self.applied and (self.apply_accepted is not True or not self.backend_stepped):
            raise ValueError("an applied action requires an accepted apply and a control step")
        _require_finite_nonnegative(self.observation_t_s, "observation_t_s")
        _require_finite_nonnegative(self.action_t_s, "action_t_s")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AppliedActionEvidence:
        return cls(
            decision_id=payload["decision_id"],
            action_index=int(payload["action_index"]),
            action_type=payload["action_type"],
            action=dict(payload["action"]),
            applied=bool(payload["applied"]),
            pre_apply_accepted=payload.get("pre_apply_accepted"),
            pre_apply_reason=payload.get("pre_apply_reason"),
            apply_accepted=payload.get("apply_accepted"),
            apply_reason=payload.get("apply_reason"),
            backend_stepped=bool(payload["backend_stepped"]),
            observation_t_s=float(payload["observation_t_s"]),
            action_t_s=payload.get("action_t_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


@dataclass(frozen=True)
class ChunkExecutionEvidence:
    """Measured counts and termination for an explicit diagnostic open-loop chunk."""

    decision_id: str
    execution_mode: str
    selected_candidate_id: str
    chunk_action_count: int
    attempted_action_count: int
    applied_action_count: int
    backend_step_count: int
    observation_count: int
    safety_validate_count: int
    safety_accepted_count: int
    safety_rejected_count: int
    apply_accepted_count: int
    apply_rejected_count: int
    missing_horizon_steps: tuple[int, ...]
    termination_reason: str
    actions: tuple[AppliedActionEvidence, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id")
        _require_text(self.execution_mode, "execution_mode")
        _require_text(self.selected_candidate_id, "selected_candidate_id")
        if self.termination_reason not in CHUNK_EXECUTION_TERMINATION_REASONS:
            raise ValueError(f"unknown chunk termination reason: {self.termination_reason}")
        counts = (
            self.chunk_action_count,
            self.attempted_action_count,
            self.applied_action_count,
            self.backend_step_count,
            self.observation_count,
            self.safety_validate_count,
            self.safety_accepted_count,
            self.safety_rejected_count,
            self.apply_accepted_count,
            self.apply_rejected_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("chunk execution counts must be nonnegative integers")
        if len(self.actions) != self.attempted_action_count:
            raise ValueError("one action record is required per attempted action")
        if self.attempted_action_count > self.chunk_action_count:
            raise ValueError("attempted actions cannot exceed the selected chunk length")
        if self.safety_validate_count != self.attempted_action_count:
            raise ValueError("every attempted action requires one pre-apply validation")
        if self.safety_accepted_count + self.safety_rejected_count != self.safety_validate_count:
            raise ValueError("safety outcomes must account for every validation")
        if self.apply_accepted_count + self.apply_rejected_count != self.safety_accepted_count:
            raise ValueError("apply outcomes must account for every safety-accepted action")
        if self.applied_action_count > self.apply_accepted_count:
            raise ValueError("applied actions cannot exceed accepted applies")
        if self.backend_step_count != self.applied_action_count:
            raise ValueError("one control step is required per applied action")
        if self.missing_horizon_steps != tuple(
            range(self.applied_action_count, self.chunk_action_count)
        ):
            raise ValueError("missing horizon steps must be the unexecuted action indices")
        if any(action.decision_id != self.decision_id for action in self.actions):
            raise ValueError("action records must belong to the chunk decision")
        if [action.action_index for action in self.actions] != list(
            range(self.attempted_action_count)
        ):
            raise ValueError("action records must be indexed sequentially from zero")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChunkExecutionEvidence:
        return cls(
            decision_id=payload["decision_id"],
            execution_mode=payload["execution_mode"],
            selected_candidate_id=payload["selected_candidate_id"],
            chunk_action_count=int(payload["chunk_action_count"]),
            attempted_action_count=int(payload["attempted_action_count"]),
            applied_action_count=int(payload["applied_action_count"]),
            backend_step_count=int(payload["backend_step_count"]),
            observation_count=int(payload["observation_count"]),
            safety_validate_count=int(payload["safety_validate_count"]),
            safety_accepted_count=int(payload["safety_accepted_count"]),
            safety_rejected_count=int(payload["safety_rejected_count"]),
            apply_accepted_count=int(payload["apply_accepted_count"]),
            apply_rejected_count=int(payload["apply_rejected_count"]),
            missing_horizon_steps=tuple(payload["missing_horizon_steps"]),
            termination_reason=payload["termination_reason"],
            actions=tuple(AppliedActionEvidence.from_dict(item) for item in payload["actions"]),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


@dataclass(frozen=True)
class ExecutionEvidence:
    """Selection-to-apply outcome; pre-apply vetoes require the optional monitor probe."""

    decision_id: str
    selected_candidate_id: str | None
    executed_candidate_id: str | None
    pre_apply_accepted: bool | None = None
    pre_apply_reason: str | None = None
    apply_accepted: bool | None = None
    apply_reason: str | None = None
    backend_stepped: bool = False
    observation_t_s: float | None = None
    action_t_s: float | None = None
    chunk: ChunkExecutionEvidence | None = None

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id")
        if self.executed_candidate_id != self.selected_candidate_id:
            raise ValueError("executed candidate must be the selected candidate")
        if (
            self.chunk is None
            and self.pre_apply_accepted is False
            and (self.apply_accepted is not None or self.backend_stepped)
        ):
            raise ValueError("a pre-apply veto must not apply or step the backend")
        if self.chunk is not None:
            if self.chunk.decision_id != self.decision_id:
                raise ValueError("chunk evidence must belong to the execution decision")
            if self.chunk.selected_candidate_id != self.selected_candidate_id:
                raise ValueError("chunk evidence must match the selected candidate")
        _require_finite_nonnegative(self.observation_t_s, "observation_t_s")
        _require_finite_nonnegative(self.action_t_s, "action_t_s")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionEvidence:
        chunk = payload.get("chunk")
        return cls(
            decision_id=payload["decision_id"],
            selected_candidate_id=payload.get("selected_candidate_id"),
            executed_candidate_id=payload.get("executed_candidate_id"),
            pre_apply_accepted=payload.get("pre_apply_accepted"),
            pre_apply_reason=payload.get("pre_apply_reason"),
            apply_accepted=payload.get("apply_accepted"),
            apply_reason=payload.get("apply_reason"),
            backend_stepped=bool(payload.get("backend_stepped", False)),
            observation_t_s=payload.get("observation_t_s"),
            action_t_s=payload.get("action_t_s"),
            chunk=None if chunk is None else ChunkExecutionEvidence.from_dict(chunk),
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


@dataclass(frozen=True)
class VerificationEvidence:
    """Observed verification result and the bounded recovery transition it caused."""

    decision_id: str
    success: bool
    reason: str
    next_action: str | None = None
    observed_t_s: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id")
        _require_finite_nonnegative(self.observed_t_s, "observed_t_s")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationEvidence:
        return cls(
            decision_id=payload["decision_id"],
            success=bool(payload["success"]),
            reason=payload["reason"],
            next_action=payload.get("next_action"),
            observed_t_s=payload.get("observed_t_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


@dataclass(frozen=True)
class DecisionEvidence:
    """One decision's additive sidecar record; no field feeds a control decision."""

    episode_id: str
    decision_id: str
    arm: str
    proposer_source: str
    decision_index: int
    candidate_count: int
    draw_seed: int | None
    draw_seeds: tuple[int, ...]
    candidates: tuple[CandidateEvidence, ...]
    selected_candidate_id: str | None
    no_safe_set: bool
    recovery_events: tuple[str, ...] = ()
    timings: PhaseTimings | None = None
    execution: ExecutionEvidence | None = None
    verification: VerificationEvidence | None = None
    prediction_pairing: PredictionPairingEvidence | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)
    schema_version: Literal[1] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.episode_id, "episode_id")
        _require_text(self.decision_id, "decision_id")
        _require_text(self.arm, "arm")
        _require_text(self.proposer_source, "proposer_source")
        if self.decision_index < 0 or self.candidate_count < 0:
            raise ValueError("decision index and candidate count must be nonnegative")
        if len(self.candidates) != self.candidate_count:
            raise ValueError("candidate_count must equal the number of candidate records")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a decision")
        if self.draw_seeds and len(self.draw_seeds) != self.candidate_count:
            raise ValueError("draw_seeds length must equal candidate_count when present")
        if any(seed < 0 for seed in self.draw_seeds):
            raise ValueError("draw seeds must be nonnegative")
        if self.draw_seed is not None and self.draw_seed < 0:
            raise ValueError("draw_seed must be nonnegative when present")
        selected = [c for c in self.candidates if c.disposition is CandidateDisposition.SELECTED]
        if self.no_safe_set:
            if self.selected_candidate_id is not None or selected:
                raise ValueError("no-safe-set decision cannot contain a selected candidate")
        elif self.selected_candidate_id is None:
            if selected:
                raise ValueError("selected candidate without decision-level selected ID")
        else:
            if len(selected) != 1 or selected[0].candidate_id != self.selected_candidate_id:
                raise ValueError("selected ID must match exactly one selected candidate")
        for value in self.provenance.values():
            _require_text(value, "provenance value")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DecisionEvidence:
        timings = payload.get("timings")
        execution = payload.get("execution")
        verification = payload.get("verification")
        pairing = payload.get("prediction_pairing")
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported decision evidence schema version: {version}")
        return cls(
            episode_id=payload["episode_id"],
            decision_id=payload["decision_id"],
            arm=payload["arm"],
            proposer_source=payload["proposer_source"],
            decision_index=int(payload["decision_index"]),
            candidate_count=int(payload["candidate_count"]),
            draw_seed=payload.get("draw_seed"),
            draw_seeds=tuple(payload.get("draw_seeds", ())),
            candidates=tuple(CandidateEvidence.from_dict(item) for item in payload["candidates"]),
            selected_candidate_id=payload.get("selected_candidate_id"),
            no_safe_set=bool(payload["no_safe_set"]),
            recovery_events=tuple(payload.get("recovery_events", ())),
            timings=None if timings is None else PhaseTimings.from_dict(timings),
            execution=None if execution is None else ExecutionEvidence.from_dict(execution),
            verification=None
            if verification is None
            else VerificationEvidence.from_dict(verification),
            prediction_pairing=None
            if pairing is None
            else PredictionPairingEvidence.from_dict(pairing),
            provenance=dict(payload.get("provenance", {})),
            schema_version=SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)
