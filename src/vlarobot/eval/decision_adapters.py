"""Optional, delegate-only instrumentation adapters for WorldMind decision evidence.

Every wrapper calls the original exactly once, returns the original object unchanged, and
records only a compact immutable sidecar observation. Nothing here is wired into production;
callers may compose these adapters in tests or an explicitly authorized diagnostic harness.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import Any

from vlarobot.core.types import (
    Action,
    ActionChunk,
    ApplyResult,
    Observation,
    VerifyResult,
    WorldState,
)
from vlarobot.eval.decision_evidence import (
    CandidateDisposition,
    CandidateEvidence,
    DecisionEvidence,
    ExecutionEvidence,
    HardCheckResult,
    MissingCostReason,
    PhaseTimings,
    PredictionPairingEvidence,
    VerificationEvidence,
    make_cost_components,
)
from vlarobot.risk.comparison import ComparisonPlanner, Selection
from vlarobot.risk.gate import CandidateAssessment
from vlarobot.world_model.rollout import Trajectory

Clock = Callable[[], int]


def candidate_id(index: int) -> str:
    """Stable sidecar ID derived only from candidate order within one decision."""
    if index < 0:
        raise ValueError("candidate index must be nonnegative")
    return f"candidate-{index:02d}"


def _elapsed_s(start: int | None, clock: Clock, enabled: bool) -> float | None:
    return None if not enabled or start is None else (clock() - start) / 1_000_000_000.0


@dataclass(frozen=True)
class GenerationObservation:
    episode_id: str
    decision_id: str
    proposer_source: str
    decision_index: int
    draw_seed: int
    draw_seeds: tuple[int, ...]
    candidate_count: int
    duration_s: float | None = None


@dataclass(frozen=True)
class PredictionObservation:
    episode_id: str
    decision_id: str
    candidate_index: int
    candidate_id: str
    proposer_source: str
    action_count: int
    available: bool
    horizon_steps: int | None = None
    trajectory_state_count: int | None = None
    hard_reasons: tuple[str, ...] = ()
    duration_s: float | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class AssessmentObservation:
    episode_id: str
    decision_id: str
    candidate_index: int
    candidate_id: str
    hard_checks: tuple[tuple[str, bool | None], ...] = ()
    components: tuple[float, float, float, float] | None = None
    duration_s: float | None = None
    error_code: str | None = None


class CandidateGenerationProbe:
    """Delegate ``propose`` and snapshot the existing ``draw_log[-1]`` after the call."""

    def __init__(
        self,
        generator: Any,
        *,
        episode_id: str,
        decision_id: str,
        sink: list[GenerationObservation] | None = None,
        timing_enabled: bool = False,
        clock: Clock = perf_counter_ns,
    ) -> None:
        self.generator = generator
        self.episode_id = episode_id
        self.decision_id = decision_id
        self.observations = sink if sink is not None else []
        self.timing_enabled = timing_enabled
        self.clock = clock

    def propose(self, task: Any, state: WorldState, memory: object) -> list[ActionChunk]:
        start = self.clock() if self.timing_enabled else None
        chunks = self.generator.propose(task, state, memory)
        duration = _elapsed_s(start, self.clock, self.timing_enabled)
        log = self.generator.draw_log[-1]
        expected = {"decision", "seed", "draw_seeds", "source"}
        if not isinstance(log, Mapping) or set(log) != expected:
            raise ValueError("generator draw_log[-1] has an unexpected contract")
        draw_seeds = tuple(int(seed) for seed in log["draw_seeds"])
        self.observations.append(
            GenerationObservation(
                episode_id=self.episode_id,
                decision_id=self.decision_id,
                proposer_source=str(log["source"]),
                decision_index=int(log["decision"]),
                draw_seed=int(log["seed"]),
                draw_seeds=draw_seeds,
                candidate_count=len(chunks),
                duration_s=duration,
            )
        )
        return chunks


class PredictionAssessmentProbe:
    """Wrap injected predict/assess callables once; never wrap ``RiskGate.select``."""

    def __init__(
        self,
        *,
        predict: Callable[[WorldState, ActionChunk], Trajectory],
        assess: Callable[[WorldState, ActionChunk, Trajectory | None], CandidateAssessment],
        episode_id: str,
        decision_id: str,
        prediction_sink: list[PredictionObservation] | None = None,
        assessment_sink: list[AssessmentObservation] | None = None,
        timing_enabled: bool = False,
        clock: Clock = perf_counter_ns,
    ) -> None:
        self._predict = predict
        self._assess = assess
        self.episode_id = episode_id
        self.decision_id = decision_id
        self.predictions = prediction_sink if prediction_sink is not None else []
        self.assessments = assessment_sink if assessment_sink is not None else []
        self.timing_enabled = timing_enabled
        self.clock = clock
        self._predict_index = 0
        self._assess_index = 0

    def predict(self, state: WorldState, chunk: ActionChunk) -> Trajectory:
        index = self._predict_index
        self._predict_index += 1
        start = self.clock() if self.timing_enabled else None
        try:
            trajectory = self._predict(state, chunk)
        except Exception as exc:
            self.predictions.append(
                PredictionObservation(
                    episode_id=self.episode_id,
                    decision_id=self.decision_id,
                    candidate_index=index,
                    candidate_id=candidate_id(index),
                    proposer_source=chunk.source,
                    action_count=len(chunk.actions),
                    available=False,
                    duration_s=_elapsed_s(start, self.clock, self.timing_enabled),
                    error_code=f"{type(exc).__name__}:{exc}",
                )
            )
            raise
        self.predictions.append(
            PredictionObservation(
                episode_id=self.episode_id,
                decision_id=self.decision_id,
                candidate_index=index,
                candidate_id=candidate_id(index),
                proposer_source=chunk.source,
                action_count=len(chunk.actions),
                available=True,
                horizon_steps=max(0, len(trajectory.states) - 1),
                trajectory_state_count=len(trajectory.states),
                hard_reasons=tuple(trajectory.hard_reasons),
                duration_s=_elapsed_s(start, self.clock, self.timing_enabled),
            )
        )
        return trajectory

    def assess(
        self, state: WorldState, chunk: ActionChunk, trajectory: Trajectory | None
    ) -> CandidateAssessment:
        index = self._assess_index
        self._assess_index += 1
        start = self.clock() if self.timing_enabled else None
        try:
            assessment = self._assess(state, chunk, trajectory)
        except Exception as exc:
            self.assessments.append(
                AssessmentObservation(
                    episode_id=self.episode_id,
                    decision_id=self.decision_id,
                    candidate_index=index,
                    candidate_id=candidate_id(index),
                    duration_s=_elapsed_s(start, self.clock, self.timing_enabled),
                    error_code=f"{type(exc).__name__}:{exc}",
                )
            )
            raise
        components = tuple(float(value) for value in assessment.components)
        if len(components) != 4:
            raise ValueError("CandidateAssessment must contain four components")
        self.assessments.append(
            AssessmentObservation(
                episode_id=self.episode_id,
                decision_id=self.decision_id,
                candidate_index=index,
                candidate_id=candidate_id(index),
                hard_checks=tuple(assessment.hard_checks),
                components=(components[0], components[1], components[2], components[3]),
                duration_s=_elapsed_s(start, self.clock, self.timing_enabled),
            )
        )
        return assessment


def instrument_comparison_planner(
    planner: ComparisonPlanner,
    *,
    episode_id: str,
    decision_id: str,
    timing_enabled: bool = False,
    clock: Clock = perf_counter_ns,
) -> tuple[ComparisonPlanner, PredictionAssessmentProbe]:
    """Return a new planner with instrumented injected hooks; the original is untouched."""
    probe = PredictionAssessmentProbe(
        predict=planner.predict,
        assess=planner.assess,
        episode_id=episode_id,
        decision_id=decision_id,
        timing_enabled=timing_enabled,
        clock=clock,
    )
    instrumented = ComparisonPlanner(
        planner.arm,
        planner.gate,
        predict=probe.predict,
        assess=probe.assess,
    )
    return instrumented, probe


def _hard_result(value: bool | None) -> HardCheckResult:
    if value is True:
        return HardCheckResult.PASS
    if value is False:
        return HardCheckResult.FAIL
    return HardCheckResult.UNKNOWN


def _hard_reason_codes(
    hard_checks: Sequence[tuple[str, bool | None]],
    required_constraints: Sequence[str],
) -> tuple[tuple[str, ...], tuple[tuple[str, HardCheckResult], ...]]:
    reasons = [name for name, result in hard_checks if result is not True]
    names = [name for name, _ in hard_checks]
    reasons.extend(f"missing:{name}" for name in required_constraints if name not in names)
    if len(set(names)) != len(names):
        reasons.append("duplicated-hard-constraints")
    if not hard_checks:
        reasons.append("missing-hard-constraints")
    results = [(name, _hard_result(result)) for name, result in hard_checks]
    for name in required_constraints:
        if name not in names:
            results.append((name, HardCheckResult.NOT_RUN))
    duplicated = {name for name in names if names.count(name) > 1}
    results = [
        (name, HardCheckResult.DUPLICATE if name in duplicated else result)
        for name, result in results
    ]
    return tuple(reasons), tuple(results)


@dataclass(frozen=True)
class DecisionEvidenceBuilder:
    """Merge delegate-only observations into one immutable ``DecisionEvidence`` record."""

    episode_id: str
    decision_id: str
    arm: str
    proposer_source: str
    decision_index: int
    candidate_count: int
    weights: tuple[float, float, float, float]
    maximize: bool
    required_constraints: tuple[str, ...]
    draw_seed: int | None = None
    draw_seeds: tuple[int, ...] = ()
    generation: GenerationObservation | None = None
    predictions: tuple[PredictionObservation, ...] = ()
    assessments: tuple[AssessmentObservation, ...] = ()
    selection: Selection | None = None
    timings: PhaseTimings | None = None
    execution: ExecutionEvidence | None = None
    verification: VerificationEvidence | None = None
    prediction_pairing: PredictionPairingEvidence | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def add_generation(self, observation: GenerationObservation) -> DecisionEvidenceBuilder:
        if (
            observation.episode_id != self.episode_id
            or observation.decision_id != self.decision_id
            or observation.proposer_source != self.proposer_source
            or observation.candidate_count != self.candidate_count
        ):
            raise ValueError("generation observation does not match this decision")
        return replace(
            self,
            generation=observation,
            draw_seed=observation.draw_seed,
            draw_seeds=observation.draw_seeds,
        )

    def add_prediction(self, observation: PredictionObservation) -> DecisionEvidenceBuilder:
        self._check_observation(observation)
        return replace(self, predictions=(*self.predictions, observation))

    def add_assessment(self, observation: AssessmentObservation) -> DecisionEvidenceBuilder:
        self._check_observation(observation)
        return replace(self, assessments=(*self.assessments, observation))

    def set_selection(self, selection: Selection) -> DecisionEvidenceBuilder:
        return replace(self, selection=selection)

    def set_timings(self, timings: PhaseTimings) -> DecisionEvidenceBuilder:
        return replace(self, timings=timings)

    def set_execution(self, evidence: ExecutionEvidence) -> DecisionEvidenceBuilder:
        return replace(self, execution=evidence)

    def set_verification(self, evidence: VerificationEvidence) -> DecisionEvidenceBuilder:
        return replace(self, verification=evidence)

    def set_prediction_pairing(
        self, evidence: PredictionPairingEvidence
    ) -> DecisionEvidenceBuilder:
        return replace(self, prediction_pairing=evidence)

    def _check_observation(
        self, observation: PredictionObservation | AssessmentObservation
    ) -> None:
        if observation.episode_id != self.episode_id or observation.decision_id != self.decision_id:
            raise ValueError("observation does not match this decision")
        if observation.candidate_id != candidate_id(observation.candidate_index):
            raise ValueError("candidate ID must match its sampling order index")
        if observation.candidate_index >= self.candidate_count:
            raise ValueError("candidate index exceeds declared candidate count")

    def _candidate(self, index: int) -> CandidateEvidence:
        assessment = next((row for row in self.assessments if row.candidate_index == index), None)
        prediction = next((row for row in self.predictions if row.candidate_index == index), None)
        generation = self.generation
        seed = (
            generation.draw_seeds[index]
            if generation is not None and index < len(generation.draw_seeds)
            else None
        )
        unevaluable: list[str] = []
        if assessment is None or assessment.error_code is not None or assessment.components is None:
            unevaluable.append("assessment_missing_or_failed")
        if prediction is not None and prediction.error_code is not None:
            unevaluable.append("prediction_failed")
        hard_checks = () if assessment is None else assessment.hard_checks
        reasons, results = _hard_reason_codes(hard_checks, self.required_constraints)
        components = (
            () if assessment is None or assessment.components is None else assessment.components
        )
        costs = make_cost_components(
            components,
            weights=self.weights,
            maximize=self.maximize,
            missing_reason=(
                MissingCostReason.ASSESSMENT_ERROR
                if unevaluable and assessment is not None and assessment.error_code
                else None
            ),
        )
        evaluated = not unevaluable
        safe: bool | None
        if not evaluated:
            safe = None if "assessment_missing_or_failed" in unevaluable else True
            disposition = CandidateDisposition.INVALID_OR_UNEVALUABLE
            rankable = False
        elif reasons:
            safe, rankable = False, False
            disposition = CandidateDisposition.HARD_VETOED
        elif costs.total_j is None:
            safe, rankable = True, False
            disposition = CandidateDisposition.INVALID_OR_UNEVALUABLE
            unevaluable.append("nonfinite_or_missing_cost_components")
        else:
            safe, rankable = True, True
            disposition = CandidateDisposition.SAFE_UNSELECTED
        return CandidateEvidence(
            episode_id=self.episode_id,
            decision_id=self.decision_id,
            candidate_id=candidate_id(index),
            candidate_index=index,
            sampling_seed=seed,
            proposer_source=self.proposer_source,
            candidate_count=self.candidate_count,
            disposition=disposition,
            hard_check_result=(
                HardCheckResult.FAIL
                if reasons
                else HardCheckResult.PASS
                if evaluated
                else HardCheckResult.UNKNOWN
            ),
            hard_veto_reasons=reasons,
            unevaluable_reasons=tuple(unevaluable),
            hard_checks=results,
            safe=safe,
            rankable=rankable,
            costs=costs,
            no_safe_set=self.selection is not None and self.selection.index is None,
            recovery_transition=(
                self.selection.recovery_events[-1]
                if self.selection is not None and self.selection.recovery_events
                else None
            ),
            prediction_available=bool(prediction is not None and prediction.available),
            prediction_horizon_steps=None if prediction is None else prediction.horizon_steps,
            prediction_hard_reasons=() if prediction is None else prediction.hard_reasons,
            trajectory_state_count=(
                None if prediction is None else prediction.trajectory_state_count
            ),
        )

    def build(self) -> DecisionEvidence:
        if len(self.assessments) != self.candidate_count:
            raise ValueError("one assessment per candidate is required")
        candidates = [self._candidate(index) for index in range(self.candidate_count)]
        selection = self.selection
        selected_id = (
            None if selection is None or selection.index is None else candidate_id(selection.index)
        )
        if selected_id is not None:
            selected_index = int(selected_id.split("-")[-1])
            candidates[selected_index] = replace(
                candidates[selected_index],
                disposition=CandidateDisposition.SELECTED,
                selected_candidate_id=selected_id,
            )
        for index, candidate in enumerate(candidates):
            if (
                candidate.disposition is not CandidateDisposition.SELECTED
                and selected_id is not None
            ):
                candidates[index] = replace(candidate, selected_candidate_id=selected_id)
        no_safe_set = selection is not None and selection.index is None
        timing_contract = self.timings or PhaseTimings.unavailable(
            {
                name: "not instrumented in this inert adapter"
                for name in (
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
            }
        )
        return DecisionEvidence(
            episode_id=self.episode_id,
            decision_id=self.decision_id,
            arm=self.arm,
            proposer_source=self.proposer_source,
            decision_index=self.decision_index,
            candidate_count=self.candidate_count,
            draw_seed=self.draw_seed,
            draw_seeds=self.draw_seeds,
            candidates=tuple(candidates),
            selected_candidate_id=selected_id,
            no_safe_set=no_safe_set,
            recovery_events=() if selection is None else selection.recovery_events,
            timings=timing_contract,
            execution=self.execution,
            verification=self.verification,
            prediction_pairing=self.prediction_pairing,
            provenance=dict(self.provenance or {}),
        )


class SafetyMonitorProbe:
    """Delegate ``validate`` and record the pre-apply verdict in a sidecar list."""

    def __init__(self, monitor: Any) -> None:
        self.monitor = monitor
        self.verdicts: list[tuple[Action, ApplyResult]] = []

    def validate(self, action: Action, state: Any) -> ApplyResult:
        verdict = self.monitor.validate(action, state)
        self.verdicts.append((action, verdict))
        return verdict


class BackendApplyProbe:
    """Delegate backend attributes and observe only ``apply`` results."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.apply_results: list[ApplyResult] = []

    def apply(self, action: Action) -> ApplyResult:
        result = self._backend.apply(action)
        self.apply_results.append(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


class VerificationProbe:
    """Delegate verifier calls and record observed results and next transitions."""

    def __init__(self, verifier: Any) -> None:
        self.verifier = verifier
        self.results: list[VerifyResult] = []
        self.transitions: list[str] = []

    def check(self, task: Any, predicted: WorldState, observed: WorldState) -> VerifyResult:
        result = self.verifier.check(task, predicted, observed)
        self.results.append(result)
        return result

    def next_action(self, result: VerifyResult) -> str:
        action = self.verifier.next_action(result)
        self.transitions.append(action)
        return action

    def __getattr__(self, name: str) -> Any:
        return getattr(self.verifier, name)


def wrap_execute_first(
    execute: Callable[[Selection, Observation, Any, Any], tuple[Observation, ApplyResult]],
    *,
    decision_id: str,
    selected_candidate_id: str | None,
) -> tuple[
    Callable[[Selection, Observation, Any, Any], tuple[Observation, ApplyResult]],
    list[ExecutionEvidence],
]:
    """Return a whole-call wrapper and its sidecar sink; delegated calls are unchanged."""
    observations: list[ExecutionEvidence] = []

    def wrapped(
        selection: Selection, observation: Observation, backend: Any, monitor: Any
    ) -> tuple[Observation, ApplyResult]:
        executed = None if selection.index is None else candidate_id(selection.index)
        if executed != selected_candidate_id:
            raise ValueError("execution wrapper selection does not match the evidence decision")
        result_observation, result = execute(selection, observation, backend, monitor)
        observations.append(
            ExecutionEvidence(
                decision_id=decision_id,
                selected_candidate_id=selected_candidate_id,
                executed_candidate_id=executed,
                apply_accepted=result.accepted,
                apply_reason=result.reason,
                backend_stepped=result_observation is not observation,
                observation_t_s=observation.t,
                action_t_s=result_observation.t if result_observation is not observation else None,
            )
        )
        return result_observation, result

    return wrapped, observations


def build_execution_evidence_from_probes(
    *,
    decision_id: str,
    selected_candidate_id: str | None,
    observation: Observation,
    result: ApplyResult,
    monitor_probe: SafetyMonitorProbe,
    backend_probe: BackendApplyProbe,
    observation_after: Observation,
) -> ExecutionEvidence:
    """Assemble pre-apply and apply evidence from delegate-only probes."""
    verdict = monitor_probe.verdicts[-1][1] if monitor_probe.verdicts else None
    apply = backend_probe.apply_results[-1] if backend_probe.apply_results else None
    return ExecutionEvidence(
        decision_id=decision_id,
        selected_candidate_id=selected_candidate_id,
        executed_candidate_id=selected_candidate_id,
        pre_apply_accepted=None if verdict is None else verdict.accepted,
        pre_apply_reason=None if verdict is None else verdict.reason,
        apply_accepted=None if apply is None else apply.accepted,
        apply_reason=None if apply is None else apply.reason,
        backend_stepped=observation_after is not observation,
        observation_t_s=observation.t,
        action_t_s=observation_after.t if observation_after is not observation else None,
    )
