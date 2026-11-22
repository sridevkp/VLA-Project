"""Offline-only composition for the frozen scripted-primary world-model ablation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import ActionChunk, ApplyResult, Observation, TaskSpec, WorldState
from vlarobot.eval.decision_adapters import (
    CandidateGenerationProbe,
    GenerationObservation,
    candidate_id,
)
from vlarobot.eval.report_contract import COMPONENTS, ReportDefinitions
from vlarobot.eval.worldmind_calibration import load_calibration_candidate
from vlarobot.eval.worldmind_config import (
    FROZEN_BASE_SEED,
    FROZEN_CANDIDATE_COUNT,
    FROZEN_CONTROL_DT_S,
    FROZEN_HORIZON_STEPS,
    FROZEN_PROPOSER,
    FROZEN_SEEDS,
    FROZEN_SPLIT,
    FROZEN_STATE_SOURCE,
    WorldMindBindings,
    WorldMindDiagnosticConfig,
    validate_worldmind_diagnostic_bindings,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import JointJitter, ScriptedChunkAdapter
from vlarobot.execution.worldmind_pipeline import _world_state  # noqa: PLC2701
from vlarobot.planning.candidates import SeededCandidateGenerator
from vlarobot.planning.execution import execute_first
from vlarobot.risk.comparison import ComparisonPlanner, Selection
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.rollout import AnalyticSurrogate, Trajectory

ARM = "vla_world_model"
# Approved report definition bytes pinned in ADR-0079.
_DEFINITIONS_SHA256 = "457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14"


@dataclass(frozen=True)
class FixtureRanking:
    """Explicit in-memory fixture inputs. They carry no calibration authority."""

    components: Mapping[str, Callable[[WorldState, ActionChunk, Trajectory | None], float]]
    weights: tuple[float, float, float, float]
    maximize: bool


@dataclass(frozen=True)
class ComparisonFixtureDecision:
    provenance: Mapping[str, str]
    generation: GenerationObservation
    candidate_ids: tuple[str, ...]
    candidates: tuple[ActionChunk, ...]
    selection: Selection
    selected_candidate_id: str | None
    apply_result: ApplyResult
    observation_after: Observation


@dataclass(frozen=True)
class WorldModelComparisonFixture:
    """One inert decision using production construction bindings and injected ranking inputs."""

    config: WorldMindDiagnosticConfig
    bindings: WorldMindBindings
    provenance: Mapping[str, str]
    planner: ComparisonPlanner
    monitor: SafetyMonitor

    def execute_fixture_tick(
        self,
        backend: FakeBackend,
        observation: Observation,
        *,
        episode_id: str,
        seed: int,
        decision_index: int = 0,
    ) -> ComparisonFixtureDecision:
        """Generate and rank a complete pool, then exercise the motor veto for one fake tick."""
        if not isinstance(backend, FakeBackend):
            raise TypeError("offline comparison execution requires FakeBackend")
        if type(seed) is not int or seed not in FROZEN_SEEDS:
            raise ValueError("offline comparison requires a frozen episode seed")
        task, state = self.prepare_fixture_input(backend, observation)
        generation, candidates = self.generate_fixture_pool(
            task, state, episode_id=episode_id, seed=seed, decision_index=decision_index
        )
        selection = self.planner.select(state, candidates)
        after, applied = execute_first(selection, observation, backend, self.monitor)
        return ComparisonFixtureDecision(
            provenance=self.provenance,
            generation=generation,
            candidate_ids=tuple(candidate_id(i) for i in range(len(candidates))),
            candidates=candidates,
            selection=selection,
            selected_candidate_id=None
            if selection.index is None
            else candidate_id(selection.index),
            apply_result=applied,
            observation_after=after,
        )

    def prepare_fixture_input(
        self, backend: FakeBackend, observation: Observation
    ) -> tuple[TaskSpec, WorldState]:
        """Expose the actual fixture task and observed world state to offline capture."""
        if not isinstance(backend, FakeBackend):
            raise TypeError("offline comparison execution requires FakeBackend")
        task = TaskSpec(
            instruction=self.config.task.instruction,
            target=self.config.task.target,
            destination=self.config.task.destination,
        )
        return task, _world_state(
            observation, backend.get_ground_truth(), self.config, target=task.target
        )

    def generate_fixture_pool(
        self, task: TaskSpec, state: WorldState, *, episode_id: str, seed: int, decision_index: int
    ) -> tuple[GenerationObservation, tuple[ActionChunk, ...]]:
        """Share the existing all-or-error generator with the acquisition adapter."""
        adapter_config = self.config.adapter
        waypoint_factory = self.bindings.waypoint_factories.get(adapter_config.waypoints_binding)
        waypoints = (
            waypoint_factory(self.config, state, task)
            if waypoint_factory is not None
            else self.bindings.waypoint_sets[adapter_config.waypoints_binding]
        )
        solver = self.bindings.pose_solvers[adapter_config.pose_solver_binding]
        bound = getattr(solver, "execution_max_joint_step_rad", None)
        if bound is None or not math.isfinite(float(bound)):
            raise ValueError("registered pose solver must expose a finite execution step limit")
        adapter = ScriptedChunkAdapter(
            waypoints,
            solver=solver,
            monitor=self.monitor,
            max_joint_step_rad=float(bound),
            horizon_steps=self.config.pipeline.horizon_steps,
            control_dt_s=self.config.pipeline.control_dt_s,
            jitter=JointJitter(
                std_rad=adapter_config.jitter.joint_std_rad,
                limit_rad=adapter_config.jitter.joint_limit_rad,
                max_attempts=adapter_config.jitter.max_attempts,
            ),
            position_tolerance_m=adapter_config.ik_position_tolerance_m,
            orientation_tolerance_rad=adapter_config.ik_orientation_tolerance_rad,
        )
        generator = SeededCandidateGenerator(
            adapter.draw(),
            seed=self.config.pipeline.decision_seed_base + seed,
            count=self.config.pipeline.candidate_count,
            source="scripted",
        )
        probe = CandidateGenerationProbe(
            generator, episode_id=episode_id, decision_id=f"{episode_id}-d{decision_index:02d}"
        )
        candidates = tuple(probe.propose(task, state, None))
        return probe.observations[0], candidates


@dataclass(frozen=True)
class OfflineWorldModelComparison:
    """Readiness result. Only an explicitly synthetic fixture can execute a tick."""

    provenance: Mapping[str, str]
    blockers: tuple[str, ...]
    fixture: WorldModelComparisonFixture | None = None


def build_offline_world_model_comparison(
    config: WorldMindDiagnosticConfig,
    bindings: WorldMindBindings,
    *,
    fixture_ranking: FixtureRanking | None = None,
    calibration_candidate: Path | None = None,
) -> OfflineWorldModelComparison:
    """Validate definitions, report scientific blockers, or compose an explicit inert fixture."""
    pipeline = config.pipeline
    if (
        config.split != FROZEN_SPLIT
        or config.backend.state_source != FROZEN_STATE_SOURCE
        or pipeline.proposer != FROZEN_PROPOSER
        or pipeline.candidate_count != FROZEN_CANDIDATE_COUNT
        or pipeline.horizon_steps != FROZEN_HORIZON_STEPS
        or pipeline.control_dt_s != FROZEN_CONTROL_DT_S
        or pipeline.decision_seed_base != FROZEN_BASE_SEED
        or config.diagnostic.scored_evaluation_allowed
    ):
        raise ValueError("offline comparison requires the frozen scripted diagnostic contract")
    report = config.report
    definitions = ReportDefinitions.load(report.definitions, report.pdf)
    definitions_sha256 = hashlib.sha256(report.definitions.read_bytes()).hexdigest()
    if (
        definitions_sha256 != _DEFINITIONS_SHA256
        or definitions != config.report_definitions
        or definitions.report_sha256 != report.sha256
    ):
        raise ValueError("approved objective definitions do not match the diagnostic config")
    provenance = {
        "arm": ARM,
        "proposer_source": "scripted",
        "scripted_adapter_binding": pipeline.proposer,
        "prediction_mode": "analytic_geometric_kinematic",
        "surrogate_binding": config.surrogate.binding,
        "veto_mode": "motor_safety_only",
        "objective_contract": "ADR-0065 Eq. 3.8",
        "objective_definitions_sha256": definitions_sha256,
        "report_sha256": definitions.report_sha256,
        "calibration_status": "absent",
        "execution_scope": "non_executable",
        "scientific_execution_approved": "false",
    }
    if fixture_ranking is None:
        if calibration_candidate is None:
            return OfflineWorldModelComparison(provenance, ("approved calibration missing",))
        approval = load_calibration_candidate(calibration_candidate)
        provenance.update(
            calibration_status="candidate_artifact_validated",
            calibration_id=approval.calibration_id,
            calibration_candidate_sha256=approval.sha256,
            calibration_evidence_id=approval.evidence.evidence_id,
            calibration_evidence_sha256=approval.evidence.sha256,
        )
        return OfflineWorldModelComparison(
            provenance, ("offline-only scope; no arm-specific scientific activation",)
        )
    if calibration_candidate is not None:
        raise ValueError("fixture ranking cannot be combined with an approved calibration artifact")
    if set(fixture_ranking.components) != set(COMPONENTS) or not all(
        callable(component) for component in fixture_ranking.components.values()
    ):
        raise ValueError("all four Eq. 3.8 fixture component bindings are required")

    validate_worldmind_diagnostic_bindings(config, bindings)
    monitor = bindings.safety[config.safety.binding](config, bindings.inputs.get("safety", {}))
    if not isinstance(monitor, SafetyMonitor):
        raise TypeError("offline comparison requires a SafetyMonitor")
    predictor = bindings.surrogates[config.surrogate.binding](
        config, bindings.inputs.get("surrogate", {})
    )
    if not isinstance(predictor, AnalyticSurrogate):
        raise TypeError("offline comparison requires the analytic surrogate")

    components = tuple(fixture_ranking.components[name] for name in COMPONENTS)

    def assess(
        state: WorldState, chunk: ActionChunk, trajectory: Trajectory | None
    ) -> CandidateAssessment:
        values = cast(
            tuple[float, float, float, float],
            tuple(float(component(state, chunk, trajectory)) for component in components),
        )
        return CandidateAssessment((), values)

    # ComparisonPlanner uses only weights and ranking direction on this arm.
    ranking = RiskGate(
        weights=fixture_ranking.weights,
        maximize=fixture_ranking.maximize,
        required_constraints=config.assessment.required_constraints,
    )
    planner = ComparisonPlanner(ARM, ranking, predict=predictor.predict_trajectory, assess=assess)
    provenance.update(
        calibration_status="fixture_inputs_only",
        objective_binding_scope="fixture_only",
        execution_scope="fixture_only",
    )
    return OfflineWorldModelComparison(
        provenance,
        ("approved calibration missing",),
        WorldModelComparisonFixture(config, bindings, provenance, planner, monitor),
    )
