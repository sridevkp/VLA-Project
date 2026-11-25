"""Offline FakeBackend composition for ADR-0084's prediction-free scripted ablation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import (
    ActionChunk,
    ApplyResult,
    Observation,
    Pose3D,
    TaskSpec,
    Vec3,
    WorldState,
)
from vlarobot.eval.decision_adapters import (
    CandidateGenerationProbe,
    GenerationObservation,
    candidate_id,
)
from vlarobot.eval.report_contract import ReportDefinitions, Table51MetricValue
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
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import JointJitter, ScriptedChunkAdapter
from vlarobot.execution.worldmind_pipeline import _inverse_pose, _world_state  # noqa: PLC2701
from vlarobot.planning.candidates import SeededCandidateGenerator
from vlarobot.planning.verification import PlacementVerifier
from vlarobot.risk.gate import RecoveryState
from vlarobot.world_model.rollout import compose

ARM = "vla_risk_veto"
FINITE_CHUNK = "finite_chunk_v1"
_DEFINITIONS_SHA256 = "457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14"


@dataclass(frozen=True)
class HeuristicCandidate:
    candidate_id: str
    draw_index: int
    draw_seed: int
    chunk: ActionChunk
    finite_chunk: bool | None
    first_command: ApplyResult | None
    disposition: Literal["selected", "eligible", "rejected"]


@dataclass(frozen=True)
class HeuristicDecision:
    generation: GenerationObservation
    candidates: tuple[HeuristicCandidate, ...]
    selected_candidate_id: str | None
    transition: str
    reason: str
    motor_verdict: ApplyResult | None = None
    backend_verdict: ApplyResult | None = None


@dataclass(frozen=True)
class HeuristicFixtureEpisode:
    provenance: Mapping[str, str]
    decisions: tuple[HeuristicDecision, ...]
    terminal_state: str
    confirmed_ticks: int
    empty_sets: int
    verification_retries: int
    prediction_error: Table51MetricValue

    @property
    def evaluated_candidates(self) -> int:
        return sum(len(decision.candidates) for decision in self.decisions)

    @property
    def rejected_candidates(self) -> int:
        return sum(
            candidate.disposition == "rejected"
            for decision in self.decisions
            for candidate in decision.candidates
        )

    @property
    def first_command_rejections(self) -> int:
        return sum(
            candidate.first_command is None or candidate.first_command.accepted is not True
            for decision in self.decisions
            for candidate in decision.candidates
        )

    @property
    def motor_vetoes(self) -> int:
        return sum(
            decision.motor_verdict is not None and decision.motor_verdict.accepted is not True
            for decision in self.decisions
        )


@dataclass(frozen=True)
class HeuristicComparisonFixture:
    """Small fixture runner. Both recovery allowances are scoped to the whole fake episode."""

    config: WorldMindDiagnosticConfig
    bindings: WorldMindBindings
    provenance: Mapping[str, str]

    def _pool(
        self,
        task: TaskSpec,
        state: WorldState,
        monitor: SafetyMonitor,
        *,
        episode_id: str,
        seed: int,
        decision_index: int,
    ) -> tuple[GenerationObservation, tuple[ActionChunk, ...]]:
        config = self.config
        adapter_spec = config.adapter
        waypoint_factory = self.bindings.waypoint_factories.get(adapter_spec.waypoints_binding)
        waypoints = (
            waypoint_factory(config, state, task)
            if waypoint_factory is not None
            else self.bindings.waypoint_sets[adapter_spec.waypoints_binding]
        )
        solver = self.bindings.pose_solvers[adapter_spec.pose_solver_binding]
        adapter = ScriptedChunkAdapter(
            waypoints,
            solver=solver,
            monitor=monitor,
            max_joint_step_rad=float(solver.execution_max_joint_step_rad),
            horizon_steps=config.pipeline.horizon_steps,
            control_dt_s=config.pipeline.control_dt_s,
            jitter=JointJitter(
                adapter_spec.jitter.joint_std_rad,
                adapter_spec.jitter.joint_limit_rad,
                adapter_spec.jitter.max_attempts,
            ),
            position_tolerance_m=adapter_spec.ik_position_tolerance_m,
            orientation_tolerance_rad=adapter_spec.ik_orientation_tolerance_rad,
        )
        generator = SeededCandidateGenerator(
            adapter.draw(),
            seed=config.pipeline.decision_seed_base + seed,
            count=config.pipeline.candidate_count,
            source="scripted",
        )
        decision_id = f"{episode_id}-d{decision_index:02d}"
        probe = CandidateGenerationProbe(generator, episode_id=episode_id, decision_id=decision_id)
        pool = tuple(probe.propose(task, state, None))
        (generation,) = probe.observations
        draw = generator.draw_log[-1]
        if (
            len(pool) != FROZEN_CANDIDATE_COUNT
            or generation.candidate_count != len(pool)
            or generation.episode_id != episode_id
            or generation.decision_id != decision_id
            or generation.proposer_source != "scripted"
            or generation.decision_index != draw["decision"]
            or generation.draw_seed != draw["seed"]
            or len(generation.draw_seeds) != len(pool)
            or generation.draw_seeds != draw["draw_seeds"]
            or any(
                chunk.source != "scripted"
                or len(chunk.actions) != FROZEN_HORIZON_STEPS
                or chunk.dt != FROZEN_CONTROL_DT_S
                for chunk in pool
            )
        ):
            raise ValueError("inconsistent frozen scripted candidate pool")
        return generation, pool

    def assess_pool(
        self,
        state: WorldState,
        observation: Observation,
        monitor: SafetyMonitor,
        generation: GenerationObservation,
        pool: tuple[ActionChunk, ...],
    ) -> tuple[list[HeuristicCandidate], int | None]:
        """Evaluate ADR-0084 eligibility in draw order without applying a command."""
        if len(pool) != FROZEN_CANDIDATE_COUNT or len(generation.draw_seeds) != len(pool):
            raise ValueError("incomplete heuristic pool")
        records: list[HeuristicCandidate] = []
        selected = None
        for index, chunk in enumerate(pool):
            finite = self.bindings.hard_checks[FINITE_CHUNK](state, chunk, None)
            first = monitor.validate(chunk.actions[0], observation.robot)
            if not isinstance(first, ApplyResult) or type(first.accepted) is not bool:
                first = None
            disposition: Literal["selected", "eligible", "rejected"] = "rejected"
            if finite is True and first is not None and first.accepted is True:
                disposition = "eligible"
                if selected is None:
                    selected = index
            records.append(
                HeuristicCandidate(
                    candidate_id(index),
                    index,
                    generation.draw_seeds[index],
                    chunk,
                    finite if isinstance(finite, bool) else None,
                    first,
                    disposition,
                )
            )
        return records, selected

    def build_verifier(self, observation: Observation) -> PlacementVerifier:
        config = self.config
        bindings = self.bindings
        base = observation.robot.base_pose_world
        if base is None:
            raise ValueError("heuristic ablation requires a world-to-base transform")
        inverse = _inverse_pose(base)
        identity = (0.0, 0.0, 0.0, 1.0)
        verification = config.verification
        verifier = bindings.verification[verification.binding](
            config,
            compose(inverse, Pose3D(Vec3(*verification.destination_world), identity)),
            compose(inverse, Pose3D(Vec3(*verification.rest_center_world), identity)),
            bindings.inputs.get("verification", {}),
        )
        if not isinstance(verifier, PlacementVerifier):
            raise TypeError("heuristic ablation requires PlacementVerifier")
        return verifier

    def execute_fixture_episode(
        self, backend: FakeBackend, observation: Observation, *, episode_id: str, seed: int
    ) -> HeuristicFixtureEpisode:
        """Exercise approved selection and recovery with an already initialized fake observation."""
        if not isinstance(backend, FakeBackend):
            raise TypeError("offline heuristic ablation requires FakeBackend")
        if type(seed) is not int or seed not in FROZEN_SEEDS:
            raise ValueError("offline heuristic ablation requires a frozen episode seed")
        config = self.config
        bindings = self.bindings
        monitor = bindings.safety[config.safety.binding](config, bindings.inputs.get("safety", {}))
        if not isinstance(monitor, SafetyMonitor):
            raise TypeError("offline heuristic ablation requires a SafetyMonitor")
        verifier = self.build_verifier(observation)
        task = TaskSpec(config.task.instruction, config.task.target, config.task.destination)
        no_safe = RecoveryState()
        decisions: list[HeuristicDecision] = []
        ticks = 0
        terminal = "WORLDMIND_BOUND_EXHAUSTED"
        while len(decisions) < config.bounds.max_decisions and ticks < config.bounds.max_steps:
            state = _world_state(
                observation, backend.get_ground_truth(), config, target=task.target
            )
            generation, pool = self._pool(
                task,
                state,
                monitor,
                episode_id=episode_id,
                seed=seed,
                decision_index=len(decisions),
            )
            records, selected = self.assess_pool(state, observation, monitor, generation, pool)
            if selected is None:
                event = no_safe.no_safe()
                if event == "reobserve":
                    observation = backend.step(1)
                    ticks += 1
                    decisions.append(
                        HeuristicDecision(
                            generation, tuple(records), None, event, "empty eligible set"
                        )
                    )
                    continue
                decisions.append(
                    HeuristicDecision(
                        generation,
                        tuple(records),
                        None,
                        "alternate_candidate_family:unavailable",
                        "empty eligible set and alternate binding is null",
                    )
                )
                terminal = "NO_SAFE_SET"
                break

            records[selected] = replace(records[selected], disposition="selected")
            picked = records[selected]
            action = picked.chunk.actions[0]
            motor = monitor.validate(action, observation.robot)
            if not isinstance(motor, ApplyResult):
                raise TypeError("invalid execution-time motor verdict")
            applied = backend.apply(action) if motor.accepted is True else None
            if applied is not None and not isinstance(applied, ApplyResult):
                raise TypeError("invalid backend apply verdict")
            if applied is None or applied.accepted is not True:
                terminal = "SAFETY_ABORT" if applied is None else "BACKEND_ABORT"
                verdict = motor if applied is None else applied
                decisions.append(
                    HeuristicDecision(
                        generation,
                        tuple(records),
                        picked.candidate_id,
                        terminal,
                        verdict.reason or terminal,
                        motor,
                        applied,
                    )
                )
                break

            observation = backend.step(1)
            ticks += 1
            observed = _world_state(
                observation, backend.get_ground_truth(), config, target=task.target
            )
            result = verifier.check_observed(task, observed)
            event = verifier.next_action(result)
            decisions.append(
                HeuristicDecision(
                    generation,
                    tuple(records),
                    picked.candidate_id,
                    event,
                    result.reason,
                    motor,
                    applied,
                )
            )
            if event != "retry_once":
                terminal = "COMPLETE" if event == "complete" else "VERIFICATION_ABORT"
                break
        return HeuristicFixtureEpisode(
            self.provenance,
            tuple(decisions),
            terminal,
            ticks,
            no_safe.no_safe_count,
            verifier.recovery.verification_retries,
            Table51MetricValue(
                "not_applicable", None, "arm_has_no_prediction", valid_paired_steps=0
            ),
        )


def build_offline_heuristic_comparison(
    config: WorldMindDiagnosticConfig, bindings: WorldMindBindings
) -> HeuristicComparisonFixture:
    """Validate the frozen contract and return a fixture with no live entry point."""
    pipe = config.pipeline
    if (
        config.split != FROZEN_SPLIT
        or config.backend.state_source != FROZEN_STATE_SOURCE
        or config.diagnostic.scored_evaluation_allowed
        or config.adapter.alternate_waypoints_binding is not None
        or config.assessment.required_constraints != (FINITE_CHUNK,)
        or config.assessment.hard_checks != (FINITE_CHUNK,)
        or not 1 <= config.bounds.max_steps <= 1059
        or not 1 <= config.bounds.max_decisions <= 1060
        or (
            pipe.proposer,
            pipe.candidate_count,
            pipe.horizon_steps,
            pipe.control_dt_s,
            pipe.decision_seed_base,
        )
        != (
            FROZEN_PROPOSER,
            FROZEN_CANDIDATE_COUNT,
            FROZEN_HORIZON_STEPS,
            FROZEN_CONTROL_DT_S,
            FROZEN_BASE_SEED,
        )
    ):
        raise ValueError(
            "offline heuristic ablation requires the frozen scripted diagnostic contract"
        )
    report = config.report
    definitions = ReportDefinitions.load(report.definitions, report.pdf)
    definitions_sha256 = hashlib.sha256(report.definitions.read_bytes()).hexdigest()
    if (
        definitions != config.report_definitions
        or definitions.report_sha256 != report.sha256
        or definitions_sha256 != _DEFINITIONS_SHA256
    ):
        raise ValueError("approved Table 5.1 definitions do not match the diagnostic config")
    for group, name in (
        (bindings.hard_checks, FINITE_CHUNK),
        (bindings.safety, config.safety.binding),
        (bindings.verification, config.verification.binding),
    ):
        if not callable(group.get(name)):
            raise ValueError(f"missing heuristic ablation binding: {name}")
    adapter = config.adapter
    if not callable(
        bindings.waypoint_factories.get(adapter.waypoints_binding)
    ) and not bindings.waypoint_sets.get(adapter.waypoints_binding):
        raise ValueError("missing primary scripted waypoint binding")
    solver = bindings.pose_solvers.get(adapter.pose_solver_binding)
    bound = getattr(solver, "execution_max_joint_step_rad", None)
    if not callable(getattr(solver, "solve", None)) or (
        type(bound) not in (float, int) or not math.isfinite(bound) or bound <= 0
    ):
        raise ValueError("missing reviewed pose solver or execution step bound")
    return HeuristicComparisonFixture(
        config,
        bindings,
        {
            "arm": ARM,
            "proposer_source": "scripted",
            "scripted_adapter_binding": pipe.proposer,
            "prediction_mode": "none",
            "veto_mode": "finite_chunk_and_first_command_prefilter_plus_motor",
            "selection_mode": "first_eligible_in_draw_order",
            "objective_score": "not_applicable",
            "objective_contract": "ADR-0084 arm-specific Eq. 3.8 ranking exception",
            "scientific_contrast": "prediction availability and selection policy both differ",
            "calibration_status": "not_applicable_no_heuristic_score",
            "objective_definitions_sha256": definitions_sha256,
            "report_sha256": definitions.report_sha256,
            "execution_scope": "fixture_only",
            "scientific_execution_approved": "false",
        },
    )
