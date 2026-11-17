"""Production WorldMind composition: scripted proposer -> surrogate -> gate -> live safety.

This module is the single production call graph for the ``full_system`` arm. It reuses
``SeededCandidateGenerator``, an owner-registered ``AnalyticSurrogate`` factory,
``ComparisonPlanner``, ``RiskGate``, ``execute_first``, ``SafetyMonitor``, ``PlacementVerifier``
and the frozen evaluation recorder. It does not implement a second planner or scorer.

Execution is one control tick per decision: the selected chunk's first action crosses the live
safety veto through ``execute_first`` and the world is re-observed before the next decision.
The H-step prediction is paired only with truthfully observed, explicitly truncated states;
the unexecuted horizon is never claimed to have been executed or scored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, cast

import numpy as np

from vlarobot.core.types import (
    GroundTruth,
    ObjectState,
    Observation,
    Pose3D,
    RobotState,
    SceneState,
    TaskSpec,
    Vec3,
    VerifyResult,
    WorldState,
)
from vlarobot.eval.decision_adapters import (
    AssessmentObservation,
    BackendApplyProbe,
    CandidateGenerationProbe,
    DecisionEvidenceBuilder,
    PredictionAssessmentProbe,
    SafetyMonitorProbe,
    VerificationProbe,
    build_execution_evidence_from_probes,
    candidate_id,
    wrap_execute_first,
)
from vlarobot.eval.decision_evidence import PhaseTimings, VerificationEvidence
from vlarobot.eval.decision_metrics import pair_executed_prediction
from vlarobot.eval.runner import EvalReport, EvalRunConfig, run_evaluation
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.eval.worldmind_config import (
    FROZEN_ARM,
    GATE_CALL_POLICY,
    WorldMindBindings,
    WorldMindConfig,
)
from vlarobot.eval.worldmind_evidence import WorldMindEvidenceWriter
from vlarobot.execution.scripted_chunk_adapter import (
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.planning.candidates import SeededCandidateGenerator
from vlarobot.planning.execution import execute_first
from vlarobot.risk.comparison import ComparisonPlanner
from vlarobot.risk.gate import CandidateAssessment, RecoveryState, RiskGate
from vlarobot.world_model.rollout import Trajectory, compose, rotate
from vlarobot.world_model.state_copy import copy_robot

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inverse_pose(pose: Pose3D) -> Pose3D:
    """Invert a world-to-base pose; reject malformed transforms instead of repairing them."""
    quaternion = np.asarray(pose.orientation, dtype=np.float64)
    position = pose.position.as_array()
    if quaternion.shape != (4,) or not np.isfinite(np.r_[quaternion, position]).all():
        raise ValueError("finite xyzw pose required for frame conversion")
    norm = float(quaternion @ quaternion)
    if not math.isclose(norm, 1.0, abs_tol=1e-8):
        raise ValueError("unit xyzw pose required for frame conversion")
    inverse = np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]))
    translated = rotate(inverse, -position)
    return Pose3D(
        Vec3(*map(float, translated)),
        (float(inverse[0]), float(inverse[1]), float(inverse[2]), float(inverse[3])),
    )


def _world_pose(ground_truth: GroundTruth, target: str) -> Pose3D:
    pose = ground_truth.object_poses.get(target)
    if pose is None:
        raise ValueError(f"oracle state source has no {target} pose")
    return pose


def _world_state(
    observation: Observation,
    ground_truth: GroundTruth,
    config: WorldMindConfig,
    *,
    target: str,
) -> WorldState:
    """Build an oracle-initialized, explicit base-frame WorldState; no perception is claimed."""
    base = observation.robot.base_pose_world
    if base is None:
        raise ValueError("oracle state source requires base_pose_world")
    pose_base = compose(_inverse_pose(base), _world_pose(ground_truth, target))
    obj = ObjectState(
        object_id=target,
        label="can",
        pose=pose_base,
        dimensions=Vec3(*config.verification.object_dimensions_m),
        confidence=1.0,
    )
    return WorldState(SceneState([obj]), copy_robot(observation.robot))


class _CountingGate(RiskGate):
    """Delegate-only wrapper recording the existing N+1 gate calls and gate-visible assessments."""

    def __init__(self, gate: RiskGate) -> None:
        super().__init__(
            weights=gate.weights,
            maximize=gate.maximize,
            required_constraints=gate.required_constraints,
        )
        self._gate = gate
        self.calls = 0
        self.full_assessments: tuple[CandidateAssessment, ...] = ()

    def select(self, candidates: list[CandidateAssessment]) -> Any:
        self.calls += 1
        if len(candidates) > 1:
            self.full_assessments = tuple(candidates)
        return self._gate.select(candidates)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gate, name)


@dataclass
class _TrajectoryCapture:
    """Keep real predictor trajectories for selected-candidate pairing."""

    predictor: Any
    trajectories: list[Trajectory] = field(default_factory=list)

    def predict(self, state: WorldState, chunk: Any) -> Trajectory:
        trajectory = self.predictor.predict_trajectory(state, chunk)
        if not isinstance(trajectory, Trajectory):
            raise TypeError("registered surrogate must return a Trajectory")
        self.trajectories.append(trajectory)
        return trajectory


def _assessment_rows(
    probe: PredictionAssessmentProbe, gate: _CountingGate
) -> tuple[AssessmentObservation, ...]:
    """Use the augmented assessments the gate actually ranked, preserving measured timings."""
    rows = list(probe.assessments)
    if not gate.full_assessments:
        return tuple(rows)
    if len(rows) != len(gate.full_assessments):
        raise ValueError("assessment observation count differs from gate-visible assessments")
    return tuple(
        replace(
            row,
            hard_checks=tuple(assessment.hard_checks),
            components=(  # type: ignore[arg-type]
                float(assessment.components[0]),
                float(assessment.components[1]),
                float(assessment.components[2]),
                float(assessment.components[3]),
            ),
        )
        for row, assessment in zip(rows, gate.full_assessments, strict=True)
    )


def _timings(
    *,
    enabled: bool,
    generation_s: float | None,
    prediction_samples: Sequence[float | None],
    planning_s: float | None,
    selection_s: float | None,
    verification_s: float | None,
    recovery_s: float | None,
) -> PhaseTimings:
    if not enabled:
        return PhaseTimings.unavailable(
            {
                "candidate_generation_s": "phase timing disabled in reviewed config",
                "surrogate_prediction_s": "phase timing disabled in reviewed config",
                "hard_safety_checks_s": "phase timing disabled in reviewed config",
                "riskgate_ranking_s": "phase timing disabled in reviewed config",
                "selection_s": "phase timing disabled in reviewed config",
                "planning_end_to_end_s": "phase timing disabled in reviewed config",
                "model_inference_s": "analytic surrogate performs no neural inference",
                "verification_s": "phase timing disabled in reviewed config",
                "recovery_s": "phase timing disabled in reviewed config",
            }
        )
    missing: dict[str, str] = {
        "hard_safety_checks_s": (
            "hard checks and Eq. 3.8 ranking are fused inside ComparisonPlanner.select; "
            "only the whole selection window is separable"
        ),
        "riskgate_ranking_s": (
            "RiskGate.select is called N+1 times inside ComparisonPlanner.select and is not "
            "separately wrapped; only whole selection_s is measured"
        ),
        "model_inference_s": "analytic surrogate performs no neural inference",
    }
    if generation_s is None:
        missing["candidate_generation_s"] = "no generation observation was recorded"
    if not prediction_samples or any(value is None for value in prediction_samples):
        missing["surrogate_prediction_s"] = "one or more surrogate predictions were not timed"
    if planning_s is None:
        missing["planning_end_to_end_s"] = "planning window was not measured"
    if selection_s is None:
        missing["selection_s"] = "selection window was not measured"
    if verification_s is None:
        missing["verification_s"] = "no verification was reached for this decision"
    if recovery_s is None:
        missing["recovery_s"] = "no recovery transition was measured for this decision"
    prediction_total = (
        None
        if "surrogate_prediction_s" in missing
        else float(sum(float(value) for value in prediction_samples if value is not None))
    )
    return PhaseTimings(
        candidate_generation_s=generation_s,
        surrogate_prediction_s=prediction_total,
        hard_safety_checks_s=None,
        riskgate_ranking_s=None,
        selection_s=selection_s,
        planning_end_to_end_s=planning_s,
        model_inference_s=None,
        verification_s=verification_s,
        recovery_s=recovery_s,
        missing_reasons=missing,
    )


def _provenance(config: WorldMindConfig, plan: Sequence[Any]) -> dict[str, str]:
    return {
        "schema_version": "1",
        "config_path": str(config.source_path),
        "config_sha256": _sha256(config.source_path),
        "report_pdf": str(config.report.pdf),
        "report_sha256": config.report.sha256,
        "definitions_path": str(config.report.definitions),
        "definitions_sha256": _sha256(config.report.definitions),
        "calibration_evidence_id": config.calibration.evidence_id,
        "calibration_id": config.calibration.calibration_id,
        "calibration_candidate_sha256": config.calibration.candidate_sha256,
        "calibration_evidence_sha256": config.calibration.evidence_sha256,
        "calibration_selected_candidate_id": config.calibration.selected_candidate_id,
        "calibration_parameters": json.dumps(dict(config.calibration.parameters), sort_keys=True),
        "calibration_provenance": json.dumps(dict(config.calibration.provenance), sort_keys=True),
        "backend": config.backend.kind,
        "backend_declaration": config.backend.declaration,
        "state_source": config.backend.state_source,
        "split": config.split,
        "base_seed": str(config.base_seed),
        "seeds": ",".join(str(value) for value in config.seeds),
        "episodes": str(len(plan)),
        "candidate_count": str(config.pipeline.candidate_count),
        "horizon_steps": str(config.pipeline.horizon_steps),
        "control_dt_s": str(config.pipeline.control_dt_s),
        "component_bindings": json.dumps(
            {name: spec.binding for name, spec in config.components.items()}, sort_keys=True
        ),
        "component_units": json.dumps(
            {name: spec.unit for name, spec in config.components.items()}, sort_keys=True
        ),
        "weights": json.dumps(
            dict(zip(("Jg", "Jl", "Jr", "Ju"), config.assessment.weights, strict=True))
        ),
        "maximize": str(config.assessment.maximize),
        "required_constraints": ",".join(config.assessment.required_constraints),
        "gate_call_policy": GATE_CALL_POLICY,
        "prediction_pairing_scope": (
            "executed candidate only; one receding-horizon tick observed per decision, "
            "paired with allow_truncated=True and never scored over the unexecuted horizon"
        ),
        "oracle_boundary": (
            "privileged get_ground_truth supplies only the target pose; robot fields come from "
            "the declared backend observation; no perception-derived performance is claimed"
        ),
        "observed_robot_state_fields": (
            "base_pose_world, joint_pos, joint_vel, ee_pose and gripper_opening from observation"
        ),
        "perception_equivalent": "false",
        "collision_proxy_channel": (
            "measured displacement/tilt proxy; a proxy hit is not a verified contact or a full-arm "
            "swept-volume certificate"
        ),
        "collision_contact_channel": (
            "backend-recorded contact nodes/steps only; never synthesized from the prediction"
        ),
        "arm_sweep_certificate_channel": (
            "production surrogate registers no full-arm sweep certificate; unverified-arm-sweep "
            "must remain a hard rejection until an owner-reviewed certificate is bound"
        ),
    }


def _home_reached(config: WorldMindConfig, robot: RobotState) -> bool:
    return all(
        name in robot.joint_pos
        and math.isfinite(float(robot.joint_pos[name]))
        and abs(float(robot.joint_pos[name]) - target) <= config.verification.home_tolerance_rad
        for name, target in config.verification.home_joints.items()
    )


def _outcome(
    *,
    verified_success: bool,
    failure_state: str,
    failure_reason: str,
    initial_world: Pose3D,
    final_world: Pose3D,
    home_reached: bool,
    config: WorldMindConfig,
    rejections: int,
    steps: int,
) -> dict[str, Any]:
    """Apply the frozen final criteria; a verifier claim cannot override failed measurements."""
    outcome = final_can_outcome(
        initial_world,
        final_world,
        place_world_m=config.verification.destination_world,
        distance_tolerance_m=config.verification.distance_tolerance_m,
        height_tolerance_m=config.verification.height_tolerance_m,
        tilt_limit_deg=math.degrees(config.verification.tilt_limit_rad),
        home_reached=home_reached,
        failure_state=failure_state or "PLACEMENT",
        failure_reason=failure_reason or "frozen final criteria failed",
    )
    success = bool(verified_success and outcome["success"])
    outcome["success"] = success
    outcome["final_state"] = "COMPLETE" if success else "FAILED"
    outcome["failure_state"] = None if success else (failure_state or "PLACEMENT")
    outcome["failure_reason"] = None if success else (failure_reason or "frozen criteria failed")
    outcome["rejections"] = int(rejections)
    outcome["worldmind_control_steps"] = int(steps)
    outcome["collision_evidence_channels"] = {
        "proxy": "recorded displacement/tilt proxy; labelled separately from contact",
        "contact": "backend-recorded contacts only; not inferred from the surrogate",
        "arm_sweep": "no production certificate; production predictions reject unverified sweeps",
    }
    return outcome


class WorldMindEpisodePolicy:
    """One episode of the production composition; the eval runner owns the recording."""

    def __init__(
        self,
        scenario: Any,
        config: WorldMindConfig,
        bindings: WorldMindBindings,
        writer: WorldMindEvidenceWriter,
    ) -> None:
        self.scenario = scenario
        self.config = config
        self.bindings = bindings
        self.writer = writer

    def execute(self, backend: Any, seed: int) -> Mapping[str, Any]:
        episode_id = str(self.scenario.name)
        task = TaskSpec(
            instruction=self.config.task.instruction,
            target=self.config.task.target,
            destination=self.config.task.destination,
        )
        rejections = 0
        steps = 0
        verified_success = False
        failure_state = "WORLDMIND_ABORT"
        failure_reason = "episode ended before a decision"
        observation: Observation | None = None
        state: WorldState | None = None
        initial_world: Pose3D | None = None
        ground_truth: GroundTruth | None = None
        try:
            observation = backend.reset(self.scenario, seed)
            ground_truth = backend.get_ground_truth()
            initial_world = _world_pose(ground_truth, task.target)
            state = _world_state(observation, ground_truth, self.config, target=task.target)
            base = observation.robot.base_pose_world
            if base is None:
                raise ValueError("oracle state source requires base_pose_world")
            base_inverse = _inverse_pose(base)
            destination_base = compose(
                base_inverse,
                Pose3D(Vec3(*self.config.verification.destination_world), _IDENTITY_QUAT),
            )
            rest_center_base = compose(
                base_inverse,
                Pose3D(Vec3(*self.config.verification.rest_center_world), _IDENTITY_QUAT),
            )
            safety_inputs = self.bindings.inputs.get("safety", {})
            surrogate_inputs = self.bindings.inputs.get("surrogate", {})
            verification_inputs = self.bindings.inputs.get("verification", {})
            monitor = self.bindings.safety[self.config.safety.binding](self.config, safety_inputs)
            predictor = self.bindings.surrogates[self.config.surrogate.binding](
                self.config, surrogate_inputs
            )
            capture = _TrajectoryCapture(predictor)
            verifier = self.bindings.verification[self.config.verification.binding](
                self.config, destination_base, rest_center_base, verification_inputs
            )
            gate = _CountingGate(
                RiskGate(
                    weights=self.config.assessment.weights,
                    maximize=self.config.assessment.maximize,
                    required_constraints=self.config.assessment.required_constraints,
                )
            )

            def assess(
                assess_state: WorldState, chunk: Any, trajectory: Trajectory | None
            ) -> CandidateAssessment:
                checks: list[tuple[str, bool | None]] = []
                for name in self.config.assessment.hard_checks:
                    value = self.bindings.hard_checks[name](assess_state, chunk, trajectory)
                    if value is not None and not isinstance(value, bool):
                        raise ValueError(f"hard check {name} must return bool or None")
                    checks.append((name, value))
                components = cast(
                    tuple[float, float, float, float],
                    tuple(
                        float(
                            self.bindings.components[self.config.components[name].binding](
                                assess_state, chunk, trajectory
                            )
                        )
                        for name in ("Jg", "Jl", "Jr", "Ju")
                    ),
                )
                return CandidateAssessment(tuple(checks), components)

            def predict(predict_state: WorldState, chunk: Any) -> Trajectory:
                return capture.predict(predict_state, chunk)

            no_safe = RecoveryState()
            use_alternate = False
            decision_index = 0

            while (
                decision_index < self.config.bounds.max_decisions
                and steps < self.config.bounds.max_steps
            ):
                assert observation is not None and state is not None
                # Re-derive reviewed waypoints from the latest oracle state each decision. The
                # frozen baseline re-reads its world-anchored targets per state; computing them
                # once per episode would make later chunks stale after a move or recovery.
                primary = self._waypoints(self.config.adapter.waypoints_binding, state, task)
                alternate = (
                    self._waypoints(self.config.adapter.alternate_waypoints_binding, state, task)
                    if self.config.adapter.alternate_waypoints_binding
                    else None
                )
                before_observation, before_state = observation, state
                decision_id = f"{episode_id}-d{decision_index:02d}"
                waypoints = alternate if use_alternate and alternate else primary
                generator = self._generator(waypoints, seed)
                generation_probe = CandidateGenerationProbe(
                    generator,
                    episode_id=episode_id,
                    decision_id=decision_id,
                    timing_enabled=self.config.bounds.timings_enabled,
                )
                probe = PredictionAssessmentProbe(
                    predict=predict,
                    assess=assess,
                    episode_id=episode_id,
                    decision_id=decision_id,
                    timing_enabled=self.config.bounds.timings_enabled,
                )
                planner = ComparisonPlanner(
                    FROZEN_ARM, gate, predict=probe.predict, assess=probe.assess
                )
                planning_start = perf_counter_ns()
                trajectory_offset = len(capture.trajectories)
                candidates = generation_probe.propose(task, state, None)
                gate.calls = 0
                gate.full_assessments = ()
                selection_start = perf_counter_ns()
                selection = planner.select(state, candidates)
                selection_s = self._seconds(selection_start)
                planning_s = self._seconds(planning_start)
                if gate.calls != len(candidates) + 1:
                    raise RuntimeError("N+1 RiskGate call contract violated")
                rejections += int(selection.veto_count)
                generation = generation_probe.observations[-1]

                base_provenance = {
                    "gate_select_calls": str(gate.calls),
                    "gate_call_policy": GATE_CALL_POLICY,
                    "surrogate_binding": self.config.surrogate.binding,
                    "calibration_id": self.config.calibration.calibration_id,
                    "state_source": self.config.backend.state_source,
                    "oracle_boundary": (
                        "target pose from privileged oracle; robot fields from observation"
                    ),
                    "perception_equivalent": "false",
                    "collision_channels": (
                        "proxy=recorded displacement/tilt; contact=backend contacts only; "
                        "sweep=no certified full-arm check"
                    ),
                }
                ranking_provenance = {
                    "selected_index": "none" if selection.index is None else str(selection.index),
                    "selected_score": "none" if selection.score is None else str(selection.score),
                    "selection_veto_count": str(selection.veto_count),
                    "selection_prediction_count": str(selection.prediction_count),
                    "selection_rationale": str(selection.rationale),
                }
                builder = DecisionEvidenceBuilder(
                    episode_id=episode_id,
                    decision_id=decision_id,
                    arm=FROZEN_ARM,
                    proposer_source="scripted",
                    decision_index=decision_index,
                    candidate_count=len(candidates),
                    weights=self.config.assessment.weights,
                    maximize=self.config.assessment.maximize,
                    required_constraints=self.config.assessment.required_constraints,
                    provenance={**base_provenance, **ranking_provenance},
                )
                builder = builder.add_generation(generation)
                for row in probe.predictions:
                    builder = builder.add_prediction(row)
                for row in _assessment_rows(probe, gate):
                    builder = builder.add_assessment(row)

                if selection.chunk is None or selection.index is None:
                    recovery_start = perf_counter_ns()
                    event = no_safe.no_safe()
                    abort = event == "abort"
                    if event == "alternate_candidate_family" and alternate is None:
                        event = "alternate_candidate_family:unavailable"
                        abort = True
                    if event == "reobserve":
                        observation = backend.step(1)
                        steps += 1
                        ground_truth = backend.get_ground_truth()
                        state = _world_state(
                            observation, ground_truth, self.config, target=task.target
                        )
                    if event == "alternate_candidate_family":
                        use_alternate = True
                    recovery_s = self._seconds(recovery_start)
                    recovery_selection = replace(selection, recovery_events=(event,))
                    timing = self._build_timings(
                        generation_s=generation.duration_s,
                        prediction_samples=[row.duration_s for row in probe.predictions],
                        planning_s=planning_s,
                        selection_s=selection_s,
                        verification_s=None,
                        recovery_s=recovery_s,
                    )
                    self.writer.record_decision(
                        builder.set_selection(recovery_selection).set_timings(timing).build()
                    )
                    failure_state, failure_reason = "NO_SAFE_SET", selection.rationale
                    if abort:
                        break
                    decision_index += 1
                    continue

                selected_id = candidate_id(selection.index)
                selected_trajectory_index = trajectory_offset + selection.index
                monitor_probe = SafetyMonitorProbe(monitor)
                backend_probe = BackendApplyProbe(backend)
                wrapped, _ = wrap_execute_first(
                    execute_first,
                    decision_id=decision_id,
                    selected_candidate_id=selected_id,
                )
                after, apply = wrapped(selection, observation, backend_probe, monitor_probe)
                execution = build_execution_evidence_from_probes(
                    decision_id=decision_id,
                    selected_candidate_id=selected_id,
                    observation=observation,
                    result=apply,
                    monitor_probe=monitor_probe,
                    backend_probe=backend_probe,
                    observation_after=after,
                )
                builder = builder.set_selection(selection).set_execution(execution)
                if execution.pre_apply_accepted is False or execution.apply_accepted is not True:
                    boundary = (
                        "pre_apply_veto"
                        if execution.pre_apply_accepted is False
                        else "backend_apply_rejected"
                    )
                    failure_state, failure_reason = (
                        "SAFETY_ABORT",
                        (execution.pre_apply_reason or execution.apply_reason or boundary),
                    )
                    pairing = self._pairing(
                        decision_id=decision_id,
                        selected_id=selected_id,
                        capture=capture,
                        index=selected_trajectory_index,
                        before=before_observation,
                        after=None,
                        before_state=before_state,
                        after_state=None,
                        abort_boundary=boundary,
                        recovery_boundary=None,
                    )
                    timing = self._build_timings(
                        generation_s=generation.duration_s,
                        prediction_samples=[row.duration_s for row in probe.predictions],
                        planning_s=planning_s,
                        selection_s=selection_s,
                        verification_s=None,
                        recovery_s=None,
                    )
                    builder = builder.set_timings(timing)
                    if pairing is not None:
                        builder = builder.set_prediction_pairing(pairing)
                    self.writer.record_decision(builder.build(), pairing)
                    break

                steps += 1
                after_ground_truth = backend.get_ground_truth()
                after_state = _world_state(
                    after, after_ground_truth, self.config, target=task.target
                )
                ground_truth = after_ground_truth
                observation, state = after, after_state
                verification_start = perf_counter_ns()
                trajectory = self._selected_trajectory(capture, selected_trajectory_index)
                verifier_probe = VerificationProbe(verifier)
                verification = verifier_probe.check(task, trajectory.states[-1], after_state)
                transition_start = perf_counter_ns()
                transition = verifier_probe.next_action(verification)
                recovery_s = self._seconds(transition_start)
                verification_s = self._seconds(verification_start)
                pairing = self._pairing(
                    decision_id=decision_id,
                    selected_id=selected_id,
                    capture=capture,
                    index=selected_trajectory_index,
                    before=before_observation,
                    after=after,
                    before_state=before_state,
                    after_state=after_state,
                    abort_boundary=None,
                    recovery_boundary=None if transition == "complete" else transition,
                )
                timing = self._build_timings(
                    generation_s=generation.duration_s,
                    prediction_samples=[row.duration_s for row in probe.predictions],
                    planning_s=planning_s,
                    selection_s=selection_s,
                    verification_s=verification_s,
                    recovery_s=recovery_s,
                )
                builder = (
                    builder.set_timings(timing)
                    .set_verification(
                        self._verification_evidence(decision_id, verification, transition, after.t)
                    )
                    .set_prediction_pairing(pairing)
                )
                self.writer.record_decision(builder.build(), pairing)
                if verification.success and transition == "complete":
                    verified_success, failure_state, failure_reason = True, "", ""
                    break
                if transition == "abort":
                    failure_state, failure_reason = "VERIFICATION_ABORT", verification.reason
                    break
                failure_state, failure_reason = "VERIFICATION_RETRY", verification.reason
                use_alternate = True
                decision_index += 1
        except Exception as error:  # noqa: BLE001 - runner must receive a recorded failure
            self.writer.record_error(episode_id, error)
            failure_state = "EXCEPTION"
            failure_reason = f"{type(error).__name__}: {error}"

        final_pose = initial_world
        if ground_truth is not None:
            final_pose = _world_pose(ground_truth, task.target)
        if initial_world is None or final_pose is None:
            raise RuntimeError("oracle ground truth was never read")
        home = _home_reached(self.config, state.robot) if state is not None else False
        return _outcome(
            verified_success=verified_success,
            failure_state=failure_state,
            failure_reason=failure_reason,
            initial_world=initial_world,
            final_world=final_pose,
            home_reached=home,
            config=self.config,
            rejections=rejections,
            steps=steps,
        )

    def _generator(
        self, waypoints: Sequence[ScriptedWaypoint], seed: int
    ) -> SeededCandidateGenerator:
        monitor = self.bindings.safety[self.config.safety.binding](
            self.config, self.bindings.inputs.get("safety", {})
        )
        adapter = ScriptedChunkAdapter(
            waypoints,
            solver=self.bindings.pose_solvers[self.config.adapter.pose_solver_binding],
            monitor=monitor,
            horizon_steps=self.config.pipeline.horizon_steps,
            control_dt_s=self.config.pipeline.control_dt_s,
            jitter=JointJitter(
                std_rad=self.config.adapter.jitter.joint_std_rad,
                limit_rad=self.config.adapter.jitter.joint_limit_rad,
                max_attempts=self.config.adapter.jitter.max_attempts,
            ),
            position_tolerance_m=self.config.adapter.ik_position_tolerance_m,
            orientation_tolerance_rad=self.config.adapter.ik_orientation_tolerance_rad,
        )
        return SeededCandidateGenerator(
            adapter.draw(),
            seed=self.config.pipeline.decision_seed_base + seed,
            count=self.config.pipeline.candidate_count,
            source="scripted",
        )

    def _waypoints(
        self, binding: str | None, state: WorldState, task: TaskSpec
    ) -> Sequence[ScriptedWaypoint]:
        if binding is None:
            raise ValueError("no reviewed waypoint set bound")
        factory = self.bindings.waypoint_factories.get(binding)
        if factory is not None:
            waypoints = factory(self.config, state, task)
        else:
            waypoints = self.bindings.waypoint_sets[binding]
        if not isinstance(waypoints, Sequence) or not waypoints:
            raise ValueError(f"waypoint binding {binding} produced no stations")
        return waypoints

    @staticmethod
    def _selected_trajectory(capture: _TrajectoryCapture, index: int) -> Trajectory:
        if index >= len(capture.trajectories):
            raise ValueError("selected candidate has no recorded trajectory")
        return capture.trajectories[index]

    def _pairing(
        self,
        *,
        decision_id: str,
        selected_id: str,
        capture: _TrajectoryCapture,
        index: int,
        before: Observation,
        after: Observation | None,
        before_state: WorldState,
        after_state: WorldState | None,
        abort_boundary: str | None,
        recovery_boundary: str | None,
    ) -> Any:
        if index >= len(capture.trajectories):
            return None
        observed = (
            [before_state] if after is None or after_state is None else [before_state, after_state]
        )
        timestamps = [float(before.t)] if after is None else [float(before.t), float(after.t)]
        return pair_executed_prediction(
            decision_id=decision_id,
            executed_candidate_id=selected_id,
            prediction_candidate_id=selected_id,
            prediction=capture.trajectories[index],
            observed=observed,
            prediction_timestamp_s=float(before.t),
            observed_timestamp_s=float(before.t),
            observed_timestamps_s=timestamps,
            allow_truncated=True,
            abort_boundary=abort_boundary,
            recovery_boundary=recovery_boundary,
        )

    def _build_timings(
        self,
        *,
        generation_s: float | None,
        prediction_samples: Sequence[float | None],
        planning_s: float | None,
        selection_s: float | None,
        verification_s: float | None,
        recovery_s: float | None,
    ) -> PhaseTimings:
        return _timings(
            enabled=self.config.bounds.timings_enabled,
            generation_s=generation_s,
            prediction_samples=prediction_samples,
            planning_s=planning_s,
            selection_s=selection_s,
            verification_s=verification_s,
            recovery_s=recovery_s,
        )

    @staticmethod
    def _seconds(start: int) -> float:
        return (perf_counter_ns() - start) / 1_000_000_000.0

    @staticmethod
    def _verification_evidence(
        decision_id: str, result: VerifyResult, transition: str | None, observed_t_s: float
    ) -> VerificationEvidence:
        return VerificationEvidence(
            decision_id=decision_id,
            success=bool(result.success),
            reason=str(result.reason),
            next_action=transition,
            observed_t_s=float(observed_t_s),
        )


@dataclass
class WorldMindPolicyFactory:
    """Create one production policy per planned scene and finalize sidecars once."""

    config: WorldMindConfig
    bindings: WorldMindBindings
    writer: WorldMindEvidenceWriter

    def __call__(self, scene: Any) -> WorldMindEpisodePolicy:
        return WorldMindEpisodePolicy(scene, self.config, self.bindings, self.writer)

    def finish(self) -> dict[str, Path]:
        return self.writer.finish()


def build_worldmind_policy_factory(
    config: WorldMindConfig,
    bindings: WorldMindBindings,
    writer: WorldMindEvidenceWriter,
) -> WorldMindPolicyFactory:
    """Return the eval-runner policy factory after sidecars have been created exclusively."""
    return WorldMindPolicyFactory(config, bindings, writer)


def run_worldmind_evaluation(
    plan: Sequence[Any],
    backend: Any,
    config: WorldMindConfig,
    out_dir: Path,
    bindings: WorldMindBindings,
    *,
    settings: Any = None,
    startup_s: float = 0.0,
) -> tuple[EvalReport, dict[str, Path]]:
    """Run the frozen recorder with the production composition and exclusive sidecars."""
    writer = WorldMindEvidenceWriter(out_dir)
    writer.start(_provenance(config, plan))
    factory = build_worldmind_policy_factory(config, bindings, writer)
    run_config = EvalRunConfig(
        out_dir=out_dir,
        pipeline=FROZEN_ARM,
        split=config.split,
        base_seed=config.base_seed,
        backend=config.backend.kind,
        measurement="synthetic" if config.backend.kind == "fake" else "measured",
        state_source=config.backend.state_source,
        record_frames=False,
    )
    try:
        report = run_evaluation(
            plan,
            backend,
            run_config,
            policy_factory=factory,
            settings=settings,
            startup_s=startup_s,
        )
    finally:
        sidecars = factory.finish()
    return report, sidecars
