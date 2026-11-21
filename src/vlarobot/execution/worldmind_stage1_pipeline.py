"""Non-scored Stage 1 evidence collection for the WorldMind bootstrap.

This module is deliberately additive. It reuses the frozen evaluator, recorder, safety chain,
surrogate, verifier and sidecar writer, but it never constructs RiskGate, an Eq. 3.8 utility,
a calibration candidate, a calibration evaluator or a scored result. Two explicit diagnostic
arm modes exist and neither can be selected implicitly:

* ``full_system_veto`` records the intact full-system hard checks and then blocks actuation;
* ``scripted_proposer`` records scripted-arm evidence under live safety only and is labelled
  ``scripted diagnostic``, never ``full_system``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import yaml

from vlarobot.core.types import (
    Action,
    ActionChunk,
    ApplyResult,
    GripperCmd,
    GroundTruth,
    JointTarget,
    Observation,
    Pose3D,
    TaskSpec,
    Vec3,
    VerifyResult,
    WorldState,
)
from vlarobot.eval.decision_adapters import (
    BackendApplyProbe,
    CandidateGenerationProbe,
    GenerationObservation,
    PredictionObservation,
    SafetyMonitorProbe,
    VerificationProbe,
    build_execution_evidence_from_probes,
    candidate_id,
    wrap_execute_first,
)
from vlarobot.eval.decision_evidence import (
    AppliedActionEvidence,
    CandidateDisposition,
    CandidateEvidence,
    ChunkExecutionEvidence,
    DecisionEvidence,
    ExecutionEvidence,
    HardCheckResult,
    PhaseTimings,
    VerificationEvidence,
)
from vlarobot.eval.decision_metrics import pair_executed_prediction
from vlarobot.eval.metrics import write_json
from vlarobot.eval.observed_state_feedback import (
    COLLISION_PROXY_ATTRIBUTION_SIDECAR_NAME,
    DIAGNOSTIC_EVENT_SIDECAR_NAME,
    FEEDBACK_EVIDENCE_SCHEMA_VERSION,
    FEEDBACK_SIDECAR_NAME,
    FEEDBACK_TERMINATION_REASONS,
    OBSERVED_STATE_FEEDBACK_EXECUTION_MODE,
    DiagnosticHomeCompletionEvidence,
    FeedbackActionEvidence,
    FeedbackAttemptEvidence,
    FeedbackBudgetEvidence,
    FeedbackConvergenceSummary,
    FeedbackPredictionProvenance,
    JointProjectionEvidence,
    ObservedStateFeedbackEvidence,
    build_collision_proxy_attribution,
)
from vlarobot.eval.runner import (
    CampaignFailStop,
    EvalEpisode,
    EvalReport,
    EvalRunConfig,
    _completed_recording,
    run_evaluation,
)
from vlarobot.eval.settings import EvalSettings, load_eval_settings
from vlarobot.eval.worldmind_config import (
    FROZEN_ARM,
    WorldMindBindings,
    WorldMindDiagnosticConfig,
)
from vlarobot.eval.worldmind_evidence import WorldMindEvidenceWriter
from vlarobot.execution.scripted_chunk_adapter import (
    CandidateIKFailure,
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.execution.worldmind_pipeline import (
    _IDENTITY_QUAT,
    _home_reached,
    _inverse_pose,
    _outcome,
    _TrajectoryCapture,
    _world_pose,
    _world_state,
)
from vlarobot.planning.candidates import SeededCandidateGenerator
from vlarobot.planning.execution import execute_first
from vlarobot.risk.comparison import Selection
from vlarobot.risk.gate import RecoveryState
from vlarobot.world_model.rollout import Trajectory, compose

Stage1ArmMode = Literal["full_system_veto", "scripted_proposer"]
Stage1ExecutionMode = Literal[
    "receding_horizon_tick",
    "diagnostic_open_loop_chunk",
    "diagnostic_observed_state_feedback_chunk",
]

STAGE1_ARM_MODES: tuple[Stage1ArmMode, ...] = ("full_system_veto", "scripted_proposer")
STAGE1_EXECUTION_MODES: tuple[Stage1ExecutionMode, ...] = (
    "receding_horizon_tick",
    "diagnostic_open_loop_chunk",
    "diagnostic_observed_state_feedback_chunk",
)
STAGE1_EXECUTION_MODE_LABELS: dict[Stage1ExecutionMode, str] = {
    "receding_horizon_tick": "legacy one-tick execution; re-observation before the next decision",
    "diagnostic_open_loop_chunk": "diagnostic open-loop chunk execution",
    "diagnostic_observed_state_feedback_chunk": (
        "diagnostic observed-state feedback chunk execution"
    ),
}
_FEEDBACK_ABORT_STATES: dict[str, str] = {
    "global_step_ceiling": "WORLDMIND_BOUND_EXHAUSTED",
    "joint_move_timeout": "JOINT_MOVE_TIMEOUT",
    "joint_stall": "JOINT_STALL",
    "gripper_move_timeout": "GRIPPER_MOVE_TIMEOUT",
    "safety_rejected": "SAFETY_ABORT",
    "backend_apply_rejected": "BACKEND_ABORT",
    "execution_exception": "EXCEPTION",
}
STAGE1_GATE_CALL_POLICY = "stage1_no_risk_gate"
STAGE1_EVIDENCE_CLASS: dict[Stage1ArmMode, str] = {
    "full_system_veto": "full-system blocked/veto diagnostic",
    "scripted_proposer": "scripted diagnostic",
}
STAGE1_EFFECTIVE_ARM: dict[Stage1ArmMode, str] = {
    "full_system_veto": FROZEN_ARM,
    "scripted_proposer": "scripted_chunk_adapter",
}
STAGE1_RECORDER_PIPELINE: dict[Stage1ArmMode, str] = {
    "full_system_veto": "full_system",
    "scripted_proposer": "scripted_chunk_adapter",
}
STAGE1_LATENCY_NOTE = (
    "Whole-phase perf_counter_ns durations from Stage 1 decision sidecars. RiskGate was not "
    "called; hard-check and selection phases are measured directly and do not include an "
    "Eq. 3.8 ranking window."
)
_STAGE1_MISSING_TIMING_REASONS = {
    "candidate_generation_s": "no generation observation was recorded",
    "surrogate_prediction_s": "one or more surrogate predictions were not timed",
    "hard_safety_checks_s": "no hard-check observation was recorded",
    "riskgate_ranking_s": "stage1 evidence_collection_only does not run RiskGate",
    "selection_s": "selection window was not measured",
    "planning_end_to_end_s": "planning window was not measured",
    "model_inference_s": "analytic surrogate performs no neural inference",
    "verification_s": "no verification was reached for this decision",
    "recovery_s": "no recovery transition was measured for this decision",
}


def validate_stage1_arm_mode(mode: str) -> Stage1ArmMode:
    """Require an explicit, authorized Stage 1 diagnostic mode; there is no default."""
    if mode not in STAGE1_ARM_MODES:
        raise ValueError(
            f"unknown Stage 1 arm mode {mode!r}; choose one of {', '.join(STAGE1_ARM_MODES)}"
        )
    return mode  # type: ignore[return-value]


def validate_stage1_execution_mode(mode: str) -> Stage1ExecutionMode:
    """Require an explicit execution mode; the open-loop chunk path is never implicit."""
    if mode not in STAGE1_EXECUTION_MODES:
        raise ValueError(
            f"unknown Stage 1 execution mode {mode!r}; "
            f"choose one of {', '.join(STAGE1_EXECUTION_MODES)}"
        )
    return mode  # type: ignore[return-value]


def _require_feedback_arm_mode(mode: Stage1ArmMode, execution_mode: Stage1ExecutionMode) -> None:
    if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE and mode != "scripted_proposer":
        raise ValueError("observed-state feedback is available only with scripted_proposer")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_sha256(plan: Sequence[Any]) -> str:
    payload = [
        {"index": entry.index, "scene": asdict(entry.scene), "seed": entry.seed} for entry in plan
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _stage1_provenance(
    config: WorldMindDiagnosticConfig,
    plan: Sequence[EvalEpisode],
    mode: Stage1ArmMode,
    execution_mode: Stage1ExecutionMode,
    runtime_values: Mapping[str, str],
) -> dict[str, str]:
    """Bind the real frozen plan and source hashes without any scored or calibration identity."""
    payload: dict[str, str] = {
        "schema_version": "1",
        "config_path": str(config.source_path),
        "config_sha256": _sha256(config.source_path),
        "status": config.status,
        "diagnostic_mode": config.diagnostic.mode,
        "calibration_stage": str(config.diagnostic.calibration_stage),
        "scored_evaluation_allowed": str(config.diagnostic.scored_evaluation_allowed).lower(),
        "evidence_collection_split": config.diagnostic.evidence_collection_split,
        "report_pdf": str(config.report.pdf),
        "report_sha256": config.report.sha256,
        "definitions_path": str(config.report.definitions),
        "definitions_sha256": _sha256(config.report.definitions),
        "backend": config.backend.kind,
        "backend_declaration": config.backend.declaration,
        "state_source": config.backend.state_source,
        "split": config.split,
        "base_seed": str(config.base_seed),
        "seeds": ",".join(str(value) for value in config.seeds),
        "episodes": str(len(plan)),
        "frozen_plan_sha256": _plan_sha256(plan),
        "proposer": config.pipeline.proposer,
        "configured_arm": config.pipeline.arm,
        "arm_mode": mode,
        "effective_arm": STAGE1_EFFECTIVE_ARM[mode],
        "evidence_class": STAGE1_EVIDENCE_CLASS[mode],
        "execution_mode": execution_mode,
        "execution_mode_label": STAGE1_EXECUTION_MODE_LABELS[execution_mode],
        "feedback_evidence_schema_version": (
            str(FEEDBACK_EVIDENCE_SCHEMA_VERSION)
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_robot_config_sha256": (
            _sha256(_robot_config_path(config))
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_step_bound_source": (
            "pose_solver.execution_max_joint_step_rad"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_convergence_source": (
            "robot.yaml:pregrasp_settle_tolerance_rad,home_settle_tolerance_rad"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_diagnostic_home_source": (
            "configs/worldmind_diagnostic.approved.yaml:verification.home_joints/home_tolerance_rad"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_gripper_completion_source": (
            "configs/robot.yaml:finger_speed,gripper_opening_limits,"
            "gripper_open_timeout_s; control_dt_s from the frozen pipeline config"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_candidate_ik_failure": (
            "actual failed draw recorded; candidate generation remains fail-fast"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "feedback_diagnostic_sidecars": (
            f"{FEEDBACK_SIDECAR_NAME},{DIAGNOSTIC_EVENT_SIDECAR_NAME},"
            f"{COLLISION_PROXY_ATTRIBUTION_SIDECAR_NAME}"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "not_applicable"
        ),
        "candidate_count": str(config.pipeline.candidate_count),
        "horizon_steps": str(config.pipeline.horizon_steps),
        "control_dt_s": str(config.pipeline.control_dt_s),
        "decision_seed_base": str(config.pipeline.decision_seed_base),
        "max_decisions": str(config.bounds.max_decisions),
        "max_steps": str(config.bounds.max_steps),
        "timings_enabled": str(config.bounds.timings_enabled).lower(),
        "hard_checks": ",".join(config.assessment.hard_checks),
        "required_constraints": ",".join(config.assessment.required_constraints),
        "surrogate_binding": config.surrogate.binding,
        "safety_binding": config.safety.binding,
        "verification_binding": config.verification.binding,
        "pose_solver_binding": config.adapter.pose_solver_binding,
        "waypoints_binding": config.adapter.waypoints_binding,
        "gate_call_policy": STAGE1_GATE_CALL_POLICY,
        "utility_ranking": "not_performed",
        "prediction_pairing_scope": (
            "original H=50 candidate prediction is immutable; observed-state feedback can add "
            "ticks and execute projected targets, so its trajectory is non-comparable and no "
            "prediction pairing is emitted"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "executed candidate only; every observed state from the diagnostic open-loop "
            "chunk is paired with the full prediction, or the available prefix on early "
            "termination"
            if execution_mode == "diagnostic_open_loop_chunk"
            else "executed candidate only; one receding-horizon tick observed per decision and "
            "paired with allow_truncated=True over the unexecuted horizon"
        ),
        "prediction_pairing_records": (
            "not_emitted_non_comparable"
            if execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else "emitted_for_executed_candidate_only"
        ),
        "oracle_boundary": (
            "privileged get_ground_truth supplies only the target pose; robot fields come from "
            "the declared backend observation; no perception-derived result is claimed"
        ),
        "observed_robot_state_fields": (
            "base_pose_world, joint_pos, joint_vel, ee_pose and gripper_opening from observation"
        ),
        "perception_equivalent": "false",
        "independent_evaluation": "false",
        "performance_claim": "not_permitted_bookkeeping_only",
        "full_system_checks_run": str(mode == "full_system_veto").lower(),
        "full_system_veto_preserved": "true",
        "diagnostic_actuation": "none" if mode == "full_system_veto" else "scripted_safety_gated",
        "collision_proxy_channel": (
            "measured displacement/tilt proxy; a proxy hit is not a verified contact or a "
            "full-arm swept-volume certificate"
        ),
        "collision_contact_channel": (
            "backend-recorded contact nodes/steps only; never synthesized from the prediction"
        ),
        "arm_sweep_certificate_channel": (
            "no owner-reviewed full-arm certificate is bound; unverified-arm-sweep stays a hard "
            "full-system rejection and is never suppressed"
        ),
    }
    for name, value in sorted(runtime_values.items()):
        payload[f"runtime.{name}"] = str(value)
    return payload


def _robot_config_path(config: WorldMindDiagnosticConfig) -> Path:
    """Locate the reviewed robot settings beside the safety config or at the project root."""
    beside_safety = config.safety.config_path.parent / "robot.yaml"
    if beside_safety.is_file():
        return beside_safety.resolve()
    root_config = Path(__file__).resolve().parents[3] / "configs" / "robot.yaml"
    if root_config.is_file():
        return root_config
    raise FileNotFoundError("reviewed configs/robot.yaml is required for feedback budgets")


@dataclass(frozen=True)
class _FeedbackContract:
    """Reviewed numerical contract; every value comes from existing config or bindings."""

    max_joint_step_rad: float
    non_home_tolerance_rad: float
    home_tolerance_rad: float
    joint_stall_steps: int
    joint_progress_epsilon_rad: float
    joint_move_timeout_steps: int
    linear_move_timeout_steps: int
    gripper_open_timeout_s: float
    finger_speed_m_s: float
    gripper_opening_limits: tuple[float, float]
    finger_names: tuple[str, ...]
    fingers_inverted: bool
    gripper_arrival_tolerance_m: float
    gripper_timeout_steps: int
    episode_max_steps: int
    robot_config_path: Path
    robot_config_sha256: str
    diagnostic_home_target_rad: Mapping[str, float]
    diagnostic_home_tolerance_rad: float

    def evidence(self) -> FeedbackBudgetEvidence:
        return FeedbackBudgetEvidence(
            max_joint_step_rad=self.max_joint_step_rad,
            non_home_tolerance_rad=self.non_home_tolerance_rad,
            home_tolerance_rad=self.home_tolerance_rad,
            joint_stall_steps=self.joint_stall_steps,
            joint_progress_epsilon_rad=self.joint_progress_epsilon_rad,
            joint_move_timeout_steps=self.joint_move_timeout_steps,
            linear_move_timeout_steps=self.linear_move_timeout_steps,
            gripper_open_timeout_s=self.gripper_open_timeout_s,
            finger_speed_m_s=self.finger_speed_m_s,
            gripper_arrival_tolerance_m=self.gripper_arrival_tolerance_m,
            gripper_timeout_steps=self.gripper_timeout_steps,
            episode_max_steps=self.episode_max_steps,
            robot_config_path=str(self.robot_config_path),
            robot_config_sha256=self.robot_config_sha256,
            applicability={
                "joint_step_bound": "applied to every JointTarget projection",
                "joint_convergence": "applied with HOME/non-HOME reviewed tolerances",
                "joint_stall": "applied to JointTarget feedback after each observed step",
                "joint_move_timeout": "applied to each JointTarget feedback loop",
                "linear_move_timeout": "not_applicable_no_linear_actions_in_scripted_chunk",
                "gripper_open_timeout": (
                    "applied to observed GripperCmd completion using the derived per-tick "
                    "arrival bound and tick ceiling"
                ),
                "episode_step_ceiling": (
                    "applied once across selected actions, diagnostic HOME completion and retries"
                ),
            },
        )


def _load_feedback_contract(config: WorldMindDiagnosticConfig, solver: Any) -> _FeedbackContract:
    """Derive every diagnostic budget from reviewed config values without inventing numbers."""
    path = _robot_config_path(config)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("robot.yaml must contain a mapping")
    names = (
        "pregrasp_settle_tolerance_rad",
        "home_settle_tolerance_rad",
        "joint_stall_steps",
        "joint_progress_epsilon_rad",
        "joint_move_timeout_steps",
        "linear_move_timeout_steps",
        "gripper_open_timeout_s",
        "finger_speed",
    )
    missing = [name for name in names if name not in raw]
    if "gripper_opening_limits" not in raw:
        missing.append("gripper_opening_limits")
    if missing:
        raise ValueError(f"robot.yaml is missing reviewed feedback budgets: {missing}")
    integers = ("joint_stall_steps", "joint_move_timeout_steps", "linear_move_timeout_steps")
    for name in integers:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"robot.yaml {name} must be a positive integer")
    floats = tuple(name for name in names if name not in integers)
    for name in floats:
        value = raw[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"robot.yaml {name} must be finite and positive")
    limits = raw["gripper_opening_limits"]
    if (
        not isinstance(limits, Sequence)
        or len(limits) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in limits)
        or not all(math.isfinite(float(value)) for value in limits)
        or float(limits[0]) > float(limits[1])
    ):
        raise ValueError("robot.yaml gripper_opening_limits must be a finite ordered pair")
    step_bound = getattr(solver, "execution_max_joint_step_rad", None)
    if step_bound is None or not math.isfinite(float(step_bound)) or float(step_bound) <= 0.0:
        raise ValueError("pose solver must expose the reviewed finite execution step bound")
    control_dt_s = float(config.pipeline.control_dt_s)
    if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise ValueError("frozen pipeline control_dt_s must be finite and positive")
    finger_speed = float(raw["finger_speed"])
    timeout_s = float(raw["gripper_open_timeout_s"])
    gripper_timeout_steps = max(1, int(math.ceil(timeout_s / control_dt_s)))
    finger_names = raw.get("finger_joints")
    inverted = raw.get("fingers_inverted")
    if (
        not isinstance(finger_names, list)
        or len(finger_names) != 2
        or any(not isinstance(name, str) or not name for name in finger_names)
        or finger_names[0] == finger_names[1]
        or type(inverted) is not bool
    ):
        raise ValueError("robot.yaml must declare two finger names and their orientation")
    diagnostic_target = dict(config.verification.home_joints)
    if not diagnostic_target:
        raise ValueError(
            "configured verification HOME vector is required for diagnostic completion"
        )
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for name, value in diagnostic_target.items()
    ):
        raise ValueError("configured verification HOME vector must contain finite named joints")
    diagnostic_tolerance = float(config.verification.home_tolerance_rad)
    if not math.isfinite(diagnostic_tolerance) or diagnostic_tolerance <= 0.0:
        raise ValueError("configured verification home_tolerance_rad must be finite and positive")
    return _FeedbackContract(
        max_joint_step_rad=float(step_bound),
        non_home_tolerance_rad=float(raw["pregrasp_settle_tolerance_rad"]),
        home_tolerance_rad=float(raw["home_settle_tolerance_rad"]),
        joint_stall_steps=int(raw["joint_stall_steps"]),
        joint_progress_epsilon_rad=float(raw["joint_progress_epsilon_rad"]),
        joint_move_timeout_steps=int(raw["joint_move_timeout_steps"]),
        linear_move_timeout_steps=int(raw["linear_move_timeout_steps"]),
        gripper_open_timeout_s=timeout_s,
        finger_speed_m_s=finger_speed,
        gripper_opening_limits=(float(limits[0]), float(limits[1])),
        finger_names=tuple(finger_names),
        fingers_inverted=inverted,
        gripper_arrival_tolerance_m=finger_speed * control_dt_s,
        gripper_timeout_steps=gripper_timeout_steps,
        episode_max_steps=int(config.bounds.max_steps),
        robot_config_path=path,
        robot_config_sha256=_sha256(path),
        diagnostic_home_target_rad=diagnostic_target,
        diagnostic_home_tolerance_rad=diagnostic_tolerance,
    )


@dataclass(frozen=True)
class _Stage1Assessment:
    hard_checks: tuple[tuple[str, bool | None], ...]
    duration_s: float | None
    error_code: str | None = None


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
    """Derive disjoint veto reasons from the checks actually evaluated."""
    names = [name for name, _ in hard_checks]
    reasons = [name for name, result in hard_checks if result is not True]
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
    return tuple(dict.fromkeys(reasons)), tuple(results)


def _timings(
    *,
    enabled: bool,
    generation_s: float | None,
    prediction_samples: Sequence[float | None],
    hard_checks_s: float | None,
    selection_s: float | None,
    planning_s: float | None,
    verification_s: float | None,
    recovery_s: float | None,
) -> PhaseTimings:
    if not enabled:
        return PhaseTimings.unavailable(
            {
                **_STAGE1_MISSING_TIMING_REASONS,
                "candidate_generation_s": "phase timing disabled in reviewed config",
                "surrogate_prediction_s": "phase timing disabled in reviewed config",
                "hard_safety_checks_s": "phase timing disabled in reviewed config",
                "selection_s": "phase timing disabled in reviewed config",
                "planning_end_to_end_s": "phase timing disabled in reviewed config",
                "verification_s": "phase timing disabled in reviewed config",
                "recovery_s": "phase timing disabled in reviewed config",
            }
        )
    missing: dict[str, str] = {
        "riskgate_ranking_s": "stage1 evidence_collection_only does not run RiskGate",
        "model_inference_s": "analytic surrogate performs no neural inference",
    }
    if generation_s is None:
        missing["candidate_generation_s"] = _STAGE1_MISSING_TIMING_REASONS["candidate_generation_s"]
    if not prediction_samples or any(value is None for value in prediction_samples):
        missing["surrogate_prediction_s"] = _STAGE1_MISSING_TIMING_REASONS["surrogate_prediction_s"]
    if hard_checks_s is None:
        missing["hard_safety_checks_s"] = _STAGE1_MISSING_TIMING_REASONS["hard_safety_checks_s"]
    if selection_s is None:
        missing["selection_s"] = _STAGE1_MISSING_TIMING_REASONS["selection_s"]
    if planning_s is None:
        missing["planning_end_to_end_s"] = _STAGE1_MISSING_TIMING_REASONS["planning_end_to_end_s"]
    if verification_s is None:
        missing["verification_s"] = _STAGE1_MISSING_TIMING_REASONS["verification_s"]
    if recovery_s is None:
        missing["recovery_s"] = _STAGE1_MISSING_TIMING_REASONS["recovery_s"]
    prediction_total = (
        None
        if "surrogate_prediction_s" in missing
        else float(sum(float(value) for value in prediction_samples if value is not None))
    )
    return PhaseTimings(
        candidate_generation_s=generation_s,
        surrogate_prediction_s=prediction_total,
        hard_safety_checks_s=hard_checks_s,
        riskgate_ranking_s=None,
        selection_s=selection_s,
        planning_end_to_end_s=planning_s,
        model_inference_s=None,
        verification_s=verification_s,
        recovery_s=recovery_s,
        missing_reasons=missing,
    )


def _candidate_evidence(
    *,
    config: WorldMindDiagnosticConfig,
    episode_id: str,
    decision_id: str,
    index: int,
    candidate_count: int,
    draw_seed: int | None,
    prediction: PredictionObservation,
    assessment: _Stage1Assessment,
    selection: Selection,
    no_safe_set: bool,
) -> CandidateEvidence:
    unevaluable: list[str] = []
    if prediction.error_code is not None or not prediction.available:
        unevaluable.append("prediction_missing_or_failed")
    if assessment.error_code is not None:
        unevaluable.append("hard_check_failed")
    reasons, results = _hard_reason_codes(
        assessment.hard_checks, config.assessment.required_constraints
    )
    hard_result = (
        HardCheckResult.FAIL
        if reasons
        else HardCheckResult.UNKNOWN
        if unevaluable
        else HardCheckResult.PASS
    )
    if unevaluable:
        safe: bool | None = None
        rankable = False
        disposition = CandidateDisposition.INVALID_OR_UNEVALUABLE
    elif reasons:
        safe, rankable = False, False
        disposition = CandidateDisposition.HARD_VETOED
    else:
        safe, rankable = True, True
        disposition = CandidateDisposition.SAFE_UNSELECTED
    return CandidateEvidence(
        episode_id=episode_id,
        decision_id=decision_id,
        candidate_id=candidate_id(index),
        candidate_index=index,
        sampling_seed=draw_seed,
        proposer_source=prediction.proposer_source,
        candidate_count=candidate_count,
        disposition=disposition,
        hard_check_result=hard_result,
        hard_veto_reasons=reasons,
        unevaluable_reasons=tuple(unevaluable),
        hard_checks=results,
        safe=safe,
        rankable=rankable,
        costs=None,
        selected_candidate_id=None,
        no_safe_set=no_safe_set,
        recovery_transition=selection.recovery_events[-1] if selection.recovery_events else None,
        prediction_available=prediction.available,
        prediction_horizon_steps=prediction.horizon_steps,
        prediction_hard_reasons=prediction.hard_reasons,
        trajectory_state_count=prediction.trajectory_state_count,
    )


def _decision_evidence(
    *,
    config: WorldMindDiagnosticConfig,
    mode: Stage1ArmMode,
    execution_mode: Stage1ExecutionMode,
    episode_id: str,
    decision_id: str,
    decision_index: int,
    generation: GenerationObservation,
    predictions: Sequence[PredictionObservation],
    assessments: Sequence[_Stage1Assessment],
    selection: Selection,
    timings: PhaseTimings,
    execution: Any = None,
    verification: VerificationEvidence | None = None,
    pairing: Any = None,
    decision_provenance: Mapping[str, str],
) -> DecisionEvidence:
    count = generation.candidate_count
    if len(predictions) != count or len(assessments) != count:
        raise ValueError("one prediction and assessment is required per generated candidate")
    no_safe_set = selection.index is None
    candidates = [
        _candidate_evidence(
            config=config,
            episode_id=episode_id,
            decision_id=decision_id,
            index=index,
            candidate_count=count,
            draw_seed=generation.draw_seeds[index] if index < len(generation.draw_seeds) else None,
            prediction=predictions[index],
            assessment=assessments[index],
            selection=selection,
            no_safe_set=no_safe_set,
        )
        for index in range(count)
    ]
    selected_id = None if selection.index is None else candidate_id(selection.index)
    if selected_id is not None:
        if selection.index is None or selection.index >= count:
            raise ValueError("selected candidate index is outside the generated pool")
        selected = candidates[selection.index]
        if selected.safe is not True or selected.disposition is CandidateDisposition.HARD_VETOED:
            raise ValueError("selected candidate failed its hard checks")
        candidates[selection.index] = replace(
            selected,
            disposition=CandidateDisposition.SELECTED,
            selected_candidate_id=selected_id,
        )
        for index in range(count):
            if index != selection.index:
                candidates[index] = replace(candidates[index], selected_candidate_id=selected_id)
    decision_provenance = {
        **dict(decision_provenance),
        "gate_call_policy": STAGE1_GATE_CALL_POLICY,
        "gate_select_calls": "0",
        "utility_ranking": "not_performed",
        "evidence_class": STAGE1_EVIDENCE_CLASS[mode],
        "arm_mode": mode,
        "execution_mode": execution_mode,
        "execution_mode_label": STAGE1_EXECUTION_MODE_LABELS[execution_mode],
        "full_system_checks_run": str(mode == "full_system_veto").lower(),
        "selection_policy": (
            "no_selection_full_system_blocked"
            if mode == "full_system_veto"
            else "first_safe_by_seed_order_no_utility"
        ),
    }
    return DecisionEvidence(
        episode_id=episode_id,
        decision_id=decision_id,
        arm=STAGE1_EFFECTIVE_ARM[mode],
        proposer_source=generation.proposer_source,
        decision_index=decision_index,
        candidate_count=count,
        draw_seed=generation.draw_seed,
        draw_seeds=generation.draw_seeds,
        candidates=tuple(candidates),
        selected_candidate_id=selected_id,
        no_safe_set=no_safe_set,
        recovery_events=selection.recovery_events,
        timings=timings,
        execution=execution,
        verification=verification,
        prediction_pairing=pairing,
        provenance=decision_provenance,
    )


@dataclass(frozen=True)
class _ChunkExecutionResult:
    observation: Observation
    ground_truth: GroundTruth
    evidence: ExecutionEvidence
    observations: tuple[Observation, ...]
    observed_states: tuple[WorldState, ...]


def _action_payload(action: Action) -> dict[str, Any]:
    if not is_dataclass(action):
        raise ValueError(f"action {type(action).__name__} is not serializable evidence")
    payload = asdict(action)
    if not isinstance(payload, dict):
        raise ValueError("action evidence must be a mapping")
    return payload


def _execute_selected_chunk(
    *,
    decision_id: str,
    selection: Selection,
    config: WorldMindDiagnosticConfig,
    task: TaskSpec,
    observation: Observation,
    state: WorldState,
    ground_truth: GroundTruth,
    backend_probe: BackendApplyProbe,
    monitor_probe: SafetyMonitorProbe,
) -> _ChunkExecutionResult:
    """Apply the whole selected chunk in open loop, one reviewed tick at a time."""
    if selection.chunk is None or not selection.chunk.actions:
        raise ValueError("diagnostic chunk execution requires a nonempty selected chunk")
    if selection.index is None:
        raise ValueError("diagnostic chunk execution requires a selected candidate")
    selected_id = candidate_id(selection.index)
    actions = tuple(selection.chunk.actions)
    records: list[AppliedActionEvidence] = []
    observations = [observation]
    observed_states = [state]
    final_ground_truth = ground_truth
    safety_calls = safety_ok = safety_rejections = 0
    apply_ok = apply_rejections = step_count = 0
    termination = "chunk_complete"
    error: str | None = None

    for index, action in enumerate(actions):
        current = observations[-1]
        record: dict[str, Any] = {
            "decision_id": decision_id,
            "action_index": index,
            "action_type": type(action).__name__,
            "action": _action_payload(action),
            "applied": False,
            "pre_apply_accepted": None,
            "pre_apply_reason": None,
            "apply_accepted": None,
            "apply_reason": None,
            "backend_stepped": False,
            "observation_t_s": float(current.t),
            "action_t_s": None,
        }
        try:
            safety_calls += 1
            verdict = monitor_probe.validate(action, current.robot)
            record["pre_apply_accepted"] = bool(verdict.accepted)
            record["pre_apply_reason"] = verdict.reason
            if not verdict.accepted:
                safety_rejections += 1
                termination = "safety_rejected"
                break
            safety_ok += 1
            applied = backend_probe.apply(action)
            record["apply_accepted"] = bool(applied.accepted)
            record["apply_reason"] = applied.reason
            if not applied.accepted:
                apply_rejections += 1
                termination = "backend_apply_rejected"
                break
            apply_ok += 1
            after = backend_probe.step(1)
            step_count += 1
            record["backend_stepped"] = True
            record["applied"] = True
            record["action_t_s"] = float(after.t)
            final_ground_truth = backend_probe.get_ground_truth()
            after_state = _world_state(after, final_ground_truth, config, target=task.target)
            observations.append(after)
            observed_states.append(after_state)
        except Exception as error_:  # noqa: BLE001 - retain measured prefix and fail closed
            error = f"{type(error_).__name__}: {error_}"
            termination = "execution_exception"
            if safety_calls > safety_ok + safety_rejections:
                safety_rejections += 1
            if safety_ok > apply_ok + apply_rejections:
                apply_rejections += 1
            break
        finally:
            records.append(AppliedActionEvidence(**record))

    applied_count = sum(item.applied for item in records)
    chunk = ChunkExecutionEvidence(
        decision_id=decision_id,
        execution_mode="diagnostic_open_loop_chunk",
        selected_candidate_id=selected_id,
        chunk_action_count=len(actions),
        attempted_action_count=len(records),
        applied_action_count=applied_count,
        backend_step_count=step_count,
        observation_count=len(observations),
        safety_validate_count=safety_calls,
        safety_accepted_count=safety_ok,
        safety_rejected_count=safety_rejections,
        apply_accepted_count=apply_ok,
        apply_rejected_count=apply_rejections,
        missing_horizon_steps=tuple(range(applied_count, len(actions))),
        termination_reason=termination,
        actions=tuple(records),
        error=error,
    )
    terminal = records[-1] if records else None
    execution = ExecutionEvidence(
        decision_id=decision_id,
        selected_candidate_id=selected_id,
        executed_candidate_id=selected_id,
        pre_apply_accepted=None if terminal is None else terminal.pre_apply_accepted,
        pre_apply_reason=None if terminal is None else terminal.pre_apply_reason,
        apply_accepted=None if terminal is None else terminal.apply_accepted,
        apply_reason=None if terminal is None else terminal.apply_reason,
        backend_stepped=step_count > 0,
        observation_t_s=float(observation.t),
        action_t_s=None if step_count == 0 else float(observations[-1].t),
        chunk=chunk,
    )
    return _ChunkExecutionResult(
        observation=observations[-1],
        ground_truth=final_ground_truth,
        evidence=execution,
        observations=tuple(observations),
        observed_states=tuple(observed_states),
    )


@dataclass(frozen=True)
class _FeedbackExecutionResult:
    observation: Observation
    ground_truth: GroundTruth
    ground_truths: tuple[GroundTruth, ...]
    evidence: ExecutionEvidence
    diagnostic: ObservedStateFeedbackEvidence
    observations: tuple[Observation, ...]
    observed_states: tuple[WorldState, ...]
    diagnostic_events: tuple[dict[str, Any], ...]
    termination_reason: str
    backend_step_count: int


def _robot_snapshot(robot: Any) -> dict[str, Any]:
    """Record observed robot fields, including the full world-to-base pose quaternion."""
    base_world = robot.base_pose_world
    opening = float(robot.gripper_opening)
    return {
        "joint_pos": {str(name): float(value) for name, value in robot.joint_pos.items()},
        "joint_vel": {str(name): float(value) for name, value in robot.joint_vel.items()},
        "gripper_opening": opening if math.isfinite(opening) else None,
        "base_pose": asdict(robot.base_pose),
        "base_pose_world": None if base_world is None else asdict(base_world),
        "ee_pose": asdict(robot.ee_pose),
    }


def _max_joint_error(action: JointTarget, observed_joint_pos: Mapping[str, float]) -> float:
    errors: list[float] = []
    for name, target in action.q.items():
        observed = observed_joint_pos.get(name)
        if observed is None or not math.isfinite(float(observed)):
            raise ValueError(f"observed position for joint {name} is unavailable")
        errors.append(abs(float(target) - float(observed)))
    return max(errors, default=0.0)


def _project_joint_target(
    action: JointTarget,
    observed_joint_pos: Mapping[str, float],
    bound_rad: float,
) -> tuple[JointTarget, tuple[str, ...]]:
    """Project a selected target to the reviewed per-joint bound from observed state."""
    executed: dict[str, float] = {}
    clamped: list[str] = []
    for name, target in action.q.items():
        observed = observed_joint_pos.get(name)
        if observed is None or not math.isfinite(float(observed)):
            raise ValueError(f"observed position for joint {name} is unavailable")
        value = float(target)
        lower, upper = float(observed) - bound_rad, float(observed) + bound_rad
        if value < lower:
            value, clamped = lower, [*clamped, str(name)]
        elif value > upper:
            value, clamped = upper, [*clamped, str(name)]
        executed[str(name)] = value
    return JointTarget(executed), tuple(clamped)


def _home_action_indices(
    waypoints: Sequence[ScriptedWaypoint], horizon_steps: int
) -> frozenset[int]:
    """Derive HOME action indices from the immutable reviewed waypoint cadence."""
    indices: set[int] = set()
    offset = 0
    for waypoint in waypoints:
        if waypoint.label.upper() == "HOME":
            indices.update(range(offset, offset + waypoint.hold_steps))
        offset += waypoint.hold_steps
    if offset != horizon_steps:
        raise ValueError("waypoint cadence does not match the frozen horizon")
    return frozenset(indices)


def _request_feedback_tick(backend: Any) -> tuple[int | None, Observation | None, str | None]:
    """Return the known physics outcome independently of the usable observation."""
    step = getattr(backend, "diagnostic_step", None)
    outcome = step() if step is not None else {"confirmed_ticks": 1, "observation": backend.step(1)}
    ticks = outcome.get("confirmed_ticks")
    observation = outcome.get("observation")
    return (
        ticks if type(ticks) is int and ticks in (0, 1) else None,
        observation if isinstance(observation, Observation) else None,
        outcome.get("error"),
    )


def _execute_observed_state_feedback_chunk(
    *,
    decision_id: str,
    episode_id: str,
    selection: Selection,
    config: WorldMindDiagnosticConfig,
    contract: _FeedbackContract,
    task: TaskSpec,
    observation: Observation,
    state: WorldState,
    ground_truth: GroundTruth,
    backend_probe: BackendApplyProbe,
    monitor_probe: SafetyMonitorProbe,
    waypoints: Sequence[ScriptedWaypoint],
    episode_steps_before: int,
) -> _FeedbackExecutionResult:
    """Execute the immutable selected chunk, then a separately labelled diagnostic HOME phase.

    Every selected action payload stays byte-identical to the reviewed candidate. Gripper and HOME
    completion use only observed state and existing reviewed budgets; no grasp, release, contact
    or prediction pairing is inferred from convergence. One global step ledger covers selected
    actions, diagnostic HOME completion and every bounded retry.
    """
    if selection.chunk is None or not selection.chunk.actions:
        raise ValueError("feedback chunk execution requires a nonempty selected chunk")
    if selection.index is None:
        raise ValueError("feedback chunk execution requires a selected candidate")
    selected_id = candidate_id(selection.index)
    actions = tuple(selection.chunk.actions)
    home_indices = _home_action_indices(waypoints, len(actions))
    remaining_steps = contract.episode_max_steps - episode_steps_before
    if remaining_steps < 0:
        raise ValueError("episode step budget is already exhausted")

    observations = [observation]
    observed_states = [state]
    ground_truths = [ground_truth]
    attempts: list[FeedbackAttemptEvidence] = []
    action_records: list[FeedbackActionEvidence] = []
    diagnostic_events: list[dict[str, Any]] = []
    final_ground_truth = ground_truth
    safety_calls = safety_ok = 0
    apply_ok = apply_rejections = step_count = 0
    stop_reason: str | None = None
    error: str | None = None
    home_completion: DiagnosticHomeCompletionEvidence | None = None

    def _record_attempt(
        *,
        selected_index: int,
        selected_action: Action,
        executed_action: Action | None,
        attempt_within_action: int,
        projection: JointProjectionEvidence | None,
        before: Observation,
        before_error: float | None,
        safety_accepted: bool | None,
        safety_reason: str | None,
        apply_accepted: bool | None,
        apply_reason: str | None,
        phase: str = "selected_chunk",
        completion_error: float | None = None,
        completion_unit: str = "rad",
        completion_target: float | None = None,
    ) -> int:
        index = len(attempts)
        attempts.append(
            FeedbackAttemptEvidence(
                attempt_index=index,
                selected_action_index=selected_index,
                selected_action_type=type(selected_action).__name__,
                selected_action=_action_payload(selected_action),
                attempt_within_action=attempt_within_action,
                executed_action_index=None,
                executed_action_type=None
                if executed_action is None
                else type(executed_action).__name__,
                executed_action=None
                if executed_action is None
                else _action_payload(executed_action),
                projection=projection,
                observed_before=_robot_snapshot(before.robot),
                observed_after=None,
                max_joint_error_before_rad=before_error,
                max_joint_error_after_rad=None,
                progress_rad=None,
                no_progress_steps=None,
                safety_accepted=safety_accepted,
                safety_reason=safety_reason,
                apply_accepted=apply_accepted,
                apply_reason=apply_reason,
                backend_stepped=False,
                backend_step_number=None,
                episode_step_number=None,
                phase=phase,
                completion_error=completion_error,
                completion_unit=completion_unit,
                completion_target=completion_target,
            )
        )
        return index

    def _update_attempt(index: int, **fields: Any) -> None:
        attempts[index] = replace(attempts[index], **fields)

    def _tick(index: int, action: Action) -> tuple[ApplyResult, Observation | None]:
        nonlocal apply_ok, apply_rejections, step_count, final_ground_truth
        apply = getattr(backend_probe, "diagnostic_apply", backend_probe.apply)
        _update_attempt(index, apply_outcome="unknown")
        result = apply(action)
        if not isinstance(result, ApplyResult) or type(result.accepted) is not bool:
            raise ValueError("diagnostic apply returned no authoritative result")
        apply_ok += int(result.accepted)
        apply_rejections += int(not result.accepted)
        _update_attempt(
            index,
            apply_outcome="accepted" if result.accepted else "rejected",
            apply_accepted=result.accepted,
            apply_reason=result.reason,
            executed_action_index=apply_ok if result.accepted else None,
        )
        record_apply = getattr(backend_probe, "record_diagnostic_apply", None)
        if record_apply is not None:
            _update_attempt(index, apply_recording="failed")
            record_apply(action, result)
            _update_attempt(index, apply_recording="completed")
        if not result.accepted:
            return result, None

        _update_attempt(index, step_outcome="unknown")
        ticks, after, step_error = _request_feedback_tick(backend_probe)
        if ticks is None:
            raise RuntimeError(step_error or "diagnostic tick outcome unknown")
        if ticks == 0:
            _update_attempt(index, step_outcome="not_executed")
            raise RuntimeError(step_error or "diagnostic tick not executed")

        step_count += 1
        _update_attempt(
            index,
            step_outcome="confirmed",
            backend_stepped=True,
            backend_step_number=step_count,
            episode_step_number=episode_steps_before + step_count,
            observation_outcome="failed",
        )
        if after is None:
            raise RuntimeError(step_error or "confirmed tick has no observation")
        _update_attempt(index, observation_outcome="processing_failed")
        _update_attempt(index, observed_after=_robot_snapshot(after.robot))
        final_ground_truth = backend_probe.get_ground_truth()
        after_state = _world_state(after, final_ground_truth, config, target=task.target)
        _update_attempt(index, observation_outcome="available")
        observations.append(after)
        observed_states.append(after_state)
        ground_truths.append(final_ground_truth)
        record_step = getattr(backend_probe, "record_diagnostic_step", None)
        if record_step is not None:
            _update_attempt(index, step_recording="failed")
            record_step(after, final_ground_truth)
            _update_attempt(index, step_recording="completed")
        return result, after

    def _opening_error(robot: Any, target: float) -> float | None:
        if config.backend.kind != "webots":
            opening = float(robot.gripper_opening)
            return abs(opening - target) if math.isfinite(opening) else None
        raw = robot.diagnostics.get("finger_joint_sensor_rad")
        minimum, maximum = contract.gripper_opening_limits
        readings: list[float] = []
        defects: dict[str, str] = {}
        for name in contract.finger_names:
            value = raw.get(name) if isinstance(raw, Mapping) else None
            if value is None:
                defects[name] = "missing"
            elif type(value) not in (int, float) or not math.isfinite(value):
                defects[name] = "invalid_or_nonfinite"
            elif not minimum <= value <= maximum:
                defects[name] = "outside_reviewed_limits"
            else:
                readings.append(float(value))
        if not defects:
            closed = maximum if contract.fingers_inverted else minimum
            return abs(sum(abs(value - closed) for value in readings) / len(readings) - target)
        diagnostic_events.append(
            {
                "schema_version": 1,
                "event_type": "invalid_gripper_telemetry",
                "episode_id": episode_id,
                "decision_id": decision_id,
                "expected_sensors": list(contract.finger_names),
                "sensor_defects": defects,
                "target_opening_m": target,
                "episode_steps": episode_steps_before + step_count,
            }
        )
        return None

    for selected_index, action in enumerate(actions):
        if stop_reason is not None:
            break
        action_attempts: list[int] = []

        if isinstance(action, JointTarget):
            tolerance = (
                contract.home_tolerance_rad
                if selected_index in home_indices
                else contract.non_home_tolerance_rad
            )
            move_steps = 0
            no_progress_steps = 0
            action_reason = "target_converged"
            final_error = 0.0
            while True:
                current = observations[-1]
                final_error = _max_joint_error(action, current.robot.joint_pos)
                if final_error <= tolerance:
                    break
                if step_count >= remaining_steps:
                    stop_reason = action_reason = "global_step_ceiling"
                    break
                if move_steps >= contract.joint_move_timeout_steps:
                    stop_reason = action_reason = "joint_move_timeout"
                    break
                attempt_index: int | None = None
                safety_accepted = False
                try:
                    executed_action, clamped = _project_joint_target(
                        action, current.robot.joint_pos, contract.max_joint_step_rad
                    )
                    projection = JointProjectionEvidence(
                        observed_joint_pos_rad={
                            str(name): float(value)
                            for name, value in current.robot.joint_pos.items()
                            if name in action.q
                        },
                        selected_joint_target_rad={
                            str(name): float(value) for name, value in action.q.items()
                        },
                        executed_joint_target_rad={
                            str(name): float(value) for name, value in executed_action.q.items()
                        },
                        clamped_joints=clamped,
                        bound_rad=contract.max_joint_step_rad,
                    )
                    safety_calls += 1
                    verdict = monitor_probe.validate(executed_action, current.robot)
                    safety_accepted = bool(verdict.accepted)
                    if safety_accepted:
                        safety_ok += 1
                    attempt_index = _record_attempt(
                        selected_index=selected_index,
                        selected_action=action,
                        executed_action=executed_action,
                        attempt_within_action=move_steps + 1,
                        projection=projection,
                        before=current,
                        before_error=final_error,
                        safety_accepted=safety_accepted,
                        safety_reason=verdict.reason,
                        apply_accepted=None,
                        apply_reason=None,
                    )
                    action_attempts.append(attempt_index)
                    if not safety_accepted:
                        stop_reason = action_reason = "safety_rejected"
                        break
                    applied, after = _tick(attempt_index, executed_action)
                    if after is None:
                        stop_reason = action_reason = "backend_apply_rejected"
                        break
                    progress = max(
                        (
                            abs(
                                float(after.robot.joint_pos[name])
                                - float(current.robot.joint_pos[name])
                            )
                            for name in action.q
                            if name in after.robot.joint_pos and name in current.robot.joint_pos
                        ),
                        default=0.0,
                    )
                    after_error = _max_joint_error(action, after.robot.joint_pos)
                    no_progress_steps = (
                        0
                        if progress > contract.joint_progress_epsilon_rad
                        else no_progress_steps + 1
                    )
                    move_steps += 1
                    final_error = after_error
                    attempts[attempt_index] = replace(
                        attempts[attempt_index],
                        executed_action_index=step_count,
                        executed_action_type=type(executed_action).__name__,
                        executed_action=_action_payload(executed_action),
                        observed_after=_robot_snapshot(after.robot),
                        max_joint_error_after_rad=after_error,
                        progress_rad=progress,
                        no_progress_steps=no_progress_steps,
                        apply_accepted=True,
                        apply_reason=applied.reason,
                        backend_stepped=True,
                        backend_step_number=step_count,
                        episode_step_number=episode_steps_before + step_count,
                    )
                    if no_progress_steps >= contract.joint_stall_steps:
                        stop_reason = action_reason = "joint_stall"
                        break
                    if after_error <= tolerance:
                        break
                except Exception as error_:  # noqa: BLE001 - fail closed with measured prefix
                    error = f"{type(error_).__name__}: {error_}"
                    stop_reason = action_reason = "execution_exception"
                    break
            action_records.append(
                FeedbackActionEvidence(
                    selected_action_index=selected_index,
                    selected_action_type=type(action).__name__,
                    selected_action=_action_payload(action),
                    home_action=selected_index in home_indices,
                    convergence_tolerance_rad=tolerance,
                    terminal_reason=action_reason,
                    attempt_indices=tuple(action_attempts),
                    executed_action_count=sum(
                        attempts[index].backend_stepped for index in action_attempts
                    ),
                    backend_step_count=sum(
                        attempts[index].backend_stepped for index in action_attempts
                    ),
                    final_max_joint_error_rad=final_error,
                )
            )
            continue

        if isinstance(action, GripperCmd):
            target = float(action.opening)
            tolerance = contract.gripper_arrival_tolerance_m
            gripper_steps = 0
            action_reason = "gripper_command_complete"
            final_completion_error: float | None = None
            while True:
                current = observations[-1]
                final_completion_error = _opening_error(current.robot, target)
                if final_completion_error is None:
                    error = "invalid_gripper_telemetry"
                    stop_reason = action_reason = "execution_exception"
                    break
                if gripper_steps > 0 and final_completion_error <= tolerance:
                    break
                if step_count >= remaining_steps:
                    stop_reason = action_reason = "global_step_ceiling"
                    break
                if gripper_steps >= contract.gripper_timeout_steps:
                    stop_reason = action_reason = "gripper_move_timeout"
                    break
                attempt_index = None
                safety_accepted = False
                try:
                    safety_calls += 1
                    verdict = monitor_probe.validate(action, current.robot)
                    safety_accepted = bool(verdict.accepted)
                    if safety_accepted:
                        safety_ok += 1
                    attempt_index = _record_attempt(
                        selected_index=selected_index,
                        selected_action=action,
                        executed_action=action,
                        attempt_within_action=gripper_steps + 1,
                        projection=None,
                        before=current,
                        before_error=None,
                        safety_accepted=safety_accepted,
                        safety_reason=verdict.reason,
                        apply_accepted=None,
                        apply_reason=None,
                        completion_error=final_completion_error,
                        completion_unit="m",
                        completion_target=target,
                    )
                    action_attempts.append(attempt_index)
                    if not safety_accepted:
                        stop_reason = action_reason = "safety_rejected"
                        break
                    applied, after = _tick(attempt_index, action)
                    if after is None:
                        stop_reason = action_reason = "backend_apply_rejected"
                        break
                    gripper_steps += 1
                    final_completion_error = _opening_error(after.robot, target)
                    if final_completion_error is None:
                        error = "invalid_gripper_telemetry"
                        stop_reason = action_reason = "execution_exception"
                        break
                    attempts[attempt_index] = replace(
                        attempts[attempt_index],
                        executed_action_index=step_count,
                        executed_action_type=type(action).__name__,
                        executed_action=_action_payload(action),
                        observed_after=_robot_snapshot(after.robot),
                        completion_error=final_completion_error,
                        completion_unit="m",
                        completion_target=target,
                        apply_accepted=True,
                        apply_reason=applied.reason,
                        backend_stepped=True,
                        backend_step_number=step_count,
                        episode_step_number=episode_steps_before + step_count,
                    )
                    if final_completion_error <= tolerance:
                        break
                except Exception as error_:  # noqa: BLE001 - fail closed with measured prefix
                    error = f"{type(error_).__name__}: {error_}"
                    stop_reason = action_reason = "execution_exception"
                    break
            action_records.append(
                FeedbackActionEvidence(
                    selected_action_index=selected_index,
                    selected_action_type=type(action).__name__,
                    selected_action=_action_payload(action),
                    home_action=False,
                    convergence_tolerance_rad=None,
                    terminal_reason=action_reason,
                    attempt_indices=tuple(action_attempts),
                    executed_action_count=sum(
                        attempts[index].backend_stepped for index in action_attempts
                    ),
                    backend_step_count=sum(
                        attempts[index].backend_stepped for index in action_attempts
                    ),
                    final_max_joint_error_rad=None,
                    completion_target=target,
                    completion_error=final_completion_error,
                    completion_unit="m",
                )
            )
            continue

        error = f"unsupported selected action type: {type(action).__name__}"
        stop_reason = "execution_exception"
        break

    if stop_reason is None:
        home_target = dict(contract.diagnostic_home_target_rad)
        home_action = JointTarget(home_target)
        home_tolerance = contract.diagnostic_home_tolerance_rad
        initial_home_error = _max_joint_error(home_action, observations[-1].robot.joint_pos)
        final_home_error = initial_home_error
        home_reason = "home_converged"
        home_attempts: list[int] = []
        home_steps_before = step_count
        home_move_steps = 0
        home_no_progress_steps = 0
        while True:
            current = observations[-1]
            final_home_error = _max_joint_error(home_action, current.robot.joint_pos)
            if final_home_error <= home_tolerance:
                break
            if step_count >= remaining_steps:
                home_reason = "global_step_ceiling"
                break
            if home_move_steps >= contract.joint_move_timeout_steps:
                home_reason = "joint_move_timeout"
                break
            home_attempt_index: int | None = None
            safety_accepted = False
            try:
                executed_action, clamped = _project_joint_target(
                    home_action, current.robot.joint_pos, contract.max_joint_step_rad
                )
                projection = JointProjectionEvidence(
                    observed_joint_pos_rad={
                        str(name): float(value)
                        for name, value in current.robot.joint_pos.items()
                        if name in home_target
                    },
                    selected_joint_target_rad={
                        str(name): float(value) for name, value in home_action.q.items()
                    },
                    executed_joint_target_rad={
                        str(name): float(value) for name, value in executed_action.q.items()
                    },
                    clamped_joints=clamped,
                    bound_rad=contract.max_joint_step_rad,
                )
                safety_calls += 1
                verdict = monitor_probe.validate(executed_action, current.robot)
                safety_accepted = bool(verdict.accepted)
                if safety_accepted:
                    safety_ok += 1
                home_attempt_index = _record_attempt(
                    selected_index=len(actions),
                    selected_action=home_action,
                    executed_action=executed_action,
                    attempt_within_action=home_move_steps + 1,
                    projection=projection,
                    before=current,
                    before_error=final_home_error,
                    safety_accepted=safety_accepted,
                    safety_reason=verdict.reason,
                    apply_accepted=None,
                    apply_reason=None,
                    phase="diagnostic_home_completion",
                )
                home_attempts.append(home_attempt_index)
                if not safety_accepted:
                    home_reason = "safety_rejected"
                    break
                applied, after = _tick(home_attempt_index, executed_action)
                if after is None:
                    home_reason = "backend_apply_rejected"
                    break
                progress = max(
                    (
                        abs(
                            float(after.robot.joint_pos[name])
                            - float(current.robot.joint_pos[name])
                        )
                        for name in home_target
                        if name in after.robot.joint_pos and name in current.robot.joint_pos
                    ),
                    default=0.0,
                )
                after_error = _max_joint_error(home_action, after.robot.joint_pos)
                home_no_progress_steps = (
                    0
                    if progress > contract.joint_progress_epsilon_rad
                    else home_no_progress_steps + 1
                )
                home_move_steps += 1
                final_home_error = after_error
                attempts[home_attempt_index] = replace(
                    attempts[home_attempt_index],
                    executed_action_index=step_count,
                    executed_action_type=type(executed_action).__name__,
                    executed_action=_action_payload(executed_action),
                    observed_after=_robot_snapshot(after.robot),
                    max_joint_error_after_rad=after_error,
                    progress_rad=progress,
                    no_progress_steps=home_no_progress_steps,
                    apply_accepted=True,
                    apply_reason=applied.reason,
                    backend_stepped=True,
                    backend_step_number=step_count,
                    episode_step_number=episode_steps_before + step_count,
                )
                if home_no_progress_steps >= contract.joint_stall_steps:
                    home_reason = "joint_stall"
                    break
                if after_error <= home_tolerance:
                    break
            except Exception as error_:  # noqa: BLE001 - fail closed with measured prefix
                error = f"{type(error_).__name__}: {error_}"
                home_reason = "execution_exception"
                break
        home_completion = DiagnosticHomeCompletionEvidence(
            phase="diagnostic_home_completion",
            source="configs/worldmind_diagnostic.approved.yaml:verification.home_joints",
            target_joint_pos_rad=home_target,
            tolerance_rad=home_tolerance,
            initial_max_joint_error_rad=initial_home_error,
            final_max_joint_error_rad=final_home_error,
            terminal_reason=home_reason,
            attempt_indices=tuple(home_attempts),
            executed_action_count=sum(attempts[index].backend_stepped for index in home_attempts),
            backend_step_count=step_count - home_steps_before,
        )
        diagnostic_events.append(
            {
                "schema_version": 1,
                "event_type": "diagnostic_home_completion",
                "episode_id": episode_id,
                "decision_id": decision_id,
                "phase": "diagnostic_home_completion",
                "source": home_completion.source,
                "tolerance_rad": home_tolerance,
                "initial_max_joint_error_rad": initial_home_error,
                "final_max_joint_error_rad": final_home_error,
                "terminal_reason": home_reason,
                "backend_step_count": home_completion.backend_step_count,
                "note": (
                    "Diagnostic-only HOME completion after selected-chunk execution; it never "
                    "rewrites selected actions or creates a prediction pairing."
                ),
            }
        )
        if home_reason != "home_converged":
            stop_reason = home_reason

    converged = [
        record
        for record in action_records
        if record.convergence_tolerance_rad is not None
        and record.terminal_reason == "target_converged"
    ]
    non_converged = [
        record
        for record in action_records
        if record.convergence_tolerance_rad is not None
        and record.terminal_reason != "target_converged"
    ]
    reasons: dict[str, int] = {}
    for record in non_converged:
        reasons[record.terminal_reason] = reasons.get(record.terminal_reason, 0) + 1
    convergence_summary = FeedbackConvergenceSummary(
        converged_action_indices=tuple(record.selected_action_index for record in converged),
        converged_action_count=len(converged),
        max_final_error_converged_rad=(
            max(
                float(record.final_max_joint_error_rad)
                for record in converged
                if record.final_max_joint_error_rad is not None
            )
            if any(record.final_max_joint_error_rad is not None for record in converged)
            else None
        ),
        non_converged_action_indices=tuple(
            record.selected_action_index for record in non_converged
        ),
        non_converged_reasons=reasons,
        non_converged_final_errors=tuple(
            (
                record.selected_action_index,
                record.terminal_reason,
                float(record.final_max_joint_error_rad),
            )
            for record in non_converged
            if record.final_max_joint_error_rad is not None
        ),
    )

    termination = stop_reason or "selected_chunk_converged"
    applied_count = apply_ok
    safety_rejections = safety_calls - safety_ok
    terminal = next(
        (item for item in reversed(attempts) if item.backend_stepped),
        attempts[-1] if attempts else None,
    )
    execution = ExecutionEvidence(
        decision_id=decision_id,
        selected_candidate_id=selected_id,
        executed_candidate_id=selected_id,
        pre_apply_accepted=None if terminal is None else terminal.safety_accepted,
        pre_apply_reason=None if terminal is None else terminal.safety_reason,
        apply_accepted=None if terminal is None else terminal.apply_accepted,
        apply_reason=None if terminal is None else terminal.apply_reason,
        backend_stepped=step_count > 0,
        observation_t_s=float(observation.t),
        action_t_s=float(observations[-1].t),
    )
    diagnostic = ObservedStateFeedbackEvidence(
        schema_version=FEEDBACK_EVIDENCE_SCHEMA_VERSION,
        execution_mode=OBSERVED_STATE_FEEDBACK_EXECUTION_MODE,
        episode_id=episode_id,
        decision_id=decision_id,
        selected_candidate_id=selected_id,
        selected_action_count=len(actions),
        attempt_count=len(attempts),
        applied_action_count=applied_count,
        backend_step_count=step_count,
        observation_count=len(observations),
        safety_validate_count=safety_calls,
        safety_accepted_count=safety_ok,
        safety_rejected_count=safety_rejections,
        apply_accepted_count=apply_ok,
        apply_rejected_count=apply_rejections,
        episode_steps_before=episode_steps_before,
        episode_steps_after=episode_steps_before + step_count,
        episode_max_steps=contract.episode_max_steps,
        termination_reason=termination,
        budgets=contract.evidence(),
        prediction_provenance=FeedbackPredictionProvenance(
            original_prediction_candidate_id=selected_id,
            original_prediction_horizon_steps=config.pipeline.horizon_steps,
        ),
        actions=tuple(action_records),
        attempts=tuple(attempts),
        convergence_summary=convergence_summary,
        error=error,
        home_completion=home_completion,
        apply_unknown_count=sum(item.apply_outcome == "unknown" for item in attempts),
        step_unknown_count=sum(item.step_outcome == "unknown" for item in attempts),
        observation_failure_count=sum(
            item.observation_outcome in ("failed", "processing_failed") for item in attempts
        ),
        recording_failure_count=sum(
            (item.apply_recording == "failed") + (item.step_recording == "failed")
            for item in attempts
        ),
    )
    return _FeedbackExecutionResult(
        observation=observations[-1],
        ground_truth=final_ground_truth,
        ground_truths=tuple(ground_truths),
        evidence=execution,
        diagnostic=diagnostic,
        observations=tuple(observations),
        observed_states=tuple(observed_states),
        diagnostic_events=tuple(diagnostic_events),
        termination_reason=termination,
        backend_step_count=step_count,
    )


class Stage1WorldMindEpisodePolicy:
    """One Stage 1 episode under an explicitly selected non-scored diagnostic mode."""

    def __init__(
        self,
        scenario: Any,
        config: WorldMindDiagnosticConfig,
        bindings: WorldMindBindings,
        writer: WorldMindEvidenceWriter,
        mode: Stage1ArmMode,
        execution_mode: Stage1ExecutionMode = "receding_horizon_tick",
        settings: EvalSettings | None = None,
    ) -> None:
        self.scenario = scenario
        self.config = config
        self.bindings = bindings
        self.writer = writer
        self.settings = settings if settings is not None else load_eval_settings()
        self.mode = validate_stage1_arm_mode(mode)
        self.execution_mode = validate_stage1_execution_mode(execution_mode)
        _require_feedback_arm_mode(self.mode, self.execution_mode)
        if self.mode == "full_system_veto" and self.execution_mode != "receding_horizon_tick":
            raise ValueError("full_system_veto cannot execute an open-loop chunk")

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
        failure_state = "WORLDMIND_STAGE1_ABORT"
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
            monitor = self.bindings.safety[self.config.safety.binding](
                self.config, self.bindings.inputs.get("safety", {})
            )
            predictor = self.bindings.surrogates[self.config.surrogate.binding](
                self.config, self.bindings.inputs.get("surrogate", {})
            )
            capture = _TrajectoryCapture(predictor)
            verifier = self.bindings.verification[self.config.verification.binding](
                self.config, destination_base, rest_center_base, {}
            )
            no_safe = RecoveryState()
            use_alternate = False
            decision_index = 0
            loop_exited_by_bound = True

            while (
                decision_index < self.config.bounds.max_decisions
                and steps < self.config.bounds.max_steps
            ):
                assert observation is not None and state is not None
                primary = self._waypoints(self.config.adapter.waypoints_binding, state, task)
                alternate = (
                    self._waypoints(self.config.adapter.alternate_waypoints_binding, state, task)
                    if self.config.adapter.alternate_waypoints_binding
                    else None
                )
                before_observation, before_state = observation, state
                decision_id = f"{episode_id}-d{decision_index:02d}"
                waypoints = alternate if use_alternate and alternate else primary
                generator = self._generator(
                    waypoints, seed, episode_id=episode_id, decision_id=decision_id
                )
                generation_probe = CandidateGenerationProbe(
                    generator,
                    episode_id=episode_id,
                    decision_id=decision_id,
                    timing_enabled=self.config.bounds.timings_enabled,
                )
                planning_start = perf_counter_ns()
                trajectory_offset = len(capture.trajectories)
                candidates = generation_probe.propose(task, state, None)
                generation = generation_probe.observations[-1]

                predictions: list[PredictionObservation] = []
                assessments: list[_Stage1Assessment] = []
                hard_total_ns = 0
                for index, chunk in enumerate(candidates):
                    prediction, trajectory = self._predict(
                        capture,
                        state,
                        chunk,
                        episode_id=episode_id,
                        decision_id=decision_id,
                        index=index,
                    )
                    predictions.append(prediction)
                    assessment = self._assess(
                        state,
                        chunk,
                        trajectory,
                        hard_checks=list(self.config.assessment.hard_checks),
                    )
                    assessments.append(assessment)
                    if assessment.duration_s is not None:
                        hard_total_ns += int(assessment.duration_s * 1_000_000_000)

                selection_start = perf_counter_ns()
                selection = self._select(candidates, predictions, assessments)
                selection_s = self._seconds(selection_start)
                rejections += int(selection.veto_count)
                planning_s = self._seconds(planning_start)
                base_provenance = self._decision_provenance(capture, predictions)
                timing = self._build_timings(
                    generation_s=generation.duration_s,
                    prediction_samples=[row.duration_s for row in predictions],
                    hard_checks_s=self._seconds_since(hard_total_ns),
                    selection_s=selection_s,
                    planning_s=planning_s,
                    verification_s=None,
                    recovery_s=None,
                )

                if selection.index is None:
                    recovery_start = perf_counter_ns()
                    if self.mode == "full_system_veto":
                        event, abort = "abort", True
                    else:
                        event = no_safe.no_safe()
                        abort = event == "abort"
                        if event == "alternate_candidate_family" and alternate is None:
                            event = "alternate_candidate_family:unavailable"
                            abort = True
                        if event == "reobserve":
                            if self.execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE:
                                outcome = {
                                    "schema_version": 1,
                                    "event_type": "diagnostic_reobserve",
                                    "episode_id": episode_id,
                                    "decision_id": decision_id,
                                    "step_outcome": "unknown",
                                    "observation_outcome": "not_available",
                                    "step_recording": "not_attempted",
                                }
                                try:
                                    ticks, after, step_error = _request_feedback_tick(backend)
                                    if ticks != 1:
                                        if ticks == 0:
                                            outcome["step_outcome"] = "not_executed"
                                        raise RuntimeError(
                                            step_error or "re-observe tick unconfirmed"
                                        )
                                    steps += 1
                                    outcome["step_outcome"] = "confirmed"
                                    if after is None:
                                        outcome["observation_outcome"] = "failed"
                                        raise RuntimeError(
                                            step_error or "confirmed tick has no observation"
                                        )
                                    outcome["observation_outcome"] = "processing_failed"
                                    observation = after
                                    ground_truth = backend.get_ground_truth()
                                    state = _world_state(
                                        observation, ground_truth, self.config, target=task.target
                                    )
                                    outcome["observation_outcome"] = "available"
                                    outcome["step_recording"] = "failed"
                                    backend.record_diagnostic_step(observation, ground_truth)
                                    outcome["step_recording"] = "completed"
                                except Exception as exc:  # noqa: BLE001 - retain outcome then fail-stop
                                    outcome["error"] = f"{type(exc).__name__}: {exc}"
                                    raise
                                finally:
                                    self.writer.record_diagnostic_event(outcome)
                            else:
                                observation = backend.step(1)
                                steps += 1
                                ground_truth = backend.get_ground_truth()
                                state = _world_state(
                                    observation, ground_truth, self.config, target=task.target
                                )
                        if event == "alternate_candidate_family":
                            use_alternate = True
                    recovery_s = self._seconds(recovery_start)
                    timed = self._with_phases(timing, recovery_s=recovery_s)
                    self.writer.record_decision(
                        _decision_evidence(
                            config=self.config,
                            mode=self.mode,
                            execution_mode=self.execution_mode,
                            episode_id=episode_id,
                            decision_id=decision_id,
                            decision_index=decision_index,
                            generation=generation,
                            predictions=predictions,
                            assessments=assessments,
                            selection=replace(selection, recovery_events=(event,)),
                            timings=timed,
                            decision_provenance=base_provenance,
                        )
                    )
                    failure_state, failure_reason = "NO_SAFE_SET", selection.rationale
                    if abort:
                        loop_exited_by_bound = False
                        break
                    decision_index += 1
                    continue

                selected_id = candidate_id(selection.index)
                selected_trajectory_index = trajectory_offset + selection.index
                monitor_probe = SafetyMonitorProbe(monitor)
                backend_probe = BackendApplyProbe(backend)
                chunk_result: _ChunkExecutionResult | None = None
                feedback_result: _FeedbackExecutionResult | None = None
                observed_states: Sequence[WorldState] = (before_state,)
                observed_timestamps: Sequence[float] = (float(before_observation.t),)
                if self.execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE:
                    solver = self.bindings.pose_solvers[self.config.adapter.pose_solver_binding]
                    contract = _load_feedback_contract(self.config, solver)
                    feedback_result = _execute_observed_state_feedback_chunk(
                        decision_id=decision_id,
                        episode_id=episode_id,
                        selection=selection,
                        config=self.config,
                        contract=contract,
                        task=task,
                        observation=observation,
                        state=state,
                        ground_truth=ground_truth,
                        backend_probe=backend_probe,
                        monitor_probe=monitor_probe,
                        waypoints=waypoints,
                        episode_steps_before=steps,
                    )
                    execution = feedback_result.evidence
                    observed_states = feedback_result.observed_states
                    observed_timestamps = tuple(item.t for item in feedback_result.observations)
                    steps += feedback_result.backend_step_count
                    self.writer.record_feedback(feedback_result.diagnostic)
                    for event in feedback_result.diagnostic_events:
                        self.writer.record_diagnostic_event(event)
                    proxy = build_collision_proxy_attribution(
                        episode_id=episode_id,
                        ground_truths=feedback_result.ground_truths,
                        observations=feedback_result.observations,
                        threshold_source=f"{self.settings.source_path}:collision_proxy",
                        max_displacement_m=self.settings.collision_proxy_max_displacement_m,
                        max_tilt_deg=self.settings.collision_proxy_max_tilt_deg,
                    )
                    self.writer.record_collision_proxy(proxy)
                    boundary = feedback_result.termination_reason
                    if boundary not in FEEDBACK_TERMINATION_REASONS:
                        raise RuntimeError(f"unknown feedback termination reason: {boundary}")
                    if boundary not in ("selected_chunk_converged", "gripper_command_complete"):
                        failure_state = _FEEDBACK_ABORT_STATES[boundary]
                        home = feedback_result.diagnostic.home_completion
                        failure_reason = feedback_result.diagnostic.error or (
                            (
                                f"diagnostic HOME completion {home.terminal_reason}: "
                                f"final_max_joint_error_rad="
                                f"{home.final_max_joint_error_rad}"
                            )
                            if home is not None and home.terminal_reason not in ("home_converged",)
                            else boundary
                        )
                        # Feedback ticks are non-comparable with the immutable H=50 rollout.
                        self.writer.record_decision(
                            _decision_evidence(
                                config=self.config,
                                mode=self.mode,
                                execution_mode=self.execution_mode,
                                episode_id=episode_id,
                                decision_id=decision_id,
                                decision_index=decision_index,
                                generation=generation,
                                predictions=predictions,
                                assessments=assessments,
                                selection=selection,
                                timings=timing,
                                execution=execution,
                                decision_provenance=base_provenance,
                            )
                        )
                        loop_exited_by_bound = boundary == "global_step_ceiling"
                        break
                elif self.execution_mode == "diagnostic_open_loop_chunk":
                    chunk_result = _execute_selected_chunk(
                        decision_id=decision_id,
                        selection=selection,
                        config=self.config,
                        task=task,
                        observation=observation,
                        state=state,
                        ground_truth=ground_truth,
                        backend_probe=backend_probe,
                        monitor_probe=monitor_probe,
                    )
                    execution = chunk_result.evidence
                    chunk = execution.chunk
                    if chunk is None:
                        raise RuntimeError("open-loop execution did not record chunk evidence")
                    observed_states = chunk_result.observed_states
                    observed_timestamps = tuple(item.t for item in chunk_result.observations)
                    # Count every backend step recorded by the chunk before any early abort
                    # break; safety/backend/exception terminations still executed a prefix.
                    steps += chunk.backend_step_count
                    if chunk.termination_reason != "chunk_complete":
                        boundary = chunk.termination_reason
                        failure_state = {
                            "safety_rejected": "SAFETY_ABORT",
                            "backend_apply_rejected": "BACKEND_ABORT",
                            "execution_exception": "EXECUTION_EXCEPTION",
                        }[boundary]
                        terminal = chunk.actions[-1] if chunk.actions else None
                        failure_reason = (
                            chunk.error
                            or (
                                terminal.pre_apply_reason
                                if boundary == "safety_rejected"
                                else terminal.apply_reason
                                if boundary == "backend_apply_rejected"
                                else None
                            )
                            or boundary
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
                            observed_states=observed_states,
                            observed_timestamps=observed_timestamps,
                        )
                        self.writer.record_decision(
                            _decision_evidence(
                                config=self.config,
                                mode=self.mode,
                                execution_mode=self.execution_mode,
                                episode_id=episode_id,
                                decision_id=decision_id,
                                decision_index=decision_index,
                                generation=generation,
                                predictions=predictions,
                                assessments=assessments,
                                selection=selection,
                                timings=timing,
                                execution=execution,
                                pairing=pairing,
                                decision_provenance=base_provenance,
                            ),
                            pairing,
                        )
                        loop_exited_by_bound = False
                        break
                else:
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
                    if (
                        execution.pre_apply_accepted is False
                        or execution.apply_accepted is not True
                    ):
                        boundary = (
                            "pre_apply_veto"
                            if execution.pre_apply_accepted is False
                            else "backend_apply_rejected"
                        )
                        failure_state, failure_reason = (
                            "SAFETY_ABORT",
                            execution.pre_apply_reason or execution.apply_reason or boundary,
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
                        self.writer.record_decision(
                            _decision_evidence(
                                config=self.config,
                                mode=self.mode,
                                execution_mode=self.execution_mode,
                                episode_id=episode_id,
                                decision_id=decision_id,
                                decision_index=decision_index,
                                generation=generation,
                                predictions=predictions,
                                assessments=assessments,
                                selection=selection,
                                timings=timing,
                                execution=execution,
                                pairing=pairing,
                                decision_provenance=base_provenance,
                            ),
                            pairing,
                        )
                        loop_exited_by_bound = False
                        break

                if feedback_result is not None:
                    after = feedback_result.observation
                    ground_truth = feedback_result.ground_truth
                    after_state = feedback_result.observed_states[-1]
                    observed_states = feedback_result.observed_states
                    observed_timestamps = tuple(item.t for item in feedback_result.observations)
                elif chunk_result is not None:
                    chunk = chunk_result.evidence.chunk
                    if chunk is None:
                        raise RuntimeError("open-loop execution did not record chunk evidence")
                    after = chunk_result.observation
                    ground_truth = chunk_result.ground_truth
                    after_state = chunk_result.observed_states[-1]
                    observed_states = chunk_result.observed_states
                    observed_timestamps = tuple(item.t for item in chunk_result.observations)
                else:
                    steps += 1
                    after_ground_truth = backend.get_ground_truth()
                    after_state = _world_state(
                        after, after_ground_truth, self.config, target=task.target
                    )
                    ground_truth = after_ground_truth
                    observed_states = (before_state, after_state)
                    observed_timestamps = (float(before_observation.t), float(after.t))
                observation, state = after, after_state
                verification_start = perf_counter_ns()
                trajectory = self._selected_trajectory(capture, selected_trajectory_index)
                verifier_probe = VerificationProbe(verifier)
                verification = verifier_probe.check(task, trajectory.states[-1], after_state)
                transition_start = perf_counter_ns()
                transition = verifier_probe.next_action(verification)
                verification_s = self._seconds(verification_start)
                recovery_s = self._seconds(transition_start)
                pairing = (
                    None
                    if self.execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
                    else self._pairing(
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
                        observed_states=observed_states,
                        observed_timestamps=observed_timestamps,
                    )
                )
                timed = self._with_phases(
                    timing, verification_s=verification_s, recovery_s=recovery_s
                )
                evidence = _decision_evidence(
                    config=self.config,
                    mode=self.mode,
                    execution_mode=self.execution_mode,
                    episode_id=episode_id,
                    decision_id=decision_id,
                    decision_index=decision_index,
                    generation=generation,
                    predictions=predictions,
                    assessments=assessments,
                    selection=selection,
                    timings=timed,
                    execution=execution,
                    verification=self._verification_evidence(
                        decision_id, verification, transition, after.t
                    ),
                    pairing=pairing,
                    decision_provenance=base_provenance,
                )
                self.writer.record_decision(evidence, pairing)
                if verification.success and transition == "complete":
                    verified_success, failure_state, failure_reason = True, "", ""
                    loop_exited_by_bound = False
                    break
                if transition == "abort":
                    failure_state, failure_reason = "VERIFICATION_ABORT", verification.reason
                    loop_exited_by_bound = False
                    break
                failure_state, failure_reason = "VERIFICATION_RETRY", verification.reason
                use_alternate = True
                decision_index += 1
        except Exception as error:  # noqa: BLE001 - runner records a fail-closed failure row
            loop_exited_by_bound = False
            self.writer.record_error(episode_id, error)
            failure_state = "EXCEPTION"
            failure_reason = f"{type(error).__name__}: {error}"

        if loop_exited_by_bound:
            reached: list[str] = []
            if decision_index >= self.config.bounds.max_decisions:
                reached.append(
                    f"max_decisions={self.config.bounds.max_decisions} after "
                    f"{decision_index} decisions"
                )
            if steps >= self.config.bounds.max_steps:
                reached.append(f"max_steps={self.config.bounds.max_steps} after {steps} steps")
            failure_state = "WORLDMIND_BOUND_EXHAUSTED"
            failure_reason = "bound exhaustion: " + (
                "; ".join(reached) if reached else "no progress possible"
            )

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

    def _predict(
        self,
        capture: _TrajectoryCapture,
        state: WorldState,
        chunk: Any,
        *,
        episode_id: str,
        decision_id: str,
        index: int,
    ) -> tuple[PredictionObservation, Trajectory | None]:
        enabled = self.config.bounds.timings_enabled
        started = perf_counter_ns() if enabled else None
        try:
            trajectory = capture.predict(state, chunk)
            error = None
        except Exception as error_:  # noqa: BLE001 - keep an explicit unevaluable reason
            trajectory, error = None, f"{type(error_).__name__}: {error_}"
        duration = None if not enabled or started is None else self._seconds(started)
        observation = PredictionObservation(
            episode_id=episode_id,
            decision_id=decision_id,
            candidate_index=index,
            candidate_id=candidate_id(index),
            proposer_source=str(chunk.source),
            action_count=len(chunk.actions),
            available=trajectory is not None and error is None,
            horizon_steps=None if trajectory is None else max(0, len(trajectory.states) - 1),
            trajectory_state_count=None if trajectory is None else len(trajectory.states),
            hard_reasons=() if trajectory is None else tuple(trajectory.hard_reasons),
            duration_s=duration,
            error_code=error,
        )
        return observation, trajectory

    def _assess(
        self,
        state: WorldState,
        chunk: Any,
        trajectory: Trajectory | None,
        *,
        hard_checks: list[str],
    ) -> _Stage1Assessment:
        enabled = self.config.bounds.timings_enabled
        started = perf_counter_ns() if enabled else None
        results: list[tuple[str, bool | None]] = []
        error: str | None = None
        try:
            for name in hard_checks:
                check = self.bindings.hard_checks.get(name)
                if not callable(check):
                    raise ValueError(f"unregistered hard check: {name}")
                value = check(state, chunk, trajectory)
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"hard check {name} must return bool or None")
                results.append((name, value))
            if self.mode == "full_system_veto":
                if trajectory is None:
                    results.append(("prediction_complete", False))
                    results.append(("prediction_constraints", False))
                else:
                    results.append(
                        ("prediction_complete", len(trajectory.states) == len(chunk.actions) + 1)
                    )
                    results.append(("prediction_constraints", not trajectory.hard_reasons))
                    if "unverified-arm-sweep" in trajectory.hard_reasons:
                        results.append(("unverified-arm-sweep", False))
                # Full system remains blocked independently of whether a sweep certificate exists.
                results.append(("stage1_full_system_blocked", False))
        except Exception as error_:  # noqa: BLE001 - fail closed with an explicit reason
            error = f"{type(error_).__name__}: {error_}"
        duration = None if not enabled or started is None else self._seconds(started)
        return _Stage1Assessment(tuple(results), duration, error)

    def _select(
        self,
        candidates: Sequence[Any],
        predictions: Sequence[PredictionObservation],
        assessments: Sequence[_Stage1Assessment],
    ) -> Selection:
        safe: list[int] = []
        for index, assessment in enumerate(assessments):
            prediction = predictions[index]
            if prediction.error_code is not None or not prediction.available:
                continue
            if assessment.error_code is not None:
                continue
            reasons, _ = _hard_reason_codes(
                assessment.hard_checks, self.config.assessment.required_constraints
            )
            if not reasons:
                safe.append(index)
        veto_count = len(candidates) - len(safe)
        if self.mode == "full_system_veto":
            return Selection(
                None,
                None,
                "full_system remains blocked; all candidates hard-vetoed without actuation",
                prediction_count=len(candidates),
                veto_count=veto_count,
            )
        if not safe:
            return Selection(
                None,
                None,
                "no hard-check-safe scripted candidate; no action applied",
                prediction_count=len(candidates),
                veto_count=veto_count,
            )
        index = safe[0]
        return Selection(
            candidates[index],
            index,
            "first safe seeded candidate; no utility ranking or calibration evidence",
            prediction_count=len(candidates),
            veto_count=veto_count,
        )

    def _decision_provenance(
        self, capture: _TrajectoryCapture, predictions: Sequence[PredictionObservation]
    ) -> dict[str, str]:
        reasons = sorted({reason for row in predictions for reason in row.hard_reasons})
        return {
            "state_source": self.config.backend.state_source,
            "surrogate_binding": self.config.surrogate.binding,
            "oracle_boundary": "target pose from privileged oracle; robot fields from observation",
            "perception_equivalent": "false",
            "prediction_hard_reasons": ",".join(reasons) if reasons else "none",
            "collision_channels": (
                "proxy=recorded displacement/tilt; contact=backend contacts only; "
                "sweep=no certified full-arm check"
            ),
        }

    def _generator(
        self,
        waypoints: Sequence[ScriptedWaypoint],
        seed: int,
        *,
        episode_id: str,
        decision_id: str,
    ) -> SeededCandidateGenerator:
        monitor = self.bindings.safety[self.config.safety.binding](
            self.config, self.bindings.inputs.get("safety", {})
        )
        solver = self.bindings.pose_solvers[self.config.adapter.pose_solver_binding]
        max_joint_step_rad = getattr(solver, "execution_max_joint_step_rad", None)
        if max_joint_step_rad is None or not math.isfinite(float(max_joint_step_rad)):
            raise ValueError("registered pose solver must expose a finite execution step limit")
        adapter = ScriptedChunkAdapter(
            waypoints,
            solver=solver,
            monitor=monitor,
            max_joint_step_rad=float(max_joint_step_rad),
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
        draw_slot = 0

        def record_failure(failure: CandidateIKFailure) -> None:
            self.writer.record_diagnostic_event(
                {
                    **asdict(failure),
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "decision_id": decision_id,
                    "draw_slot": draw_slot,
                    # Execution begins only after propose returns the entire pool.
                    "backend_actions_applied": 0,
                }
            )

        draw = adapter.draw(
            on_ik_failure=record_failure
            if self.execution_mode == OBSERVED_STATE_FEEDBACK_EXECUTION_MODE
            else None
        )

        def tracked_draw(
            task: TaskSpec, state: WorldState, memory: object, seed: int
        ) -> ActionChunk:
            nonlocal draw_slot
            chunk = draw(task, state, memory, seed)
            draw_slot += 1
            return chunk

        return SeededCandidateGenerator(
            tracked_draw,
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

    @staticmethod
    def _pairing(
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
        observed_states: Sequence[WorldState] | None = None,
        observed_timestamps: Sequence[float] | None = None,
    ) -> Any:
        if index >= len(capture.trajectories):
            return None
        observed = (
            list(observed_states)
            if observed_states is not None
            else [before_state]
            if after is None or after_state is None
            else [before_state, after_state]
        )
        timestamps = (
            list(observed_timestamps)
            if observed_timestamps is not None
            else [float(before.t)]
            if after is None
            else [float(before.t), float(after.t)]
        )
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
        hard_checks_s: float | None,
        selection_s: float | None,
        planning_s: float | None,
        verification_s: float | None,
        recovery_s: float | None,
    ) -> PhaseTimings:
        return _timings(
            enabled=self.config.bounds.timings_enabled,
            generation_s=generation_s,
            prediction_samples=prediction_samples,
            hard_checks_s=hard_checks_s,
            selection_s=selection_s,
            planning_s=planning_s,
            verification_s=verification_s,
            recovery_s=recovery_s,
        )

    def _with_phases(self, timing: PhaseTimings, **values: float) -> PhaseTimings:
        """Attach late-measured phases only when the reviewed config enabled timings."""
        if not self.config.bounds.timings_enabled:
            return timing
        return replace(
            timing,
            **values,
            missing_reasons={
                key: reason for key, reason in timing.missing_reasons.items() if key not in values
            },
        )

    @staticmethod
    def _seconds(start: int) -> float:
        return (perf_counter_ns() - start) / 1_000_000_000.0

    @staticmethod
    def _seconds_since(total_ns: int) -> float:
        return total_ns / 1_000_000_000.0

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
class Stage1PolicyFactory:
    """Create one Stage 1 policy per scene and finalize additive sidecars once."""

    config: WorldMindDiagnosticConfig
    bindings: WorldMindBindings
    writer: WorldMindEvidenceWriter
    mode: Stage1ArmMode
    execution_mode: Stage1ExecutionMode = "receding_horizon_tick"
    settings: EvalSettings = field(default_factory=load_eval_settings)

    def __call__(self, scene: Any) -> Stage1WorldMindEpisodePolicy:
        return Stage1WorldMindEpisodePolicy(
            scene,
            self.config,
            self.bindings,
            self.writer,
            self.mode,
            self.execution_mode,
            self.settings,
        )

    def finish(self) -> dict[str, Path]:
        return self.writer.finish(
            gate_policy=STAGE1_GATE_CALL_POLICY,
            expected_gate_calls_per_decision=0,
            latency_note=STAGE1_LATENCY_NOTE,
        )


STAGE1_QUARANTINE_FILE = "quarantine.json"


def _quarantine_path(out_dir: Path) -> Path:
    return out_dir / "worldmind-sidecars" / STAGE1_QUARANTINE_FILE


def _assert_no_resume_quarantine(out_dir: Path) -> None:
    """Refuse a directory previously marked as containing incomplete episode artifacts."""
    path = _quarantine_path(out_dir)
    if path.exists():
        raise ValueError(f"resume quarantined by {path}; do not use this evidence directory")


def _validate_resume_coverage(
    out_dir: Path, plan: Sequence[EvalEpisode], writer: WorldMindEvidenceWriter, pipeline: str
) -> None:
    """Refuse resume when completed recorder episodes and sidecar coverage disagree."""
    _assert_no_resume_quarantine(out_dir)
    completed = {
        entry.scene.name
        for entry in plan
        if _completed_recording(out_dir, entry, pipeline) is not None
    }
    recorded = writer.episode_ids
    if completed != recorded:
        write_json(
            {
                "reason": "incomplete episode artifacts; resume refused",
                "completed_recordings": sorted(completed),
                "sidecar_episodes": sorted(recorded),
            },
            _quarantine_path(out_dir),
        )
        raise ValueError(
            "resume quarantined: coverage mismatch: "
            f"completed_recordings={sorted(completed)} sidecar_episodes={sorted(recorded)}"
        )


def run_stage1_evidence(
    plan: Sequence[EvalEpisode],
    backend: Any,
    config: WorldMindDiagnosticConfig,
    out_dir: Path,
    bindings: WorldMindBindings,
    *,
    arm_mode: Stage1ArmMode,
    execution_mode: Stage1ExecutionMode = "receding_horizon_tick",
    runtime_values: Mapping[str, str],
    settings: EvalSettings | None = None,
    startup_s: float = 0.0,
    resume: bool = False,
) -> tuple[EvalReport, dict[str, Path]]:
    """Run the frozen recorder with non-scored Stage 1 evidence and exclusive sidecars."""
    mode = validate_stage1_arm_mode(arm_mode)
    execution_mode = validate_stage1_execution_mode(execution_mode)
    _require_feedback_arm_mode(mode, execution_mode)
    if mode == "full_system_veto" and execution_mode != "receding_horizon_tick":
        raise ValueError("full_system_veto cannot execute an open-loop chunk")
    resolved_settings = settings if settings is not None else load_eval_settings()
    provenance = _stage1_provenance(config, plan, mode, execution_mode, runtime_values)
    writer = WorldMindEvidenceWriter(out_dir)
    writer.start(provenance, resume=resume)
    pipeline = STAGE1_RECORDER_PIPELINE[mode]
    if resume:
        _validate_resume_coverage(out_dir, plan, writer, pipeline)
    factory = Stage1PolicyFactory(
        config, bindings, writer, mode, execution_mode, settings=resolved_settings
    )
    run_config = EvalRunConfig(
        out_dir=out_dir,
        pipeline=pipeline,
        split=config.split,
        base_seed=config.base_seed,
        backend=config.backend.kind,
        measurement="synthetic" if config.backend.kind == "fake" else "measured",
        state_source=config.backend.state_source,
        record_frames=False,
        resume=resume,
        stop_on_exception=True,
    )
    try:
        report = run_evaluation(
            plan,
            backend,
            run_config,
            policy_factory=factory,
            settings=resolved_settings,
            startup_s=startup_s,
        )
    finally:
        sidecars = factory.finish()
    report.summary["stage1_diagnostic"] = {
        "schema_version": 1,
        "scored_evaluation_allowed": False,
        "independent_evaluation": False,
        "performance_claim": "not_permitted_bookkeeping_only",
        "arm_mode": mode,
        "configured_arm": config.pipeline.arm,
        "effective_arm": STAGE1_EFFECTIVE_ARM[mode],
        "evidence_class": STAGE1_EVIDENCE_CLASS[mode],
        "execution_mode": execution_mode,
        "execution_mode_label": STAGE1_EXECUTION_MODE_LABELS[execution_mode],
        "resumed": resume,
        "sidecars": sorted(sidecars),
    }
    write_json(report.summary, report.summary_json)
    if report.summary.get("campaign_status") == "incomplete_fail_stop":
        fail_stop = report.summary["fail_stop"]
        raise CampaignFailStop(str(fail_stop["reason"]))
    return report, sidecars
