"""Versioned diagnostic-only evidence for observed-state feedback execution.

This sidecar is intentionally separate from the frozen decision-evidence schema. It records the
selected action, the projected action actually applied, the observation used for projection and
the reviewed budgets consumed. Schema 2 adds an explicitly labelled diagnostic HOME completion
phase, a non-converged-action classification summary and the complete observed world-to-base pose.
Feedback trajectories are not paired with the original fixed-horizon prediction: their tick counts
and executed targets can differ, so no ordinary prediction-error claim is supported.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from vlarobot.eval.decision_evidence import evidence_to_dict
from vlarobot.eval.metrics import tilt_angle_deg

OBSERVED_STATE_FEEDBACK_EXECUTION_MODE: Literal["diagnostic_observed_state_feedback_chunk"] = (
    "diagnostic_observed_state_feedback_chunk"
)
FEEDBACK_EVIDENCE_SCHEMA_VERSION: Literal[3] = 3
SUPPORTED_FEEDBACK_EVIDENCE_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2, 3)
FEEDBACK_SIDECAR_NAME = "observed-state-feedback.jsonl"
DIAGNOSTIC_EVENT_SIDECAR_NAME = "diagnostic-events.jsonl"
COLLISION_PROXY_ATTRIBUTION_SIDECAR_NAME = "collision-proxy-attribution.jsonl"

FEEDBACK_TERMINATION_REASONS: tuple[str, ...] = (
    "selected_chunk_converged",
    "gripper_command_complete",
    "gripper_move_timeout",
    "global_step_ceiling",
    "joint_move_timeout",
    "joint_stall",
    "safety_rejected",
    "backend_apply_rejected",
    "execution_exception",
)
HOME_COMPLETION_TERMINAL_REASONS: tuple[str, ...] = (
    "home_converged",
    "global_step_ceiling",
    "joint_move_timeout",
    "joint_stall",
    "safety_rejected",
    "backend_apply_rejected",
    "execution_exception",
)
FEEDBACK_ATTEMPT_PHASES: tuple[str, ...] = (
    "selected_chunk",
    "diagnostic_home_completion",
)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class FeedbackBudgetEvidence:
    """Reviewed limits actually used by the diagnostic executor."""

    max_joint_step_rad: float
    non_home_tolerance_rad: float
    home_tolerance_rad: float
    joint_stall_steps: int
    joint_progress_epsilon_rad: float
    joint_move_timeout_steps: int
    linear_move_timeout_steps: int
    gripper_open_timeout_s: float
    finger_speed_m_s: float
    gripper_arrival_tolerance_m: float
    gripper_timeout_steps: int
    episode_max_steps: int
    robot_config_path: str
    robot_config_sha256: str
    applicability: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "max_joint_step_rad",
            "non_home_tolerance_rad",
            "home_tolerance_rad",
            "joint_progress_epsilon_rad",
            "linear_move_timeout_steps",
            "gripper_open_timeout_s",
            "finger_speed_m_s",
            "gripper_arrival_tolerance_m",
        ):
            _number(float(getattr(self, name)), name)
        for name in (
            "joint_stall_steps",
            "joint_move_timeout_steps",
            "gripper_timeout_steps",
            "episode_max_steps",
        ):
            _integer(getattr(self, name), name, minimum=1)
        _text(self.robot_config_path, "robot_config_path")
        _text(self.robot_config_sha256, "robot_config_sha256")
        expected = {
            "joint_step_bound",
            "joint_convergence",
            "joint_stall",
            "joint_move_timeout",
            "linear_move_timeout",
            "gripper_open_timeout",
            "episode_step_ceiling",
        }
        if set(self.applicability) != expected:
            raise ValueError("feedback budget applicability must name every reviewed budget")


@dataclass(frozen=True)
class FeedbackPredictionProvenance:
    """Explicit statement that feedback execution cannot be scored against the frozen rollout."""

    original_prediction_candidate_id: str
    original_prediction_horizon_steps: int
    comparability: Literal["non_comparable"] = "non_comparable"
    reason: str = (
        "observed-state feedback may add simulator ticks and execute projected targets that "
        "differ from the immutable selected candidate; no one-to-one pairing is emitted"
    )
    pairing_records_emitted: int = 0

    def __post_init__(self) -> None:
        _text(self.original_prediction_candidate_id, "original_prediction_candidate_id")
        _integer(
            self.original_prediction_horizon_steps, "original_prediction_horizon_steps", minimum=1
        )
        if self.comparability != "non_comparable":
            raise ValueError("feedback execution is non-comparable with fixed-horizon prediction")
        if self.pairing_records_emitted != 0:
            raise ValueError("feedback execution must not emit prediction pairings")


@dataclass(frozen=True)
class JointProjectionEvidence:
    """The observed-to-executed joint target projection for one attempt."""

    observed_joint_pos_rad: Mapping[str, float]
    selected_joint_target_rad: Mapping[str, float]
    executed_joint_target_rad: Mapping[str, float]
    clamped_joints: tuple[str, ...]
    bound_rad: float

    def __post_init__(self) -> None:
        _number(self.bound_rad, "bound_rad")
        if set(self.observed_joint_pos_rad) < set(self.selected_joint_target_rad):
            raise ValueError("observed state must contain every projected joint")
        if self.executed_joint_target_rad.keys() != self.selected_joint_target_rad.keys():
            raise ValueError("projection must preserve the selected target's joint set")
        if not set(self.clamped_joints) <= set(self.selected_joint_target_rad):
            raise ValueError("clamped joints must be selected target joints")


@dataclass(frozen=True)
class FeedbackAttemptEvidence:
    """One safety-validated command, whether or not it was accepted and applied."""

    attempt_index: int
    selected_action_index: int
    selected_action_type: str
    selected_action: Mapping[str, Any]
    attempt_within_action: int
    executed_action_index: int | None
    executed_action_type: str | None
    executed_action: Mapping[str, Any] | None
    projection: JointProjectionEvidence | None
    observed_before: Mapping[str, Any]
    observed_after: Mapping[str, Any] | None
    max_joint_error_before_rad: float | None
    max_joint_error_after_rad: float | None
    progress_rad: float | None
    no_progress_steps: int | None
    safety_accepted: bool | None
    safety_reason: str | None
    apply_accepted: bool | None
    apply_reason: str | None
    backend_stepped: bool
    backend_step_number: int | None
    episode_step_number: int | None
    phase: str = "selected_chunk"
    completion_error: float | None = None
    completion_unit: str = "rad"
    completion_target: float | None = None

    apply_outcome: str = "not_attempted"
    step_outcome: str = "not_attempted"
    observation_outcome: str = "not_available"
    apply_recording: str = "not_attempted"
    step_recording: str = "not_attempted"

    def __post_init__(self) -> None:
        _integer(self.attempt_index, "attempt_index")
        _integer(self.selected_action_index, "selected_action_index")
        _integer(self.attempt_within_action, "attempt_within_action", minimum=1)
        _text(self.selected_action_type, "selected_action_type")
        if self.executed_action_index is not None:
            _integer(self.executed_action_index, "executed_action_index", minimum=1)
        if self.backend_step_number is not None:
            _integer(self.backend_step_number, "backend_step_number", minimum=1)
        if self.episode_step_number is not None:
            _integer(self.episode_step_number, "episode_step_number", minimum=1)
        if self.backend_stepped and self.apply_accepted is not True:
            raise ValueError("a stepped attempt requires an accepted apply")
        if self.progress_rad is not None:
            _number(self.progress_rad, "progress_rad")
        if self.phase not in FEEDBACK_ATTEMPT_PHASES:
            raise ValueError(f"unknown feedback attempt phase: {self.phase}")
        if self.completion_error is not None:
            _number(self.completion_error, "completion_error")
        if self.completion_target is not None:
            _number(self.completion_target, "completion_target")
        if self.completion_unit not in ("rad", "m"):
            raise ValueError("completion_unit must be rad or m")


@dataclass(frozen=True)
class FeedbackActionEvidence:
    """One immutable selected action and the feedback attempts it caused."""

    selected_action_index: int
    selected_action_type: str
    selected_action: Mapping[str, Any]
    home_action: bool
    convergence_tolerance_rad: float | None
    terminal_reason: str
    attempt_indices: tuple[int, ...]
    executed_action_count: int
    backend_step_count: int
    final_max_joint_error_rad: float | None
    completion_target: float | None = None
    completion_error: float | None = None
    completion_unit: str = "rad"

    def __post_init__(self) -> None:
        _integer(self.selected_action_index, "selected_action_index")
        _text(self.selected_action_type, "selected_action_type")
        _text(self.terminal_reason, "terminal_reason")
        _integer(self.executed_action_count, "executed_action_count")
        _integer(self.backend_step_count, "backend_step_count")
        if self.convergence_tolerance_rad is not None:
            _number(self.convergence_tolerance_rad, "convergence_tolerance_rad")
        if self.final_max_joint_error_rad is not None:
            _number(self.final_max_joint_error_rad, "final_max_joint_error_rad")
        if self.completion_target is not None:
            _number(self.completion_target, "completion_target")
        if self.completion_error is not None:
            _number(self.completion_error, "completion_error")
        if self.completion_unit not in ("rad", "m"):
            raise ValueError("completion_unit must be rad or m")


@dataclass(frozen=True)
class FeedbackConvergenceSummary:
    """Counts derived only from converged selected actions; non-converged errors stay visible."""

    converged_action_indices: tuple[int, ...]
    converged_action_count: int
    max_final_error_converged_rad: float | None
    non_converged_action_indices: tuple[int, ...]
    non_converged_reasons: Mapping[str, int]
    non_converged_final_errors: tuple[tuple[int, str, float], ...]

    def __post_init__(self) -> None:
        _integer(self.converged_action_count, "converged_action_count")
        if len(self.converged_action_indices) != self.converged_action_count:
            raise ValueError("converged action indices must match the converged count")
        if self.max_final_error_converged_rad is not None:
            _number(self.max_final_error_converged_rad, "max_final_error_converged_rad")
        for index, reason, error in self.non_converged_final_errors:
            _integer(index, "non-converged action index")
            _text(reason, "non-converged terminal reason")
            _number(error, "non-converged final error")


@dataclass(frozen=True)
class DiagnosticHomeCompletionEvidence:
    """Separately labelled HOME completion phase; it never rewrites selected actions."""

    phase: Literal["diagnostic_home_completion"]
    source: str
    target_joint_pos_rad: Mapping[str, float]
    tolerance_rad: float
    initial_max_joint_error_rad: float
    final_max_joint_error_rad: float
    terminal_reason: str
    attempt_indices: tuple[int, ...]
    executed_action_count: int
    backend_step_count: int

    def __post_init__(self) -> None:
        if self.phase != "diagnostic_home_completion":
            raise ValueError("HOME completion phase must be explicitly labelled")
        _text(self.source, "source")
        if not self.target_joint_pos_rad:
            raise ValueError("configured HOME vector is required")
        for name, value in self.target_joint_pos_rad.items():
            _text(str(name), "target joint name")
            _number(float(value), "target joint position")
        _number(self.tolerance_rad, "tolerance_rad")
        _number(self.initial_max_joint_error_rad, "initial_max_joint_error_rad")
        _number(self.final_max_joint_error_rad, "final_max_joint_error_rad")
        if self.terminal_reason not in HOME_COMPLETION_TERMINAL_REASONS:
            raise ValueError(f"unknown HOME completion reason: {self.terminal_reason}")
        _integer(self.executed_action_count, "executed_action_count")
        _integer(self.backend_step_count, "backend_step_count")


@dataclass(frozen=True)
class CollisionProxyObjectEvidence:
    """Per-object displacement/tilt attribution; this is not contact detection."""

    name: str
    max_displacement_m: float
    final_displacement_m: float
    max_tilt_deg: float
    final_tilt_deg: float
    displacement_hit: bool
    tilt_hit: bool

    def __post_init__(self) -> None:
        _text(self.name, "name")
        for name in (
            "max_displacement_m",
            "final_displacement_m",
            "max_tilt_deg",
            "final_tilt_deg",
        ):
            _number(float(getattr(self, name)), name)

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


@dataclass(frozen=True)
class CollisionProxyAttributionEvidence:
    """Versioned diagnostic-only per-object proxy attribution for one feedback episode."""

    schema_version: Literal[1]
    episode_id: str
    execution_mode: Literal["diagnostic_observed_state_feedback_chunk"]
    threshold_source: str
    max_displacement_threshold_m: float
    max_tilt_threshold_deg: float
    objects: tuple[CollisionProxyObjectEvidence, ...]
    proxy_hit: bool
    backend_contact_steps_observed: int
    contact_channel: str = (
        "backend-recorded contacts.jsonl only; proxy displacement/tilt is never promoted to "
        "a verified contact"
    )
    note: str = (
        "Diagnostic attribution of the already-reported collision proxy. It changes no frozen "
        "report metric and makes no grasp, release or swept-arm claim."
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported collision-proxy attribution schema version")
        _text(self.episode_id, "episode_id")
        if self.execution_mode != OBSERVED_STATE_FEEDBACK_EXECUTION_MODE:
            raise ValueError("proxy attribution belongs only to feedback diagnostic mode")
        _text(self.threshold_source, "threshold_source")
        _number(self.max_displacement_threshold_m, "max_displacement_threshold_m")
        _number(self.max_tilt_threshold_deg, "max_tilt_threshold_deg")
        _integer(self.backend_contact_steps_observed, "backend_contact_steps_observed")
        if any(
            name <= 0 for name in (self.max_displacement_threshold_m, self.max_tilt_threshold_deg)
        ):
            raise ValueError("proxy thresholds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


def build_collision_proxy_attribution(
    *,
    episode_id: str,
    ground_truths: Sequence[Any],
    observations: Sequence[Any],
    threshold_source: str,
    max_displacement_m: float,
    max_tilt_deg: float,
) -> CollisionProxyAttributionEvidence:
    """Attribute the existing proxy to each object from observed oracle poses only."""
    if not ground_truths or len(ground_truths) != len(observations):
        raise ValueError("one ground-truth snapshot is required per observed state")
    names = sorted(ground_truths[0].object_poses)
    objects: list[CollisionProxyObjectEvidence] = []
    for name in names:
        positions: list[tuple[float, float, float]] = []
        quaternions: list[tuple[float, float, float, float]] = []
        for truth in ground_truths:
            pose = truth.object_poses.get(name)
            if pose is None:
                continue
            positions.append(
                (
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                )
            )
            quaternions.append(tuple(float(value) for value in pose.orientation))
        if not positions:
            continue
        reset_position = positions[0]
        displacements = [math.dist(position, reset_position) for position in positions]
        reset_quaternion = quaternions[0]
        tilts = [tilt_angle_deg(reset_quaternion, quaternion) for quaternion in quaternions]
        objects.append(
            CollisionProxyObjectEvidence(
                name=str(name),
                max_displacement_m=max(displacements),
                final_displacement_m=displacements[-1],
                max_tilt_deg=max(tilts),
                final_tilt_deg=tilts[-1],
                displacement_hit=max(displacements) > float(max_displacement_m),
                tilt_hit=max(tilts) > float(max_tilt_deg),
            )
        )
    contact_steps = 0
    for observation in observations:
        diagnostics = observation.robot.diagnostics
        if isinstance(diagnostics, dict) and diagnostics.get("contact_nodes"):
            contact_steps += 1
    return CollisionProxyAttributionEvidence(
        schema_version=1,
        episode_id=episode_id,
        execution_mode=OBSERVED_STATE_FEEDBACK_EXECUTION_MODE,
        threshold_source=threshold_source,
        max_displacement_threshold_m=float(max_displacement_m),
        max_tilt_threshold_deg=float(max_tilt_deg),
        objects=tuple(objects),
        proxy_hit=any(item.displacement_hit or item.tilt_hit for item in objects),
        backend_contact_steps_observed=contact_steps,
    )


@dataclass(frozen=True)
class ObservedStateFeedbackEvidence:
    """Complete versioned record of one feedback-executed selected chunk."""

    schema_version: Literal[3]
    execution_mode: Literal["diagnostic_observed_state_feedback_chunk"]
    episode_id: str
    decision_id: str
    selected_candidate_id: str
    selected_action_count: int
    attempt_count: int
    applied_action_count: int
    backend_step_count: int
    observation_count: int
    safety_validate_count: int
    safety_accepted_count: int
    safety_rejected_count: int
    apply_accepted_count: int
    apply_rejected_count: int
    episode_steps_before: int
    episode_steps_after: int
    episode_max_steps: int
    termination_reason: str
    budgets: FeedbackBudgetEvidence
    prediction_provenance: FeedbackPredictionProvenance
    actions: tuple[FeedbackActionEvidence, ...]
    attempts: tuple[FeedbackAttemptEvidence, ...]
    convergence_summary: FeedbackConvergenceSummary
    error: str | None = None
    home_completion: DiagnosticHomeCompletionEvidence | None = None

    apply_unknown_count: int = 0
    step_unknown_count: int = 0
    observation_failure_count: int = 0
    recording_failure_count: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != FEEDBACK_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported observed-state feedback evidence version")
        if self.execution_mode != OBSERVED_STATE_FEEDBACK_EXECUTION_MODE:
            raise ValueError("feedback evidence has the wrong execution mode")
        _text(self.episode_id, "episode_id")
        _text(self.decision_id, "decision_id")
        _text(self.selected_candidate_id, "selected_candidate_id")
        if self.termination_reason not in FEEDBACK_TERMINATION_REASONS:
            raise ValueError(f"unknown feedback termination reason: {self.termination_reason}")
        for name in (
            "selected_action_count",
            "attempt_count",
            "applied_action_count",
            "backend_step_count",
            "observation_count",
            "safety_validate_count",
            "safety_accepted_count",
            "safety_rejected_count",
            "apply_accepted_count",
            "apply_rejected_count",
            "apply_unknown_count",
            "step_unknown_count",
            "observation_failure_count",
            "recording_failure_count",
            "episode_steps_before",
            "episode_steps_after",
            "episode_max_steps",
        ):
            _integer(getattr(self, name), name)
        if len(self.attempts) != self.attempt_count:
            raise ValueError("one attempt record is required per attempt")
        if len(self.actions) > self.selected_action_count:
            raise ValueError("more action records than selected actions")
        if [item.attempt_index for item in self.attempts] != list(range(self.attempt_count)):
            raise ValueError("attempt records must be indexed sequentially from zero")
        if self.applied_action_count != self.apply_accepted_count:
            raise ValueError("every accepted apply must be an applied action")
        if self.backend_step_count > self.applied_action_count:
            raise ValueError("confirmed steps cannot outnumber accepted applies")
        if self.safety_validate_count != self.safety_accepted_count + self.safety_rejected_count:
            raise ValueError("safety outcomes must account for every validation")
        if (
            self.apply_accepted_count + self.apply_rejected_count + self.apply_unknown_count
            > self.safety_accepted_count
        ):
            raise ValueError("apply outcomes cannot outnumber safety-accepted attempts")
        if self.episode_steps_after != self.episode_steps_before + self.backend_step_count:
            raise ValueError("episode step consumption must match the recorded control steps")
        if self.episode_steps_after > self.episode_max_steps:
            raise ValueError("feedback execution exceeded the episode step ceiling")
        if not 1 <= self.observation_count <= self.backend_step_count + 1:
            raise ValueError("observations cannot outnumber the initial state and confirmed steps")
        if self.error is None and (
            self.backend_step_count != self.applied_action_count
            or self.observation_count != self.backend_step_count + 1
            or self.apply_accepted_count + self.apply_rejected_count != self.safety_accepted_count
        ):
            raise ValueError("successful feedback operations must reconcile")

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)


def validate_feedback_evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted feedback row, accepting historical and current rows."""
    version = row.get("schema_version")
    if version not in SUPPORTED_FEEDBACK_EVIDENCE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported observed-state feedback evidence version: {version}")
    if row.get("execution_mode") != OBSERVED_STATE_FEEDBACK_EXECUTION_MODE:
        raise ValueError("feedback evidence has the wrong execution mode")
    reason = row.get("termination_reason")
    if reason not in FEEDBACK_TERMINATION_REASONS:
        raise ValueError(f"unknown feedback termination reason: {reason}")
    return dict(row)


def read_feedback_evidence(path: Path) -> list[dict[str, Any]]:
    """Read historical and current feedback rows without rewriting them."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(validate_feedback_evidence_row(json.loads(line)))
    return rows
