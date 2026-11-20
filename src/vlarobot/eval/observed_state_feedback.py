"""Versioned diagnostic-only evidence for observed-state feedback execution.

This sidecar is intentionally separate from the frozen decision-evidence schema. It records the
selected action, the projected action actually applied, the observation used for projection and
the reviewed budgets consumed. Feedback trajectories are not paired with the original fixed-horizon
prediction: their tick counts and executed targets can differ, so no ordinary prediction-error
claim is supported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from vlarobot.eval.decision_evidence import evidence_to_dict

OBSERVED_STATE_FEEDBACK_EXECUTION_MODE: Literal["diagnostic_observed_state_feedback_chunk"] = (
    "diagnostic_observed_state_feedback_chunk"
)
FEEDBACK_EVIDENCE_SCHEMA_VERSION: Literal[1] = 1
FEEDBACK_SIDECAR_NAME = "observed-state-feedback.jsonl"

FEEDBACK_TERMINATION_REASONS: tuple[str, ...] = (
    "selected_chunk_converged",
    "gripper_command_complete",
    "global_step_ceiling",
    "joint_move_timeout",
    "joint_stall",
    "safety_rejected",
    "backend_apply_rejected",
    "execution_exception",
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
        ):
            _number(float(getattr(self, name)), name)
        for name in (
            "joint_stall_steps",
            "joint_move_timeout_steps",
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


@dataclass(frozen=True)
class ObservedStateFeedbackEvidence:
    """Complete versioned record of one feedback-executed selected chunk."""

    schema_version: Literal[1]
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
    error: str | None = None

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
        if self.backend_step_count != self.applied_action_count:
            raise ValueError("one control step is required per applied action")
        if self.safety_validate_count != self.safety_accepted_count + self.safety_rejected_count:
            raise ValueError("safety outcomes must account for every validation")
        if self.apply_accepted_count + self.apply_rejected_count != self.safety_accepted_count:
            raise ValueError("apply outcomes must account for every safety-accepted attempt")
        if self.episode_steps_after != self.episode_steps_before + self.backend_step_count:
            raise ValueError("episode step consumption must match the recorded control steps")
        if self.episode_steps_after > self.episode_max_steps:
            raise ValueError("feedback execution exceeded the episode step ceiling")
        if self.observation_count != self.backend_step_count + 1:
            raise ValueError("one observation is required before and after each control step")

    def to_dict(self) -> dict[str, Any]:
        return evidence_to_dict(self)
