"""Candidate enumeration and CPU utility fitting for the V4 TRAIN prototype."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from vlarobot.core.interfaces import RobotBackend
from vlarobot.core.types import (
    ActionChunk,
    Decision,
    EEPoseTarget,
    GripperCmd,
    Observation,
    Pose3D,
    Vec3,
    WorldState,
)
from vlarobot.execution.skills import Skill, SkillAction, SkillExecutor, SkillOutcome
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.rollout import AnalyticSurrogate, Attachment, Trajectory


def enumerate_skills(
    state: WorldState,
    targets: dict[Skill, Vec3],
    *,
    position_tolerance_m: float,
    open_tolerance_m: float,
    open_position_m: float,
) -> tuple[SkillAction, ...]:
    """Enumerate geometrically distinct motions without consulting an expert phase."""
    tool = state.robot.ee_pose.position.as_array()
    choices = [
        SkillAction(skill, target)
        for skill, target in targets.items()
        if np.linalg.norm(target.as_array() - tool) > position_tolerance_m
    ]
    if state.robot.gripper_opening < open_position_m - open_tolerance_m:
        choices.append(SkillAction(Skill.OPEN))
    if state.robot.gripper_opening > open_tolerance_m:
        choices.append(SkillAction(Skill.CLOSE))
    return tuple(choices)


class ContextMode(StrEnum):
    NONE = "none"
    PREDICTED = "predicted"
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class ScoringInput:
    action: SkillAction
    trajectory: Trajectory
    hard_checks: tuple[tuple[str, bool | None], ...]


@dataclass(frozen=True)
class UtilityExample:
    scene_group: str
    state: WorldState
    candidate: ScoringInput
    utility: float
    source: str


def _features(state: WorldState, action: SkillAction, future: WorldState | None) -> np.ndarray:
    tool = state.robot.ee_pose.position.as_array()
    can = next(
        obj.pose.position.as_array() for obj in state.scene.objects if obj.object_id == "CAN"
    )
    target = tool if action.target_base is None else action.target_base.as_array()
    identity = np.array([skill == action.skill for skill in Skill], dtype=float)
    current = np.r_[
        tool,
        can - tool,
        target - tool,
        state.robot.gripper_opening,
        float(state.robot.gripper_holding is True),
    ]
    if future is None:
        return np.r_[identity, current, np.outer(identity, current).ravel()]
    future_tool = future.robot.ee_pose.position.as_array()
    future_can = next(
        obj.pose.position.as_array() for obj in future.scene.objects if obj.object_id == "CAN"
    )
    lookahead = np.r_[
        future_tool - tool,
        future_can - can,
        future_can - future_tool,
        future.robot.gripper_opening - state.robot.gripper_opening,
        float(future.robot.gripper_holding is True),
    ]
    return np.r_[
        identity,
        current,
        np.outer(identity, current).ravel(),
        lookahead,
        np.outer(identity, lookahead).ravel(),
    ]


@dataclass(frozen=True)
class UtilityModel:
    prediction_context: bool
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def score(self, state: WorldState, action: SkillAction, future: WorldState | None) -> float:
        if self.prediction_context != (future is not None):
            raise ValueError("feature mode does not match the fitted model")
        features = _features(state, action, future)
        return float((features - self.mean) / self.scale @ self.coefficients + self.intercept)


def fit_utility(
    examples: list[UtilityExample],
    *,
    train_groups: set[str],
    prediction_context: bool,
    ridge: float,
) -> UtilityModel:
    """Fit only on the specified TRAIN scene groups and empirical or demonstrated labels."""
    training = [row for row in examples if row.scene_group in train_groups]
    if not training or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("training examples and positive finite regularization required")
    if any(row.source not in {"demonstration", "simulator-observed"} for row in training):
        raise ValueError("utility targets require demonstration or simulator evidence")
    x = np.stack(
        [
            _features(
                row.state,
                row.candidate.action,
                row.candidate.trajectory.states[-1] if prediction_context else None,
            )
            for row in training
        ]
    )
    y = np.array([row.utility for row in training])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite training features and utility required")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ (y - intercept))
    return UtilityModel(prediction_context, mean, scale, coefficients, intercept)


def select_skill(
    state: WorldState,
    candidates: tuple[ScoringInput, ...],
    model: UtilityModel,
    gate: RiskGate,
    *,
    context_mode: ContextMode,
    seed: int,
    intervention: tuple[WorldState, ...] | None = None,
) -> tuple[Decision, tuple[float, ...]]:
    """Apply interventions to model inputs while preserving the original safety assessments."""
    futures = tuple(candidate.trajectory.states[-1] for candidate in candidates)
    if intervention is not None:
        if context_mode is not ContextMode.PREDICTED or len(intervention) != len(candidates):
            raise ValueError("intervention requires one predicted state per candidate")
        futures = intervention
    if context_mode is ContextMode.SHUFFLED:
        order = np.random.default_rng(seed).permutation(len(futures))
        futures = tuple(futures[index] for index in order)
    scores = tuple(
        model.score(state, candidate.action, None if context_mode is ContextMode.NONE else future)
        for candidate, future in zip(candidates, futures, strict=True)
    )
    decision = gate.select(
        [
            CandidateAssessment(candidate.hard_checks, (score, None, None, None))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
    )
    return decision, scores


def predict_candidates(
    state: WorldState,
    actions: tuple[SkillAction, ...],
    surrogate: AnalyticSurrogate,
    *,
    horizon_s: float,
    open_position_m: float,
    attachment: Attachment | None = None,
) -> tuple[ScoringInput, ...]:
    """Predict a nominal candidate endpoint and retain all surrogate hard vetoes."""
    candidates = []
    for action in actions:
        if action.target_base is None:
            command = GripperCmd(open_position_m if action.skill is Skill.OPEN else 0.0)
        else:
            command = EEPoseTarget(
                Pose3D(action.target_base, state.robot.ee_pose.orientation),
                gripper=None,
                duration=horizon_s,
            )
        trajectory = surrogate.predict_trajectory(
            state, ActionChunk([command], horizon_s, "vla"), attachment=attachment
        )
        checks = (("surrogate", not trajectory.hard_reasons),) + tuple(
            (reason, False) for reason in dict.fromkeys(trajectory.hard_reasons)
        )
        candidates.append(ScoringInput(action, trajectory, checks))
    return tuple(candidates)


@dataclass(frozen=True)
class SkillDecision:
    decision: Decision
    candidates: tuple[ScoringInput, ...]
    scores: tuple[float, ...]
    feasible_skill_count: int
    outcome: SkillOutcome | None


def execute_decision(
    state: WorldState,
    observation: Observation,
    candidates: tuple[ScoringInput, ...],
    model: UtilityModel,
    executor: SkillExecutor,
    backend: RobotBackend,
    *,
    context_mode: ContextMode,
    seed: int,
    intervention: tuple[WorldState, ...] | None = None,
) -> SkillDecision:
    """Evaluate one fresh decision boundary and dispatch only the winning feasible skill."""
    gate = RiskGate(
        weights=(1.0, 0.0, 0.0, 0.0),
        maximize=True,
        required_constraints=("surrogate",),
        inactive_components=(1, 2, 3),
    )
    decision, scores = select_skill(
        state,
        candidates,
        model,
        gate,
        context_mode=context_mode,
        seed=seed,
        intervention=intervention,
    )
    feasible = {
        candidate.action.skill
        for candidate, score in zip(candidates, scores, strict=True)
        if gate.select(
            [CandidateAssessment(candidate.hard_checks, (score, None, None, None))]
        ).selected_index
        is not None
    }
    outcome = None
    if decision.selected_index is not None:
        outcome = executor.run_skill(
            backend, observation, candidates[decision.selected_index].action
        )
    return SkillDecision(decision, candidates, scores, len(feasible), outcome)
