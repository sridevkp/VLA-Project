"""Minimum learned Eq. 3.2 decision head: prediction is a real policy input.

The submitted report defines ``a_t:t+H-1 = pi_theta(I_t, L, s_hat, c_t)``. V1
implemented prediction outside the proposer, which does not satisfy that
equation. This module provides the smallest learned mechanism in which the
predicted state sequence is an argument of the decision itself. It is a
preparation artifact: it is not wired into any scored campaign until learned
TRAIN admission passes. Activation is a separate, documented decision.

Inputs are learned/observable quantities only. Ground truth, expert phase
labels and scripted manipulation decisions never enter this head. Training
samples are recorded decision outcomes from TRAIN episodes; the rationale
returned by :meth:`PredictionConditionedDecision.select` is built from the
learned scores and the supplied predicted features, not from free text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vlarobot.core.types import Decision

ACTION_DIM = 7
OBSERVATION_DIM = 8
CONTEXT_DIM = 5


@dataclass(frozen=True)
class PredictedContext:
    """Numeric summary of one predicted state sequence, in SI units.

    The world-model adapter fills these fields from a rollout. Neither the
    object ground truth nor the expert phase is consulted here.
    """

    tool_to_target_m: float
    clearance_margin_m: float
    object_progress_m: float
    gripper_endpoint_m: float
    valid: bool = True

    def vector(self, scales: tuple[float, float, float, float]) -> tuple[float, ...]:
        """Return a finite, explicitly normalized context vector.

        A missing or nonfinite prediction fails closed: a decision must consume
        a prediction rather than silently fall back to action-only ranking.
        """
        if not self.valid:
            raise ValueError("prediction-conditioned decision requires a valid rollout")
        raw = (
            self.tool_to_target_m,
            self.clearance_margin_m,
            self.object_progress_m,
            self.gripper_endpoint_m,
        )
        if not all(math.isfinite(value) for value in raw):
            raise ValueError("predicted context contains a nonfinite value")
        if not all(math.isfinite(scale) and scale > 0 for scale in scales):
            raise ValueError("explicit positive context scales are required")
        return tuple(value / scale for value, scale in zip(raw, scales, strict=True)) + (1.0,)


@dataclass(frozen=True)
class ConditionedCandidate:
    """One learned action payload paired with its predicted consequence."""

    action: tuple[float, ...]
    predicted: PredictedContext


@dataclass(frozen=True)
class ConditionedInput:
    """Eq. 3.2 inputs: observation, language identifier, candidates and predictions."""

    observation: tuple[float, ...]
    task_id: int
    candidates: tuple[ConditionedCandidate, ...]


@dataclass(frozen=True)
class TrainingSample:
    """A recorded TRAIN decision with the exact inputs shown to the head."""

    decision_input: ConditionedInput
    chosen_index: int


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class PredictionConditionedDecision:
    """Deterministic softmax decision head trained on recorded TRAIN decisions.

    The score of every candidate is a linear function of the learned
    observation, language, action and *predicted context* features. Because
    the last group enters the same score, changing only the prediction while
    holding image, state, language and action payloads fixed can change the
    selected index. External objective ranking is not used here.
    """

    def __init__(
        self,
        *,
        task_count: int,
        context_scales: tuple[float, float, float, float],
        seed: int,
        learning_rate: float = 0.1,
        epochs: int = 200,
    ) -> None:
        if task_count < 1 or seed < 0 or epochs < 1:
            raise ValueError("explicit task count, nonnegative seed and positive epochs required")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("explicit positive learning rate required")
        self.task_count = task_count
        self.context_scales = context_scales
        self.seed = seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: np.ndarray | None = None

    def _features(self, decision_input: ConditionedInput) -> np.ndarray:
        observation = np.asarray(decision_input.observation, dtype=np.float64)
        if observation.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(observation)):
            raise ValueError("expected eight finite learned observation features")
        if (
            not isinstance(decision_input.task_id, int)
            or not 0 <= decision_input.task_id < self.task_count
        ):
            raise ValueError("language task identifier outside the declared task set")
        if not decision_input.candidates:
            raise ValueError("at least one candidate is required")
        task = np.zeros(self.task_count, dtype=np.float64)
        task[decision_input.task_id] = 1.0
        rows = []
        for candidate in decision_input.candidates:
            action = np.asarray(candidate.action, dtype=np.float64)
            if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                raise ValueError("expected a finite seven-dimensional learned action")
            context = np.asarray(candidate.predicted.vector(self.context_scales), dtype=np.float64)
            if context.shape != (CONTEXT_DIM,):
                raise ValueError("invalid predicted context dimension")
            rows.append(np.concatenate((observation, task, action, context)))
        return np.stack(rows)

    def fit(self, samples: Sequence[TrainingSample]) -> PredictionConditionedDecision:
        """Fit the head from TRAIN-only recorded decisions with a fixed seed."""
        if not samples:
            raise ValueError("training requires recorded TRAIN decision samples")
        features = [self._features(sample.decision_input) for sample in samples]
        widths = {matrix.shape[:2] for matrix in features}
        if len(widths) != 1:
            raise ValueError("all samples must present the same candidate count")
        matrix = np.stack(features)
        chosen = np.asarray([sample.chosen_index for sample in samples], dtype=np.int64)
        count = matrix.shape[1]
        if not np.all((chosen >= 0) & (chosen < count)):
            raise ValueError("recorded chosen index outside its candidate set")
        rng = np.random.default_rng(self.seed)
        dimension = matrix.shape[2]
        weights = rng.normal(0.0, 0.01, size=dimension)
        targets = np.arange(len(samples), dtype=np.int64)
        for _ in range(self.epochs):
            scores = np.einsum("snd,d->sn", matrix, weights)
            errors = _softmax(scores)
            errors[targets, chosen] -= 1.0
            gradient = np.einsum("snd,sn->d", matrix, errors) / len(samples)
            weights -= self.learning_rate * gradient
        self.weights = weights
        return self

    def select(self, decision_input: ConditionedInput) -> Decision:
        """Select a candidate using observation, language, actions and prediction."""
        if self.weights is None:
            raise RuntimeError("decision head must be fitted before selection")
        features = self._features(decision_input)
        scores = features @ self.weights
        probabilities = _softmax(scores[None, :])[0]
        selected = int(np.argmax(scores))
        contexts = [
            candidate.predicted.vector(self.context_scales)
            for candidate in decision_input.candidates
        ]
        rationale = (
            f"selected={selected}; n={len(scores)}; task={decision_input.task_id}; "
            f"scores={[round(float(value), 6) for value in scores]}; "
            f"p={[round(float(value), 6) for value in probabilities]}; "
            f"predicted_context={[[round(float(value), 6) for value in row] for row in contexts]}; "
            f"weights_norm={round(float(np.linalg.norm(self.weights)), 6)}"
        )
        return Decision(selected, rationale)
