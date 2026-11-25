"""Unit evidence for the minimal Eq. 3.2 prediction-conditioned decision head."""

from __future__ import annotations

import numpy as np
import pytest

from vlarobot.core.types import Decision
from vlarobot.planning.prediction_conditioned import (
    ConditionedCandidate,
    ConditionedInput,
    PredictedContext,
    PredictionConditionedDecision,
    TrainingSample,
)

SCALES = (0.1, 0.05, 0.02, 0.045)
OBSERVATION = (0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.05, 0.045)

GOOD = PredictedContext(
    tool_to_target_m=0.01,
    clearance_margin_m=0.03,
    object_progress_m=0.02,
    gripper_endpoint_m=0.0,
)
BAD = PredictedContext(
    tool_to_target_m=0.6,
    clearance_margin_m=-0.02,
    object_progress_m=0.0,
    gripper_endpoint_m=0.045,
)
ACTIONS = (
    (0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045),
    (0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


def _decision_input(first: PredictedContext, second: PredictedContext) -> ConditionedInput:
    return ConditionedInput(
        observation=OBSERVATION,
        task_id=0,
        candidates=(
            ConditionedCandidate(ACTIONS[0], first),
            ConditionedCandidate(ACTIONS[1], second),
        ),
    )


def _fitted() -> PredictionConditionedDecision:
    samples = [TrainingSample(_decision_input(BAD, GOOD), chosen_index=1) for _ in range(4)]
    head = PredictionConditionedDecision(task_count=1, context_scales=SCALES, seed=0)
    return head.fit(samples)


def test_fit_is_deterministic_and_trains_the_recorded_choice() -> None:
    first = _fitted()
    second = _fitted()
    assert first.weights is not None and second.weights is not None
    np.testing.assert_array_equal(first.weights, second.weights)
    result = first.select(_decision_input(BAD, GOOD))
    assert isinstance(result, Decision)
    assert result.selected_index == 1
    assert "predicted_context=" in result.rationale
    assert "scores=" in result.rationale


def test_changing_only_predicted_context_can_change_the_decision() -> None:
    head = _fitted()
    baseline = head.select(_decision_input(BAD, GOOD))
    # Observation, language identifier and both action payloads are unchanged.
    varied = head.select(_decision_input(GOOD, BAD))
    assert baseline.selected_index == 1
    assert varied.selected_index == 0
    assert baseline.rationale != varied.rationale


def test_missing_prediction_fails_closed() -> None:
    head = _fitted()
    missing = PredictedContext(
        tool_to_target_m=0.0,
        clearance_margin_m=0.0,
        object_progress_m=0.0,
        gripper_endpoint_m=0.0,
        valid=False,
    )
    with pytest.raises(ValueError, match="valid rollout"):
        head.select(_decision_input(missing, GOOD))


def test_invalid_task_or_action_is_rejected() -> None:
    head = PredictionConditionedDecision(task_count=2, context_scales=SCALES, seed=0)
    bad_task = ConditionedInput(
        observation=OBSERVATION,
        task_id=2,
        candidates=(ConditionedCandidate(ACTIONS[0], GOOD),),
    )
    with pytest.raises(ValueError, match="task identifier"):
        head._features(bad_task)
    bad_action = ConditionedInput(
        observation=OBSERVATION,
        task_id=0,
        candidates=(ConditionedCandidate((0.0,) * 6, GOOD),),
    )
    with pytest.raises(ValueError, match="seven-dimensional"):
        head._features(bad_action)
