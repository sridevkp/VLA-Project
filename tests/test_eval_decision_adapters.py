"""Inert tests proving instrumentation delegates once and leaves inputs/results unchanged."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vlarobot.core.types import (
    ActionChunk,
    ApplyResult,
    CameraInfo,
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    VerifyResult,
    WorldState,
)
from vlarobot.eval.decision_adapters import (
    AssessmentObservation,
    BackendApplyProbe,
    CandidateGenerationProbe,
    DecisionEvidenceBuilder,
    GenerationObservation,
    PredictionAssessmentProbe,
    PredictionObservation,
    SafetyMonitorProbe,
    VerificationProbe,
    wrap_execute_first,
)
from vlarobot.eval.decision_evidence import CandidateDisposition
from vlarobot.risk.comparison import Selection
from vlarobot.risk.gate import CandidateAssessment
from vlarobot.world_model.rollout import Trajectory


class _TinyGenerator:
    def __init__(self) -> None:
        self.draw_log: list[dict[str, object]] = []

    def propose(self, task: Any, state: WorldState, memory: object) -> list[ActionChunk]:
        self.draw_log.append(
            {"decision": 0, "seed": 900, "draw_seeds": (7, 8), "source": "scripted"}
        )
        return [ActionChunk([], 0.032, "scripted") for _ in range(2)]


def _state() -> WorldState:
    robot = RobotState(
        {"joint": 0.5},
        {},
        Pose2D(0.0, 0.0, 0.0),
        Pose3D(Vec3(0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        0.02,
        True,
    )
    obj = ObjectState("can", "can", Pose3D(Vec3(0, 0, 0.4), (0, 0, 0, 1)), Vec3(0.1, 0.1, 0.1), 1.0)
    return WorldState(SceneState([obj]), robot)


def _trajectory() -> Trajectory:
    return Trajectory((_state(), _state()), 0.032, (), (), ())


def _observation(t: float) -> Any:
    camera = CameraInfo(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        2,
        2,
    )
    return type(
        "SyntheticObservation",
        (),
        {
            "t": t,
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": None,
            "cam_info": camera,
            "robot": _state().robot,
        },
    )()


def _clock(values: list[int]):
    iterator = iter(values)

    def clock() -> int:
        return next(iterator)

    return clock


def test_generation_probe_delegates_once_and_snapshots_existing_draw_log() -> None:
    generator = _TinyGenerator()
    probe = CandidateGenerationProbe(
        generator,
        episode_id="episode-0",
        decision_id="decision-0",
        timing_enabled=True,
        clock=_clock([1_000_000_000, 2_000_000_000]),
    )
    chunks = probe.propose(None, _state(), None)
    assert len(chunks) == 2
    assert probe.observations[0].draw_seeds == (7, 8)
    assert probe.observations[0].duration_s == pytest.approx(1.0)
    assert len(generator.draw_log) == 1


def test_prediction_and_assessment_wrappers_return_original_objects() -> None:
    state = _state()
    chunk = ActionChunk([], 0.032, "scripted")
    trajectory = _trajectory()
    assessment = CandidateAssessment((("ok", True),), (0.1, 0.2, 0.3, 0.4))
    probe = PredictionAssessmentProbe(
        predict=lambda *_args: trajectory,
        assess=lambda *_args: assessment,
        episode_id="episode-0",
        decision_id="decision-0",
    )
    assert probe.predict(state, chunk) is trajectory
    assert probe.assess(state, chunk, trajectory) is assessment
    assert probe.predictions[0].candidate_id == "candidate-00"
    assert probe.assessments[0].components == (0.1, 0.2, 0.3, 0.4)


def test_builder_classifies_evaluated_and_invalid_candidates() -> None:
    generation = GenerationObservation(
        "episode-0", "decision-0", "scripted", 0, 900, (7, 8), 2, None
    )
    predictions = [
        PredictionObservation(
            "episode-0",
            "decision-0",
            index,
            f"candidate-{index:02d}",
            "scripted",
            1,
            True,
            1,
            2,
            (),
        )
        for index in range(2)
    ]
    assessments = [
        AssessmentObservation(
            "episode-0",
            "decision-0",
            index,
            f"candidate-{index:02d}",
            (("ok", True),),
            (0.1, 0.2, 0.3, 0.4),
        )
        for index in range(2)
    ]
    selection = Selection(ActionChunk([], 0.032, "scripted"), 1, "synthetic")
    builder = DecisionEvidenceBuilder(
        episode_id="episode-0",
        decision_id="decision-0",
        arm="full_system",
        proposer_source="scripted",
        decision_index=0,
        candidate_count=2,
        weights=(1.0, 1.0, 1.0, 1.0),
        maximize=False,
        required_constraints=("ok",),
    )
    evidence = (
        builder.add_generation(generation)
        .add_prediction(predictions[0])
        .add_prediction(predictions[1])
        .add_assessment(assessments[0])
        .add_assessment(assessments[1])
        .set_selection(selection)
        .build()
    )
    assert evidence.candidates[0].disposition is CandidateDisposition.SAFE_UNSELECTED
    assert evidence.candidates[1].disposition is CandidateDisposition.SELECTED
    assert evidence.selected_candidate_id == "candidate-01"
    assert evidence.timings is not None
    assert evidence.timings.candidate_generation_s is None


def test_execution_wrapper_records_delegated_result_without_changing_it() -> None:
    before, after = _observation(1.0), _observation(1.032)
    calls: list[tuple[object, object, object, object]] = []

    def execute(selection: Selection, observation: Any, backend: Any, monitor: Any):
        calls.append((selection, observation, backend, monitor))
        return after, ApplyResult(True, None)

    wrapper, sink = wrap_execute_first(
        execute, decision_id="decision-0", selected_candidate_id="candidate-00"
    )
    selection = Selection(ActionChunk([], 0.032, "scripted"), 0, "synthetic")
    returned_observation, result = wrapper(selection, before, object(), object())
    assert returned_observation is after
    assert result.accepted
    assert len(calls) == 1
    assert sink[0].backend_stepped
    assert sink[0].action_t_s == pytest.approx(1.032)


def test_probe_wrappers_delegate_verdicts_and_verification() -> None:
    class Monitor:
        def validate(self, action: Any, state: Any) -> ApplyResult:
            return ApplyResult(False, "synthetic veto")

    class Backend:
        def apply(self, action: Any) -> ApplyResult:
            return ApplyResult(True, None)

        def step(self, count: int) -> None:
            return None

    class Verifier:
        def check(self, task: Any, predicted: WorldState, observed: WorldState) -> VerifyResult:
            return VerifyResult(True, "synthetic pass")

        def next_action(self, result: VerifyResult) -> str:
            return "complete"

    monitor_probe = SafetyMonitorProbe(Monitor())
    backend_probe = BackendApplyProbe(Backend())
    verifier_probe = VerificationProbe(Verifier())
    action = JointTarget({"joint": 0.5})
    verdict = monitor_probe.validate(action, _state().robot)
    applied = backend_probe.apply(action)
    checked = verifier_probe.check(None, _state(), _state())
    transition = verifier_probe.next_action(checked)
    assert verdict == ApplyResult(False, "synthetic veto")
    assert applied.accepted
    assert checked.success
    assert transition == "complete"
    assert backend_probe.step(1) is None
    assert verifier_probe.results[0] is checked
