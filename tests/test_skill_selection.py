from dataclasses import replace

from vlarobot.backends.fake.backend import FakeBackend
from vlarobot.core.types import ObjectState, Pose3D, ScenarioSpec, SceneState, Vec3, WorldState
from vlarobot.execution.skill_selection import (
    ContextMode,
    ScoringInput,
    UtilityExample,
    enumerate_skills,
    fit_utility,
    select_skill,
)
from vlarobot.execution.skills import Skill
from vlarobot.risk.gate import RiskGate
from vlarobot.world_model.rollout import Trajectory


def test_learned_preference_responds_to_prediction_and_cannot_override_veto() -> None:
    backend = FakeBackend()
    obs = backend.reset(ScenarioSpec(), 0)
    can = ObjectState(
        "CAN", "can", Pose3D(Vec3(0.5, 0.0, 0.7), (0.0, 0.0, 0.0, 1.0)), Vec3(0.1, 0.1, 0.1), 1.0
    )
    state = WorldState(SceneState([can]), obs.robot)
    actions = enumerate_skills(
        state,
        {Skill.APPROACH: Vec3(0.5, 0.0, 0.9), Skill.RETREAT: Vec3(0.5, 0.3, 0.9)},
        position_tolerance_m=0.01,
        open_tolerance_m=0.002,
        open_position_m=0.045,
    )
    assert [action.skill for action in actions[:2]] == [Skill.APPROACH, Skill.RETREAT]
    higher = WorldState(
        SceneState([replace(can, pose=replace(can.pose, position=Vec3(0.5, 0.0, 0.8)))]),
        obs.robot,
    )
    outcomes = (state, higher)
    candidates = tuple(
        ScoringInput(action, Trajectory((state, future), 1.0, (), (), ()), (("safe", True),))
        for action, future in zip(actions[:2], outcomes, strict=True)
    )
    examples = [
        UtilityExample(
            "train-a",
            state,
            replace(candidate, trajectory=replace(candidate.trajectory, states=(state, future))),
            float(index),
            "simulator-observed",
        )
        for candidate in candidates
        for index, future in enumerate(outcomes)
    ]
    model = fit_utility(examples, train_groups={"train-a"}, prediction_context=True, ridge=0.1)
    gate = RiskGate(
        weights=(1, 0, 0, 0),
        maximize=True,
        required_constraints=("safe",),
        inactive_components=(1, 2, 3),
    )
    decision, _ = select_skill(
        state, candidates, model, gate, context_mode=ContextMode.PREDICTED, seed=0
    )
    assert decision.selected_index == 1
    decision, _ = select_skill(
        state,
        candidates,
        model,
        gate,
        context_mode=ContextMode.PREDICTED,
        seed=0,
        intervention=(higher, state),
    )
    assert decision.selected_index == 0
    vetoed = (candidates[0], replace(candidates[1], hard_checks=(("safe", False),)))
    decision, _ = select_skill(
        state, vetoed, model, gate, context_mode=ContextMode.PREDICTED, seed=0
    )
    assert decision.selected_index == 0
    no_context = fit_utility(
        examples, train_groups={"train-a"}, prediction_context=False, ridge=0.1
    )
    decision, scores = select_skill(
        state, candidates, no_context, gate, context_mode=ContextMode.NONE, seed=0
    )
    assert decision.selected_index == 0
    assert scores == (0.5, 0.5)
    shuffled, _ = select_skill(
        state, candidates, model, gate, context_mode=ContextMode.SHUFFLED, seed=3
    )
    assert shuffled.selected_index == 0
