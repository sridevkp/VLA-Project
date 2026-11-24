from __future__ import annotations

import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import ApplyResult, GripperCmd, ScenarioSpec
from vlarobot.execution.worldmind_stateful_chunk import StatefulChunkBackend


def test_chunks_preserve_active_target_and_observed_phase():
    backend = FakeBackend()
    scenario = ScenarioSpec()
    initial = backend.reset(scenario, 0)
    plans = []

    def decide(cursor, action, continuation):
        plans.append((cursor.phase, cursor.ticks, cursor.observation.t, action, continuation))
        if not continuation:
            return backend.apply(action), action
        return ApplyResult(True), action

    cursor = StatefulChunkBackend(
        initial,
        scenario,
        0,
        horizon=2,
        max_ticks=6,
        decide=decide,
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
    )
    assert cursor.reset(scenario, 0) is initial
    cursor.phase = "PREGRASP"
    action = GripperCmd(0.04)
    assert cursor.apply(action).accepted
    for _ in range(5):
        cursor.step(1)
    assert [(phase, t, cmd, cont) for phase, t, _, cmd, cont in plans] == [
        ("PREGRASP", 0, action, False),
        ("PREGRASP", 2, action, True),
        ("PREGRASP", 4, action, True),
    ]
    assert plans[0][2] < plans[1][2] < plans[2][2]
    cursor.phase = "DESCEND"
    assert cursor.apply(action).accepted
    assert plans[-1][0] == "DESCEND"
    cursor.step(1)
    with pytest.raises(RuntimeError, match="WORLDMIND_BOUND_EXHAUSTED"):
        cursor.step(1)
    with pytest.raises(ValueError, match="inconsistent reset"):
        cursor.reset(scenario, 0)


def test_rejected_continuation_does_not_tick():
    backend = FakeBackend()
    scenario = ScenarioSpec()
    initial = backend.reset(scenario, 0)
    cursor = StatefulChunkBackend(
        initial,
        scenario,
        0,
        horizon=1,
        max_ticks=5,
        decide=lambda _cursor, action, continuation: (
            ApplyResult(not continuation, "veto"),
            action,
        ),
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
    )
    cursor.reset(scenario, 0)
    assert cursor.apply(GripperCmd(0.04)).accepted
    cursor.step(1)
    with pytest.raises(RuntimeError, match="continuation rejected"):
        cursor.step(1)
    assert cursor.ticks == 1


def test_only_explicit_home_retry_may_reopen_init_and_counters_stay_global():
    backend = FakeBackend()
    scenario = ScenarioSpec()
    initial = backend.reset(scenario, 0)
    cursor = StatefulChunkBackend(
        initial,
        scenario,
        0,
        horizon=5,
        max_ticks=10,
        decide=lambda _cursor, action, _continuation: (backend.apply(action), action),
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
    )
    cursor.reset(scenario, 0)
    cursor.phase = "PREGRASP"
    cursor.apply(GripperCmd(0.04))
    after = cursor.step(1)
    with pytest.raises(ValueError, match="invalid verification retry"):
        cursor.retry_verification()
    cursor.phase = "HOME"
    cursor.retry_verification()
    assert cursor.phase == "INIT"
    assert cursor.reset(scenario, 0) is after
    assert (cursor.ticks, cursor.decisions) == (1, 1)
    cursor.phase = "HOME"
    with pytest.raises(ValueError, match="invalid verification retry"):
        cursor.retry_verification()


def test_empty_set_reobserves_once_without_restarting_the_waypoint():
    backend = FakeBackend()
    scenario = ScenarioSpec()
    initial = backend.reset(scenario, 0)
    plans = []

    def decide(cursor, action, _continuation):
        plans.append((cursor.phase, cursor.observation.t))
        return ApplyResult(False, "no eligible stateful candidate"), action

    cursor = StatefulChunkBackend(
        initial,
        scenario,
        0,
        horizon=5,
        max_ticks=10,
        decide=decide,
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
        reobserve=lambda _cursor: backend.step(1),
    )
    cursor.reset(scenario, 0)
    cursor.phase = "DESCEND"
    with pytest.raises(RuntimeError, match="NO_SAFE_SET"):
        cursor.apply(GripperCmd(0.04))
    assert tuple(phase for phase, _ in plans) == ("DESCEND", "DESCEND")
    assert plans[1][1] > plans[0][1]
    assert (cursor.ticks, cursor.decisions) == (1, 2)


def test_new_waypoint_discards_the_held_suffix_and_uses_the_selected_command():
    backend = FakeBackend()
    scenario = ScenarioSpec()
    initial = backend.reset(scenario, 0)
    plans = []
    actual = GripperCmd(0.03)

    def decide(cursor, action, continuation):
        plans.append((cursor.ticks, action, continuation))
        chosen = actual if len(plans) == 1 else action
        return backend.apply(chosen), chosen

    cursor = StatefulChunkBackend(
        initial,
        scenario,
        0,
        horizon=50,
        max_ticks=100,
        decide=decide,
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
    )
    cursor.reset(scenario, 0)
    cursor.apply(GripperCmd(0.04))
    assert cursor.target == actual
    cursor.step(1)
    close = GripperCmd(0.0)
    cursor.apply(close)
    assert (cursor.used, cursor.target) == (0, close)
    cursor.step(1)
    assert plans == [(0, GripperCmd(0.04), False), (1, close, False)]


def test_boundary_waypoint_draws_are_deterministic_without_rejection_exhaustion():
    import numpy as np

    from vlarobot.execution.worldmind_stateful_chunk import bounded_normal_target

    for reference in (-1.39, 1.39):

        def draws(reference):
            rng = np.random.default_rng(0)
            return [
                bounded_normal_target(reference, -1.39, 1.39, 0.001, 0.005, rng)
                for _ in range(1024)
            ]

        actual = draws(reference)
        assert actual == draws(reference)
        assert len(set(actual)) == len(actual)
        assert all(-1.39 <= value <= 1.39 and abs(value - reference) <= 0.005 for value in actual)


def test_final_verifier_is_not_called_before_observed_home(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from vlarobot.core.types import GroundTruth, Pose3D, Vec3
    from vlarobot.execution import worldmind_stateful_chunk as module

    @dataclass
    class Result:
        home_reached: bool = True
        rejections: int = 0

    class EarlyExpert:
        def __init__(self, **_kwargs):
            pass

        def execute(self, cursor, _seed):
            cursor.phase = "PREGRASP"
            return Result()

    def unexpected(*_args):
        pytest.fail("observed verifier ran before HOME")

    monkeypatch.setattr(module, "ChunkExpert", EarlyExpert)
    backend = FakeBackend()
    scene = ScenarioSpec()
    initial = backend.reset(scene, 0)
    truth = GroundTruth({"CAN": Pose3D(Vec3(0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0))})
    assert (
        module.run_stateful_expert(
            SimpleNamespace(client=SimpleNamespace(get_ground_truth=lambda: truth)),
            {},
            tmp_path,
            SimpleNamespace(seed=0, scene=scene),
            SimpleNamespace(
                adapter=SimpleNamespace(jitter=None),
                pipeline=SimpleNamespace(horizon_steps=50),
                bounds=SimpleNamespace(max_steps=1059),
            ),
            None,
            SimpleNamespace(limits=SimpleNamespace(joint_position={})),
            initial,
            SimpleNamespace(check_observed=unexpected),
            SimpleNamespace(target="CAN"),
            unexpected,
            scored=False,
        )
        == "PLACEMENT_FAILED"
    )
    assert list(tmp_path.glob("verification-*.json")) == []


def test_feedback_missing_predicted_completion_replans_before_suffix():
    backend = FakeBackend()
    scene = ScenarioSpec()
    initial = backend.reset(scene, 0)
    plans = []

    def decide(cursor, action, continuation):
        plans.append((cursor.ticks, cursor.observation.t, continuation))
        cursor.active_prefix = 1
        return ApplyResult(True), action

    cursor = StatefulChunkBackend(
        initial,
        scene,
        0,
        horizon=50,
        max_ticks=10,
        decide=decide,
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
    )
    cursor.reset(scene, 0)
    cursor.apply(GripperCmd(0.04))
    cursor.step(1)
    # The expert asks for a second step because no observed completion occurred.
    cursor.step(1)
    assert plans[0][0] == 0 and plans[1][0] == 1
    assert plans[1][2] and plans[1][1] > plans[0][1]


def test_vetoed_continuation_uses_three_rounds_without_a_tick_or_new_waypoint():
    backend = FakeBackend()
    scene = ScenarioSpec()
    initial = backend.reset(scene, 0)
    calls = []

    def decide(cursor, action, continuation):
        calls.append((cursor.waypoint_index, cursor.replacement_round, cursor.ticks))
        if continuation:
            return ApplyResult(False, "no eligible stateful candidate"), action
        return backend.apply(action), action

    cursor = StatefulChunkBackend(
        initial,
        scene,
        0,
        horizon=1,
        max_ticks=10,
        decide=decide,
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
        max_replacements=3,
        reobserve=lambda _cursor: pytest.fail("must not step with a vetoed held target"),
    )
    cursor.reset(scene, 0)
    cursor.apply(GripperCmd(0.045))
    cursor.step(1)
    with pytest.raises(RuntimeError, match="NO_SAFE_SET"):
        cursor.step(1)
    assert calls == [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1)]
    assert cursor.ticks == 1


def test_accepted_replacement_becomes_target_without_resetting_waypoint_allowance():
    backend = FakeBackend()
    scene = ScenarioSpec()
    initial = backend.reset(scene, 0)
    replacement = GripperCmd(0.03)

    def decide(cursor, action, continuation):
        if not continuation:
            return backend.apply(action), action
        if cursor.replacement_round:
            return backend.apply(replacement), replacement
        return ApplyResult(False, "no eligible stateful candidate"), action

    cursor = StatefulChunkBackend(
        initial,
        scene,
        0,
        horizon=1,
        max_ticks=10,
        decide=decide,
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
        max_replacements=3,
    )
    cursor.reset(scene, 0)
    initial_target = GripperCmd(0.045)
    cursor.apply(initial_target)
    cursor.step(1)
    cursor.step(1)
    assert cursor.target == replacement and cursor.waypoint_reference == initial_target
    assert cursor.waypoint_index == 0 and cursor.replacements_used == 1
    cursor.step(1)
    assert cursor.replacements_used == 2
    cursor.apply(GripperCmd(0.02))
    assert cursor.waypoint_index == 1 and cursor.replacements_used == 0


def test_live_terminal_home_matches_verifier_and_leaves_intermediate_settle_unchanged():
    from vlarobot.backends.webots.config import ARM_JOINTS
    from vlarobot.core.types import ObjectState, Pose3D, SceneState, TaskSpec, Vec3, WorldState
    from vlarobot.execution.worldmind_stateful_chunk import ChunkExpert
    from vlarobot.planning.verification import PlacementVerifier

    q = dict.fromkeys(ARM_JOINTS, 0.0)
    expert = object.__new__(ChunkExpert)
    expert.verified_home = (q, 0.02)
    command = {**q, ARM_JOINTS[0]: 0.00225}
    pose = Pose3D(Vec3(0, 0, 1), (0.0, 0.0, 0.0, 1.0))
    verifier = PlacementVerifier(
        destination_base=pose,
        rest_center_base=pose,
        up_axis_base=(0.0, 0.0, 1.0),
        home_joints=q,
        distance_tolerance_m=0.1,
        height_tolerance_m=0.02,
        tilt_limit_rad=0.2,
        home_tolerance_rad=0.02,
    )
    robot = FakeBackend().reset(ScenarioSpec(), 0).robot
    scene = SceneState([ObjectState("CAN", "can", pose, Vec3(0.06, 0.06, 0.12), 1.0)])
    for value, expected in ((0.02195, False), (0.02, True), (0.019, True), (float("nan"), False)):
        measured = {**q, ARM_JOINTS[0]: value}
        robot.joint_pos = measured
        observed = WorldState(scene, robot)
        result = expert._joint_waypoint_settled(measured, command, 0.02, terminal_home=True)
        assert result is expected
        assert (
            verifier.check_observed(TaskSpec("place", "CAN", "goal"), observed).success is expected
        )
    measured = {**q, ARM_JOINTS[0]: 0.02195}
    assert expert._joint_waypoint_settled(measured, command, 0.02, terminal_home=False)
