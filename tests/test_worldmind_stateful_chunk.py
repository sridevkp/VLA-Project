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
