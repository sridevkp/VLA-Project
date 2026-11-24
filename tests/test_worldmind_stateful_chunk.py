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
            return backend.apply(action)
        return ApplyResult(True)

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
    with pytest.raises(RuntimeError, match="budget"):
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
        decide=lambda _cursor, _action, continuation: ApplyResult(not continuation, "veto"),
        advance=lambda _cursor: backend.step(1),
        truth=backend.get_ground_truth,
    )
    cursor.reset(scenario, 0)
    assert cursor.apply(GripperCmd(0.04)).accepted
    cursor.step(1)
    with pytest.raises(RuntimeError, match="continuation rejected"):
        cursor.step(1)
    assert cursor.ticks == 1
