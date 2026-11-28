from unittest.mock import patch

from vlarobot.backends.fake.backend import FakeBackend
from vlarobot.core.types import ApplyResult, GripperCmd, ScenarioSpec
from vlarobot.execution.skills import Skill, SkillAction, SkillExecutor


def test_open_skill_only_sends_a_gripper_command() -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    executor = SkillExecutor(0, initial)
    with patch.object(backend, "apply", wraps=backend.apply) as apply:
        outcome = executor.run_skill(backend, initial, SkillAction(Skill.OPEN))
    assert outcome.settled
    assert apply.call_count == 1
    assert isinstance(apply.call_args.args[0], GripperCmd)
    assert outcome.can_rise_m == 0.0


def test_monitor_veto_stops_dispatch() -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    executor = SkillExecutor(0, initial)
    with (
        patch.object(executor.monitor, "validate", return_value=ApplyResult(False, "veto")),
        patch.object(backend, "apply", wraps=backend.apply) as apply,
    ):
        outcome = executor.run_skill(backend, initial, SkillAction(Skill.CLOSE))
    assert not outcome.settled
    assert outcome.vetoes == ("veto",)
    apply.assert_not_called()
