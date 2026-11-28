from unittest.mock import patch

import pytest

from vlarobot.backends.fake.backend import FakeBackend
from vlarobot.backends.webots.config import GRIPPER_OPEN
from vlarobot.core.types import ApplyResult, GripperCmd, JointTarget, ScenarioSpec, Vec3
from vlarobot.execution.skills import Skill, SkillAction, SkillExecutor, _CheckedBackend


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


def test_live_profile_supplies_velocity_limits_for_all_arm_joints() -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    monitor = SkillExecutor(0, initial).monitor
    assert monitor.limits.joint_position.keys() == monitor.limits.joint_velocity.keys()
    assert len(monitor.limits.joint_velocity) == 7


@pytest.mark.parametrize(
    "skill", [Skill.DESCEND, Skill.LIFT, Skill.TRANSFER, Skill.LOWER, Skill.RETREAT]
)
def test_cartesian_skill_calls_only_its_linear_primitive(skill: Skill) -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    executor = SkillExecutor(0, initial)
    with (
        patch.object(executor, "_move_linear", return_value=(None, initial)) as motion,
        patch.object(backend, "apply", wraps=backend.apply) as apply,
    ):
        result = executor.run_skill(backend, initial, SkillAction(skill, Vec3(0.5, 0, 0.8)))
    assert result.settled
    assert motion.call_count == 1
    assert motion.call_args.args[-1] == skill.value
    apply.assert_not_called()


@pytest.mark.parametrize("skill, opening", [(Skill.CLOSE, 0.0), (Skill.RELEASE, GRIPPER_OPEN)])
def test_gripper_skill_dispatches_only_its_gripper_command(skill: Skill, opening: float) -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    executor = SkillExecutor(0, initial)
    result = executor.run_skill(backend, initial, SkillAction(skill))
    assert result.settled
    assert len(result.dispatched) == 1
    command = result.dispatched[0]
    assert isinstance(command, GripperCmd)
    assert command.opening == opening


def test_predictive_audit_rechecks_each_exact_command_and_rejects_mutation() -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    joint = next(iter(initial.robot.joint_pos))
    original = initial.robot.joint_pos[joint]
    inspected = []

    def audit(action, observation, truth, terminal):
        inspected.append((action.q.copy(), observation.t))
        if len(inspected) == 2:
            action.q[joint] += 0.001
        return ApplyResult(True)

    checked = _CheckedBackend(backend, backend.safety_monitor, initial, audit)
    first = JointTarget({joint: original})
    second = JointTarget({joint: original + 0.01})
    with patch.object(backend, "apply", wraps=backend.apply) as dispatched:
        assert checked.apply(first).accepted
        checked.step()
        assert not checked.apply(second).accepted
    dispatched.assert_called_once_with(first)
    assert inspected == [({joint: original}, 0.0), ({joint: original + 0.01}, 0.032)]
    assert checked.vetoes == ["joint command changed during predictive audit"]


def test_proxy_rejection_prevents_dispatch() -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    executor = SkillExecutor(0, initial)
    with patch.object(backend, "apply", wraps=backend.apply) as apply:
        result = executor.run_skill(
            backend,
            initial,
            SkillAction(Skill.OPEN),
            audit=lambda *args: ApplyResult(False, "modeled_obstacles"),
        )
    assert not result.settled
    assert result.vetoes == ("modeled_obstacles",)
    apply.assert_not_called()
