from dataclasses import replace
from types import SimpleNamespace

import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import ApplyResult, GripperCmd, JointTarget, ScenarioSpec
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.worldmind_expert_parity import EPISODE, ReplayPlan, execute_timed_candidate

DT = 0.032


@pytest.mark.parametrize("opening_at_close", (0.045, 0.0173))
def test_candidate_00_replays_holds_and_checks_measured_opening(tmp_path, opening_at_close):
    """Keep expert command timestamps and reject a v1-sized pre-CLOSE opening."""
    backend = FakeBackend()
    raw_names = ("gripper_left_finger_joint", "gripper_right_finger_joint")
    initial = backend.reset(ScenarioSpec(), 0)
    t0 = initial.t
    trace = []
    tick = 0

    def observed(obs, opening):
        diag = {
            **obs.robot.diagnostics,
            "finger_joint_sensor_rad": dict.fromkeys(raw_names, opening),
        }
        pos = {**dict.fromkeys((f"arm_{i}_joint" for i in range(1, 8)), 0.0), **obs.robot.joint_pos}
        vel = {**dict.fromkeys(pos, 0.0), **obs.robot.joint_vel}
        return replace(
            obs,
            t=t0 + tick * DT,
            robot=replace(
                obs.robot,
                gripper_opening=opening,
                diagnostics=diag,
                joint_pos=pos,
                joint_vel=vel,
            ),
        )

    def apply(action):
        trace.append((tick, action))
        return ApplyResult(True)

    def step():
        nonlocal tick
        tick += 1
        return {
            "confirmed_ticks": 1,
            "observation": observed(backend.step(1), opening_at_close if tick >= 48 else 0.045),
        }

    actions = (
        (0, GripperCmd(0.045)),
        (45, JointTarget({"arm_1_joint": 0.0})),
        (48, GripperCmd(0.0, 50.0)),
    )
    phases = ((0, "INIT"), (45, "PREGRASP"), (47, "DESCEND"), (48, "CLOSE"))
    plan = ReplayPlan(EPISODE, 0, t0, 50, actions, phases, 0.045, {})
    monitor = SafetyMonitor(
        SafetyLimits(joint_position={"arm_1_joint": (-1, 1)}, joint_velocity={"arm_1_joint": 1})
    )
    owned = SimpleNamespace(client=SimpleNamespace(apply=apply, diagnostic_step=step))
    if opening_at_close < 0.044:
        with pytest.raises(RuntimeError, match="more than 1 mm"):
            execute_timed_candidate(plan, owned, {}, tmp_path, observed(initial, 0), monitor)
        assert trace == list(actions[:-1])
        assert tick == 48
    else:
        final, opening = execute_timed_candidate(
            plan, owned, {}, tmp_path, observed(initial, 0), monitor
        )
        assert opening == 0.045
        assert trace == list(actions)
        assert final.t == pytest.approx(t0 + 50 * DT)
        assert tick == 50
    for phase_tick, _ in phases:
        assert (tmp_path / f"tick-{phase_tick:04d}/phase.json").exists()


@pytest.mark.parametrize(("remaining", "success"), ((60, True), (10, False)))
def test_worldmind_init_wait_uses_expert_criterion_and_shared_budget(
    monkeypatch,
    tmp_path,
    remaining,
    success,
):
    from vlarobot.execution import worldmind_heuristic_qualification as runner

    observation = FakeBackend().reset(ScenarioSpec(), 0)
    start = replace(observation, robot=replace(observation.robot, gripper_opening=0.00192))
    ticks = []

    def tick(_, session, before, intent, scope):
        assert intent["phase"] == "INIT"
        ticks.append(intent["hold_tick"])
        return "one_tick_observed", replace(
            before,
            t=before.t + DT,
            robot=replace(before.robot, gripper_opening=before.robot.gripper_opening + 0.00096),
        )

    monkeypatch.setattr(runner, "_tick", tick)
    status, after, count = runner.wait_for_expert_opening(
        tmp_path / "hold",
        SimpleNamespace(),
        start,
        remaining_ticks=remaining,
        elapsed_open_ticks=2,
    )
    assert ticks == list(range(count))
    if success:
        assert status == "opening_reached" and count == 43
        assert after.robot.gripper_opening >= 0.043
    else:
        assert status == "WORLDMIND_BOUND_EXHAUSTED" and count == 10 and after is None
