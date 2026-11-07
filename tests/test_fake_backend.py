import numpy as np
import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import EEPoseTarget, GripperCmd, Pose3D, ScenarioSpec, Vec3


def test_fake_backend_reset_is_deterministic_for_same_seed() -> None:
    backend = FakeBackend()

    first = backend.reset(scenario=ScenarioSpec(), seed=7)
    second = backend.reset(scenario=ScenarioSpec(), seed=7)

    assert first.t == second.t == 0.0
    assert first.robot == second.robot
    np.testing.assert_array_equal(first.rgb, second.rgb)
    np.testing.assert_array_equal(first.depth, second.depth)


def test_fake_backend_steps_and_returns_independent_observations() -> None:
    backend = FakeBackend()
    initial = backend.reset(scenario=ScenarioSpec(), seed=0)

    later = backend.step(3)
    initial.rgb.fill(255)

    assert later.t == pytest.approx(0.096)
    assert later.rgb.shape == (48, 64, 3)
    assert not later.rgb.any()
    assert backend.get_ground_truth().object_poses == {}


def test_fake_backend_rejects_unsafe_actions_before_state_changes() -> None:
    backend = FakeBackend()
    before = backend.reset(scenario=ScenarioSpec(), seed=0)
    unsafe = EEPoseTarget(
        pose=Pose3D(Vec3(2.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        gripper=None,
        duration=1.0,
    )

    result = backend.apply(unsafe)

    assert not result.accepted
    assert "workspace" in (result.reason or "")
    assert backend.step().robot == before.robot


def test_fake_backend_applies_safe_commands() -> None:
    backend = FakeBackend()
    backend.reset(scenario=ScenarioSpec(), seed=0)

    result = backend.apply(GripperCmd(opening=0.02))

    assert result.accepted
    assert backend.step().robot.gripper_opening == 0.02


def test_fake_backend_rejects_invalid_step_and_stopped_operations() -> None:
    backend = FakeBackend()
    backend.reset(scenario=ScenarioSpec(), seed=0)

    with pytest.raises(ValueError, match="positive"):
        backend.step(0)

    backend.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        backend.step()
    with pytest.raises(RuntimeError, match="stopped"):
        backend.apply(GripperCmd(opening=0.02))
