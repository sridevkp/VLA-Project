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
    gt = backend.get_ground_truth()
    assert list(gt.object_poses) == ["CAN"]
    np.testing.assert_allclose(
        gt.object_poses["CAN"].position.as_array(), [-0.799879, -4.38998, 0.810949]
    )
    assert gt.object_poses["CAN"].orientation == (0.0, 0.0, 0.0, 1.0)


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


def test_fake_backend_can_world_xyz_matches_scenario_config() -> None:
    # Drift guard: FakeBackend's default CAN pose mirrors configs/scenarios.yaml (ADR-0033).
    from vlarobot.backends.fake.backend import CAN_WORLD_XYZ
    from vlarobot.eval.scenarios import load_scenario_library

    assert CAN_WORLD_XYZ == load_scenario_library().world_can_xyz


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


def test_fake_backend_export_kinematics_is_chunked_and_separate_from_reset() -> None:
    backend = FakeBackend()
    # Reset stays fast and must not attach/require an export snapshot.
    obs = backend.reset(scenario=ScenarioSpec(), seed=0)
    assert "sim_kinematics_snapshot" not in obs.robot.diagnostics

    status = backend.export_kinematics("start")
    assert status["done"] is False
    assert status["state"] == "in_progress"
    polls = 0
    while not status["done"] and polls < 10:
        status = backend.export_kinematics("poll")
        polls += 1
    assert status["done"] is True
    snapshot = status.get("snapshot")
    assert isinstance(snapshot, dict)
    assert snapshot.get("schema_version") == 1
    per_joint = status.get("per_joint") or []
    assert isinstance(per_joint, list) and per_joint
    first = per_joint[0]
    assert isinstance(first, dict)
    assert first["reason"] == "ok"
    assert first["measured_angle_moved_rad"] == pytest.approx(0.05)
