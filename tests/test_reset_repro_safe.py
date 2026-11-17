from dataclasses import replace

import numpy as np
import pytest

from vlarobot.core.types import Observation, Pose2D, Pose3D, RobotState, ScenarioSpec, Vec3
from vlarobot.eval.reset_repro import (
    ResetSnapshot,
    _quat_angle_diff,
    compare_snapshots,
    snapshot_from,
)

pytestmark = pytest.mark.safe_ci


def _snapshot(**changes: object) -> ResetSnapshot:
    original = ResetSnapshot(
        "scene",
        0,
        0.0,
        {"j": 0.0},
        {"j": 0.0},
        0.04,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        {},
    )
    return replace(original, **changes)


def test_missing_joint_keys_are_explicit_mismatch() -> None:
    result = compare_snapshots(_snapshot(), _snapshot(joint_pos={"other": 0.0}))
    assert not result.ok
    assert result.diffs["joint_pos"] == float("inf")
    assert any("keys differ" in note for note in result.notes)


def test_quaternion_sign_equivalence() -> None:
    assert _quat_angle_diff((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)) == pytest.approx(0.0)


def test_snapshot_without_ground_truth_has_no_object_poses() -> None:
    pose = Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot = RobotState({}, {}, Pose2D(0.0, 0.0, 0.0), pose, 0.04, base_pose_world=pose)
    observation = Observation(0.0, np.zeros((1, 1, 3), dtype=np.uint8), None, None, robot)
    assert snapshot_from(observation, None, ScenarioSpec("scene"), 0).object_poses == {}


def test_object_pose_shape_is_checked() -> None:
    pose = Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot = RobotState({}, {}, Pose2D(0.0, 0.0, 0.0), pose, 0.04, base_pose_world=pose)
    observation = Observation(0.0, np.zeros((1, 1, 3), dtype=np.uint8), None, None, robot)

    class BadPose:
        position = Vec3(1.0, 2.0, 3.0)
        orientation = (0.0, 0.0, 1.0)

    from vlarobot.core.types import GroundTruth

    with pytest.raises(ValueError, match="invalid object pose shape"):
        snapshot_from(observation, GroundTruth({"bad": BadPose()}), ScenarioSpec("scene"), 0)  # type: ignore[dict-item]
