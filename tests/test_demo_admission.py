"""A terminal success flag never admits a demonstration without measured physical motion."""

from copy import deepcopy

import numpy as np
import pytest

from vlarobot.core.types import Pose3D, Vec3
from vlarobot.eval.demo_audit import DemoAuditSpec, audit_arrays
from vlarobot.vla.dataset_export import derive_command_labels


@pytest.fixture
def physical_episode():
    spec = DemoAuditSpec(
        (-1, -4.3, 0.821), 0.1, 0.02, 15, 0.04, 0.03175, 0.008, 0.045, 0.002, 0.032
    )
    can = np.array(
        [[-0.775, -4.3, 0.811], [-0.8, -4.3, 0.91], [-0.95, -4.3, 0.91], [-1, -4.3, 0.811]]
    )
    t = np.arange(4) * 0.032
    arrays = {
        "t": t,
        "gt_names": np.array(["CAN"]),
        "gt_xyz": can[:, None, :],
        "gt_quat": np.tile([0, 0, 0, 1.0], (4, 1, 1)),
    }
    meta = {
        "frame_count": 4,
        "scenario": {"name": "train-clear-unit", "metadata": {"split": "train"}},
        "outcome": {
            "success": True,
            "home_reached": True,
            "place_distance_ok": True,
            "can_height_ok": True,
            "can_tilt_ok": True,
            "rejections": 0,
        },
    }
    physical = []
    for i, c in enumerate(can):
        midpoint = c + [0, 0, 0.02]
        physical.append(
            {
                "t": float(t[i]),
                "finger_positions_m": [0.031, 0.031],
                "pad_positions_world_m": [
                    (midpoint + [0, -0.031, 0]).tolist(),
                    (midpoint + [0, 0.031, 0]).tolist(),
                ],
            }
        )
    return meta, arrays, physical, spec


def test_success_requires_lift_enclosure_transport_and_final_placement(physical_episode):
    meta, arrays, physical, spec = physical_episode
    result = audit_arrays(meta, arrays, physical, spec)
    assert result["accepted"]
    assert all(result["checks"].values())
    assert result["max_can_rise_m"] == pytest.approx(0.099)


def test_home_and_correct_placement_without_lift_are_rejected(physical_episode):
    meta, arrays, physical, spec = physical_episode
    arrays["gt_xyz"][:, :, 2] = 0.811
    result = audit_arrays(meta, arrays, physical, spec)
    assert not result["accepted"]
    assert not result["checks"]["lift"]
    assert not result["checks"]["transport"]


@pytest.mark.parametrize(
    "fault", ["failed_outcome", "validation", "missing_frames", "lost_object", "missing_final_flag"]
)
def test_failed_held_out_or_incomplete_episodes_are_never_admitted(physical_episode, fault):
    meta, arrays, physical, spec = physical_episode
    if fault == "failed_outcome":
        meta["outcome"]["success"] = False
    elif fault == "validation":
        meta["scenario"]["metadata"]["split"] = "validation"
    elif fault == "missing_frames":
        meta["frame_count"] = 3
    elif fault == "lost_object":
        for row in physical:
            row["pad_positions_world_m"] = [[1, 1, 1], [1, 1.06, 1]]
    elif fault == "missing_final_flag":
        del meta["outcome"]["can_height_ok"]
    assert not audit_arrays(meta, arrays, physical, spec)["accepted"]


def test_command_labels_do_not_teach_observed_stall_feedback_as_close_command():
    arrays = {
        "t": np.array([0.0, 0.032]),
        "joint_names": np.array(["arm_1_joint"]),
        "joint_pos": np.array([[0.0], [0.01]]),
        "ee_xyz": np.array([[0, 0, 0], [0.01, 0, 0]]),
        "ee_quat": np.tile([0.0, 0, 0, 1], (2, 1)),
        "gripper": np.array([0.031, 0.031]),
    }
    commands = [
        {"t": 0.0, "accepted": True, "action": {"q": {"arm_1_joint": 0.02}}},
        {"t": 0.0, "accepted": True, "action": {"opening": 0.0, "force": 50.0}},
    ]
    labels = derive_command_labels(
        arrays, commands, lambda q: Pose3D(Vec3(q["arm_1_joint"], 0, 0), (0, 0, 0, 1))
    )
    np.testing.assert_allclose(labels[:, 0], [0.02, 0.01])
    np.testing.assert_array_equal(labels[:, 6], [0.0, 0.0])
    bad = deepcopy(commands)
    bad[0]["accepted"] = False
    with pytest.raises(ValueError, match="safety-validated"):
        derive_command_labels(arrays, bad, lambda q: Pose3D(Vec3(0, 0, 0), (0, 0, 0, 1)))
