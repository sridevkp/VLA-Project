"""Physical validation reporting requires every pair and independent CAN feedback."""

import json

import numpy as np
import pytest

from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.validation_audit import audit_validation

PARAMETERS = dict(
    place_world_m=(0.4, 0.0, 1.0),
    distance_tolerance_m=0.1,
    height_tolerance_m=0.02,
    tilt_limit_deg=15.0,
    home_tolerance_rad=0.02,
    arm_names=("arm_1_joint",),
)


def test_incomplete_validation_never_produces_final_rate(tmp_path):
    with pytest.raises(ValueError, match="not complete"):
        audit_validation(tmp_path, **PARAMETERS)


def test_measured_can_pose_and_joint_return_must_agree_with_declared_outcome(tmp_path):
    from dataclasses import asdict

    plan = build_episode_plan("validation", 0)
    for entry in plan:
        stem = f"episode-{entry.index:02d}"
        meta = {
            "scenario": asdict(entry.scene),
            "reset_seed": entry.seed,
            "pipeline": "baseline",
            "outcome": {"success": True, "home_reached": True},
        }
        (tmp_path / (stem + ".meta.json")).write_text(json.dumps(meta), encoding="utf-8")
        np.savez(
            tmp_path / (stem + ".steps.npz"),
            gt_names=["CAN"],
            joint_names=["arm_1_joint"],
            gt_xyz=[[[0.0, 0.0, 1.0]], [[0.4, 0.0, 1.0]]],
            gt_quat=[[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
            joint_pos=[[0.0], [0.0]],
            t=[0.0, 0.032],
        )
    result = audit_validation(tmp_path, **PARAMETERS)
    assert result["episodes"] == 18 and result["overall"]["successes"] == 18
    bad = tmp_path / "episode-00.steps.npz"
    np.savez(
        bad,
        gt_names=["CAN"],
        joint_names=["arm_1_joint"],
        gt_xyz=[[[0.0, 0.0, 1.0]], [[0.4, 0.0, 1.2]]],
        gt_quat=[[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
        joint_pos=[[0.0], [0.0]],
        t=[0.0, 0.032],
    )
    with pytest.raises(ValueError, match="physical evidence disagreement"):
        audit_validation(tmp_path, **PARAMETERS)
