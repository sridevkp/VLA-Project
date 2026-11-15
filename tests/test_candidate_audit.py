"""Contact-feedback error is reported separately and diagnostic trajectories are never demos."""

import json

import numpy as np
import pytest

from vlarobot.eval.candidate_audit import audit_candidates


def test_diagnostic_feedback_error_and_success_exclusion(tmp_path):
    folder = tmp_path / "closure"
    folder.mkdir()
    meta = folder / "episode-00.meta.json"
    meta.write_text(
        json.dumps({"outcome": {"success": False, "not_a_demonstration": True}}), encoding="utf-8"
    )
    (folder / "episode-00.predicted.jsonl").write_text(
        json.dumps({"t": 0.032, "predicted": {"gripper_m": 0.0}}) + "\n", encoding="utf-8"
    )
    np.savez(
        folder / "episode-00.steps.npz",
        t=[0.0, 0.032],
        gripper=[0.045, 0.031],
        gt_names=["CAN"],
        gt_xyz=[[[0.0, 0.0, 0.81]], [[0.0, 0.0, 0.81]]],
    )
    result = audit_candidates(tmp_path)
    assert result["cases"][0]["gripper_feedback_rmse_m"] == pytest.approx(0.031)
    assert result["training_admission"] is False
    meta.write_text(
        json.dumps({"outcome": {"success": True, "not_a_demonstration": True}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not be successful demonstrations"):
        audit_candidates(tmp_path)
