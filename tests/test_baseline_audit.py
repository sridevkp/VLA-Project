"""Completed HOME metadata does not manufacture measured lift or placement evidence."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from vlarobot.eval import baseline_audit
from vlarobot.eval.demo_audit import DemoAuditSpec
from vlarobot.eval.runner import build_episode_plan


@pytest.mark.parametrize("rise,expected_lifts", [(0.0, 0), (0.079, 1)])
def test_baseline_audit_measures_physical_lift_and_checks_scene_geometry(
    tmp_path, monkeypatch, rise, expected_lifts
):
    entry = build_episode_plan("train", 0)[0]
    monkeypatch.setattr(baseline_audit, "build_episode_plan", lambda *args: [entry])
    monkeypatch.setattr(baseline_audit, "_completed_recording", lambda *args: object())
    documents = {
        "campaign-status.json": {"status": "completed"},
        "summary.json": {"n_episodes": 1, "success": {"successes": 1}},
        "provenance.json": {},
        "episode-00.meta.json": {
            "scenario": asdict(entry.scene),
            "outcome": {"success": True, "final_state": "home"},
        },
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    np.savez(
        tmp_path / "episode-00.steps.npz",
        gt_names=np.array(["CAN"]),
        t=np.array([0, 0.032, 0.064]),
        gt_xyz=np.array([[[0, 0, 0.821]], [[0.05, 0, 0.821 + rise]], [[0.1, 0, 0.821]]]),
        gt_quat=np.array([[[0, 0, 0, 1]]] * 3),
    )
    for suffix in ["actions.jsonl", "contacts.jsonl"]:
        (tmp_path / ("episode-00." + suffix)).write_text("", encoding="utf-8")
    spec = DemoAuditSpec((0.1, 0, 0.821), 0.1, 0.02, 15, 0.04, 0.03175, 0.008, 0.045, 0.002, 0.032)
    report = baseline_audit.audit_completed_baseline(tmp_path, spec)
    assert report["physical_successes_with_rise_ge_40mm"] == expected_lifts
    assert report["successful_outcome_placement_disagreements"] == []
    assert report["episodes"][0]["max_can_rise_m"] == pytest.approx(rise)
    meta = json.loads((tmp_path / "episode-00.meta.json").read_text(encoding="utf-8"))
    meta["scenario"]["can_xy_yaw_offset"][0] += 0.001
    (tmp_path / "episode-00.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry differs"):
        baseline_audit.audit_completed_baseline(tmp_path, spec)
