"""Admission health detects changed immutable streams without replaying a policy."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image

from vlarobot.eval.demos import DEFAULT_INSTRUCTION, build_demo_plan
from vlarobot.execution import collection_health as health


def test_integrity_monitor_validates_frames_and_rejects_changed_recording(tmp_path, monkeypatch):
    plan = build_demo_plan(1, base_seed=100)
    frozen = json.loads(json.dumps([asdict(p) for p in plan]))
    root = tmp_path / "collection"
    root.mkdir()
    raw = root / "attempts/attempt-0000"
    (raw / "frames/episode-00").mkdir(parents=True)
    meta = {"frame_count": 2, "step_count": 2}
    (raw / "episode-00.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(
        raw / "episode-00.steps.npz",
        joint_pos=np.zeros((2, 7)),
        gripper=np.zeros(2),
        ee_xyz=np.zeros((2, 3)),
        ee_quat=np.tile([0, 0, 0, 1], (2, 1)),
    )
    for name in ["actions", "physical"]:
        (raw / f"episode-00.{name}.jsonl").write_text("{}\n", encoding="utf-8")
    for i in range(2):
        Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
            raw / "frames/episode-00" / f"{i:06d}.jpg"
        )
    row = {
        "attempt": 0,
        "seed": 100,
        "scene": frozen[0]["scene"],
        "raw_dir": str(raw),
        "status": "accepted",
    }
    (root / "accepted.json").write_text(
        json.dumps({"count": 1, "episodes": [row]}), encoding="utf-8"
    )
    (root / "attempts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "provenance.json").write_text(
        json.dumps({"seed_start": 100, "attempt_plan": frozen, "instruction": DEFAULT_INSTRUCTION}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "audit_demo", lambda *args: {"accepted": True})
    cache = tmp_path / "integrity.json"
    report = health.inspect_collection(root, cache)
    assert (
        report["admitted"] == 1 and report["verified_frames"] == 2 and report["admission_rate"] == 1
    )
    (raw / "episode-00.actions.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stream changed"):
        health.inspect_collection(root, cache)
