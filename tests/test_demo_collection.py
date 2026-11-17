"""Fresh collection preserves failed attempts and freezes its TRAIN provenance on resume."""

import hashlib
import json
from pathlib import Path

import pytest

from vlarobot.backends.webots.config import SETTINGS_PATH
from vlarobot.execution import demo_collection as collection


def baseline(tmp_path, monkeypatch):
    root = tmp_path / "baseline"
    root.mkdir()
    collection.write_json(root / "campaign-status.json", {"status": "completed"})
    collection.write_json(
        root / "summary.json",
        {
            "n_episodes": 24,
            "measurement": "measured",
            "split": "train",
            "success": {"successes": 20, "rate": 20 / 24},
        },
    )
    project = SETTINGS_PATH.parent.parent
    names = [
        "configs/calibration/urdf-base-train-v1.yaml",
        "src/vlarobot/backends/webots/images.py",
    ]
    collection.write_json(
        root / "provenance.json",
        {
            "files_sha256": {
                name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in names
            },
        },
    )
    monkeypatch.setenv("VLAROBOT_URDF_REGISTRATION_FILE", "calibration/urdf-base-train-v1.yaml")
    return root


def test_collection_resume_preserves_quarantine_and_selects_only_train_successes(
    tmp_path, monkeypatch
):
    frozen = baseline(tmp_path, monkeypatch)
    calls = []

    def attempt(entry, out, **kwargs):
        calls.append(entry.index)
        assert entry.scene.metadata["split"] == "train"
        assert entry.seed >= 100
        destination = out / "attempts" / f"attempt-{entry.index:04d}"
        destination.mkdir(parents=True)
        result = {
            "attempt": entry.index,
            "seed": entry.seed,
            "status": "quarantined" if entry.index == 0 else "accepted",
            "raw_dir": str(destination),
        }
        collection.write_json(destination / "result.json", result)
        return result

    monkeypatch.setattr(collection, "_attempt", attempt)
    out = tmp_path / "demos"
    result = collection.collect_verified_training(
        out, frozen, webots="unused", target=2, max_attempts=4
    )
    assert result["status"] == "completed"
    assert result["quarantined_attempts"] == 1
    accepted = json.loads((out / "accepted.json").read_text(encoding="utf-8"))
    assert [r["attempt"] for r in accepted["episodes"]] == [1, 2]
    assert accepted["training_approved"] is False
    called = list(calls)
    collection.collect_verified_training(
        out, frozen, webots="unused", target=2, max_attempts=4, resume=True
    )
    assert calls == called
    with pytest.raises(FileExistsError, match="identical"):
        collection.collect_verified_training(
            out, frozen, webots="unused", target=2, max_attempts=4, resume=True, seed=200
        )


def test_collection_refuses_partial_baseline_zero_success_and_other_world(tmp_path, monkeypatch):
    frozen = baseline(tmp_path, monkeypatch)
    collection.write_json(frozen / "campaign-status.json", {"status": "running"})
    with pytest.raises(ValueError, match="must finish"):
        collection.collect_verified_training(tmp_path / "out", frozen, webots="unused")
    collection.write_json(frozen / "campaign-status.json", {"status": "completed"})
    with pytest.raises(ValueError, match="frozen world"):
        collection.collect_verified_training(
            tmp_path / "out", frozen, webots="unused", world=Path("other.wbt")
        )
    summary = json.loads((frozen / "summary.json").read_text(encoding="utf-8"))
    summary["success"]["successes"] = 0
    collection.write_json(frozen / "summary.json", summary)
    with pytest.raises(ValueError, match="zero-success"):
        collection.collect_verified_training(tmp_path / "out", frozen, webots="unused")
