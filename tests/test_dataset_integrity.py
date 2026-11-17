"""Full native validation detects corrupted images, episode boundaries and statistics."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from vlarobot.vla.dataset_integrity import validate_dataset_integrity
from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES

pa = pytest.importorskip("pyarrow", reason="optional installed native integrity integration")
pq = pytest.importorskip("pyarrow.parquet")


def fixture_dataset(tmp_path):
    raw = tmp_path / "raw"
    (raw / "frames/episode-00").mkdir(parents=True)
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    scene = {"name": "train-clear-i00-s100", "metadata": {"split": "train"}}
    meta = {
        "seed": 100,
        "scenario": scene,
        "step_count": 2,
        "frame_count": 2,
        "outcome": {
            "success": True,
            "home_reached": True,
            "place_distance_ok": True,
            "can_height_ok": True,
            "can_tilt_ok": True,
        },
    }
    (raw / "episode-00.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    entry = {
        "attempt": 0,
        "seed": 100,
        "scene": scene,
        "raw_dir": str(raw),
        "audit": {
            "accepted": True,
            "checks": {"grasp": True, "lift": True, "transport": True, "placement": True},
        },
        "meta_sha256": hashlib.sha256((raw / "episode-00.meta.json").read_bytes()).hexdigest(),
    }
    for name, key in [
        ("episode-00.steps.npz", "steps_sha256"),
        ("episode-00.actions.jsonl", "commands_sha256"),
        ("episode-00.physical.jsonl", "physical_sha256"),
        ("policy.jsonl", "policy_trace_sha256"),
    ]:
        (raw / name).write_bytes(b"unit test source identity only")
        entry[key] = hashlib.sha256((raw / name).read_bytes()).hexdigest()
    rows = []
    digest = hashlib.sha256()
    for i in range(2):
        image = raw / "frames/episode-00" / f"{i:06d}.jpg"
        Image.fromarray(np.full((480, 640, 3), i, dtype=np.uint8)).save(image)
        content = image.read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        rows.append(
            {
                "episode_index": 0,
                "frame_index": i,
                "index": i,
                "timestamp": i / 31.25,
                "observation.state": [0.0] * 8,
                "action": [0.0] * 7,
                CAMERA_KEY: {"bytes": content, "path": image.name},
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/record.parquet")
    source = {
        "camera_key": CAMERA_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "sources": [entry],
        "fps": 31.25,
        "image_storage": {"source_jpeg_bytes_sha256": digest.hexdigest()},
    }
    (root / "source-provenance.json").write_text(json.dumps(source), encoding="utf-8")
    (root / "meta/info.json").write_text(
        json.dumps({"fps": 31.25, "total_frames": 2, "total_episodes": 1}), encoding="utf-8"
    )
    stats = {
        "observation.state": {"mean": [0.0] * 8, "std": [0.0] * 8},
        "action": {"mean": [0.0] * 7, "std": [0.0] * 7},
    }
    (root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return root, rows


def test_full_scan_decodes_all_rows_and_rejects_boundary_tampering(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    report = validate_dataset_integrity(root, gripper_limits_m=(0, 0.045))
    assert report["verified"] and report["frames"] == 2 and report["all_rgb_bytes_verified"]
    rows[1]["frame_index"] = 0
    pq.write_table(pa.Table.from_pylist(rows), root / "data/record.parquet")
    with pytest.raises(ValueError, match="discontinuity"):
        validate_dataset_integrity(root, gripper_limits_m=(0, 0.045))


def test_full_scan_rejects_normalization_or_source_modification(tmp_path):
    root, _ = fixture_dataset(tmp_path)
    stats = json.loads((root / "meta/stats.json").read_text())
    stats["action"]["mean"][0] = 1
    (root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(AssertionError):
        validate_dataset_integrity(root, gripper_limits_m=(0, 0.045))
    (tmp_path / "raw/policy.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stream changed"):
        validate_dataset_integrity(root, gripper_limits_m=(0, 0.045))
