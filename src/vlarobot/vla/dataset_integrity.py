"""Complete native-dataset verification before training approval (no simulator imports)."""

from __future__ import annotations

import hashlib
import io
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES


def dataset_file_hashes(dataset: Path) -> dict[str, str]:
    """Fingerprint immutable training payload and metadata, excluding derived approvals."""
    paths = [
        *sorted((dataset / "data").rglob("*.parquet")),
        *sorted((dataset / "meta").rglob("*")),
        dataset / "source-provenance.json",
    ]
    return {
        str(p.relative_to(dataset).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def validate_dataset_integrity(
    dataset: Path, *, gripper_limits_m: tuple[float, float]
) -> dict[str, Any]:
    """Read every row/image, enforce source boundaries and recompute state/action statistics."""
    parquet = import_module("pyarrow.parquet")
    source = json.loads((dataset / "source-provenance.json").read_text(encoding="utf-8"))
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset / "meta/stats.json").read_text(encoding="utf-8"))
    entries = source["sources"]
    if source.get("camera_key") != CAMERA_KEY or tuple(source["state_names"]) != STATE_NAMES:
        raise ValueError("native observation contract differs from TIAGo")
    if tuple(source["action_names"]) != ACTION_NAMES or not entries:
        raise ValueError("native action contract or successful sources missing")
    if len({row["attempt"] for row in entries}) != len(entries):
        raise ValueError("duplicate source attempts in native dataset")
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0 or fps != source["fps"]:
        raise ValueError("native sample rate differs from source provenance")
    expected_counts = []
    for entry in entries:
        raw = Path(entry["raw_dir"])
        meta_path = raw / "episode-00.meta.json"
        if hashlib.sha256(meta_path.read_bytes()).hexdigest() != entry["meta_sha256"]:
            raise ValueError("source metadata changed")
        for name, key in [
            ("episode-00.steps.npz", "steps_sha256"),
            ("episode-00.actions.jsonl", "commands_sha256"),
            ("episode-00.physical.jsonl", "physical_sha256"),
            ("policy.jsonl", "policy_trace_sha256"),
        ]:
            if hashlib.sha256((raw / name).read_bytes()).hexdigest() != entry[key]:
                raise ValueError("source recording stream changed")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["scenario"] != entry["scene"] or meta["seed"] != entry["seed"]:
            raise ValueError("source scene/seed identity mismatch")
        if meta["scenario"]["metadata"]["split"] != "train" or not meta["outcome"]["success"]:
            raise ValueError("native data contains failed or held-out source")
        if (
            not meta["scenario"]["name"].startswith("train-")
            or not all(
                meta["outcome"].get(k) is True
                for k in ["home_reached", "place_distance_ok", "can_height_ok", "can_tilt_ok"]
            )
            or not all(
                entry["audit"]["checks"].get(k) is True
                for k in ["grasp", "lift", "transport", "placement"]
            )
        ):
            raise ValueError("native source lacks independently verified physical task criteria")
        if meta["frame_count"] != meta["step_count"] or entry["audit"]["accepted"] is not True:
            raise ValueError("source admission/cadence incomplete")
        expected_counts.append(meta["step_count"])
    counts = [0] * len(entries)
    sums = {"observation.state": np.zeros(8), "action": np.zeros(7)}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    image_digest = hashlib.sha256()
    total = 0
    last_episode = 0
    for path in sorted((dataset / "data").rglob("*.parquet")):
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                episode, frame = row["episode_index"], row["frame_index"]
                if not 0 <= episode < len(entries) or episode < last_episode:
                    raise ValueError("invalid native episode ordering")
                if episode > last_episode + 1 or frame != counts[episode] or row["index"] != total:
                    raise ValueError("native boundary, frame or index discontinuity")
                if not np.isclose(row["timestamp"], frame / fps, atol=1e-5, rtol=0):
                    raise ValueError("native timestamp differs from full lockstep cadence")
                for key, width in [("observation.state", 8), ("action", 7)]:
                    value = np.asarray(row[key], dtype=np.float64)
                    if value.shape != (width,) or not np.all(np.isfinite(value)):
                        raise ValueError("nonfinite or incorrectly shaped native state/action")
                    sums[key] += value
                    squares[key] += value * value
                grip = float(row["action"][6])
                if (
                    not float(np.float32(gripper_limits_m[0]))
                    <= grip
                    <= float(np.float32(gripper_limits_m[1]))
                ):
                    raise ValueError("gripper action is outside per-finger command limits")
                encoded = row[CAMERA_KEY]["bytes"]
                raw = Path(entries[episode]["raw_dir"]) / "frames/episode-00" / f"{frame:06d}.jpg"
                if encoded != raw.read_bytes():
                    raise ValueError("native RGB bytes differ from original recorded JPEG")
                with Image.open(io.BytesIO(encoded)) as picture:
                    if picture.mode != "RGB" or picture.size != (640, 480):
                        raise ValueError("native RGB schema changed")
                    picture.load()
                image_digest.update(len(encoded).to_bytes(8, "little"))
                image_digest.update(encoded)
                counts[episode] += 1
                last_episode = episode
                total += 1
    if (
        counts != expected_counts
        or total != info["total_frames"]
        or len(counts) != info["total_episodes"]
    ):
        raise ValueError("native episode/frame counts do not match audited source recordings")
    for key in sums:
        mean = sums[key] / total
        std = np.sqrt(np.maximum(squares[key] / total - mean * mean, 0))
        np.testing.assert_allclose(mean, stats[key]["mean"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(std, stats[key]["std"], rtol=1e-4, atol=1e-6)
    if image_digest.hexdigest() != source["image_storage"]["source_jpeg_bytes_sha256"]:
        raise ValueError("native image digest differs from export provenance")
    return {
        "verified": True,
        "episodes": len(entries),
        "frames": total,
        "episode_frame_counts": counts,
        "fps": fps,
        "all_rgb_bytes_verified": True,
        "all_rgb_decoded": True,
        "state_action_statistics_recomputed": True,
        "files_sha256": dataset_file_hashes(dataset),
    }
