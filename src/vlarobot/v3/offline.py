"""Offline V3 waypoint relabelling and CPU-only baselines from frozen TRAIN demos.

The tool reads the existing LeRobot-convertible dataset and recorded scripted policy
traces. It never calls Webots, never reads the sealed holdout and never changes a frozen
V1/V2 result. PyArrow is imported lazily because the core project environment does not
ship the parquet reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from vlarobot.v3.model import CompactEncoder, benchmark_encoder

BASE_SCENE_RE = re.compile(r"-s\d+$")
DEFAULT_DATASET = Path("/home/tj330/programs/VLA-Project/runs/datasets/tiago-train-v1")
DEFAULT_OUT = Path("runs/v3/offline")
DEFAULT_EVIDENCE = Path("docs/evidence/v3-offline-baselines.json")


@dataclass
class WaypointSample:
    episode_index: int
    scene: str
    base_scene: str
    family: str
    t: float
    frame_index: int
    state: np.ndarray
    action: np.ndarray
    target_xyz: np.ndarray
    gripper_target: float
    phase: str


def base_scene_name(scene: str) -> str:
    """Strip the seed suffix so correlated demos stay in the same split group."""
    return BASE_SCENE_RE.sub("", scene)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(dataset_root: Path) -> list[dict[str, Any]]:
    provenance = json.loads((dataset_root / "source-provenance.json").read_text(encoding="utf-8"))
    sources = provenance.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset source provenance has no sources")
    return sources


def load_policy_events(raw_dir: Path) -> list[dict[str, Any]]:
    """Return recorded waypoint target changes with their phase and time.

    Waypoint rows emitted before the first simulator step carry no ``sim_time_s``;
    they are the initial command and are anchored at t=0.
    """
    events: list[dict[str, Any]] = []
    previous: tuple[float, float, float] | None = None
    seen: set[tuple[float, tuple[float, ...]]] = set()
    with (raw_dir / "policy.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            target = row.get("commanded_ee_base_xyz_m")
            if target is None:
                continue
            key = tuple(round(float(value), 9) for value in target)
            if previous is not None and key == previous:
                continue
            previous = key
            t = float(row.get("sim_time_s", 0.0))
            dedupe = (round(t, 9), key)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            events.append(
                {
                    "t": t,
                    "target": [float(value) for value in target],
                    "phase": str(row.get("state", "UNKNOWN")),
                }
            )
    return events


def load_metadata_rows(dataset_root: Path) -> dict[int, list[dict[str, Any]]]:
    """Load only state/action/timestamp rows; image bytes are loaded in a second pass."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("V3 offline build requires pyarrow") from error
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet data under {dataset_root / 'data'}")
    for path in paths:
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
            ],
        )
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            rows_by_episode.setdefault(episode, []).append(row)
    for rows in rows_by_episode.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return rows_by_episode


def align_samples(
    sources: list[dict[str, Any]],
    rows_by_episode: dict[int, list[dict[str, Any]]],
) -> list[WaypointSample]:
    samples: list[WaypointSample] = []
    for episode_index, source in enumerate(sources):
        rows = rows_by_episode.get(episode_index)
        if not rows:
            continue
        scene = source["scene"]
        times = np.asarray([float(row["timestamp"]) for row in rows], dtype=float)
        for event in load_policy_events(Path(source["raw_dir"])):
            frame_index = int(np.argmin(np.abs(times - float(event["t"]))))
            row = rows[frame_index]
            action = np.asarray(row["action"], dtype=float)
            samples.append(
                WaypointSample(
                    episode_index=episode_index,
                    scene=str(scene["name"]),
                    base_scene=base_scene_name(str(scene["name"])),
                    family=str(scene.get("metadata", {}).get("family", "unknown")),
                    t=float(event["t"]),
                    frame_index=int(row["frame_index"]),
                    state=np.asarray(row["observation.state"], dtype=float),
                    action=action,
                    target_xyz=np.asarray(event["target"], dtype=float),
                    gripper_target=float(action[6]),
                    phase=str(event["phase"]),
                )
            )
    if not samples:
        raise ValueError("no waypoint samples aligned to dataset rows")
    return samples


def choose_validation_bases(samples: list[WaypointSample]) -> tuple[str, ...]:
    """Choose one base scene per family, deterministically, for the development split."""
    by_family: dict[str, list[str]] = {}
    for sample in samples:
        by_family.setdefault(sample.family, []).append(sample.base_scene)
    chosen = [sorted(set(values))[0] for _, values in sorted(by_family.items()) if values]
    return tuple(sorted(chosen))


def split_by_bases(
    samples: list[WaypointSample], validation_bases: tuple[str, ...]
) -> tuple[list[WaypointSample], list[WaypointSample]]:
    validation = set(validation_bases)
    train = [sample for sample in samples if sample.base_scene not in validation]
    held_out = [sample for sample in samples if sample.base_scene in validation]
    if not train or not held_out:
        raise ValueError("scene-grouped split must leave samples on both sides")
    return train, held_out


def loso_folds(
    samples: list[WaypointSample],
) -> list[tuple[str, list[WaypointSample], list[WaypointSample]]]:
    folds = []
    for base_scene in sorted({sample.base_scene for sample in samples}):
        train = [sample for sample in samples if sample.base_scene != base_scene]
        held_out = [sample for sample in samples if sample.base_scene == base_scene]
        folds.append((base_scene, train, held_out))
    return folds


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float = 1.0
) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("ridge features/targets must be 2-D and aligned")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    normalized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(normalized)), normalized])
    return design @ model["coefficients"]


def mean_abs_error(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(prediction) - np.asarray(target))
    axis = absolute.mean(axis=0)
    return {
        "overall_m": round(float(axis.mean()), 6),
        "x_m": round(float(axis[0]), 6),
        "y_m": round(float(axis[1]), 6),
        "z_m": round(float(axis[2]), 6),
    }


def image_feature(image_bytes: bytes, size: int = 16) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("image baselines require Pillow") from error
    with Image.open(BytesIO(image_bytes)) as image:
        gray = image.convert("L").resize((size, size))
        array = np.asarray(gray, dtype=np.float32) / 255.0
    return array.reshape(-1)


def load_image_bytes(
    dataset_root: Path, samples: list[WaypointSample]
) -> dict[tuple[int, int], bytes]:
    """Load the encoded JPEG bytes for each distinct (episode, frame) waypoint sample."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("V3 offline build requires pyarrow") from error
    selected: dict[int, set[int]] = {}
    for sample in samples:
        selected.setdefault(sample.episode_index, set()).add(sample.frame_index)
    encoded: dict[tuple[int, int], bytes] = {}
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.images.camera1"],
        )
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            if frame not in selected.get(episode, set()):
                continue
            image = row["observation.images.camera1"]
            if image is None:
                continue
            encoded[(episode, frame)] = image["bytes"]
    return encoded


def load_image_features(
    dataset_root: Path, samples: list[WaypointSample], size: int = 16
) -> dict[tuple[int, int], np.ndarray]:
    encoded = load_image_bytes(dataset_root, samples)
    return {key: image_feature(value, size=size) for key, value in encoded.items()}


def _matrix(samples: list[WaypointSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.stack([sample.state for sample in samples])
    actions = np.stack([sample.action for sample in samples])
    targets = np.stack([sample.target_xyz for sample in samples])
    gripper = np.asarray([sample.gripper_target for sample in samples], dtype=float)
    return states, actions, targets, gripper


def _ridge_result(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, Any]:
    model = fit_ridge(features[train_index], targets[train_index], alpha=alpha)
    return {
        "name": name,
        "train": mean_abs_error(predict_ridge(model, features[train_index]), targets[train_index]),
        "validation": mean_abs_error(
            predict_ridge(model, features[validation_index]), targets[validation_index]
        ),
    }


def _constant_result(
    train: np.ndarray, validation: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    prediction = targets[train].mean(axis=0)
    return {
        "name": "constant_train_mean",
        "train": mean_abs_error(np.broadcast_to(prediction, targets[train].shape), targets[train]),
        "validation": mean_abs_error(
            np.broadcast_to(prediction, targets[validation].shape), targets[validation]
        ),
    }


def _persistence_result(
    samples: list[WaypointSample],
    targets: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
) -> dict[str, Any]:
    by_episode: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        by_episode.setdefault(sample.episode_index, []).append(index)
    for indices in by_episode.values():
        indices.sort(key=lambda index: samples[index].t)
    previous: dict[int, int] = {}
    for indices in by_episode.values():
        for previous_index, current_index in zip(indices, indices[1:], strict=False):
            previous[current_index] = previous_index
    fallback = targets[train_index].mean(axis=0)

    def predict(indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                targets[previous[int(index)]] if int(index) in previous else fallback
                for index in indices
            ]
        )

    return {
        "name": "previous_waypoint_persistence",
        "train": mean_abs_error(predict(train_index), targets[train_index]),
        "validation": mean_abs_error(predict(validation_index), targets[validation_index]),
    }


def _gripper_summary(
    gripper: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> dict[str, Any]:
    values, counts = np.unique(np.round(gripper, 6), return_counts=True)
    majority = float(values[int(np.argmax(counts))])
    accuracy = float(np.mean(np.isclose(gripper[validation], majority, atol=1e-6)))
    return {
        "unique_targets_m": [float(value) for value in values],
        "counts": [int(count) for count in counts],
        "majority_target_m": majority,
        "validation_majority_accuracy": round(accuracy, 6),
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
    }


def _loso_summary(
    samples: list[WaypointSample],
    features: np.ndarray,
    targets: np.ndarray,
    folds: list[tuple[str, list[WaypointSample], list[WaypointSample]]],
    alpha: float = 1.0,
) -> dict[str, Any]:
    index_by_id = {id(sample): index for index, sample in enumerate(samples)}
    rows = []
    for base_scene, train, held_out in folds:
        train_index = np.asarray([index_by_id[id(sample)] for sample in train])
        validation_index = np.asarray([index_by_id[id(sample)] for sample in held_out])
        model = fit_ridge(features[train_index], targets[train_index], alpha=alpha)
        rows.append(
            {
                "base_scene": base_scene,
                "train_samples": int(len(train)),
                "validation_samples": int(len(held_out)),
                "train": mean_abs_error(
                    predict_ridge(model, features[train_index]), targets[train_index]
                ),
                "validation": mean_abs_error(
                    predict_ridge(model, features[validation_index]),
                    targets[validation_index],
                ),
            }
        )
    return {
        "folds": rows,
        "mean_validation_mae_m": round(
            float(np.mean([row["validation"]["overall_m"] for row in rows])), 6
        ),
    }


def _image_timing(
    dataset_root: Path, samples: list[WaypointSample], size: int, limit: int = 50
) -> dict[str, Any]:
    """Time JPEG decode plus grayscale resize per distinct waypoint frame."""
    encoded = load_image_bytes(dataset_root, samples[:limit])
    if not encoded:
        return {"status": "unavailable", "samples": 0}
    values = list(encoded.values())
    samples_ms: list[float] = []
    for value in values:
        started = perf_counter()
        image_feature(value, size=size)
        samples_ms.append((perf_counter() - started) * 1000.0)
    samples_ms.sort()
    return {
        "status": "measured",
        "samples": len(values),
        "mean_ms": round(float(np.mean(samples_ms)), 6),
        "p95_ms": round(samples_ms[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))], 6),
        "operation": "jpeg decode + grayscale resize + float normalise",
    }


def build_offline_evidence(
    dataset_root: Path,
    *,
    out_dir: Path,
    evidence_path: Path,
    image_size: int = 16,
    alpha: float = 1.0,
    encoder_iterations: int = 50,
) -> dict[str, Any]:
    sources = load_sources(dataset_root)
    rows = load_metadata_rows(dataset_root)
    samples = align_samples(sources, rows)
    validation_bases = choose_validation_bases(samples)
    train, validation = split_by_bases(samples, validation_bases)
    states, actions, targets, gripper = _matrix(samples)
    train_ids = {id(sample) for sample in train}
    validation_ids = {id(sample) for sample in validation}
    train_index = np.asarray(
        [index for index, sample in enumerate(samples) if id(sample) in train_ids]
    )
    validation_index = np.asarray(
        [index for index, sample in enumerate(samples) if id(sample) in validation_ids]
    )
    features = {
        "state": states,
        "state_action": np.column_stack([states, actions]),
    }
    image_features = load_image_features(dataset_root, samples, size=image_size)
    image_keys = [(sample.episode_index, sample.frame_index) for sample in samples]
    missing_images = sum(1 for key in image_keys if key not in image_features)
    if missing_images == 0:
        images = np.stack([image_features[key] for key in image_keys])
        features["image"] = images
        features["state_image"] = np.column_stack([states, images])
    baselines = [_constant_result(train_index, validation_index, targets)]
    for name in ("state", "state_action", "image", "state_image"):
        if name in features:
            baselines.append(
                _ridge_result(
                    f"{name}_ridge",
                    features[name],
                    targets,
                    train_index,
                    validation_index,
                    alpha=alpha,
                )
            )
    baselines.append(_persistence_result(samples, targets, train_index, validation_index))
    folds = loso_folds(samples)
    loso_features = features["state_action"]
    loso = _loso_summary(samples, loso_features, targets, folds, alpha=alpha)

    encoder = CompactEncoder(input_dim=image_size * image_size)
    encoder_results = benchmark_encoder(encoder, iterations=encoder_iterations)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "waypoint_samples.npz",
        state=states,
        action=actions,
        target_xyz=targets,
        gripper_target=gripper,
        episode_index=np.asarray([sample.episode_index for sample in samples]),
        frame_index=np.asarray([sample.frame_index for sample in samples]),
        t=np.asarray([sample.t for sample in samples]),
        phase=np.asarray([sample.phase for sample in samples]),
        base_scene=np.asarray([sample.base_scene for sample in samples]),
    )
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "measured-offline",
        "dataset": str(dataset_root),
        "source_provenance_sha256": sha256_file(dataset_root / "source-provenance.json"),
        "sample_count": len(samples),
        "episode_count": len({sample.episode_index for sample in samples}),
        "base_scene_count": len({sample.base_scene for sample in samples}),
        "base_scene_counts": dict(sorted(Counter(sample.base_scene for sample in samples).items())),
        "phase_counts": dict(sorted(Counter(sample.phase for sample in samples).items())),
        "development_split": {
            "validation_bases": list(validation_bases),
            "train_samples": len(train),
            "validation_samples": len(validation),
        },
        "baselines": baselines,
        "image_coverage": {
            "waypoint_samples": len(samples),
            "missing": int(missing_images),
            "image_baselines_available": "image" in features,
        },
        "loso_state_action_ridge": loso,
        "gripper": _gripper_summary(gripper, train_index, validation_index),
        "encoder": [
            {
                "input_dim": row.input_dim,
                "hidden_dim": row.hidden_dim,
                "output_dim": row.output_dim,
                "batch_size": row.batch_size,
                "iterations": row.iterations,
                "mean_ms": row.mean_ms,
                "p95_ms": row.p95_ms,
                "images_per_s": row.images_per_s,
            }
            for row in encoder_results
        ],
        "image_feature_timing": _image_timing(dataset_root, samples, image_size),
        "waypoint_samples_path": str(out_dir / "waypoint_samples.npz"),
        "notes": [
            "Waypoints come from recorded commanded_ee_base_xyz_m changes.",
            "Runtime expert phase labels and CAN ground truth are not model inputs.",
            "The development split is grouped by base scene identity.",
            "LOSO is a memorization check, not a holdout evaluation.",
        ],
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--encoder-iterations", type=int, default=50)
    args = parser.parse_args()
    result = build_offline_evidence(
        args.dataset,
        out_dir=args.out,
        evidence_path=args.evidence,
        image_size=args.image_size,
        alpha=args.alpha,
        encoder_iterations=args.encoder_iterations,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
