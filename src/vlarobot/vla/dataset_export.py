"""Command-derived VLA labels and optional native LeRobot export of audited TRAIN episodes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.core.types import Pose3D, Vec3
from vlarobot.vla.action_adapter import as_quaternion, pose_delta
from vlarobot.vla.encoded_images import embed_recorded_jpegs
from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES, observation_state


def derive_command_labels(
    arrays: dict[str, np.ndarray],
    commands: list[dict[str, Any]],
    forward: Callable[[dict[str, float]], Pose3D],
) -> np.ndarray:
    """Use accepted controller targets, never measured future motion or stall opening labels."""
    t = arrays["t"]
    names = arrays["joint_names"].tolist()
    active = dict(zip(names, map(float, arrays["joint_pos"][0]), strict=True))
    grip_target = 0.0  # reset's closed per-finger command (the first apply overrides this).
    labels = []
    index = 0
    previous_t = float("-inf")
    for row in commands:
        if row.get("accepted") is not True or not np.isfinite(row["t"]) or row["t"] < previous_t:
            raise ValueError("demonstrations require ordered, accepted safety-validated commands")
        previous_t = row["t"]
    for i, timestamp in enumerate(t):
        while index < len(commands) and commands[index]["t"] <= timestamp + 1e-9:
            command = commands[index]["action"]
            if "q" in command:
                active.update({k: float(v) for k, v in command["q"].items()})
            elif "opening" in command:
                grip_target = float(command["opening"])
            else:
                raise ValueError("unsupported expert command representation")
            index += 1
        source = Pose3D(Vec3(*map(float, arrays["ee_xyz"][i])), as_quaternion(arrays["ee_quat"][i]))
        target = forward(active)
        labels.append(np.r_[pose_delta(source, target), grip_target])
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape != (len(t), len(ACTION_NAMES)) or not np.all(np.isfinite(labels)):
        raise ValueError("derived command labels are not finite 7-D actions")
    return labels


def export_verified_dataset(
    collection: Path,
    out: Path,
    *,
    repo_id: str,
    forward: Callable[[dict[str, float]], Pose3D],
    reaudit: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Write a native image-based LeRobot dataset; full training approval is a later check."""
    from PIL import Image

    LeRobotDataset = import_module("lerobot.datasets.lerobot_dataset").LeRobotDataset

    if out.exists():
        raise FileExistsError("native datasets are versioned, never overwritten")
    provenance = json.loads((collection / "provenance.json").read_text(encoding="utf-8"))
    accepted = json.loads((collection / "accepted.json").read_text(encoding="utf-8"))["episodes"]
    if not accepted or provenance.get("measurement") != "measured":
        raise ValueError("no measured successful TRAIN episodes available")
    features = {
        CAMERA_KEY: {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": list(ACTION_NAMES)},
        "observation.finger_positions_m": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_m", "right_m"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=provenance["frame_hz"],
        features=features,
        root=out,
        robot_type="tiago",
        use_videos=False,
        video_backend="pyav",
    )
    source_rows = []
    anchors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    total_frames = 0
    try:
        for entry in accepted:
            raw = Path(entry["raw_dir"])
            audit = reaudit(raw / "episode-00.meta.json")
            if not audit["accepted"]:
                raise ValueError(f"episode failed independent re-audit: {raw}: {audit['reasons']}")
            with np.load(raw / "episode-00.steps.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            commands = [
                json.loads(line)
                for line in (raw / "episode-00.actions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            physical = [
                json.loads(line)
                for line in (raw / "episode-00.physical.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            labels = derive_command_labels(arrays, commands, forward)
            names = arrays["joint_names"].tolist()
            for i in range(len(labels)):
                image = np.array(
                    Image.open(raw / "frames/episode-00" / f"{i:06d}.jpg").convert("RGB")
                )
                if image.shape != (480, 640, 3):
                    raise ValueError("camera dimensions changed mid-dataset")
                joints = dict(zip(names, map(float, arrays["joint_pos"][i]), strict=True))
                state = observation_state(joints, float(arrays["gripper"][i]))
                if total_frames == 0 or i == len(labels) - 1:
                    # Only first and most recent last are needed for actual pixel/schema replay.
                    if total_frames != 0:
                        for old in list(anchors):
                            if old != 0:
                                del anchors[old]
                    anchors[total_frames] = (image.copy(), state.copy(), labels[i].copy())
                dataset.add_frame(
                    {
                        CAMERA_KEY: image,
                        "observation.state": state,
                        "observation.finger_positions_m": np.asarray(
                            physical[i]["finger_positions_m"], dtype=np.float32
                        ),
                        "action": labels[i],
                        "task": entry["instruction"],
                    }
                )
                total_frames += 1
            dataset.save_episode()
            source_rows.append(
                {
                    "attempt": entry["attempt"],
                    "seed": entry["seed"],
                    "scene": entry["scene"],
                    "raw_dir": str(raw),
                    "audit": audit,
                    "meta_sha256": hashlib.sha256(
                        (raw / "episode-00.meta.json").read_bytes()
                    ).hexdigest(),
                    "steps_sha256": hashlib.sha256(
                        (raw / "episode-00.steps.npz").read_bytes()
                    ).hexdigest(),
                    "commands_sha256": hashlib.sha256(
                        (raw / "episode-00.actions.jsonl").read_bytes()
                    ).hexdigest(),
                    "physical_sha256": hashlib.sha256(
                        (raw / "episode-00.physical.jsonl").read_bytes()
                    ).hexdigest(),
                    "policy_trace_sha256": hashlib.sha256(
                        (raw / "policy.jsonl").read_bytes()
                    ).hexdigest(),
                }
            )
    finally:
        dataset.finalize()
    image_storage = embed_recorded_jpegs(out, [Path(e["raw_dir"]) for e in accepted])
    loaded = LeRobotDataset(repo_id, root=out, video_backend="pyav")
    for i in [0, len(loaded) - 1]:
        sample = loaded[i]
        if (
            sample["observation.state"].shape != (8,)
            or sample["action"].shape != (7,)
            or sample[CAMERA_KEY].shape != (3, 480, 640)
        ):
            raise ValueError("installed LeRobot reload differs from the TIAGo schema")
        image, state, action = anchors[i]
        np.testing.assert_allclose(
            sample[CAMERA_KEY].permute(1, 2, 0).numpy() * 255, image, atol=1e-4, rtol=0
        )
        np.testing.assert_array_equal(sample["observation.state"].numpy(), state)
        np.testing.assert_array_equal(sample["action"].numpy(), action)
    report = {
        "status": "native LeRobot exported and reloaded",
        "episodes": loaded.num_episodes,
        "frames": len(loaded),
        "fps": loaded.fps,
        "repo_id": repo_id,
        "camera_key": CAMERA_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "sources": source_rows,
        "collection_provenance": provenance,
        "collection_provenance_sha256": hashlib.sha256(
            (collection / "provenance.json").read_bytes()
        ).hexdigest(),
        "label_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "first_last_image_state_action_roundtrip": True,
        "image_storage": image_storage,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["lerobot", "torch", "numpy", "ikpy", "scipy", "pyarrow"]
        },
        "training_approved": False,
        "reason": "real-dataset-statistics installed forward pass still required",
    }
    (out / "source-provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
