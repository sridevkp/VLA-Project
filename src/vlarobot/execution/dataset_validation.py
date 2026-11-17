"""Compose native export and measured installed-policy validation without sending actions."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from vlarobot.execution.demo_collection import audit_spec, write_json
from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES
from vlarobot.vla.settings import load_vla_settings


def export_collection(collection: Path, out: Path, *, repo_id: str) -> dict[str, Any]:
    """Use the exact registered expert geometry and re-audit every exported episode."""
    from vlarobot.backends.webots.backend import _rotation_to_quaternion
    from vlarobot.backends.webots.config import SETTINGS_PATH
    from vlarobot.backends.webots.kinematics import Kinematics
    from vlarobot.core.types import Pose3D, Vec3
    from vlarobot.eval.demo_audit import audit_demo
    from vlarobot.vla.dataset_export import export_verified_dataset

    provenance = json.loads((collection / "provenance.json").read_text(encoding="utf-8"))
    registration = SETTINGS_PATH.parent / "calibration/urdf-base-train-v1.yaml"
    if hashlib.sha256(registration.read_bytes()).hexdigest() != provenance["registration_sha256"]:
        raise ValueError("collection registration no longer matches export geometry")
    previous = os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE")
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = str(registration.resolve())
    try:
        kinematics = Kinematics()
    finally:
        if previous is None:
            os.environ.pop("VLAROBOT_URDF_REGISTRATION_FILE", None)
        else:
            os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = previous

    def forward(joints: dict[str, float]) -> Pose3D:
        import numpy as np

        q = np.zeros(len(kinematics.chain.links))
        for name, value in joints.items():
            q[kinematics.idx[name]] = value
        transform = kinematics.forward(q)
        return Pose3D(
            Vec3(*map(float, transform[:3, 3])),
            _rotation_to_quaternion(transform[:3, :3]),
        )

    return export_verified_dataset(
        collection,
        out,
        repo_id=repo_id,
        forward=forward,
        reaudit=lambda path: audit_demo(path, audit_spec(1)),
    )


def validate_native_forward(dataset: Path, smoke: Path, *, seed: int) -> dict[str, Any]:
    """Strictly load pinned weights and exercise actual dataset statistics and RGB/state."""
    torch = import_module("torch")
    LeRobotDataset = import_module("lerobot.datasets.lerobot_dataset").LeRobotDataset
    PreTrainedConfig = import_module("lerobot.configs.policies").PreTrainedConfig
    types = import_module("lerobot.configs.types")
    SmolVLAPolicy = import_module("lerobot.policies.smolvla.modeling_smolvla").SmolVLAPolicy
    processors = import_module("lerobot.policies.smolvla.processor_smolvla")
    source = json.loads((dataset / "source-provenance.json").read_text(encoding="utf-8"))
    smoke_report = json.loads((smoke / "report.json").read_text(encoding="utf-8"))
    if (dataset / "installed-forward.json").exists():
        raise FileExistsError("preserve completed native forward evidence")
    integrity_verified = False
    if not source.get("schema_only_not_demonstrations", False):
        from vlarobot.backends.webots.config import SETTINGS
        from vlarobot.vla.dataset_integrity import validate_dataset_integrity

        integrity = validate_dataset_integrity(
            dataset, gripper_limits_m=SETTINGS.gripper_opening_limits
        )
        write_json(dataset / "dataset-integrity.json", integrity)
        integrity_verified = True
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    loaded = LeRobotDataset(source["repo_id"], root=dataset, video_backend="pyav")
    if not source.get("schema_only_not_demonstrations", False):
        if loaded.num_episodes != len(source["sources"]):
            raise ValueError("native episode count differs from audited sources")
        for row in source["sources"]:
            if row["audit"].get("accepted") is not True or not all(
                row["audit"]["checks"].get(key) is True
                for key in ["grasp", "lift", "transport", "placement"]
            ):
                raise ValueError("native dataset includes an unverified physical source")
    config = PreTrainedConfig.from_pretrained(smoke / "base")
    config.device = "cpu"
    config.load_vlm_weights = False
    config.vlm_model_name = str((smoke / "processor").resolve())
    config.input_features = {
        "observation.state": types.PolicyFeature(type=types.FeatureType.STATE, shape=(8,)),
        CAMERA_KEY: types.PolicyFeature(type=types.FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    config.output_features = {
        "action": types.PolicyFeature(type=types.FeatureType.ACTION, shape=(7,))
    }
    policy = SmolVLAPolicy.from_pretrained(smoke / "base", config=config, strict=True)
    stats = {
        key: {name: torch.as_tensor(value, dtype=torch.float32) for name, value in values.items()}
        for key, values in loaded.meta.stats.items()
    }
    pre, post = processors.make_smolvla_pre_post_processors(config, dataset_stats=stats)
    sample = loaded[0]
    batch = pre({k: sample[k] for k in ["observation.state", CAMERA_KEY, "task"]})
    expected_state = (sample["observation.state"] - stats["observation.state"]["mean"]) / (
        stats["observation.state"]["std"] + 1e-8
    )
    torch.testing.assert_close(batch["observation.state"], expected_state.unsqueeze(0))
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    if tuple(chunk.shape) != (1, config.chunk_size, 7) or not torch.isfinite(chunk).all():
        raise ValueError("native data produced an invalid action chunk")
    decoded = post(chunk[:, 0])
    torch.testing.assert_close(
        decoded,
        chunk[:, 0] * (stats["action"]["std"] + 1e-8) + stats["action"]["mean"],
    )
    from vlarobot.vla.padding import install_padding_alias

    training_data = LeRobotDataset(
        source["repo_id"],
        root=dataset,
        video_backend="pyav",
        delta_timestamps={"action": [i / loaded.fps for i in range(config.chunk_size)]},
    )
    training_sample = training_data[len(training_data) - 1]
    training_batch = pre(
        torch.utils.data.default_collate(
            [
                {
                    k: training_sample[k]
                    for k in ["observation.state", CAMERA_KEY, "task", "action", "action_is_pad"]
                }
            ]
        )
    )
    install_padding_alias(SmolVLAPolicy)
    with torch.inference_mode():
        training_loss, losses = policy.forward(training_batch)
    if not torch.isfinite(training_loss):
        raise ValueError("installed training forward produced a nonfinite loss")
    masked = losses["losses_after_in_ep_bound"]
    if not torch.all(masked[training_batch["action_is_pad"]] == 0):
        raise ValueError("native tail-padding mask was not applied by installed training forward")
    report = {
        "schema_verified": True,
        "installed_forward_verified": True,
        "dataset_integrity_verified": integrity_verified,
        "dataset_integrity_sha256": (
            hashlib.sha256((dataset / "dataset-integrity.json").read_bytes()).hexdigest()
            if integrity_verified
            else None
        ),
        "stats_sha256": hashlib.sha256((dataset / "meta/stats.json").read_bytes()).hexdigest(),
        "dataset_repo_id": source["repo_id"],
        "source_provenance_sha256": hashlib.sha256(
            (dataset / "source-provenance.json").read_bytes()
        ).hexdigest(),
        "seed": seed,
        "model_revision": smoke_report["model_revision"],
        "camera_key": CAMERA_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "chunk_shape": list(chunk.shape),
        "robot_commands_sent": 0,
        "installed_training_forward_verified": True,
        "training_loss": float(training_loss),
        "native_tail_padding_mask_verified": True,
        "optimizer_steps": 0,
        "schema_only_not_demonstrations": source.get("schema_only_not_demonstrations", False),
        "target_successes": load_vla_settings().successful_demo_target,
        "episode_frame_counts": integrity["episode_frame_counts"] if integrity_verified else [],
        "episodes": [
            {
                "split": "train",
                "success": True,
                **row["audit"]["checks"],
                "evidence_path": row["raw_dir"],
                "meta_sha256": row["meta_sha256"],
            }
            for row in source["sources"]
        ],
    }
    write_json(dataset / "installed-forward.json", report)
    if (
        not report["schema_only_not_demonstrations"]
        and len(report["episodes"]) >= report["target_successes"]
    ):
        write_json(dataset / "training_approval.json", report)
    return report
