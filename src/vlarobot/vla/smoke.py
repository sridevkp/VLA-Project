"""Actual pretrained SmolVLA inference with TIAGo features; never commands a robot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES, observation_state

MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "4cde9d208e19519c36917aa170fffc7e381aa0ad"


def run_installed_forward(out: Path, recording: Path, *, seed: int) -> dict[str, Any]:
    """Download pinned weights, strictly load, preprocess real RGB/state, and infer on CPU.

    Non-identity synthetic normalization stats deliberately exercise arithmetic. These are
    compatibility-test stats, never training statistics or a claim that the policy is trained.
    """
    from importlib import import_module

    torch = import_module("torch")
    hub = import_module("huggingface_hub")
    hf_hub_download, snapshot_download = hub.hf_hub_download, hub.snapshot_download
    PreTrainedConfig = import_module("lerobot.configs.policies").PreTrainedConfig
    types = import_module("lerobot.configs.types")
    FeatureType, PolicyFeature = types.FeatureType, types.PolicyFeature
    SmolVLAConfig = import_module("lerobot.policies.smolvla.configuration_smolvla").SmolVLAConfig
    SmolVLAPolicy = import_module("lerobot.policies.smolvla.modeling_smolvla").SmolVLAPolicy
    make_smolvla_pre_post_processors = import_module(
        "lerobot.policies.smolvla.processor_smolvla"
    ).make_smolvla_pre_post_processors
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError("Completed forward evidence must not be overwritten")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    started = time.perf_counter()
    model_dir = out / "base"
    for name in ["config.json", "model.safetensors"]:
        hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=model_dir)
    vlm_revision = VLM_REVISION
    processor_dir = Path(
        snapshot_download(
            VLM_ID,
            revision=vlm_revision,
            local_dir=out / "processor",
            allow_patterns=["*.json", "*.txt", "*.model", "merges.*", "vocab.*"],
        )
    )
    config = PreTrainedConfig.from_pretrained(model_dir)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("Downloaded policy configuration is not SmolVLA")
    published = {
        "input_features": str(config.input_features),
        "output_features": str(config.output_features),
    }
    config.device = "cpu"
    config.load_vlm_weights = False  # strict full SmolVLA load below includes backbone weights.
    config.vlm_model_name = str(processor_dir.resolve())
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(len(STATE_NAMES),)),
        CAMERA_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(len(ACTION_NAMES),))
    }
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=config, strict=True)
    loaded_s = time.perf_counter() - started
    prefix = recording / "episode-00"
    with np.load(prefix.with_suffix(".steps.npz"), allow_pickle=False) as samples:
        joint_pos = dict(zip(samples["joint_names"].tolist(), samples["joint_pos"][0], strict=True))
        state = observation_state(joint_pos, float(samples["gripper"][0]))
    frame = recording / "frames" / "episode-00" / "000000.jpg"
    rgb = np.array(Image.open(frame).convert("RGB"))
    if rgb.shape != (480, 640, 3):
        raise ValueError(f"Unexpected actual camera shape: {rgb.shape}")
    mean = torch.linspace(-0.2, 0.2, len(STATE_NAMES))
    std = torch.linspace(0.5, 1.2, len(STATE_NAMES))
    action_mean = torch.tensor([0, 0, 0, 0, 0, 0, 0.0225])
    action_std = torch.tensor([0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01])
    stats = {
        "observation.state": {"mean": mean, "std": std},
        "action": {"mean": action_mean, "std": action_std},
    }
    pre, post = make_smolvla_pre_post_processors(config, dataset_stats=stats)
    batch = pre(
        {
            "observation.state": torch.from_numpy(state),
            CAMERA_KEY: torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            "task": "Pick up the can and place it at the target location.",
        }
    )
    torch.testing.assert_close(
        batch["observation.state"], ((torch.from_numpy(state) - mean) / (std + 1e-8)).unsqueeze(0)
    )
    policy.reset()
    infer_started = time.perf_counter()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    infer_s = time.perf_counter() - infer_started
    assert tuple(chunk.shape) == (1, config.chunk_size, len(ACTION_NAMES))
    assert torch.isfinite(chunk).all()
    decoded = post(chunk[:, 0, :])
    torch.testing.assert_close(decoded, chunk[:, 0, :] * (action_std + 1e-8) + action_mean)
    assert not any(p.is_symlink() for p in out.rglob("*"))
    report = {
        "status": "MEASURED installed pretrained inference; untrained TIAGo behavior",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vlm_processor": VLM_ID,
        "vlm_processor_revision": vlm_revision,
        "seed": seed,
        "device": "cpu",
        "strict_weight_load": True,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["lerobot", "torch", "torchvision", "transformers", "torchcodec", "av"]
        },
        "published_features": published,
        "camera_key": CAMERA_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "input_shapes": {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")},
        "chunk_shape": list(chunk.shape),
        "decoded_action": decoded.tolist(),
        "state_normalization_verified": True,
        "action_unnormalization_verified": True,
        "stats_provenance": "synthetic non-identity smoke-test stats; training stats UNVERIFIED",
        "frame": str(frame),
        "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        "load_s": loaded_s,
        "inference_s": infer_s,
        "robot_commands_sent": 0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config.save_pretrained(out / "tiago-config")
    return report
