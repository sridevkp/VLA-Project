"""Drive-safe checkpoint discovery and guarded LeRobot training; no training on import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHECKPOINT_FILES = (
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors",
    "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "training_state/training_step.json",
    "training_state/optimizer_state.safetensors",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
    "training_state/rng_state.safetensors",
)


def mark_checkpoint_complete(checkpoint: Path) -> Path:
    """Replace LeRobot's optional `last` symlink with an audited JSON completion pointer."""
    if not checkpoint.name.isdigit():
        raise ValueError("checkpoint folder must be a numeric step")
    sizes = {}
    for relative in CHECKPOINT_FILES:
        path = checkpoint / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"incomplete checkpoint file: {path}")
        sizes[relative] = path.stat().st_size
    step = json.loads(
        (checkpoint / "training_state/training_step.json").read_text(encoding="utf-8")
    )["step"]
    if int(checkpoint.name) != step:
        raise ValueError("checkpoint step does not match directory")
    hashes = {}
    for relative in CHECKPOINT_FILES:
        with (checkpoint / relative).open("rb") as stream:
            hashes[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {"schema_version": 2, "step": step, "files_and_sizes": sizes, "files_sha256": hashes}
    saved_config = json.loads(
        (checkpoint / "pretrained_model/train_config.json").read_text(encoding="utf-8")
    )
    if "dataset" in saved_config:
        dataset_root = Path(saved_config["dataset"]["root"])
        record["dataset_identity"] = {
            name: hashlib.sha256((dataset_root / name).read_bytes()).hexdigest()
            for name in ["meta/stats.json", "source-provenance.json"]
        }
    complete = checkpoint / "COMPLETE.json"
    temporary_complete = complete.with_suffix(".tmp")
    temporary_complete.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary_complete.replace(complete)
    pointer = checkpoint.parent / "latest.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"checkpoint": checkpoint.name, "step": step}) + "\n", encoding="utf-8"
    )
    temporary.replace(pointer)
    return checkpoint


def latest_complete_checkpoint(output_dir: Path) -> Path | None:
    """Ignore partially written/disconnected checkpoints and all symlinks."""
    candidates = sorted(
        (p for p in (output_dir / "checkpoints").glob("*") if p.name.isdigit()),
        key=lambda p: int(p.name),
        reverse=True,
    )
    for path in candidates:
        if not path.name.isdigit() or path.is_symlink() or not (path / "COMPLETE.json").is_file():
            continue
        try:
            record = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("step") != int(path.name) or record.get("schema_version") != 2:
            continue
        if all(
            (path / name).is_file()
            and not (path / name).is_symlink()
            and (path / name).stat().st_size == record.get("files_and_sizes", {}).get(name)
            for name in CHECKPOINT_FILES
        ):
            hashes_match = True
            for name in CHECKPOINT_FILES:
                with (path / name).open("rb") as stream:
                    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_hash != record.get("files_sha256", {}).get(name):
                    hashes_match = False
                    break
            if hashes_match:
                return path
    return None


def validate_training_approval(dataset: Path) -> dict[str, Any]:
    """Require actual successful TRAIN data and its installed-policy/statistics validation."""
    approval = json.loads((dataset / "training_approval.json").read_text(encoding="utf-8"))
    if (
        approval.get("schema_verified") is not True
        or approval.get("installed_forward_verified") is not True
        or approval.get("installed_training_forward_verified") is not True
        or approval.get("native_tail_padding_mask_verified") is not True
        or approval.get("dataset_integrity_verified") is not True
        or approval.get("schema_only_not_demonstrations") is True
    ):
        raise ValueError("training requires schema and installed-policy verification")
    stats = dataset / "meta/stats.json"
    integrity_path = dataset / "dataset-integrity.json"
    if hashlib.sha256(integrity_path.read_bytes()).hexdigest() != approval.get(
        "dataset_integrity_sha256"
    ):
        raise ValueError("dataset integrity report changed since approval")
    from vlarobot.vla.dataset_integrity import dataset_file_hashes

    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (
        integrity.get("verified") is not True
        or dataset_file_hashes(dataset) != integrity["files_sha256"]
    ):
        raise ValueError("native dataset files changed since complete integrity verification")
    if hashlib.sha256(stats.read_bytes()).hexdigest() != approval.get("stats_sha256"):
        raise ValueError("dataset statistics changed since installed forward validation")
    if hashlib.sha256(
        (dataset / "source-provenance.json").read_bytes()
    ).hexdigest() != approval.get("source_provenance_sha256"):
        raise ValueError("audited dataset provenance changed since installed forward validation")
    episodes = approval.get("episodes", [])
    from vlarobot.vla.settings import load_vla_settings

    if len(episodes) < load_vla_settings().successful_demo_target:
        raise ValueError("successful demonstration target is not ready")
    for episode in episodes:
        if episode.get("split") != "train" or not all(
            episode.get(k) is True for k in ["success", "grasp", "lift", "transport", "placement"]
        ):
            raise ValueError("training approval contains failed, unverified or held-out data")
    return approval


def training_arguments(
    dataset: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    base_model: Path | None = None,
    save_freq: int = 1000,
) -> list[str]:
    """Produce real LeRobot arguments; resume uses the complete checkpoint config and state."""
    approval = validate_training_approval(dataset)
    checkpoint = latest_complete_checkpoint(output)
    if checkpoint is not None:
        marker = json.loads((checkpoint / "COMPLETE.json").read_text(encoding="utf-8"))
        if marker.get("dataset_identity") != {
            "meta/stats.json": approval["stats_sha256"],
            "source-provenance.json": approval["source_provenance_sha256"],
        }:
            raise ValueError("resume dataset fingerprint differs from the complete checkpoint")
        saved = json.loads(
            (checkpoint / "pretrained_model/train_config.json").read_text(encoding="utf-8")
        )
        if (
            saved["dataset"]["repo_id"] != approval["dataset_repo_id"]
            or Path(saved["dataset"]["root"]).resolve() != dataset.resolve()
        ):
            raise ValueError("resume dataset differs from the complete checkpoint")
        return [
            f"--config_path={checkpoint / 'pretrained_model/train_config.json'}",
            "--resume=true",
        ]
    if output.exists():
        raise FileExistsError("preserve incomplete run; choose a new versioned training output")
    input_features = {
        "observation.state": {"type": "STATE", "shape": [8]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    output_features = {"action": {"type": "ACTION", "shape": [7]}}
    return [
        f"--policy.path={base_model or 'lerobot/smolvla_base'}",
        "--policy.input_features=" + json.dumps(input_features),
        "--policy.output_features=" + json.dumps(output_features),
        "--policy.load_vlm_weights=false",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=" + approval["dataset_repo_id"],
        f"--dataset.root={dataset}",
        f"--output_dir={output}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--seed={seed}",
        f"--save_freq={save_freq}",
        "--wandb.enable=false",
    ]


def _launch_training_impl(
    dataset: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    save_freq: int = 1000,
) -> None:
    """Train approved data; replace only optional checkpoint-pointer creation."""
    import importlib
    import sys

    approval = validate_training_approval(dataset)
    # Resolve a pinned regular-file base snapshot outside the training output directory.
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "approved training requires the authorized CUDA environment; no GPU found"
        )
    from vlarobot.vla.smoke import MODEL_ID, MODEL_REVISION, VLM_ID, VLM_REVISION

    hub = importlib.import_module("huggingface_hub")
    base = output.parent / (f"smolvla-base-{MODEL_REVISION[:12]}-{approval['stats_sha256'][:12]}")
    for name in ["config.json", "model.safetensors"]:
        hub.hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=base)
    processor = Path(
        hub.snapshot_download(
            VLM_ID,
            revision=VLM_REVISION,
            local_dir=base / "processor",
            allow_patterns=["*.json", "*.txt", "*.model", "merges.*", "vocab.*"],
        )
    )
    prepare_training_processors(dataset, base, processor)
    args = training_arguments(
        dataset,
        output,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        base_model=base,
        save_freq=save_freq,
    )
    module = importlib.import_module("lerobot.scripts.lerobot_train")
    from vlarobot.vla.padding import install_padding_alias
    from vlarobot.vla.training_evidence import OptimizerStepEvidence

    original_factory = module.make_optimizer_and_scheduler

    def optimizer_with_evidence(config: Any, policy: Any) -> Any:
        optimizer, scheduler = original_factory(config, policy)
        resume_step = 0
        if config.resume:
            resume_step = json.loads(
                (config.checkpoint_path / "training_state/training_step.json").read_text(
                    encoding="utf-8"
                )
            )["step"]
        OptimizerStepEvidence(
            optimizer,
            output / "optimizer-steps.jsonl",
            start_step=resume_step,
            identity={
                "run_kind": "smolvla-finetune",
                "seed": seed,
                "stats_sha256": approval["stats_sha256"],
                "source_provenance_sha256": approval["source_provenance_sha256"],
                "approved_successful_episodes": len(approval["episodes"]),
            },
        )
        return optimizer, scheduler

    module.__dict__["make_optimizer_and_scheduler"] = optimizer_with_evidence

    install_padding_alias(
        importlib.import_module("lerobot.policies.smolvla.modeling_smolvla").SmolVLAPolicy
    )
    module.__dict__["update_last_checkpoint"] = mark_checkpoint_complete
    sys.argv = ["lerobot-train", *args]
    module.main()


def launch_approved_training(
    dataset: Path,
    output: Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    save_freq: int = 1000,
) -> None:
    """Resume complete state or preserve incomplete output; prevent duplicate GPU jobs."""
    from vlarobot.vla.training_runs import claim_training_run, select_training_output

    selected = select_training_output(output)
    lock = claim_training_run(selected)
    try:
        _launch_training_impl(
            dataset, selected, steps=steps, batch_size=batch_size, seed=seed, save_freq=save_freq
        )
    finally:
        lock.unlink(missing_ok=True)


def prepare_training_processors(dataset: Path, base: Path, processor: Path) -> None:
    """Save TIAGo processors required by the installed training factory's pretrained path."""
    from importlib import import_module

    from vlarobot.vla.schema import ACTION_NAMES, CAMERA_KEY, STATE_NAMES

    torch = import_module("torch")
    # Register the choice type in a fresh training process before generic config decoding.
    import_module("lerobot.policies.smolvla.configuration_smolvla")
    config = import_module("lerobot.configs.policies").PreTrainedConfig.from_pretrained(base)
    types = import_module("lerobot.configs.types")
    config.device = "cpu"  # training's factory overrides the processor device to CUDA.
    config.load_vlm_weights = False
    config.vlm_model_name = str(processor.resolve())
    config.input_features = {
        "observation.state": types.PolicyFeature(
            type=types.FeatureType.STATE, shape=(len(STATE_NAMES),)
        ),
        CAMERA_KEY: types.PolicyFeature(type=types.FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    config.output_features = {
        "action": types.PolicyFeature(type=types.FeatureType.ACTION, shape=(len(ACTION_NAMES),))
    }
    stats = json.loads((dataset / "meta/stats.json").read_text(encoding="utf-8"))
    tensor_stats = {
        key: {name: torch.as_tensor(value, dtype=torch.float32) for name, value in values.items()}
        for key, values in stats.items()
    }
    pre, post = import_module(
        "lerobot.policies.smolvla.processor_smolvla"
    ).make_smolvla_pre_post_processors(config, dataset_stats=tensor_stats)
    config.save_pretrained(base)
    pre.save_pretrained(base)
    post.save_pretrained(base)
