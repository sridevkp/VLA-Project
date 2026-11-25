"""Portable local-inference exports, including processor assets stored outside checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any

from vlarobot.vla.training import latest_complete_checkpoint


def export_complete_checkpoint(training: Path, out: Path) -> Path:
    """Copy the latest complete model and its pinned tokenizer/config metadata, no symlinks."""
    checkpoint = latest_complete_checkpoint(training)
    if checkpoint is None:
        raise ValueError("no complete checkpoint to export")
    if out.exists():
        raise FileExistsError("preserve existing export; choose a new version")
    model = checkpoint / "pretrained_model"
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    processor = Path(config["vlm_model_name"])
    if not processor.is_dir() or processor.is_symlink():
        raise ValueError("pinned local VLM processor metadata is unavailable")
    paths = [*model.rglob("*"), *processor.rglob("*")]
    if any(p.is_symlink() for p in paths):
        raise ValueError("portable checkpoint source must not contain symlinks")
    shutil.copytree(model, out)
    shutil.copytree(processor, out / "processor", ignore=shutil.ignore_patterns(".cache"))
    step_evidence = training / "optimizer-steps.jsonl"
    training_evidence = None
    if step_evidence.is_file():
        records = [
            json.loads(line) for line in step_evidence.read_text(encoding="utf-8").splitlines()
        ]
        real = [
            r
            for r in records
            if r.get("run_kind") == "smolvla-finetune"
            and r.get("actual_post_step_hook") is True
            and r.get("optimizer_step", 0) > 0
        ]
        if real:
            saved = json.loads((model / "train_config.json").read_text(encoding="utf-8"))
            dataset = Path(saved["dataset"]["root"])
            from vlarobot.vla.training import validate_training_approval

            validate_training_approval(dataset)
            shutil.copyfile(step_evidence, out / "optimizer-steps.jsonl")
            shutil.copyfile(dataset / "training_approval.json", out / "training-approval.json")
            training_evidence = real[-1]
    manifest = {
        "checkpoint_step": int(checkpoint.name),
        "training_evidence": training_evidence,
        "files_sha256": {
            str(p.relative_to(out).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*")
            if p.is_file()
        },
        "loader": "vlarobot.vla.checkpoint_export.load_portable_policy",
    }
    (out / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return out


def require_trained_export(export: Path) -> dict[str, Any]:
    """Require actual optimizer evidence and complete successful TRAIN approval for evaluation."""
    manifest = json.loads((export / "export-manifest.json").read_text(encoding="utf-8"))
    evidence = manifest.get("training_evidence") or {}
    if (
        manifest.get("checkpoint_step", 0) <= 0
        or evidence.get("run_kind") != "smolvla-finetune"
        or evidence.get("actual_post_step_hook") is not True
        or evidence.get("optimizer_step", 0) <= 0
    ):
        raise ValueError("trained VLA evaluation requires actual SmolVLA optimizer-step evidence")
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((export / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"trained export fingerprint mismatch: {name}")
    approval = json.loads((export / "training-approval.json").read_text(encoding="utf-8"))
    if (
        not approval.get("dataset_integrity_verified")
        or len(approval.get("episodes", [])) < 80
        or evidence["stats_sha256"] != approval["stats_sha256"]
        or evidence["source_provenance_sha256"] != approval["source_provenance_sha256"]
    ):
        raise ValueError("trained export differs from approved successful TRAIN data")
    return approval


def load_portable_policy(export: Path, *, device: str = "cpu") -> tuple[Any, Any, Any]:
    """Resolve asset paths in memory after moving the export between machines."""
    manifest = json.loads((export / "export-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((export / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"portable checkpoint fingerprint changed: {name}")
    policy_class = import_module("lerobot.policies.smolvla.modeling_smolvla").SmolVLAPolicy
    config = import_module("lerobot.configs.policies").PreTrainedConfig.from_pretrained(export)
    config.device = device
    config.load_vlm_weights = False
    config.vlm_model_name = str((export / "processor").resolve())
    config.pretrained_path = export
    policy = policy_class.from_pretrained(export, config=config, strict=True)
    pre, post = import_module("lerobot.policies.factory").make_pre_post_processors(
        config,
        pretrained_path=str(export),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": config.vlm_model_name},
        },
    )
    return policy, pre, post
