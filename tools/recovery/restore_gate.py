"""Verify actual installed training restoration before any policy update."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkpoint_guard import require_isolated, sha256, verify_checkpoint


class RestoreVerified(Exception):
    """Intentional stop after full restoration, before training/data iteration."""


def equal_tensors(actual: dict[str, Any], saved: Path) -> int:
    import torch
    from safetensors import safe_open

    with safe_open(saved, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        aliases = handle.metadata() or {}
        if not keys.issubset(actual):
            raise ValueError(f"restored state has missing keys: {saved.name}")
        for name in keys:
            expected = handle.get_tensor(name)
            observed = actual[name].detach().cpu()
            if observed.dtype != expected.dtype or not torch.equal(observed, expected):
                raise ValueError(f"restored tensor mismatch: {saved.name}:{name}")
        for name in set(actual) - keys:
            target = aliases.get(name)
            if target not in keys or not torch.equal(actual[name], actual[target]):
                raise ValueError(f"unverified extra/alias tensor: {saved.name}:{name}")
    return len(keys)


def json_form(value: Any) -> Any:
    return json.loads(json.dumps(value))


def install_restore_gate(
    original: Path,
    recovery: Path,
    dataset: Path,
    *,
    stop_after_restore: bool,
    processor: Path | None = None,
) -> None:
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.utils import flatten_dict
    from lerobot.scripts import lerobot_train as module
    from lerobot.utils.random_utils import serialize_rng_state

    from vlarobot.vla.training import CHECKPOINT_FILES

    require_isolated(original, recovery)
    checkpoint = recovery / "checkpoints/017000"
    verify_checkpoint(checkpoint, CHECKPOINT_FILES)
    original_validate = TrainPipelineConfig.validate
    original_make = module.make_policy
    original_load = module.load_training_state
    original_save = module.save_checkpoint
    original_update = module.update_policy
    original_processors = module.make_pre_post_processors
    captured: dict[str, Any] = {"passed": False}

    def validate(config: Any) -> None:
        original_validate(config)
        if (
            not config.resume
            or Path(config.output_dir).resolve() != recovery.resolve()
            or Path(config.checkpoint_path).resolve() != checkpoint.resolve()
            or Path(config.dataset.root).resolve() != dataset.resolve()
        ):
            raise ValueError("recovery output/checkpoint/dataset redirection rejected")
        for name, expected in {
            "steps": 20000,
            "batch_size": 8,
            "seed": 0,
            "save_freq": 1000,
        }.items():
            if getattr(config, name) != expected:
                raise ValueError(f"frozen runtime setting changed: {name}")
        if processor is not None:
            from account_migration import relocate_policy

            relocate_policy(config.policy, processor)

    def processors(*args: Any, **kwargs: Any) -> Any:
        if processor is not None:
            from account_migration import processor_overrides

            kwargs = processor_overrides(kwargs, processor)
        return original_processors(*args, **kwargs)

    def make(*args: Any, **kwargs: Any) -> Any:
        policy = original_make(*args, **kwargs)
        if type(policy).__name__ != "SmolVLAPolicy":
            raise ValueError("restore gate requires actual installed SmolVLA")
        captured["policy"] = policy
        return policy

    def load(path: Path, optimizer: Any, scheduler: Any) -> Any:
        if Path(path).resolve() != checkpoint.resolve() or not isinstance(
            optimizer, torch.optim.AdamW
        ):
            raise ValueError("unexpected checkpoint or optimizer in restore")
        result = original_load(path, optimizer, scheduler)
        step, optimizer, scheduler = result
        if step != 17000 or scheduler is None or not torch.cuda.is_available():
            raise ValueError("restore gate requires step17000, scheduler and CUDA")
        policy = captured["policy"]
        if next(policy.parameters()).device.type != "cuda":
            raise ValueError("actual SmolVLA is not on CUDA")
        state = checkpoint / "training_state"
        # Compare RNG before any verification operation could consume randomness.
        rng_count = equal_tensors(serialize_rng_state(), state / "rng_state.safetensors")
        model_count = equal_tensors(
            policy.state_dict(), checkpoint / "pretrained_model/model.safetensors"
        )
        optimizer_state = optimizer.state_dict()
        optimizer_count = equal_tensors(
            flatten_dict({"state": optimizer_state["state"]}), state / "optimizer_state.safetensors"
        )
        if json_form(optimizer_state["param_groups"]) != json.loads(
            (state / "optimizer_param_groups.json").read_text()
        ):
            raise ValueError("restored AdamW parameter groups differ")
        if json_form(scheduler.state_dict()) != json.loads(
            (state / "scheduler_state.json").read_text()
        ):
            raise ValueError("restored scheduler differs")
        record = verify_checkpoint(checkpoint, CHECKPOINT_FILES)
        report = {
            "actual_installed_smolvla": True,
            "device": "cuda",
            "restored_step": step,
            "model_tensor_values_verified": model_count,
            "optimizer_tensor_values_verified": optimizer_count,
            "rng_tensors_verified": rng_count,
            "adamw_parameter_groups_verified": True,
            "scheduler_verified": True,
            "all_checkpoint_files_verified": len(record["files_sha256"]),
            "checkpoint_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
            "output_directory": str(recovery.resolve()),
            "original_directory_write_protection": True,
            "optimizer_updates_at_restore_gate": 0,
            "dataloader_cursor_restored": False,
            "relocated_processor_directory": str(processor) if processor else None,
        }
        captured["passed"] = True
        filename = "restore-check.json" if stop_after_restore else "training-restore-check.json"
        (recovery / filename).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("RESTORE_GATE_PASSED " + json.dumps(report), flush=True)
        if stop_after_restore:
            raise RestoreVerified()
        return result

    def update(*args: Any, **kwargs: Any) -> Any:
        if not captured["passed"]:
            raise ValueError("optimizer update vetoed: restoration not verified")
        return original_update(*args, **kwargs)

    def save(*args: Any, **kwargs: Any) -> Any:
        destination = Path(kwargs["checkpoint_dir"]).resolve()
        if recovery.resolve() / "checkpoints" not in destination.parents:
            raise ValueError("checkpoint write outside recovery directory vetoed")
        return original_save(*args, **kwargs)

    TrainPipelineConfig.validate = validate
    module.make_policy = make
    module.load_training_state = load
    module.update_policy = update
    module.save_checkpoint = save
    module.make_pre_post_processors = processors
