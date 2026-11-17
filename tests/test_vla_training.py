"""Resumption never mistakes a partial Drive checkpoint for a complete optimizer/RNG state."""

import json

import pytest

from vlarobot.vla.training import (
    CHECKPOINT_FILES,
    latest_complete_checkpoint,
    mark_checkpoint_complete,
    training_arguments,
)


def make_checkpoint(root, step):
    checkpoint = root / "checkpoints" / f"{step:06d}"
    for name in CHECKPOINT_FILES:
        p = checkpoint / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"test")
    (checkpoint / "training_state/training_step.json").write_text(
        json.dumps({"step": step}), encoding="utf-8"
    )
    (checkpoint / "pretrained_model/train_config.json").write_text("{}", encoding="utf-8")
    return checkpoint


def test_numbered_checkpoint_resume_skips_partial_newer_save(tmp_path):
    complete = make_checkpoint(tmp_path, 1000)
    mark_checkpoint_complete(complete)
    incomplete = make_checkpoint(tmp_path, 2000)
    (incomplete / "training_state/optimizer_state.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="incomplete"):
        mark_checkpoint_complete(incomplete)
    assert latest_complete_checkpoint(tmp_path) == complete
    assert not (tmp_path / "checkpoints/last").exists()
    assert (
        json.loads((tmp_path / "checkpoints/latest.json").read_text(encoding="utf-8"))["step"]
        == 1000
    )
    (complete / "training_state/rng_state.safetensors").write_bytes(b"evil")
    assert latest_complete_checkpoint(tmp_path) is None


def test_training_does_not_start_from_notebook_or_config_alone(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "training_approval.json").write_text(
        json.dumps({"schema_verified": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="installed-policy"):
        training_arguments(dataset, tmp_path / "out", steps=20000, batch_size=8, seed=0)
