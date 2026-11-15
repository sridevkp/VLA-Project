"""Missing, modified and unlisted assets never qualify as recoverable state."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "checkpoint_guard", Path(__file__).parents[1] / "tools/recovery/checkpoint_guard.py"
)
assert spec is not None and spec.loader is not None
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
REQUIRED = (
    "pretrained_model/model.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/training_step.json",
)


def checkpoint(root, step):
    path = root / "checkpoints" / f"{step:06d}"
    for name in (*REQUIRED, "extra-state.bin"):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"original state")
    (path / REQUIRED[2]).write_text(json.dumps({"step": step}), encoding="utf-8")
    files = {n: (path / n).stat().st_size for n in (*REQUIRED, "extra-state.bin")}
    record = {
        "schema_version": 2,
        "step": step,
        "files_and_sizes": files,
        "files_sha256": {n: guard.sha256(path / n) for n in files},
    }
    (path / "COMPLETE.json").write_text(json.dumps(record), encoding="utf-8")
    return path


def test_reject_three_stale_complete_markers_and_select_17000(tmp_path):
    good = checkpoint(tmp_path, 17000)
    for step, missing in ((18000, (REQUIRED[0],)), (19000, REQUIRED[:2]), (20000, REQUIRED[:2])):
        path = checkpoint(tmp_path, step)
        for name in missing:
            (path / name).unlink()
        with pytest.raises(ValueError, match="missing"):
            guard.verify_checkpoint(path, REQUIRED)
    assert guard.latest(tmp_path, REQUIRED) == good


@pytest.mark.parametrize("name", (*REQUIRED[:2], "extra-state.bin"))
def test_all_manifest_files_hashed_including_nonstandard_entries(tmp_path, name):
    path = checkpoint(tmp_path, 17000)
    original = (path / name).read_bytes()
    (path / name).write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="hash mismatch"):
        guard.verify_checkpoint(path, REQUIRED)
    assert guard.latest(tmp_path, REQUIRED) is None


@pytest.mark.parametrize("listed", (True, False))
def test_export_missing_model_fails_even_when_manifest_omits_it(tmp_path, listed):
    record = {"checkpoint_step": 20000, "files_sha256": {}}
    if listed:
        record["files_sha256"]["model.safetensors"] = "0" * 64
    (tmp_path / "export-manifest.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="missing|omits"):
        guard.verify_export(tmp_path, REQUIRED, 20000)


@pytest.mark.parametrize("failure", ("copied_model_missing", "source_optimizer_missing"))
def test_export_wrapper_readback_rejects_post_copy_losses(tmp_path, monkeypatch, failure):
    source = checkpoint(tmp_path / "training", 20000)

    def original_export(output, out):
        shutil.copytree(source / "pretrained_model", out)
        hashes = {p.name: guard.sha256(p) for p in out.iterdir()}
        (out / "export-manifest.json").write_text(
            json.dumps({"checkpoint_step": 20000, "files_sha256": hashes}), encoding="utf-8"
        )
        if failure == "copied_model_missing":
            (out / "model.safetensors").unlink()
        else:
            (source / "training_state/optimizer_state.safetensors").unlink()
        return out

    training = SimpleNamespace(CHECKPOINT_FILES=REQUIRED, mark_checkpoint_complete=lambda p: p)
    exporter = SimpleNamespace(
        export_complete_checkpoint=original_export, require_trained_export=lambda p: None
    )
    monkeypatch.setitem(
        sys.modules, "vlarobot.vla", SimpleNamespace(training=training, checkpoint_export=exporter)
    )
    guard.install_guards()
    with pytest.raises(ValueError, match="missing"):
        exporter.export_complete_checkpoint(tmp_path / "training", tmp_path / "export")


@pytest.mark.parametrize("relative", ("production", "production/recovery", "."))
def test_original_directory_and_overlapping_recovery_are_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="overlap"):
        guard.require_isolated(tmp_path / "production", tmp_path / relative)


def test_separate_recovery_directory_is_allowed(tmp_path):
    guard.require_isolated(tmp_path / "production", tmp_path / "recovery/run-v1")
