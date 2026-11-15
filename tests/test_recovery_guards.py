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


def link(alias, target):
    try:
        alias.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink fixture unavailable on this platform: {error}")


@pytest.mark.parametrize("shortcut", (False, True))
def test_real_colab_project_shortcut_layout_resolved_once(tmp_path, shortcut):
    drive = tmp_path / "content/drive"
    mydrive = drive / "MyDrive"
    mydrive.mkdir(parents=True)
    alias = mydrive / "vlarobot_smolvla"
    if shortcut:
        target = (
            drive / ".shortcut-targets-by-id/1xSPhmxF4hK0yTXPkPXGS8bziahI5LsVK/vlarobot_smolvla"
        )
        target.mkdir(parents=True)
        link(alias, target)
    else:
        alias.mkdir()
        target = alias
    original = target / "training-tiago-v1"
    saved = checkpoint(original, 17000)
    if shortcut:
        audit = guard.path_audit(alias / "training-tiago-v1/checkpoints/017000")
        assert audit["entries_leaf_first"][0]["is_symlink"] is False
        assert any(e.get("is_symlink") for e in audit["entries_leaf_first"])
        with pytest.raises(ValueError, match="ancestor"):
            guard.verify_checkpoint(alias / "training-tiago-v1/checkpoints/017000", REQUIRED)
    resolved = guard.resolve_project_root(alias, target)
    guard.require_canonical_path(resolved / "recovery/not-yet-created")
    assert resolved == target
    assert guard.verify_checkpoint(resolved / "training-tiago-v1/checkpoints/017000", REQUIRED)
    guard.require_isolated(resolved / "training-tiago-v1", resolved / "recovery/run-v2")
    assert guard.latest(resolved / "training-tiago-v1", REQUIRED) == saved


@pytest.mark.parametrize("relative", ("COMPLETE.json", REQUIRED[0], "extra-state.bin"))
def test_checkpoint_file_symlinks_rejected_even_with_same_hash(tmp_path, relative):
    saved = checkpoint(tmp_path / "production", 17000)
    outside = tmp_path / "outside.bin"
    outside.write_bytes((saved / relative).read_bytes())
    (saved / relative).unlink()
    link(saved / relative, outside)
    with pytest.raises(ValueError, match="symlink"):
        guard.verify_checkpoint(saved, REQUIRED)


@pytest.mark.parametrize("component", ("pretrained_model", "unlisted-child"))
def test_descendant_directory_symlinks_never_followed(tmp_path, component):
    saved = checkpoint(tmp_path / "production", 17000)
    outside = tmp_path / "outside"
    if (saved / component).exists():
        shutil.move(str(saved / component), outside)
    else:
        outside.mkdir()
    link(saved / component, outside)
    with pytest.raises(ValueError, match="symlink"):
        guard.require_canonical_path(saved / component / "new-output")
    with pytest.raises(ValueError, match="symlink"):
        guard.verify_checkpoint(saved, REQUIRED)


def test_checkpoint_root_symlink_rejected(tmp_path):
    saved = checkpoint(tmp_path / "production", 17000)
    other = tmp_path / "alias/017000"
    other.parent.mkdir()
    link(other, saved)
    with pytest.raises(ValueError, match="symlink"):
        guard.verify_checkpoint(other, REQUIRED)


@pytest.mark.parametrize("name", ("../outside", "/etc/passwd", "C:/escape", "a\\b", "", "."))
def test_manifest_traversal_absolute_and_empty_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe"):
        guard.regular(tmp_path, name)


def test_shortcut_must_match_confirmed_target_and_cannot_redirect_pinned_paths(tmp_path):
    first, other = tmp_path / "first", tmp_path / "other"
    first.mkdir()
    other.mkdir()
    alias = tmp_path / "project-shortcut"
    link(alias, first)
    with pytest.raises(ValueError, match="target"):
        guard.resolve_project_root(alias, other)
    canonical = guard.resolve_project_root(alias, first)
    saved = checkpoint(canonical / "production", 17000)
    alias.unlink()
    link(alias, other)
    assert guard.verify_checkpoint(saved, REQUIRED)


def test_isolation_uses_canonical_paths_for_alias_overlap(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    alias = tmp_path / "recovery-alias"
    link(alias, original)
    with pytest.raises(ValueError, match="overlap"):
        guard.require_isolated(original, alias / "nested")


def test_resolved_path_escape_is_rejected_even_after_component_checks(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="traversal"):
        guard.require_canonical_path(tmp_path / "../outside")
    with pytest.raises(ValueError, match="absolute"):
        guard.require_canonical_path(Path("relative-output"))
    leaf = tmp_path / "data.bin"
    leaf.write_bytes(b"original")
    outside = tmp_path.parent / "outside.bin"
    original_resolve = Path.resolve

    def raced_resolve(path, *args, **kwargs):
        if path == leaf:
            return outside
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", raced_resolve)
    with pytest.raises(ValueError, match="escapes"):
        guard.regular(tmp_path, "data.bin")
