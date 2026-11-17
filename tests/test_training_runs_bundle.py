"""Incomplete runs survive automatic restarts and source payloads remain versioned and bounded."""

import subprocess
import zipfile

from test_vla_training import make_checkpoint

from vlarobot.execution.colab_bundle import file_sha256, prepare_source_bundle
from vlarobot.vla.training import mark_checkpoint_complete
from vlarobot.vla.training_runs import select_training_output


def test_restart_selection_preserves_partial_runs_and_resumes_highest_valid_state(tmp_path):
    base = tmp_path / "training"
    assert select_training_output(base) == base
    base.mkdir()
    (base / "partial").write_bytes(b"preserve")
    restart = select_training_output(base)
    assert restart.name == "training.restart-0001"
    assert (base / "partial").read_bytes() == b"preserve"
    first = make_checkpoint(base, 1000)
    mark_checkpoint_complete(first)
    second = make_checkpoint(restart, 2000)
    mark_checkpoint_complete(second)
    assert select_training_output(base) == restart
    (second / "training_state/rng_state.safetensors").write_bytes(b"evil")
    assert select_training_output(base) == base


def test_source_bundle_excludes_trajectories_and_keeps_exact_source_bytes(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    (project / "src").mkdir()
    (project / "src/sample.py").write_text("x = 1\n", encoding="utf-8")
    (project / "runs").mkdir()
    (project / "runs/private-data").write_bytes(b"not source")
    out = tmp_path / "bundle"
    report = prepare_source_bundle(project, out)
    assert file_sha256(out / "source.zip") == report["source_zip_sha256"]
    assert report["training_started"] is False
    with zipfile.ZipFile(out / "source.zip") as archive:
        assert archive.read("src/sample.py") == b"x = 1\n"
        assert not any(name.startswith("runs/") for name in archive.namelist())
