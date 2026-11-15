"""Verify approved Colab code/config and exclusion of diagnostic checkpoints without GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve verification evidence")
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    from vlarobot.vla.checkpoint_export import require_trained_export
    from vlarobot.vla.training import CHECKPOINT_FILES, mark_checkpoint_complete
    from vlarobot.vla.training_runs import select_training_output

    with zipfile.ZipFile(project / "runs/colab-prep-v3/source.zip") as archive:
        files = json.loads(archive.read("snapshot-manifest.json"))
    protected = [
        "configs/training.yaml",
        *[
            f"src/vlarobot/vla/{n}.py"
            for n in ("training", "training_evidence", "training_runs", "checkpoint_export")
        ],
    ]
    hashes = {}
    for name in protected:
        hashes[name] = hashlib.sha256((project / name).read_bytes()).hexdigest()
        if hashes[name] != files[name]:
            raise ValueError("approved production code/config changed: " + name)

    def checkpoint(folder, step):
        path = folder / "checkpoints" / f"{step:06d}"
        for name in CHECKPOINT_FILES:
            p = path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"namespace fixture only")
        (path / "training_state/training_step.json").write_text(
            json.dumps({"step": step}), encoding="utf-8"
        )
        (path / "pretrained_model/train_config.json").write_text("{}", encoding="utf-8")
        mark_checkpoint_complete(path)

    with tempfile.TemporaryDirectory(prefix="worldmind-production-namespace-") as directory:
        drive = Path(directory)
        production = drive / "training-tiago-v1"
        diagnostic = drive / "preflight/t4-v001"
        checkpoint(diagnostic, 20)
        assert select_training_output(production) == production and not production.exists()
        checkpoint(production, 1000)
        checkpoint(drive / "preflight/t4-v002", 20000)
        assert select_training_output(production) == production
        fixture = drive / "diagnostic-export"
        fixture.mkdir()
        (fixture / "export-manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint_step": 20,
                    "training_evidence": {
                        "run_kind": "smolvla-t4-preflight",
                        "actual_post_step_hook": True,
                        "optimizer_step": 20,
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            require_trained_export(fixture)
        except ValueError:
            pass
        else:
            raise AssertionError("diagnostic export was accepted as trained evidence")
    report = {
        "approved_production_sources_verified": hashes,
        "diagnostic_step20_excluded_from_fresh_production": True,
        "diagnostic_step20000_excluded_from_production_resume": True,
        "diagnostic_export_rejected": True,
        "toy_namespace_test_only": True,
        "expected_first_production_event": {
            "run_kind": "smolvla-finetune",
            "optimizer_step": 1,
            "restored_training_step": 0,
            "actual_post_step_hook": True,
        },
        "actual_first_production_event_verified": False,
        "authenticated_drive_files_verified": False,
        "gpu_optimizer_steps_executed_here": 0,
        "production_settings_changed": False,
    }
    args.out.mkdir(parents=True)
    (args.out / "production-namespace.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
