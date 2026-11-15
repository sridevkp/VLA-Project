"""Explicit read-only audit, isolated preparation, and opt-in recovery execution."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from checkpoint_guard import (
    install_guards,
    latest,
    require_isolated,
    sha256,
    verify_checkpoint,
    verify_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "prepare", "restore-check", "train", "export"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--execute-recovery", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve() / "src"))
    from vlarobot.vla import training

    install_guards()
    required = training.CHECKPOINT_FILES
    source = args.original / "checkpoints/017000"
    require_isolated(args.original, args.recovery)
    if args.mode == "audit":
        results = {}
        for number in (17000, 18000, 19000, 20000):
            path = args.original / "checkpoints" / f"{number:06d}"
            try:
                record = verify_checkpoint(path, required)
                results[str(number)] = {"verified": True, "files": len(record["files_sha256"])}
            except (OSError, ValueError, KeyError, TypeError) as error:
                results[str(number)] = {"verified": False, "reason": str(error)}
        selected = latest(args.original, required)
        print(json.dumps({"checkpoints": results, "selected": str(selected)}, indent=2))
        return
    if args.mode == "prepare":
        record = verify_checkpoint(source, required)
        if latest(args.original, required) != source:
            raise ValueError(
                "verified latest checkpoint is not 017000; investigate before recovery"
            )
        approval = training.validate_training_approval(args.dataset)
        archive = args.original.parent / "source.zip"
        with zipfile.ZipFile(archive) as bundle:
            approved_sources = json.loads(bundle.read("snapshot-manifest.json"))
        source_hashes = {}
        for name in (
            "configs/training.yaml",
            "src/vlarobot/vla/training.py",
            "src/vlarobot/vla/training_evidence.py",
            "src/vlarobot/vla/training_runs.py",
            "src/vlarobot/vla/checkpoint_export.py",
        ):
            source_hashes[name] = sha256(args.project / name)
            if source_hashes[name] != approved_sources[name]:
                raise ValueError(f"approved training source changed: {name}")
        config = json.loads((source / "pretrained_model/train_config.json").read_text())
        for key, expected in {
            "steps": 20000,
            "batch_size": 8,
            "seed": 0,
            "save_freq": 1000,
        }.items():
            if config.get(key) != expected:
                raise ValueError(f"saved frozen setting differs: {key}")
        # Also enforces saved dataset paths/repo and COMPLETE dataset fingerprints.
        training.training_arguments(args.dataset, args.original, steps=20000, batch_size=8, seed=0)
        if args.recovery.exists():
            raise FileExistsError("preserve existing recovery output")
        destination = args.recovery / "checkpoints/017000"
        shutil.copytree(source, destination, symlinks=False)
        verify_checkpoint(destination, required)
        if sha256(source / "COMPLETE.json") != sha256(destination / "COMPLETE.json"):
            raise ValueError("copied checkpoint manifest differs")
        provenance = {
            "original": str(args.original.resolve()),
            "source_checkpoint": str(source.resolve()),
            "source_complete_sha256": sha256(source / "COMPLETE.json"),
            "approved_source_hashes": source_hashes,
            "approved_source_zip_sha256": sha256(archive),
            "source_files_sha256": record["files_sha256"],
            "stats_sha256": approval["stats_sha256"],
            "source_provenance_sha256": approval["source_provenance_sha256"],
            "restored_step": 17000,
            "target_step": 20000,
            "additional_updates": 3000,
            "original_log_sha256": sha256(args.original / "optimizer-steps.jsonl"),
            "dataloader_cursor_restored": False,
            "helper_hashes": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        }
        (args.recovery / "recovery-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(provenance, indent=2))
        return
    provenance = json.loads((args.recovery / "recovery-provenance.json").read_text())
    if provenance["source_complete_sha256"] != sha256(source / "COMPLETE.json"):
        raise ValueError("original recovery evidence changed")
    for name, digest in provenance["approved_source_hashes"].items():
        if sha256(args.project / name) != digest:
            raise ValueError(f"approved source changed since recovery preparation: {name}")
    for name, digest in provenance["helper_hashes"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"recovery helper changed since preparation: {name}")
    if args.mode in ("restore-check", "train"):
        if args.mode == "train" and not args.execute_recovery:
            raise ValueError("no training started: explicit --execute-recovery required")
        if args.mode == "train":
            restored = json.loads((args.recovery / "restore-check.json").read_text())
            if (
                restored.get("restored_step") != 17000
                or restored.get("device") != "cuda"
                or restored.get("checkpoint_complete_sha256")
                != sha256(args.recovery / "checkpoints/017000/COMPLETE.json")
            ):
                raise ValueError("actual CUDA restoration must pass before recovery training")
        from restore_gate import RestoreVerified, install_restore_gate

        install_restore_gate(
            args.original,
            args.recovery,
            args.dataset,
            stop_after_restore=args.mode == "restore-check",
        )
        original_arguments = training.training_arguments

        def recovery_arguments(dataset: Path, output: Path, **kwargs: object) -> list[str]:
            checkpoint = latest(output, required)
            if checkpoint is None or not 17000 <= int(checkpoint.name) < 20000:
                raise ValueError("recovery requires valid state in [17000,20000)")
            config = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
            for key, value in {
                "steps": 20000,
                "batch_size": 8,
                "seed": 0,
                "save_freq": 1000,
            }.items():
                if config.get(key) != value:
                    raise ValueError(f"frozen configuration changed: {key}")
            result = original_arguments(dataset, output, **kwargs)
            return [*result, f"--output_dir={output}"]

        training.training_arguments = recovery_arguments
        try:
            training.launch_approved_training(
                args.dataset, args.recovery, steps=20000, batch_size=8, seed=0, save_freq=1000
            )
        except RestoreVerified:
            if args.mode != "restore-check":
                raise
            print("RESTORE_CHECK_COMPLETE: no optimizer update executed", flush=True)
            return
    if args.mode in ("train", "export"):
        final = args.recovery / "checkpoints/020000"
        verify_checkpoint(final, required)
        rows = [
            json.loads(s)
            for s in (args.recovery / "optimizer-steps.jsonl").read_text().splitlines()
        ]
        if not rows or rows[0]["optimizer_step"] != 17001 or rows[-1]["optimizer_step"] != 20000:
            raise ValueError("unexpected recovery update boundaries")
        # Each invocation's final record supplies its actual number of optimizer calls.
        # With cadence 1000, every interrupted invocation must also have its final
        # progress retained; do not infer an exact total across undocumented retries.
        if (
            rows[-1]["restored_training_step"] != 17000
            or rows[-1]["actual_calls_this_invocation"] != 3000
        ):
            raise ValueError("multi-invocation recovery requires separate update-count audit")
        if args.mode == "export":
            if args.export is None:
                raise ValueError("--export required")
            from vlarobot.vla.checkpoint_export import export_complete_checkpoint

            export_complete_checkpoint(args.recovery, args.export)
            for source_file, target_name in (
                (args.recovery / "recovery-provenance.json", "recovery-provenance.json"),
                (source / "COMPLETE.json", "recovery-source-COMPLETE.json"),
                (final / "COMPLETE.json", "recovery-final-COMPLETE.json"),
            ):
                shutil.copyfile(source_file, args.export / target_name)
            manifest_path = args.export / "export-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name in (
                "recovery-provenance.json",
                "recovery-source-COMPLETE.json",
                "recovery-final-COMPLETE.json",
            ):
                manifest["files_sha256"][name] = sha256(args.export / name)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            verify_export(args.export, required, 20000)
            print(
                "Final checkpoint and portable export verified; transfer must be checked separately"
            )


if __name__ == "__main__":
    main()
