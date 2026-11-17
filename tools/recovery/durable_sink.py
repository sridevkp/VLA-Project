"""Fail-closed verified checkpoint handoff on the frozen save cadence."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from checkpoint_guard import sha256, verify_checkpoint, verify_export
from durable_transfer import identifier, send, ssh_command


def archive_tree(folder: Path, output: Path, *, extras: dict[str, Path] | None = None) -> Path:
    if output.exists() or folder.is_symlink():
        raise ValueError("preserve existing archive; source must not be a symlink")
    files = [
        (p.relative_to(folder).as_posix(), p) for p in sorted(folder.rglob("*")) if p.is_file()
    ]
    if any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError("archive source contains symlink")
    for name, path in (extras or {}).items():
        if not name.startswith("recovery-evidence/") or ".." in Path(name).parts:
            raise ValueError("invalid evidence archive path")
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or symlinked recovery evidence")
        files.append((name, path))
    expected = {}
    for manifest_name in ("export-manifest.json", "COMPLETE.json"):
        if (folder / manifest_name).is_file():
            record = json.loads((folder / manifest_name).read_text(encoding="utf-8"))
            expected.update(record.get("files_sha256", {}))
    names = [n for n, _ in files]
    if len(set(names)) != len(names) or not set(expected).issubset(names):
        raise ValueError("archive would omit/duplicate manifest-listed files")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, path in files:
            archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC mismatch")
        for name, path in files:
            with archive.open(name) as stream:
                import hashlib

                actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != sha256(path) or actual != expected.get(name, actual):
                    raise ValueError("archive differs from source")
    return output


def validate_canary(config_file: Path) -> dict[str, Any]:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("transport") == "drive_api":
        from drive_handoff import validate_canary as validate_drive

        return validate_drive(config_file)
    identifier(config["run_id"] + "-checkpoint-020000-attempt-3")
    identifier(config["run_id"] + "-portable-020000-attempt-3")
    ssh_command(config)
    receipt = json.loads(Path(config["canary_receipt"]).read_text(encoding="utf-8"))
    if (
        receipt.get("receiver_readback_verified") is not True
        or receipt.get("file_fsync") is not True
        or receipt.get("directory_fsync") is not True
        or not Path(receipt["receiver_path"]).is_absolute()
        or receipt.get("sha256") != config["canary_sha256"]
        or receipt.get("bytes", 0) < 1024 * 1024
        or receipt.get("transfer_config_sha256") != sha256(config_file)
        or not Path(receipt["receiver_path"]).is_relative_to(Path(config["remote_root"]))
    ):
        raise ValueError("independent storage canary/config receipt required before training")
    return config


def install_sink(recovery: Path, config_file: Path, *, local_root: Path = Path("/content")) -> None:
    """Block the optimizer until the selected persistence backend verifies the save."""
    from vlarobot.vla import checkpoint_export, training

    config = validate_canary(config_file)
    recovery = recovery.resolve()
    if not recovery.is_relative_to(local_root) or recovery.is_relative_to(local_root / "drive"):
        raise ValueError("durable Colab training must write under /content, outside Drive")
    original_mark = training.mark_checkpoint_complete
    required = training.CHECKPOINT_FILES
    initial = verify_checkpoint(recovery / "checkpoints/017000", required)
    initial_bytes = sum(initial["files_and_sizes"].values())
    model_bytes = initial["files_and_sizes"]["pretrained_model/model.safetensors"]
    local_required = 8 * initial_bytes + 3 * model_bytes
    if shutil.disk_usage(recovery).free < local_required:
        raise ValueError("insufficient Colab-local headroom for frozen saves/archives/wire files")
    drive_mode = config.get("transport") == "drive_api"
    if drive_mode:
        from drive_handoff import DriveAPI
        from drive_handoff import send as send_drive

        DriveAPI(config).capacity(3 * initial_bytes + model_bytes + 64 * 1024**2)
        sender = send_drive
    else:
        receiver_required = 12 * initial_bytes + 4 * model_bytes
        canary = json.loads(Path(config["canary_receipt"]).read_text(encoding="utf-8"))
        if canary.get("receiver_free_bytes_after", 0) < receiver_required:
            raise ValueError("independent receiver headroom not established by canary receipt")
        sender = send

    def deliver(artifact: Path, artifact_id: str) -> None:
        for attempt in range(1, 4):
            attempt_id = artifact_id + f"-attempt-{attempt}"
            receipt = recovery / "receipts" / (attempt_id + ".json")
            try:
                sender(artifact, config_file, attempt_id, receipt)
            except Exception as error:
                receipt.parent.mkdir(exist_ok=True)
                failure = receipt.with_suffix(".failed.json")
                with failure.open("x", encoding="utf-8") as stream:
                    json.dump({"attempt": attempt, "error_type": type(error).__name__}, stream)
                if drive_mode or attempt == 3:
                    raise
            else:
                return

    def mark(checkpoint: Path) -> Path:
        result = original_mark(checkpoint)
        record = verify_checkpoint(result, required)
        step = record["step"]
        if step not in (18000, 19000, 20000):
            raise ValueError("unexpected frozen recovery save step")
        spool = recovery / "handoff"
        spool.mkdir(exist_ok=True)
        if step == 20000:
            rows = [
                json.loads(s) for s in (recovery / "optimizer-steps.jsonl").read_text().splitlines()
            ]
            if (
                rows[-1]["optimizer_step"] != 20000
                or rows[-1]["restored_training_step"] != 17000
                or rows[-1]["actual_calls_this_invocation"] != 3000
            ):
                raise ValueError("final handoff requires exactly 3000 real recovered updates")
            export = spool / "portable-step-020000"
            checkpoint_export.export_complete_checkpoint(recovery, export)
            shutil.copyfile(
                recovery / "recovery-provenance.json", export / "recovery-provenance.json"
            )
            manifest_path = export / "export-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files_sha256"]["recovery-provenance.json"] = sha256(
                export / "recovery-provenance.json"
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            verify_export(export, required, 20000)
            # Highest priority: independently preserve usable model/processor export.
            deliver(
                archive_tree(export, spool / "portable-step-020000.zip"),
                config["run_id"] + "-portable-020000",
            )
        package = spool / f"checkpoint-{step:06d}.zip"
        evidence = {
            "recovery-evidence/optimizer-steps.jsonl": recovery / "optimizer-steps.jsonl",
            "recovery-evidence/recovery-provenance.json": recovery / "recovery-provenance.json",
            "recovery-evidence/training-restore-check.json": recovery
            / "training-restore-check.json",
        }
        provenance = json.loads((recovery / "recovery-provenance.json").read_text(encoding="utf-8"))
        relocation = provenance.get("account_migration")
        if relocation:
            processor = Path(relocation["processor_directory"])
            for name, expected in relocation["processor_files_sha256"].items():
                asset = processor / name
                if sha256(asset) != expected:
                    raise ValueError("processor snapshot changed before checkpoint handoff")
                evidence["recovery-evidence/processor/" + name] = asset
            evidence["recovery-evidence/processor-files.sha256.json"] = (
                processor.parent / "processor-files.sha256.json"
            )
        archive_tree(checkpoint, package, extras=evidence)
        deliver(package, config["run_id"] + f"-checkpoint-{step:06d}")
        label = (
            "DRIVE_API_CHECKPOINT_VERIFIED"
            if drive_mode
            else "INDEPENDENT_CHECKPOINT_RECEIPT_VERIFIED"
        )
        print(f"{label} step={step}", flush=True)
        return result

    training.mark_checkpoint_complete = mark
