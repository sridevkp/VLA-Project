"""Independent artifact receipts, corruption handling and fail-closed canary gates."""

import hashlib
import importlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/recovery"))
    return importlib.import_module("durable_transfer"), importlib.import_module("durable_sink")


def wire(payload, artifact_id="run1-model", expected=None):
    header = {
        "artifact_id": artifact_id,
        "filename": "model.bin",
        "bytes": len(payload),
        "sha256": expected or hashlib.sha256(payload).hexdigest(),
    }
    return io.BytesIO(json.dumps(header).encode() + b"\n" + payload)


def test_receiver_flushes_readbacks_and_preserves_duplicate_evidence(tmp_path, modules):
    transfer, _ = modules
    root = tmp_path / "independent"
    report = transfer.receive(wire(b"trained model"), root)
    assert report["receiver_readback_verified"] and report["file_fsync"]
    assert Path(report["receiver_path"]).read_bytes() == b"trained model"
    assert (Path(report["receiver_path"]).parent / "receipt.json").is_file()
    with pytest.raises(FileExistsError):
        transfer.receive(wire(b"different model"), root)
    assert Path(report["receiver_path"]).read_bytes() == b"trained model"


@pytest.mark.parametrize("corruption", ("hash", "truncated", "trailing"))
def test_failed_receive_keeps_partial_but_never_issues_receipt(tmp_path, modules, corruption):
    transfer, _ = modules
    data = wire(b"payload", expected="0" * 64 if corruption == "hash" else None).getvalue()
    if corruption == "truncated":
        data = data[:-2]
    if corruption == "trailing":
        data += b"unexpected"
    with pytest.raises(ValueError):
        transfer.receive(io.BytesIO(data), tmp_path)
    assert not list(tmp_path.rglob("receipt.json"))
    assert list(tmp_path.rglob("*.partial"))


def test_receiver_rejects_traversal(tmp_path, modules):
    transfer, _ = modules
    with pytest.raises(ValueError, match="identifier"):
        transfer.receive(wire(b"model", artifact_id="../escape"), tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="receiver endpoint is explicitly Linux")
def test_real_process_handoff_and_missing_receipt_veto(tmp_path, modules):
    transfer, _ = modules
    payload = tmp_path / "portable.zip"
    payload.write_bytes(b"artifact bytes" * 100)
    script = Path(transfer.__file__)
    command = [sys.executable, str(script), "receive", "--root", str(tmp_path / "linux")]
    result = transfer.send_command(payload, command, "run1-portable", tmp_path / "ack.json", 30)
    assert result["sha256"] == transfer.digest(payload)
    assert result["directory_fsync"]
    fake = [sys.executable, "-c", "print('{}')"]
    with pytest.raises(ValueError, match="receipt"):
        transfer.send_command(payload, fake, "run2-portable", tmp_path / "bad-ack.json", 30)
    assert not (tmp_path / "bad-ack.json").exists()


def test_canary_is_bound_to_endpoint_config_and_hash(tmp_path, modules):
    transfer, sink = modules
    config = {
        "destination": "owner@example.invalid",
        "remote_python": "/usr/bin/python3",
        "remote_helper": "/home/owner/durable_transfer.py",
        "remote_root": "/home/owner/received",
        "identity_file": "/content/private-key",
        "known_hosts": "/content/known_hosts",
        "canary_receipt": str(tmp_path / "canary.json"),
        "canary_sha256": "a" * 64,
        "run_id": "recovery-v4",
    }
    path = tmp_path / "transfer.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    receipt = {
        "sha256": "a" * 64,
        "receiver_readback_verified": True,
        "file_fsync": True,
        "directory_fsync": True,
        "bytes": 1024 * 1024,
        "receiver_path": "/home/owner/received/canary/payload.zip",
        "transfer_config_sha256": transfer.digest(path),
    }
    (tmp_path / "canary.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert sink.validate_canary(path)["run_id"] == "recovery-v4"
    config["destination"] = "other@example.invalid"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="canary"):
        sink.validate_canary(path)


def test_sender_pins_ssh_host_key_verification(modules, tmp_path, monkeypatch):
    transfer, _ = modules
    config = {
        "destination": "owner@host",
        "remote_python": "/usr/bin/python3",
        "remote_helper": "/path with spaces/transfer.py",
        "remote_root": "/backups",
        "identity_file": "/secret/key",
        "known_hosts": "/secret/hosts",
    }
    command = transfer.ssh_command(config)
    assert "StrictHostKeyChecking=yes" in command and "BatchMode=yes" in command
    assert "'/path with spaces/transfer.py'" in command[-1]
    config_file = tmp_path / "ssh.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")

    def fake_emit(command, *, stdout, **kwargs):
        stdout.write(b"verified input")

    monkeypatch.setattr(transfer.subprocess, "run", fake_emit)
    expected = hashlib.sha256(b"verified input").hexdigest()
    transfer.pull("/backup/golden.zip", tmp_path / "golden.zip", expected, 14, config_file)
    assert (tmp_path / "golden.zip").read_bytes() == b"verified input"
    with pytest.raises(ValueError, match="SHA-256"):
        transfer.pull("/backup/golden.zip", tmp_path / "bad.zip", "0" * 64, 14, config_file)
    assert not (tmp_path / "bad.zip").exists()
    assert (tmp_path / "bad.zip.partial").exists()


def test_archive_is_verified_and_never_overwritten(tmp_path, modules):
    _, sink = modules
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"weights")
    archive = sink.archive_tree(folder, tmp_path / "portable.zip")
    assert archive.is_file()
    with pytest.raises(ValueError, match="preserve"):
        sink.archive_tree(folder, archive)
    (folder / "export-manifest.json").write_text(
        json.dumps({"files_sha256": {"missing-model.safetensors": "0" * 64}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="omit"):
        sink.archive_tree(folder, tmp_path / "missing-model.zip")
    saved, required = checkpoint_fixture(tmp_path / "golden", 17000)
    backup = tmp_path / "checkpoint-backup-017000"
    saved.rename(backup)
    assert sink.verify_checkpoint(backup, required, expected_step=17000)["step"] == 17000
    with pytest.raises(ValueError, match="step"):
        sink.verify_checkpoint(backup, required, expected_step=20000)


def checkpoint_fixture(recovery, step):
    names = (
        "pretrained_model/model.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/training_step.json",
    )
    path = recovery / "checkpoints" / f"{step:06d}"
    for name in names:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"state")
    (path / names[-1]).write_text(json.dumps({"step": step}), encoding="utf-8")
    record = {
        "schema_version": 2,
        "step": step,
        "files_and_sizes": {n: (path / n).stat().st_size for n in names},
        "files_sha256": {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in names},
    }
    (path / "COMPLETE.json").write_text(json.dumps(record), encoding="utf-8")
    return path, names


def test_final_hook_prioritizes_portable_model_and_retries_without_updates(
    tmp_path, modules, monkeypatch
):
    _, sink = modules
    recovery = tmp_path / "content/run-v4"
    saved, required = checkpoint_fixture(recovery, 20000)
    checkpoint_fixture(recovery, 17000)
    canary = tmp_path / "canary.json"
    canary.write_text(json.dumps({"receiver_free_bytes_after": 10**12}), encoding="utf-8")
    (recovery / "recovery-provenance.json").write_text("{}", encoding="utf-8")
    (recovery / "training-restore-check.json").write_text("{}", encoding="utf-8")
    (recovery / "optimizer-steps.jsonl").write_text(
        json.dumps(
            {
                "optimizer_step": 20000,
                "restored_training_step": 17000,
                "actual_calls_this_invocation": 3000,
            }
        ),
        encoding="utf-8",
    )

    def export(output, out):
        out.mkdir()
        model = out / "model.safetensors"
        model.write_bytes(b"trained")
        (out / "export-manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint_step": 20000,
                    "files_sha256": {
                        "model.safetensors": hashlib.sha256(model.read_bytes()).hexdigest()
                    },
                }
            ),
            encoding="utf-8",
        )

    training = SimpleNamespace(CHECKPOINT_FILES=required, mark_checkpoint_complete=lambda p: p)
    monkeypatch.setitem(
        sys.modules,
        "vlarobot.vla",
        SimpleNamespace(
            training=training, checkpoint_export=SimpleNamespace(export_complete_checkpoint=export)
        ),
    )
    monkeypatch.setattr(
        sink, "validate_canary", lambda p: {"run_id": "durable-v4", "canary_receipt": str(canary)}
    )
    attempts = []

    def transfer(artifact, config, artifact_id, receipt):
        attempts.append(artifact_id)
        if len(attempts) == 1:
            raise TimeoutError("synthetic network interruption")

    monkeypatch.setattr(sink, "send", transfer)
    sink.install_sink(recovery, tmp_path / "config", local_root=tmp_path / "content")
    assert training.mark_checkpoint_complete(saved) == saved
    assert attempts == [
        "durable-v4-portable-020000-attempt-1",
        "durable-v4-portable-020000-attempt-2",
        "durable-v4-checkpoint-020000-attempt-1",
    ]
    assert (saved / "pretrained_model/model.safetensors").read_bytes() == b"state"


def test_all_transfer_failures_stop_callback_preserving_local_state(tmp_path, modules, monkeypatch):
    _, sink = modules
    recovery = tmp_path / "content/run"
    saved, required = checkpoint_fixture(recovery, 18000)
    checkpoint_fixture(recovery, 17000)
    canary = tmp_path / "canary.json"
    canary.write_text(json.dumps({"receiver_free_bytes_after": 10**12}), encoding="utf-8")
    for name in (
        "recovery-provenance.json",
        "training-restore-check.json",
        "optimizer-steps.jsonl",
    ):
        (recovery / name).write_text("{}", encoding="utf-8")
    training = SimpleNamespace(CHECKPOINT_FILES=required, mark_checkpoint_complete=lambda p: p)
    monkeypatch.setitem(
        sys.modules,
        "vlarobot.vla",
        SimpleNamespace(training=training, checkpoint_export=SimpleNamespace()),
    )
    monkeypatch.setattr(
        sink, "validate_canary", lambda p: {"run_id": "failure-v4", "canary_receipt": str(canary)}
    )

    def fail(*args):
        raise TimeoutError("synthetic outage")

    monkeypatch.setattr(sink, "send", fail)
    sink.install_sink(recovery, tmp_path / "config", local_root=tmp_path / "content")
    with pytest.raises(TimeoutError):
        training.mark_checkpoint_complete(saved)
    assert len(list((recovery / "receipts").glob("*.failed.json"))) == 3
    assert (saved / "training_state/optimizer_state.safetensors").read_bytes() == b"state"


def test_drive_diagnostics_is_read_only_and_handles_denied_quota(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/recovery"))
    diagnostics = importlib.import_module("drive_diagnostics")
    calls = []

    class ReadOnly:
        def about(self):
            return self

        def files(self):
            return self

        def get(self, **kwargs):
            calls.append(("get", kwargs))
            self.quota = "fileId" not in kwargs
            return self

        def list(self, **kwargs):
            calls.append(("list", kwargs))
            self.quota = False
            return self

        def execute(self):
            if self.quota:
                raise PermissionError("quota inaccessible")
            return {"files": []}

    report = diagnostics.inspect_drive(ReadOnly(), "project-id")
    assert "account_quota_error" in report
    assert report["drive_write_operations"] == 0 and report["root_cause_proven"] is False
    assert all(name in ("get", "list") for name, _ in calls)
