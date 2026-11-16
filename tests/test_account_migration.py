"""Account changes may relocate paths but never alter approved bytes or model settings."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def migration(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/recovery"))
    return importlib.import_module("account_migration")


class Drive:
    def __init__(self, payload=b"approved"):
        self.payload = payload
        self.calls = []

    def about(self):
        return self

    def files(self):
        return self

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        self.request = kwargs
        return self

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        self.request = kwargs
        return self

    def execute(self):
        if "fileId" in self.request:
            if self.request["fileId"] == "new-shortcut":
                return {
                    "id": "new-shortcut",
                    "mimeType": "application/vnd.google-apps.shortcut",
                    "shortcutDetails": {"targetId": "original-shared-folder"},
                }
            return {
                "id": "original-shared-folder",
                "mimeType": "application/vnd.google-apps.folder",
                "owners": [{"permissionId": "original-owner"}],
                "trashed": False,
            }
        if "q" in self.request:
            return {
                "files": [
                    {
                        "id": "binary-id",
                        "name": "source.zip",
                        "sha256Checksum": hashlib.sha256(self.payload).hexdigest(),
                        "capabilities": {"canDownload": True},
                    }
                ]
            }
        return {
            "user": {"emailAddress": "second@example.invalid", "permissionId": "second-id"},
            "storageQuota": {"usage": "100"},
        }


def test_new_shortcut_targets_original_owner_and_wrong_auth_stops(migration):
    service = Drive()
    record = migration.account_context(service, "new-shortcut", "second@example.invalid")
    assert record["target_folder"]["id"] == "original-shared-folder"
    assert (
        record["target_folder"]["owners"][0]["permissionId"]
        != record["drive_api_principal_permission_id"]
    )
    assert record["compute_account_verified_programmatically"] is False
    assert "emailAddress" not in json.dumps(record)
    with pytest.raises(ValueError, match="wrong account"):
        migration.account_context(service, "new-shortcut", "first@example.invalid")
    assert all(name in ("get", "list") for name, _ in service.calls)


def test_mount_api_and_immutable_sha_must_all_match(tmp_path, migration):
    root = tmp_path / "second-account-mounted-folder"
    root.mkdir()
    (root / "source.zip").write_bytes(b"approved")
    pin = hashlib.sha256(b"approved").hexdigest()
    bindings = migration.stage_payloads(
        Drive(), "original-shared-folder", root, tmp_path / "stage", {"source.zip": pin}
    )
    assert bindings["source.zip"]["api_bound"]
    with pytest.raises(ValueError, match="API folder"):
        migration.stage_payloads(
            Drive(b"different"),
            "original-shared-folder",
            root,
            tmp_path / "wrong-account-mount",
            {"source.zip": pin},
        )
    assert not (tmp_path / "wrong-account-mount/source.zip").exists()
    with pytest.raises(ValueError, match="SHA-256"):
        migration.stage_payloads(
            Drive(), "original-shared-folder", root, tmp_path / "bad-pin", {"source.zip": "0" * 64}
        )


def test_verified_processor_relocation_changes_only_runtime_path(tmp_path, migration):
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "config.json").write_text("{}", encoding="utf-8")
    snapshot = {"config.json": hashlib.sha256(b"{}").hexdigest()}
    (tmp_path / "processor-files.sha256.json").write_text(json.dumps(snapshot), encoding="utf-8")
    saved = tmp_path / "saved-policy.json"
    saved.write_text('{"vlm_model_name":"/old-account/processor"}', encoding="utf-8")
    before = saved.read_bytes()
    config = SimpleNamespace(vlm_model_name="/old-account/processor", chunk_size=50, state_dim=8)
    migration.relocate_policy(config, processor)
    assert config.vlm_model_name == str(processor)
    assert (config.chunk_size, config.state_dim) == (50, 8)
    assert saved.read_bytes() == before
    overrides = {
        "preprocessor_overrides": {
            "normalizer_processor": {"stats": [1, 2]},
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": "/old-account"},
        }
    }
    original = copy.deepcopy(overrides)
    changed = migration.processor_overrides(overrides, processor)
    assert changed["preprocessor_overrides"]["tokenizer_processor"]["tokenizer_name"] == str(
        processor
    )
    assert (
        changed["preprocessor_overrides"]["normalizer_processor"]
        == original["preprocessor_overrides"]["normalizer_processor"]
    )
    assert overrides == original
    (processor / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        migration.relocate_policy(config, processor)


@pytest.mark.parametrize("problem", ("duplicate", "download-denied", "no-server-checksum"))
def test_ambiguous_or_restricted_shared_files_stop_before_staging(tmp_path, migration, problem):
    class Restricted(Drive):
        def execute(self):
            result = super().execute()
            if "files" in result:
                if problem == "duplicate":
                    result["files"].append(dict(result["files"][0]))
                elif problem == "download-denied":
                    result["files"][0]["capabilities"]["canDownload"] = False
                else:
                    result["files"][0].pop("sha256Checksum")
            return result

    root = tmp_path / "root"
    root.mkdir()
    (root / "source.zip").write_bytes(b"approved")
    with pytest.raises(ValueError):
        migration.stage_payloads(
            Restricted(),
            "folder",
            root,
            tmp_path / "stage",
            {"source.zip": hashlib.sha256(b"approved").hexdigest()},
        )
    assert not (tmp_path / "stage").exists()
