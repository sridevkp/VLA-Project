"""Owned Drive destination, upload acknowledgements and API read-back fail closed."""

import datetime
import hashlib
import importlib
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def drive(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/recovery"))
    return importlib.import_module("drive_handoff")


class Response(io.BytesIO):
    def __init__(self, data=b"", status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def fake_client(module, **options):
    class Fake(module.DriveAPI):
        def __init__(self):
            self.config = {
                "principal_permission_id": "second",
                "folder_id": "new-folder",
                "my_drive_root_id": "second-root",
                "original_shared_folder_id": "old-shared",
                "run_id": "second-run",
            }
            self.data = {}
            self.sequence = 0
            self.downloads = 0
            self.metadata_calls = 0

        def json_request(self, method, suffix, body=None):
            if suffix.startswith("about"):
                return {
                    "user": {"permissionId": options.get("principal", "second")},
                    "storageQuota": {"limit": "100000000000", "usage": "0"},
                }
            self.sequence += 1
            return {"ids": ["file" + str(self.sequence)]}

        def metadata(self, file_id):
            owner = options.get("owner", "second")
            result = {
                "id": file_id,
                "ownedByMe": owner == "second",
                "owners": [{"permissionId": owner}],
                "version": "1",
                "trashed": False,
            }
            if file_id == "new-folder":
                result.update(
                    mimeType="application/vnd.google-apps.folder",
                    parents=[options.get("folder_parent", "second-root")],
                )
            else:
                data = self.data[file_id]
                result.update(
                    size=str(len(data)),
                    parents=["new-folder"],
                    md5Checksum=hashlib.md5(data).hexdigest(),
                )
                if options.get("server_sha", True):
                    result["sha256Checksum"] = (
                        "0" * 64 if options.get("bad_sha") else hashlib.sha256(data).hexdigest()
                    )
                if options.get("wrong_size"):
                    result["size"] = "1"
                self.metadata_calls += 1
                if options.get("changed_version") and self.metadata_calls > 1:
                    result["version"] = "2"
            return result

        def request(self, method, url, data=None, headers=None):
            if method == "POST":
                self.upload_metadata = json.loads(data)
                self.buffer = bytearray()
                return Response(headers={"Location": "https://www.googleapis.com/upload/session"})
            if method == "PUT":
                self.buffer.extend(data)
                if options.get("transient_once") and not getattr(self, "tripped", False):
                    self.tripped = True
                    raise module.TransientDriveError("synthetic timeout after receiving bytes")
                size = int(headers["Content-Range"].split("/")[-1])
                if len(self.buffer) < size or options.get("no_completion"):
                    return Response(
                        status=308, headers={"Range": f"bytes=0-{len(self.buffer) - 1}"}
                    )
                file_id = self.upload_metadata["id"]
                self.data[file_id] = bytes(self.buffer)
                return Response(json.dumps({"id": file_id}).encode())
            self.downloads += 1
            file_id = url.split("files/")[1].split("?")[0]
            payload = self.data[file_id]
            if options.get("bad_readback"):
                payload = b"x" * len(payload)
            return Response(payload)

    return Fake()


@pytest.mark.parametrize("server_sha", (True, False))
def test_resumable_upload_server_checksum_or_actual_api_readback(tmp_path, drive, server_sha):
    client = fake_client(drive, server_sha=server_sha)
    source = tmp_path / "checkpoint.zip"
    source.write_bytes(b"a" * (9 * 1024 * 1024))  # requires a308 and final200
    result = client.upload(source, "run-checkpoint.zip")
    assert result["bytes"] == source.stat().st_size
    assert result["server_sha256_verified"] == server_sha
    assert result["api_readback_sha256_verified"] == (not server_sha)
    assert client.downloads == (0 if server_sha else 1)
    assert client.upload_metadata["parents"] == ["new-folder"]


@pytest.mark.parametrize(
    "problem",
    (
        {"principal": "first"},
        {"owner": "first"},
        {"folder_parent": "old-shared"},
        {"bad_sha": True},
        {"wrong_size": True},
        {"changed_version": True},
        {"server_sha": False, "bad_readback": True},
        {"no_completion": True},
    ),
)
def test_mismatches_and_incomplete_uploads_never_issue_success(tmp_path, drive, problem):
    source = tmp_path / "checkpoint.zip"
    source.write_bytes(b"trained weights")
    with pytest.raises(drive.PersistenceError):
        fake_client(drive, **problem).upload(source, "checkpoint.zip")


def test_canary_forces_readback_even_with_server_sha(tmp_path, drive):
    source = tmp_path / "canary.bin"
    source.write_bytes(b"canary")
    client = fake_client(drive)
    receipt = client.upload(source, "canary", force_readback=True)
    assert receipt["api_readback_sha256_verified"] and client.downloads == 1


def test_timed_out_chunk_probes_server_before_continuing(tmp_path, drive, monkeypatch):
    monkeypatch.setattr(drive.time, "sleep", lambda delay: None)
    source = tmp_path / "checkpoint.zip"
    source.write_bytes(b"r" * (9 * 1024 * 1024))
    client = fake_client(drive, transient_once=True, server_sha=False)
    receipt = client.upload(source, "checkpoint.zip")
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(client.data[receipt["remote_file_id"]]) == source.stat().st_size


def test_remote_receipt_and_fresh_canary_readback_are_required(tmp_path, drive, monkeypatch):
    client = fake_client(drive)
    monkeypatch.setattr(drive, "DriveAPI", lambda cfg: client)
    source = tmp_path / "canary.bin"
    source.write_bytes(b"0" * 1024 * 1024)
    config = {
        **client.config,
        "transport": "drive_api",
        "canary_receipt": str(tmp_path / "receipt.json"),
        "canary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = drive.send(source, path, "second-run-canary", tmp_path / "receipt.json")
    assert result["remote_receipt_file_id"] in client.data
    assert drive.validate_canary(path)["folder_id"] == "new-folder"
    assert client.downloads >= 3
    config["folder_id"] = "old-shared"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(drive.PersistenceError, match="canary"):
        drive.validate_canary(path)


def test_expired_or_nonprivate_token_fails_before_any_network(tmp_path, drive):
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"expires_at": 0, "access_token": "synthetic-test-only"}))
    token.chmod(0o600)
    client = drive.DriveAPI({"token_file": str(token)})
    with pytest.raises(drive.PersistenceError, match="expired"):
        client.request("GET", drive.BASE + "about")
    token.chmod(0o644)
    with pytest.raises(drive.PersistenceError, match="private"):
        client.request("GET", drive.BASE + "about")


def test_kernel_setup_creates_only_a_new_second_owned_root_folder(tmp_path, monkeypatch):
    import types

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/recovery"))
    setup = importlib.import_module("drive_handoff_setup")

    class Credentials:
        token = "synthetic-unit-test-token"
        expiry = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)

        def refresh(self, request):
            pass

    class Service:
        def __init__(self, principal="second"):
            self.principal = principal
            self.created = []

        def about(self):
            self.operation = "about"
            return self

        def files(self):
            return self

        def get(self, **kwargs):
            if "fileId" in kwargs:
                self.operation = "root"
            return self

        def create(self, **kwargs):
            self.created.append(kwargs["body"])
            self.operation = "create"
            return self

        def execute(self):
            if self.operation == "about":
                return {
                    "user": {
                        "emailAddress": "second@example.invalid",
                        "permissionId": self.principal,
                    }
                }
            if self.operation == "root":
                return {"id": "second-root"}
            return {
                "id": "new-dest",
                "ownedByMe": True,
                "parents": ["second-root"],
                "owners": [{"permissionId": "second"}],
            }

    monkeypatch.setattr(
        setup, "DriveAPI", lambda cfg: types.SimpleNamespace(capacity=lambda n: None)
    )
    monkeypatch.setattr(setup, "send", lambda *args: None)
    monkeypatch.setattr(setup, "validate_canary", lambda *args: None)
    context = {"drive_api_principal_permission_id": "second", "target_folder": {"id": "old-shared"}}
    bad = Service("first")
    with pytest.raises(ValueError, match="wrong account"):
        setup.setup(
            context,
            "second@example.invalid",
            "wrong",
            {"bytes": 1},
            work=tmp_path,
            credentials=Credentials(),
            service=bad,
        )
    assert not bad.created
    service = Service()
    config_file, folder, stop = setup.setup(
        context,
        "second@example.invalid",
        "newrun",
        {"bytes": 1},
        work=tmp_path,
        credentials=Credentials(),
        service=service,
    )
    stop.set()
    config = json.loads(config_file.read_text())
    assert folder["id"] != "old-shared" and service.created[0]["parents"] == ["root"]
    token = Path(config["token_file"])
    assert token.stat().st_mode & 0o077 == 0
    assert "access_token" not in config_file.read_text()
