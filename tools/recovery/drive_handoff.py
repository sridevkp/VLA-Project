"""Drive REST persistence receipts, independent of mounted Drive caches or SSH."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from checkpoint_guard import sha256

BASE = "https://www.googleapis.com/drive/v3/"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id"
FIELDS = (
    "id,name,mimeType,size,md5Checksum,sha256Checksum,version,trashed,parents,"
    "ownedByMe,owners(permissionId),driveId"
)


class PersistenceError(RuntimeError):
    """No further optimizer update is permitted after a failed persistence gate."""


class TransientDriveError(PersistenceError):
    """An upload session may be queried, never assumed committed, after this error."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DriveAPI:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(
        self,
        method: str,
        url: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "www.googleapis.com":
            raise PersistenceError("unexpected Drive API/upload endpoint")
        token_file = Path(self.config["token_file"])
        if (
            token_file.is_symlink()
            or token_file.stat().st_mode & 0o077
            or token_file.resolve().is_relative_to(Path("/content/drive"))
        ):
            raise PersistenceError("token file must be local, private and regular")
        token = json.loads(token_file.read_text(encoding="utf-8"))
        if token.get("expires_at", 0) <= time.time() + 15:
            raise PersistenceError("Drive token broker expired; persistence not verified")
        hdr = {"Authorization": "Bearer " + token["access_token"], "Accept-Encoding": "identity"}
        hdr.update(headers or {})
        request = urllib.request.Request(url, data=data, headers=hdr, method=method)
        try:
            return self.opener.open(request, timeout=180)
        except urllib.error.HTTPError as error:
            if error.code == 308:
                return error
            if error.code in (429, 500, 502, 503, 504):
                raise TransientDriveError(f"temporary Drive HTTP {error.code}") from None
            raise PersistenceError(f"Drive API request failed with HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError):
            raise TransientDriveError("Drive network failure; query resumable session") from None

    def json_request(self, method: str, suffix: str, body: dict | None = None) -> dict:
        with self.request(
            method,
            BASE + suffix,
            json.dumps(body).encode() if body is not None else None,
            {"Content-Type": "application/json"},
        ) as response:
            return json.load(response)

    def metadata(self, file_id: str) -> dict[str, Any]:
        return self.json_request(
            "GET",
            "files/"
            + urllib.parse.quote(file_id, safe="")
            + "?fields="
            + urllib.parse.quote(FIELDS),
        )

    def destination(self) -> dict[str, Any]:
        about = self.json_request("GET", "about?fields=user(permissionId),storageQuota")
        principal = self.config["principal_permission_id"]
        if about.get("user", {}).get("permissionId") != principal:
            raise PersistenceError("Drive token belongs to the wrong Google account")
        folder = self.metadata(self.config["folder_id"])
        self.check_owner(folder)
        if (
            folder["id"] == self.config["original_shared_folder_id"]
            or folder.get("mimeType") != "application/vnd.google-apps.folder"
            or self.config["my_drive_root_id"] not in folder.get("parents", [])
        ):
            raise PersistenceError("original shared folder must not receive checkpoint writes")
        return about.get("storageQuota", {})

    def check_owner(self, metadata: dict[str, Any]) -> None:
        if (
            metadata.get("ownedByMe") is not True
            or metadata.get("trashed")
            or metadata.get("driveId")
            or self.config["principal_permission_id"]
            not in {o.get("permissionId") for o in metadata.get("owners", [])}
        ):
            raise PersistenceError(
                "destination/object is not owned by the second-account principal"
            )

    def capacity(self, required: int) -> None:
        quota = self.destination()
        if "usage" not in quota:
            raise PersistenceError("Drive quota usage could not be verified")
        if "limit" in quota:
            if "usage" not in quota or int(quota["limit"]) - int(quota["usage"]) < required:
                raise PersistenceError("insufficient/unverified Drive capacity; do not train")

    def media_hash(self, file_id: str, expected_size: int, output: Path | None = None) -> str:
        url = BASE + "files/" + urllib.parse.quote(file_id, safe="") + "?alt=media"
        digest = hashlib.sha256()
        count = 0
        stream = output.open("xb") if output else None
        try:
            with self.request("GET", url) as response:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > expected_size:
                        raise PersistenceError("API read-back exceeds expected size")
                    digest.update(chunk)
                    if stream:
                        stream.write(chunk)
            if count != expected_size:
                raise PersistenceError("API read-back/download truncated")
            if stream:
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if stream:
                stream.close()
        return digest.hexdigest()

    def download_input(self, file_id: str, output: Path, size: int, expected: str) -> None:
        before = self.metadata(file_id)
        self.check_owner(before)
        if int(before.get("size", -1)) != size or output.exists():
            raise PersistenceError("wrong input size or existing destination")
        partial = output.with_name(output.name + ".partial")
        actual = self.media_hash(file_id, size, partial)
        after = self.metadata(file_id)
        self.check_owner(after)
        if (
            actual != expected
            or not before.get("version")
            or before.get("version") != after.get("version")
            or before.get("size") != after.get("size")
        ):
            raise PersistenceError("Linux-golden input SHA/version mismatch; partial retained")
        partial.rename(output)

    def upload(self, artifact: Path, name: str, *, force_readback: bool = False) -> dict[str, Any]:
        size = artifact.stat().st_size
        expected, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                expected.update(chunk)
                md5.update(chunk)
        expected_sha = expected.hexdigest()
        self.capacity(size + 65536)
        generated = self.json_request("GET", "files/generateIds?count=1&space=drive")
        file_id = generated["ids"][0]
        mime = "application/json" if artifact.suffix == ".json" else "application/zip"
        body = {
            "id": file_id,
            "name": name,
            "mimeType": mime,
            "parents": [self.config["folder_id"]],
            "appProperties": {"worldmind_run": self.config["run_id"], "sha256": expected_sha},
        }
        with self.request(
            "POST",
            UPLOAD,
            json.dumps(body).encode(),
            {
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(size),
                "X-Upload-Content-Type": mime,
            },
        ) as response:
            location = response.headers.get("Location")
        if not location:
            raise PersistenceError("Drive resumable upload session missing")
        offset = 0
        completed = False
        retries = 0
        with artifact.open("rb") as source:
            while offset < size:
                source.seek(offset)
                data = source.read(8 * 1024 * 1024)
                try:
                    response = self.request(
                        "PUT",
                        location,
                        data,
                        {
                            "Content-Length": str(len(data)),
                            "Content-Range": f"bytes {offset}-{offset + len(data) - 1}/{size}",
                        },
                    )
                except TransientDriveError:
                    retries += 1
                    if retries > 3:
                        raise
                    time.sleep(2 ** (retries - 1))
                    # Probe server state before resending; a timed-out chunk may
                    # already have been committed. Never duplicate or skip bytes.
                    response = self.request(
                        "PUT",
                        location,
                        b"",
                        {
                            "Content-Length": "0",
                            "Content-Range": f"bytes */{size}",
                        },
                    )
                    with response:
                        if response.status in (200, 201):
                            if json.load(response).get("id") != file_id:
                                raise PersistenceError("wrong completed upload ID") from None
                            offset, completed = size, True
                        elif response.status == 308:
                            ack = response.headers.get("Range", "")
                            offset = int(ack.rsplit("-", 1)[-1]) + 1 if ack else 0
                            if not 0 <= offset < size:
                                raise PersistenceError(
                                    "invalid pending-upload server offset"
                                ) from None
                        else:
                            raise PersistenceError("resumable status probe failed") from None
                    continue
                with response:
                    if response.status == 308:
                        ack = response.headers.get("Range", "")
                        acknowledged = int(ack.rsplit("-", 1)[-1]) + 1 if ack else 0
                        if not offset < acknowledged <= offset + len(data):
                            raise PersistenceError("invalid resumable upload acknowledgement")
                        offset = acknowledged
                    elif response.status in (200, 201):
                        result = json.load(response)
                        if result.get("id") != file_id or offset + len(data) != size:
                            raise PersistenceError("premature/wrong Drive upload completion")
                        offset = size
                        completed = True
                    else:
                        raise PersistenceError("Drive upload did not complete")
        if not completed:
            raise PersistenceError("Drive never acknowledged final upload completion")
        before = self.metadata(file_id)
        self.check_owner(before)
        if int(before.get("size", -1)) != size or self.config["folder_id"] not in before.get(
            "parents", []
        ):
            raise PersistenceError("uploaded object size/folder mismatch")
        if before.get("md5Checksum") and before["md5Checksum"] != md5.hexdigest():
            raise PersistenceError("uploaded object MD5 mismatch")
        server_sha = before.get("sha256Checksum")
        if server_sha and server_sha != expected_sha:
            raise PersistenceError("uploaded object SHA-256 mismatch")
        readback = force_readback or not server_sha
        if readback and self.media_hash(file_id, size) != expected_sha:
            raise PersistenceError("independent API read-back SHA-256 mismatch")
        after = self.metadata(file_id)
        self.check_owner(after)
        if (
            not before.get("version")
            or before.get("version") != after.get("version")
            or after.get("size") != before.get("size")
            or self.config["folder_id"] not in after.get("parents", [])
        ):
            raise PersistenceError("Drive object changed during verification")
        if sha256(artifact) != expected_sha:
            raise PersistenceError("local artifact changed during upload")
        return {
            "transport": "drive_api",
            "remote_file_id": file_id,
            "folder_id": self.config["folder_id"],
            "principal_permission_id": self.config["principal_permission_id"],
            "bytes": size,
            "sha256": expected_sha,
            "server_size_verified": True,
            "server_sha256_verified": bool(server_sha),
            "api_readback_sha256_verified": readback,
            "version": after.get("version"),
            "verified_at_unix": time.time(),
        }


def send(artifact: Path, config_file: Path, artifact_id: str, receipt_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    client = DriveAPI(config)
    report = client.upload(
        artifact, artifact_id + ".zip", force_readback=artifact_id.endswith("-canary")
    )
    report["artifact_id"] = artifact_id
    report["transfer_config_sha256"] = sha256(config_file)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    pending = receipt_path.with_suffix(".pending.json")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    remote_receipt = client.upload(pending, artifact_id + ".receipt.json", force_readback=True)
    report["remote_receipt_file_id"] = remote_receipt["remote_file_id"]
    report["handoff_elapsed_s"] = time.perf_counter() - started
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("DRIVE_API_ARTIFACT_VERIFIED " + json.dumps(report), flush=True)
    return report


def validate_canary(config_file: Path) -> dict:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    receipt = json.loads(Path(config["canary_receipt"]).read_text(encoding="utf-8"))
    if (
        receipt.get("transport") != "drive_api"
        or receipt.get("folder_id") != config["folder_id"]
        or receipt.get("principal_permission_id") != config["principal_permission_id"]
        or receipt.get("sha256") != config["canary_sha256"]
        or receipt.get("transfer_config_sha256") != sha256(config_file)
        or receipt.get("api_readback_sha256_verified") is not True
        or receipt.get("server_size_verified") is not True
        or receipt.get("bytes", 0) < 1024 * 1024
        or not receipt.get("remote_receipt_file_id")
    ):
        raise PersistenceError("new-owner API canary/config receipt required before training")
    client = DriveAPI(config)
    client.destination()
    remote = client.metadata(receipt["remote_file_id"])
    client.check_owner(remote)
    previous_version = str(receipt.get("version", ""))
    current_version = str(remote.get("version", ""))
    if (
        remote.get("id") != receipt["remote_file_id"]
        or int(remote.get("size", -1)) != receipt["bytes"]
        or config["folder_id"] not in remote.get("parents", [])
        or not previous_version.isascii()
        or not previous_version.isdecimal()
        or not current_version.isascii()
        or not current_version.isdecimal()
        or int(previous_version) < 1
        or int(current_version) < int(previous_version)
        or remote.get("sha256Checksum", receipt["sha256"]) != receipt["sha256"]
    ):
        raise PersistenceError("API canary no longer matches the verified object")
    if client.media_hash(remote["id"], receipt["bytes"]) != receipt["sha256"]:
        raise PersistenceError("API canary fresh read-back failed")
    after = client.metadata(receipt["remote_file_id"])
    client.check_owner(after)
    if (
        after.get("id") != receipt["remote_file_id"]
        or after.get("version") != remote.get("version")
        or after.get("size") != remote.get("size")
        or set(after.get("parents", [])) != set(remote.get("parents", []))
        or after.get("sha256Checksum", receipt["sha256"]) != receipt["sha256"]
    ):
        raise PersistenceError("API canary changed during fresh read-back")
    if current_version != previous_version:
        # Drive version tracks metadata/server changes too. Never rewrite the
        # upload receipt; record content-verified revalidation separately.
        event = {
            "remote_file_id": remote["id"],
            "folder_id": config["folder_id"],
            "principal_permission_id": config["principal_permission_id"],
            "receipt_version": previous_version,
            "readback_version": current_version,
            "bytes": receipt["bytes"],
            "sha256": receipt["sha256"],
            "api_readback_sha256_verified": True,
            "stable_version_during_readback": True,
            "receipt_sha256": sha256(Path(config["canary_receipt"])),
            "transfer_config_sha256": sha256(config_file),
            "verified_at_unix": time.time(),
        }
        audit = Path(config["canary_receipt"]).with_suffix(".revalidations.jsonl")
        if audit.is_symlink():
            raise PersistenceError("symlinked canary revalidation log")
        with audit.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print("DRIVE_CANARY_CONTENT_REVALIDATED " + json.dumps(event), flush=True)
    return config
