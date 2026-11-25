"""Read-only account binding and hash-verified processor-path relocation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from checkpoint_guard import regular, sha256


def account_context(service: Any, entry_id: str, expected_email: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", entry_id):
        raise ValueError("supply a Drive folder/shortcut ID, not a URL")
    principal = service.about().get(fields="user(emailAddress,permissionId),storageQuota").execute()
    user = principal["user"]
    if not expected_email or user.get("emailAddress", "").casefold() != expected_email.casefold():
        raise ValueError("Drive API authenticated as the wrong account; stop and reauthenticate")
    fields = "id,name,mimeType,trashed,owners(permissionId),driveId,shortcutDetails"
    entry = service.files().get(fileId=entry_id, fields=fields, supportsAllDrives=True).execute()
    if entry.get("mimeType") == "application/vnd.google-apps.shortcut":
        target = entry["shortcutDetails"]["targetId"]
        folder = service.files().get(fileId=target, fields=fields, supportsAllDrives=True).execute()
    else:
        folder = entry
    if folder.get("trashed") or folder.get("mimeType") != "application/vnd.google-apps.folder":
        raise ValueError("selected shared target is not an accessible non-trashed folder")
    return {
        "drive_api_principal_permission_id": user["permissionId"],
        "authenticated_account_email_match": True,
        "compute_account_verified_programmatically": False,
        "mount_account_verified_programmatically": False,
        "entry_id": entry_id,
        "target_folder": folder,
        "authenticated_principal_storage_quota": principal.get("storageQuota", {}),
        "drive_write_operations": 0,
    }


def stage_payloads(
    service: Any, folder_id: str, root: Path, staging: Path, pins: dict[str, str]
) -> dict[str, Any]:
    """Bind mounted bytes to selected API folder children and immutable approvals."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise ValueError("invalid shared folder ID")
    if root.resolve(strict=True) != root:
        raise ValueError("resolve the selected project entrypoint once before staging")
    children = []
    token = None
    while True:
        page = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                pageToken=token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                fields="nextPageToken,files(id,name,md5Checksum,sha256Checksum,capabilities(canDownload))",
            )
            .execute()
        )
        children.extend(page.get("files", []))
        token = page.get("nextPageToken")
        if not token:
            break
    selected = {}
    for name in pins:
        matches = [f for f in children if f.get("name") == name]
        if len(matches) != 1 or matches[0].get("capabilities", {}).get("canDownload") is not True:
            raise ValueError(f"missing/ambiguous/download-restricted shared payload: {name}")
        if not (matches[0].get("sha256Checksum") or matches[0].get("md5Checksum")):
            raise ValueError(f"no API checksum for mounted-folder binding: {name}")
        selected[name] = matches[0]
    staging.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for name, expected in pins.items():
        source = regular(root, name)
        sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        partial = staging / (name + ".partial")
        with source.open("rb") as incoming, partial.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                outgoing.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        entry = selected[name]
        if sha.hexdigest() != expected:
            raise ValueError(f"immutable payload SHA-256 mismatch: {name}")
        if entry.get("sha256Checksum"):
            bound = sha.hexdigest() == entry["sha256Checksum"]
        else:
            bound = md5.hexdigest() == entry["md5Checksum"]
        if not bound:
            raise ValueError(f"mounted bytes do not match selected API folder child: {name}")
        partial.rename(staging / name)
        bindings[name] = {"file_id": entry["id"], "sha256": sha.hexdigest(), "api_bound": True}
    return bindings


def verify_processor(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("processor metadata must not contain symlinks")
    record = json.loads(
        regular(directory.parent, "processor-files.sha256.json").read_text(encoding="utf-8")
    )
    if not record or set(record) != {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }:
        raise ValueError("processor inventory differs from verified metadata snapshot")
    for name, expected in record.items():
        if sha256(regular(directory, name)) != expected:
            raise ValueError(f"processor metadata fingerprint mismatch: {name}")
    return record


def relocate_policy(config: Any, processor: Path) -> None:
    """Only a runtime filesystem address changes; saved checkpoint bytes stay intact."""
    verify_processor(processor)
    config.vlm_model_name = str(processor.resolve(strict=True))


def processor_overrides(kwargs: dict[str, Any], processor: Path) -> dict[str, Any]:
    """Keep normalization/device settings; replace only the tokenizer asset address."""
    result = dict(kwargs)
    overrides = dict(result.get("preprocessor_overrides", {}))
    tokenizer = dict(overrides.get("tokenizer_processor", {}))
    tokenizer["tokenizer_name"] = str(processor.resolve(strict=True))
    overrides["tokenizer_processor"] = tokenizer
    result["preprocessor_overrides"] = overrides
    return result
