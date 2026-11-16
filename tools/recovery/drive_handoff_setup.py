"""Colab-kernel OAuth broker and new-owner folder setup; no model/optimizer calls."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from drive_handoff import DriveAPI, send, validate_canary


def setup(
    context: dict[str, Any],
    expected_email: str,
    run_id: str,
    gold: dict[str, Any],
    *,
    work: Path = Path("/content"),
    credentials: Any = None,
    service: Any = None,
    refresh_request: Any = None,
) -> tuple[Path, dict[str, Any], threading.Event]:
    """Owner executes this in the existing kernel; credentials never enter archives."""
    config_file = work / (run_id + "-drive-config.json")
    token_file = work / (run_id + "-private-token.json")
    if config_file.exists() or token_file.exists():
        raise ValueError("preserve previous configuration; use a new run ID")
    if credentials is None:
        import google.auth
        from google.auth.transport.requests import Request

        credentials, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
        )
        refresh_request = Request()
    credentials.refresh(refresh_request)
    if service is None:
        from googleapiclient.discovery import build

        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    who = service.about().get(fields="user(emailAddress,permissionId),storageQuota").execute()
    if (
        who["user"]["emailAddress"].casefold() != expected_email.casefold()
        or who["user"]["permissionId"] != context["drive_api_principal_permission_id"]
    ):
        raise ValueError("wrong account; no destination folder created")
    root_id = service.files().get(fileId="root", fields="id").execute()["id"]
    folder = (
        service.files()
        .create(
            body={
                "name": run_id + "-checkpoints",
                "mimeType": "application/vnd.google-apps.folder",
                "parents": ["root"],
            },
            fields="id,name,mimeType,parents,ownedByMe,owners(permissionId),driveId",
        )
        .execute()
    )
    if (
        folder.get("ownedByMe") is not True
        or folder.get("driveId")
        or root_id not in folder.get("parents", [])
        or who["user"]["permissionId"] not in {o["permissionId"] for o in folder["owners"]}
        or folder["id"] == context["target_folder"]["id"]
    ):
        raise ValueError("new destination is not a second-account-owned My Drive folder")
    stop = threading.Event()

    def write_token(refresh: bool) -> None:
        if refresh:
            credentials.refresh(refresh_request)
        expiry = credentials.expiry
        if expiry is not None:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=datetime.UTC)
            expires_at = expiry.timestamp()
        else:
            expires_at = time.time() + 90
        temporary = token_file.with_suffix(".next")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump({"access_token": credentials.token, "expires_at": expires_at}, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(token_file)

    def refresh_loop() -> None:
        while not stop.wait(60):
            try:
                write_token(True)
            except Exception:
                print(
                    "DRIVE_TOKEN_REFRESH_FAILED: keep session alive; persistence may stop",
                    flush=True,
                )
                return

    write_token(False)
    threading.Thread(target=refresh_loop, daemon=True).start()
    canary = work / (run_id + "-canary.zip")
    with zipfile.ZipFile(canary, "x", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("canary.bin", b"0" * 1024 * 1024)
    config = {
        "transport": "drive_api",
        "run_id": run_id,
        "folder_id": folder["id"],
        "my_drive_root_id": root_id,
        "original_shared_folder_id": context["target_folder"]["id"],
        "principal_permission_id": who["user"]["permissionId"],
        "token_file": str(token_file),
        "canary_receipt": str(work / (run_id + "-canary-receipt.json")),
        "canary_sha256": hashlib.sha256(canary.read_bytes()).hexdigest(),
    }
    with config_file.open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2)
    try:
        DriveAPI(config).capacity(3 * gold["bytes"] + 1_197_789_224 + 64 * 1024**2)
        send(canary, config_file, run_id + "-canary", Path(config["canary_receipt"]))
        validate_canary(config_file)
    except Exception:
        stop.set()
        raise
    print("SECOND_ACCOUNT_API_FOLDER_AND_READBACK_VERIFIED", flush=True)
    print("Destination: https://drive.google.com/drive/folders/" + folder["id"], flush=True)
    return config_file, folder, stop
