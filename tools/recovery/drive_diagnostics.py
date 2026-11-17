"""Read-only Drive API metadata diagnostics; no mutation or tensor download."""

from __future__ import annotations

from typing import Any


def inspect_drive(service: Any, project_id: str) -> dict[str, Any]:
    """Use an owner-authenticated Drive v3 service; never print tokens or emails."""
    report: dict[str, Any] = {"root_cause_proven": False, "drive_write_operations": 0}
    try:
        report["account_quota"] = (
            service.about().get(fields="storageQuota,user(permissionId)").execute()
        )
    except Exception as error:
        report["account_quota_error"] = str(error)
    try:
        report["project_metadata"] = (
            service.files()
            .get(
                fileId=project_id,
                supportsAllDrives=True,
                fields="id,name,mimeType,driveId,owners(permissionId),shortcutDetails,trashed",
            )
            .execute()
        )
    except Exception as error:
        report["project_metadata_error"] = str(error)
    query = "(name = 'model.safetensors' or name = 'optimizer_state.safetensors')"
    items = []
    token = None
    for _ in range(20):
        try:
            page = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    pageSize=100,
                    pageToken=token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    fields=(
                        "nextPageToken,files(id,name,size,md5Checksum,sha256Checksum,trashed,"
                        "parents,modifiedTime,quotaBytesUsed,version,owners(permissionId),driveId)"
                    ),
                )
                .execute()
            )
        except Exception as error:
            report["file_inventory_error"] = str(error)
            break
        items.extend(page.get("files", []))
        token = page.get("nextPageToken")
        if not token:
            break
    report["tensor_metadata_candidates"] = items
    report["inventory_truncated"] = bool(token)
    report["limitations"] = [
        "quota is for the authenticated principal; shared/pooled ownership must be checked",
        "metadata absence does not distinguish deletion, permissions or account mismatch",
        "mount df/fsync and cached reads do not establish Drive server persistence",
        "revisions/activity/retention policy remain owner-side investigation",
    ]
    return report
