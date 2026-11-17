# v7 canary correction — current Colab session only

Run these **two cells in the existing kernel**, after the v6 canary upload failed
on version1 versus version2. Do not rerun migration Cells1–4 or v6 Cells5A/5B.
No training or evaluation is started. No original checkpoint, receipt, config,
folder or remote file is overwritten, deleted, moved or recreated.

Upload from Linux:
`/home/tj330/programs/VLA-Project-analytic/runs/drive-canary-fix-v7/recovery-tools-v7.zip`

SHA-256: `d3626c42ceac5b93dc5ebc1b82c144aba7db1430b90cc669c5ed39784e2b1bd2`.
The archive has the unchanged processor snapshots and all recovery helpers; only
canary validation and token-broker reconnection behavior differ from v6.

## Cause and scope

The exact failing predicate was historical receipt-version equality. Google defines
`version` as a monotonically increasing file counter for every server-side change,
including changes invisible to the user; it is not a hash or content revision.
See [Drive File resource](https://developers.google.com/workspace/drive/api/reference/rest/v3/files).
Changing a canary's metadata can therefore change its version without changing bytes.
The code creates the receipt using a freshly generated, distinct file ID. It never
PATCHes/updates the canary after upload. Creating the sibling receipt is not a direct
canary metadata/content update in this implementation. The supplied record does not
identify which server action caused1→2: delayed server bookkeeping, metadata activity
or another actor remain possibilities, not proven causes. Equal bytes at two checks
also do not prove that no intervening write occurred. No live account access exists here.

The correction permits only a nondecreasing positive numeric historical version,
then requires the exact file ID, current second-account ownership, expected parent,
size and **full API media SHA-256**, plus matching metadata/version before and after
that read. A version change during verification, rollback, missing version, changed
content/owner/parent/size or server-SHA mismatch still fails closed. Higher version
alone never passes. Original receipt/config remain byte-identical; successful advances
are appended to a separate local `.revalidations.jsonl` audit log. Checkpoint upload,
COMPLETE validation, source selection and exporter integrity checks are unchanged.

The failed v6 setup stopped its token broker before returning CONFIG/folder. CellB
recovers those values from the existing config and restarts refresh using your already
authenticated second-account credentials. It creates no new remote artifacts. If Google
requires renewed consent, choose the second account; deny/invalid-scope is a STOP.

## CellA — load the v7 patch without repeating setup

```python
import hashlib
import importlib
import json
import sys
import urllib.parse
from pathlib import Path

from google.colab import files

assert all(
    name in globals()
    for name in ("DRIVE_RUN_ID", "context", "gold", "EXPECTED_SECOND_ACCOUNT_EMAIL", "safe_extract")
), "STOP: use the current kernel where v6 canary upload completed"
DRIVE_RUN_ID = globals()["DRIVE_RUN_ID"]
context = globals()["context"]
gold = globals()["gold"]
EXPECTED_SECOND_ACCOUNT_EMAIL = globals()["EXPECTED_SECOND_ACCOUNT_EMAIL"]
safe_extract = globals()["safe_extract"]
CONFIG = Path("/content") / (DRIVE_RUN_ID + "-drive-config.json")
assert CONFIG.is_file(), "STOP: retain the original v6 config; do not create a new one"
assert not (Path("/content") / (DRIVE_RUN_ID + "-recovery")).exists(), (
    "STOP: already-prepared recovery needs its existing processor-path binding preserved"
)
V7_HELPERS = Path("/content") / (DRIVE_RUN_ID + "-helpers-v7")
PATCH7 = Path("/content") / (DRIVE_RUN_ID + "-patch-v7.zip")
assert not V7_HELPERS.exists() and not PATCH7.exists(), "Preserve existing v7 attempts"
uploaded = files.upload()  # select ONLY recovery-tools-v7.zip
assert set(uploaded) == {"recovery-tools-v7.zip"}
assert (
    hashlib.sha256(uploaded["recovery-tools-v7.zip"]).hexdigest()
    == "d3626c42ceac5b93dc5ebc1b82c144aba7db1430b90cc669c5ed39784e2b1bd2"
)
with PATCH7.open("xb") as stream:
    stream.write(uploaded["recovery-tools-v7.zip"])
safe_extract(PATCH7, V7_HELPERS)
sys.path.insert(0, str(V7_HELPERS))
importlib.invalidate_caches()
for module_name in ("drive_handoff", "drive_handoff_setup"):
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve().parent == V7_HELPERS.resolve()
DRIVE_HELPERS = V7_HELPERS  # subsequent existing Cell6 uses the versioned helpers/assets
print("V7_PATCH_LOADED: original v6 files and completed Cells1-4 preserved")
```

Expected: `V7_PATCH_LOADED`. STOP on a different SHA, missing config or already-prepared
recovery directory. Do not delete anything to satisfy the checks.

## CellB — revalidate the original canary and reconnect its broker

```python
drive_handoff = sys.modules["drive_handoff"]
drive_handoff_setup = sys.modules["drive_handoff_setup"]
config = json.loads(CONFIG.read_text(encoding="utf-8"))
receipt_path = Path(config["canary_receipt"])
original_config_bytes = CONFIG.read_bytes()
original_receipt_bytes = receipt_path.read_bytes()
receipt = json.loads(original_receipt_bytes)
assert config["run_id"] == DRIVE_RUN_ID
assert receipt["bytes"] == 1_048_694
assert (
    receipt["sha256"]
    == config["canary_sha256"]
    == ("aa6b8846a8fdd5d5ac12d5b8950914f2cd4dddec04146d7cf5b3f1bf0d96127b")
), "STOP: this patch procedure is pinned to the reported existing canary"
if "BROKER_STOP" in globals():
    globals()["BROKER_STOP"].set()
# Refresh current second-account credentials; no folder/file creation or upload.
CONFIG, folder, BROKER_STOP = drive_handoff_setup.resume_existing(
    CONFIG, context, EXPECTED_SECOND_ACCOUNT_EMAIL
)
client = drive_handoff.DriveAPI(config)
assert CONFIG.read_bytes() == original_config_bytes
assert receipt_path.read_bytes() == original_receipt_bytes

# Read-only diagnostics; Drive version is not a content-revision identifier.
# Historical metadata at version1 is unavailable, so this cannot prove its cause.
file_id = urllib.parse.quote(receipt["remote_file_id"], safe="")
diagnostics = {
    "receipt": receipt,
    "config_sha256": hashlib.sha256(original_config_bytes).hexdigest(),
    "exact_server_side_version_trigger": "not established",
}
for label, suffix in (
    (
        "current_metadata",
        "files/"
        + file_id
        + "?"
        + urllib.parse.urlencode(
            {
                "fields": "id,version,headRevisionId,modifiedTime,viewedByMeTime,"
                "lastModifyingUser(permissionId),size,sha256Checksum,md5Checksum,parents,"
                "ownedByMe,owners(permissionId)"
            }
        ),
    ),
    (
        "revision_page",
        "files/"
        + file_id
        + "/revisions?"
        + urllib.parse.urlencode(
            {
                "pageSize": 1000,
                "fields": "nextPageToken,revisions(id,modifiedTime,size,md5Checksum,"
                "lastModifyingUser(permissionId))",
            }
        ),
    ),
):
    try:
        diagnostics[label] = client.json_request("GET", suffix)
    except drive_handoff.PersistenceError as error:
        diagnostics[label] = {"unavailable": str(error)}  # diagnostics only; gate already passed
inspection = CONFIG.with_name(CONFIG.stem + "-v7-inspection.json")
with inspection.open("x", encoding="utf-8") as stream:
    json.dump(diagnostics, stream, indent=2)
print("V7_EXISTING_CANARY_PASS: ID/owner/parent/size/full-SHA/stable-read verified")
print("Preserved receipt:", receipt_path)
print("Revalidation log:", receipt_path.with_suffix(".revalidations.jsonl"))
print("Read-only inspection:", inspection)
print("STOP: no training/evaluation started. Readiness Cell6/7 may follow; do not run old5A/5B.")
```

Expected for the reported canary:
- `DRIVE_CANARY_CONTENT_REVALIDATED` with receipt_version1, readback_version2,
  bytes1048694 and SHAaa6b8846…d96127b (a later stable version is accepted only after
  the same complete content verification, never by version number alone).
- `EXISTING_DRIVE_CANARY_VERIFIED_NO_NEW_UPLOADS`, then `V7_EXISTING_CANARY_PASS`.
- Original receipt/config SHA unchanged; new audit and read-only inspection files.

Any integrity, identity, auth or token-refresh error is a STOP. Do not edit the receipt
version to force a pass. Diagnostic revisions/metadata are best-effort GETs and cannot
prove a historical server trigger without the absent earlier metadata/activity record.
No new OAuth scope or service is introduced. Keep the kernel alive for refreshed tokens.

After both cells pass, use the **existing v6 readiness Cells6 and7**, in order, if they
have not run. They will now use DRIVE_HELPERS=v7 and the original CONFIG/canary/folder.
They audit/prepare017000 locally and check actual CUDA restoration with zero updates.
Leave training Cell8 commented out. Do not recreate the canary or rerun migration.

The model/dataset, target20000, source17000, batch8, seed0, save1000, approved80 and all
frozen evaluation criteria are unchanged. Drive retention risk and the need to promptly
copy the final portable export to Linux remain. No GPU training or TEST access occurred.

## Verification performed locally

377 CPU tests passed, one legacy TEST-geometry generation test excluded. Focused
transport/recovery/notebook tests:74 passed. Scoped Ruff and global formatting pass;
the inherited E402 in the unchanged approved training notebook remains. v7 ZIP
read-back verifies SHA/CRC and all helper contents. Only drive_handoff.py and
drive_handoff_setup.py differ from preserved v6; all processor/source checkpoint
guards and exporter bytes are unchanged. No live authenticated API operation,
GPU restoration, optimizer update or evaluation was executed locally.
