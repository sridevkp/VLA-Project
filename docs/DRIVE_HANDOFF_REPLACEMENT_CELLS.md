# Drive API checkpoint handoff — replacement cells after completed migration Cells 1–4

**Existing canary version mismatch:** if 5B uploaded successfully but rejected
receipt version1/current version2, apply [the two v7 correction cells](DRIVE_CANARY_V7_CURRENT_SESSION.md).
Do not repeat 5A/5B or discard the existing folder/config/receipt. Then continue
readiness Cells6/7 using the v7 helpers, with training still disabled.

**Do not rerun completed setup.** Paste these cells into the CURRENT migration
notebook after Cell 4. The companion replacement notebook is a copy source, not
an independent fresh-runtime workflow. Existing source/data/venv and verified
account context are reused. Do not run the old SSH Cells 5–8.

Upload the small patch ZIP from Linux:
`/home/tj330/programs/VLA-Project-analytic/runs/drive-handoff-prep-v1/recovery-tools-v6.zip`.
SHA sidecar is beside it. Original approved files,017000 backup and all previous
recovery evidence remain untouched. Training stays local; no SSH/VPS/tunnel setup.

## Approvals and remaining blockers BEFORE optimizer updates

- The user has requested new-account-owned API output. When executing Replacement 5B,
  approve the Google authorization for read access plus app-created file/folder
  writes in the SECOND account. The helper requests `drive.readonly` + `drive.file`,
  not a new infrastructure service. If current Colab credentials lack that consent,
  a 403/invalid-scope error is a STOP: reauthorize correctly, never bypass ownership.
- A NEW folder is created directly under the second account's My Drive root, not
  inside the original shared folder. Principal ID, ownership and parent are checked
  before uploads. No existing file is updated/deleted; all remote objects use new IDs.
- Need about 6.1 GB free for three complete checkpoints and final portable export,
  plus input ZIP space if uploaded there and any prior attempts. API quota is
  checked before training and before each upload. Do not delete old evidence to fit.
- The 1 MiB API upload/read-back canary replaces the SSH full-size canary to save
  time. It verifies the new write/read path, NOT large-upload throughput. No real
  Google upload has been exercised by this local agent; mocked API tests are not
  evidence of your session's permissions or persistence.
- 017000 must reach local storage. Prefer the already-shared 017000 ONLY when all
  files and COMPLETE hash match the independently verified Linux manifest. Otherwise
  supply the verified Linux ZIP (existing local copy or second-account Drive UI
  upload and file ID). This is an input transfer, never a different checkpoint.
- Actual T4 model/AdamW/scheduler/RNG restoration at 17000, zero updates, remains
  mandatory. The training command below is deliberately commented out.

## Verification mechanism and time tradeoff

8MiB resumable API chunks (256KiB-aligned) use bounded status probes after transient
errors. Completion requires a final200/201, then fresh file metadata: owner,
parent, size, version and checksums. Server SHA256 must equal local SHA256 when
exposed. Otherwise the helper streams an independent `alt=media` read-back from
Drive API and hashes every byte; MD5 alone never passes. The small canary always
exercises full API read-back. Mounted-filesystem cache reads are not verification.
Receipts are also uploaded/read-back verified beside artifacts; no fsync claim is
made for Drive. On verification failure, the callback aborts before another update.
It does not repeatedly create duplicate multi-GB objects after a failed gate.

At 18000/19000, archive full model/optimizer/scheduler/RNG, normalization and processor
snapshot plus logs/provenance. At 20000 upload/verify PORTABLE MODEL FIRST, then full
resumable state. Exact target20000 / batch 8 / seed 0 / save 1000 / approved 80 remain unchanged.
The worker uses only stdlib HTTPS; the existing Colab-kernel auth library refreshes
an access-token file every minute. It is 0600, outside checkpoints/archives and never
uploaded. Keep the kernel alive. An expired broker token fails persistence closed.

Plan ~47 minutes compute from earlier preflight PLUS about 6 GB upload and potentially
another 6 GB download/read-back when server SHA is unavailable. No claim is made that
this fits the remaining session; do not shorten the run or alter frozen settings.
API verification is point-in-time proof, not indefinite-retention or independent-
provider storage. Account quota/policy, later deletion and session expiry remain
risks. Download the verified final portable ZIP to Linux immediately after completion.

Google references: [resumable uploads](https://developers.google.com/workspace/drive/api/guides/manage-uploads),
[file checksums/ownership](https://developers.google.com/workspace/drive/api/reference/rest/v3/files).

# Drive API handoff — replacement cells after migration Cells 1–4
Run these in the SAME existing notebook/kernel, not a new session. Cells 1–4 remain completed.
No SSH, tunnel, VPS or reinstall. Keep the T4 session alive. Original source/checkpoint/evidence stay intact.
All executable cells stop before optimizer updates; the training command is commented out.


## Replacement 5A — load only the small v6 operational patch
Upload recovery-tools-v6.zip from Linux runs/drive-handoff-prep-v1/ using the chooser below.
Do NOT upload the large 017000 ZIP here. No approved source/data/environment setup is rerun.


```python
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from google.colab import files

assert all(
    name in globals()
    for name in (
        "context",
        "gold",
        "RUN_ID",
        "DRIVE_ROOT",
        "SOURCE",
        "DATASET",
        "STAGING",
        "PYTHON",
        "versions",
        "safe_extract",
        "EXPECTED_SECOND_ACCOUNT_EMAIL",
    )
), "STOP: return to the existing session where migration Cells1-4 passed"
context = globals()["context"]
gold = globals()["gold"]
RUN_ID = globals()["RUN_ID"]
DRIVE_ROOT = globals()["DRIVE_ROOT"]
SOURCE = globals()["SOURCE"]
DATASET = globals()["DATASET"]
STAGING = globals()["STAGING"]
PYTHON = globals()["PYTHON"]
versions = globals()["versions"]
safe_extract = globals()["safe_extract"]
EXPECTED_SECOND_ACCOUNT_EMAIL = globals()["EXPECTED_SECOND_ACCOUNT_EMAIL"]
assert PYTHON.is_file() and SOURCE.is_dir() and DATASET.is_dir()
DRIVE_RUN_ID = RUN_ID + "-drive-v1"  # new, unused; preserve all earlier attempts
DRIVE_HELPERS = Path("/content") / (DRIVE_RUN_ID + "-helpers-v6")
uploaded = files.upload()  # select ONLY recovery-tools-v6.zip (~small metadata/code)
assert set(uploaded) == {"recovery-tools-v6.zip"}
patch_bytes = uploaded["recovery-tools-v6.zip"]
assert hashlib.sha256(patch_bytes).hexdigest() == (
    "cc30fc162fae8f23462c32d71195bd26390e23a08631eb1c3e1a6a8d36a3b47d"
)
PATCH = Path("/content") / (DRIVE_RUN_ID + "-patch.zip")
with PATCH.open("xb") as stream:
    stream.write(patch_bytes)
safe_extract(PATCH, DRIVE_HELPERS)
sys.path.insert(0, str(DRIVE_HELPERS))

print("V6_PATCH_VERIFIED: completed setup reused; SSH unused")
```

## Replacement 5B — authorize API writes, create a NEW second-owned folder and verify a 1 MiB canary
This is the only additional Google authorization: read existing input and create app-owned output files.
Choose the second account again if prompted. If scopes/creation are denied, STOP; do not skip checks.
The broker refreshes a short-lived bearer token in a 0600 local file outside all archives. Never upload it.


```python
from drive_handoff import DriveAPI
from drive_handoff_setup import setup

CONFIG, folder, BROKER_STOP = setup(context, EXPECTED_SECOND_ACCOUNT_EMAIL, DRIVE_RUN_ID, gold)
config = json.loads(CONFIG.read_text(encoding="utf-8"))
client = DriveAPI(config)
print("API_DESTINATION_READY: second-account-owned, read-back verified")
```

Expected: new folder owned by the second principal; successful resumable upload and actual API
SHA-256 read-back of the small canary. This is not a full-size upload speed test. Quota must have
about 6.1 GB free for the new checkpoints/export (more if a Linux input ZIP is uploaded to this account).
No mounted-Drive writes are used. STOP on any auth, ownership, quota, size, checksum or receipt error.

## Replacement 6 — stage only bytes matching the Linux 017000 backup, then prepare
First try the shared 017000, read-only, against EVERY Linux-pinned file hash AND COMPLETE hash.
If unavailable, either provide an existing local Linux-golden ZIP or upload that verified ZIP via
Drive UI while signed into the second account (e.g. into the new folder above), then paste its file ID.
Linux ZIP: runs/durable-recovery-prep-v1/checkpoint- 017000.zip, 1,610,503,770 bytes,
SHA e34c9596d0fdaef579b734829d45a60c257370d9de4e17348055de6734366109.
No incomplete 018000–020000 is read/selected. All original files remain untouched.


```python
from checkpoint_guard import sha256, verify_checkpoint

EXISTING_LOCAL_GOLD_ZIP = ""  # optional: already verified Linux ZIP in this runtime
GOLDEN_INPUT_FILE_ID = ""  # only needed if shared017000 is incomplete and no local ZIP
INPUT = Path("/content") / (DRIVE_RUN_ID + "-input")
ORIGINAL = INPUT / "training-tiago-v1"
RECOVERY = Path("/content") / (DRIVE_RUN_ID + "-recovery")
assert not INPUT.exists() and not RECOVERY.exists(), "Preserve existing attempts"
required = tuple(gold["files_sha256"])
shared17 = DRIVE_ROOT / "training-tiago-v1/checkpoints/017000"
shared_ok = False
try:
    record = verify_checkpoint(shared17, required)
    shared_ok = (
        record["files_sha256"] == gold["files_sha256"]
        and sha256(shared17 / "COMPLETE.json") == gold["source_complete_sha256"]
    )
except (OSError, ValueError, KeyError, TypeError):
    pass
saved = ORIGINAL / "checkpoints/017000"
if shared_ok:
    shutil.copytree(shared17, saved)  # source read only; hashes are pinned by Linux backup
else:
    if EXISTING_LOCAL_GOLD_ZIP:
        input_zip = Path(EXISTING_LOCAL_GOLD_ZIP)
    else:
        assert GOLDEN_INPUT_FILE_ID, (
            "STOP: provide the Linux-golden ZIP; do not use incomplete checkpoints"
        )
        input_zip = Path("/content") / (DRIVE_RUN_ID + "-golden017000.zip")
        client.download_input(GOLDEN_INPUT_FILE_ID, input_zip, gold["bytes"], gold["sha256"])
    assert input_zip.stat().st_size == gold["bytes"]
    assert sha256(input_zip) == gold["sha256"]
    with zipfile.ZipFile(input_zip) as archive:
        assert set(archive.namelist()) == set(required) | {"COMPLETE.json"}
    safe_extract(input_zip, saved)
record = verify_checkpoint(saved, required)
assert record["files_sha256"] == gold["files_sha256"]
assert sha256(saved / "COMPLETE.json") == gold["source_complete_sha256"]
assert sha256(STAGING / "source.zip") == (
    "2d3cb4bd06ac8313b1bb3cdb5aae7f916c2b614dd3884172c4570be9ff36b2f5"
)
shutil.copyfile(STAGING / "source.zip", INPUT / "source.zip")
shutil.copyfile(
    DRIVE_HELPERS / "evidence/original-optimizer-steps.jsonl", ORIGINAL / "optimizer-steps.jsonl"
)
SESSION_CONTEXT = Path("/content") / (DRIVE_RUN_ID + "-account-session.json")
session = dict(context)
session["installed_versions"] = versions
session["handoff_transport"] = "drive_api"
session["destination_folder"] = folder
session["api_canary_receipt_sha256"] = sha256(Path(config["canary_receipt"]))
with SESSION_CONTEXT.open("x", encoding="utf-8") as stream:
    json.dump(session, stream, indent=2)
COMMAND = [str(PYTHON), str(DRIVE_HELPERS / "recover_training.py")]
PATHS = [
    "--project",
    str(SOURCE),
    "--dataset",
    str(DATASET),
    "--original",
    str(ORIGINAL),
    "--recovery",
    str(RECOVERY),
    "--processor-dir",
    str(DRIVE_HELPERS / "processor"),
    "--session-context",
    str(SESSION_CONTEXT),
]
subprocess.run([*COMMAND, "audit", *PATHS], check=True)
subprocess.run([*COMMAND, "prepare", *PATHS], check=True)
print("LINUX_PINNED_017000_LOCAL_PREPARATION_VERIFIED", RECOVERY)
```

If source verification/copy fails, STOP and preserve the partial attempt; do not delete evidence or
substitute another checkpoint. No setup from completed Cells 1–4 is rerun.

## Replacement 7 — actual T4 restoration, zero optimizer updates


```python
gpu_check = """
import torch
assert torch.cuda.is_available(), 'STOP: CUDA unavailable'
assert 'T4' in torch.cuda.get_device_name(0), 'STOP: approved T4 required'
print('ACTUAL_T4_AVAILABLE', torch.cuda.get_device_name(0))
"""
subprocess.run([str(PYTHON), "-c", gpu_check], check=True)
subprocess.run([*COMMAND, "restore-check", *PATHS], check=True)
restored = json.loads((RECOVERY / "restore-check.json").read_text())
assert restored["restored_step"] == 17000 and restored["device"] == "cuda"
assert restored["optimizer_updates_at_restore_gate"] == 0
assert restored["actual_installed_smolvla"] and restored["scheduler_verified"]
assert restored["adamw_parameter_groups_verified"]
assert restored["model_tensor_values_verified"] > 0
assert restored["optimizer_tensor_values_verified"] > 0
assert restored["rng_tensors_verified"] > 0
assert restored["all_checkpoint_files_verified"] == len(required)
assert restored["relocated_processor_directory"] == str(DRIVE_HELPERS / "processor")
print("STOP_READY: model/AdamW/scheduler/RNG restored17000; zero optimizer updates")
```

## Replacement 8 — manual future launch only (commented out)
Run only after all preceding gates pass and the owner chooses to launch. Target20000, batch 8, seed 0,
approved 80 episodes and save 1000 remain unchanged. The callback uploads full018000/019000 synchronously;
at 20000 it uploads/verifies the portable inference model FIRST, then full resumable state.
Each upload requires API ownership/size/checksum verification; absent server SHA requires streamed
API SHA-256 read-back. MD5 alone never passes. Verified receipts are uploaded beside the artifacts.


```python
# No training starts as delivered. Owner manually launches only after all gates pass:
# subprocess.run([*COMMAND, 'train', *PATHS, '--execute-recovery',
#                 '--durability-config', str(CONFIG)], check=True)
```

Expected future first event:17001/restored17000/calls1. Final:20000/restored17000/calls3000.
Expect DRIVE_API_ARTIFACT_VERIFIED and DRIVE_API_CHECKPOINT_VERIFIED logs with file IDs.
Keep the notebook kernel alive for token refresh. Do not copy the token file into Drive or archives.
If an API gate fails, training stops; preserve local/remote partial objects and logs. Do not blindly
restart from 017000 after a durable 018000/019000; the existing partial-resume limitation remains.
If final portable succeeds but full state fails, preserve/download the portable and repair upload
only—do not redo optimizer updates. No automatic evaluation or TEST use.

Remaining risk: API verification is point-in-time evidence, not proof of indefinite persistence.
The new owner's quota/policy, future deletion and Colab session expiry can still cause loss. Download
the single verified final portable ZIP to Linux immediately after completion and verify SHA there.
Large upload speed is not measured by the 1 MiB canary. Plan ~47 minutes compute PLUS about 6 GB upload,
and potentially up to 6 GB API read-back when server SHA is absent. Do not reduce frozen training settings.


## After a successful final handoff

1. Open the new folder URL printed by Replacement 5B, while signed into the second account.
2. Download the file ending `-portable-020000-attempt-1.zip` and its
   `.receipt.json` immediately to Linux. Keep the resumable checkpoint ZIPs and
   receipts as well; no original or partial artifacts should be deleted.
3. On Linux, compare the portable ZIP's size and SHA-256 with the receipt's
   `bytes` and `sha256` fields. The console `DRIVE_API_ARTIFACT_VERIFIED` event
   contains the same values and the exact remote file ID. A matching download
   establishes the independent local copy; a Drive listing alone does not.
4. Preserve the full Colab log, restore-check record, progress and all receipts.
   Strict local import/inference validation is still required before evaluation.
   This workflow does not launch evaluation.

A failed upload or expired authentication is a stop, not permission to reduce
checkpoint cadence or resume blindly. Preserve the local checkpoint and repair
persistence first. No additional SSH credentials, public endpoint or VPS approval
is needed. The remaining owner actions are Google consent, sufficient storage,
and manually launching the commented command only after all stop gates pass.

## Local preparation verification

PASS: 363 CPU tests; one legacy TEST-geometry generation test deliberately
excluded. Focused recovery/transport checks: 60 passed. Scoped Ruff and global
format check pass. Global lint retains one pre-existing E402 in the approved
training notebook, which is unchanged. No live Google upload, T4 restoration,
optimizer update or evaluation was executed locally.

The v6 ZIP is 1,236,675 bytes; its SHA-256 is
`cc30fc162fae8f23462c32d71195bd26390e23a08631eb1c3e1a6a8d36a3b47d`.
It contains 12 source helpers and all 11 pinned processor assets, with archive
read-back checked against local source and the delivery manifest.

GO for running the replacement readiness cells; NO-GO for optimizer updates
until their live account, persistence, source-integrity and CUDA gates pass.
