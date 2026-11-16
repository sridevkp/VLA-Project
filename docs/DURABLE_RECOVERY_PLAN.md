# Durable recovery plan — second-account migration (v5)

**Prepared, not executed. NO-GO until the new session passes all stop gates.**
No training, evaluation, credential creation or service startup is performed by
this preparation. The original approved notebook, source/data bundles, 017000
backup and prior recovery artifacts are unchanged. The v4 procedure is retained
in `docs/history/DURABLE_RECOVERY_PLAN_v4.md` and Git history.

## Blockers before GPU execution

- The second account's identity, sharing/download permissions and T4 availability
  have not been tested by this local agent. The owner must supply its primary
  email and the shared folder/new shortcut ID in Cell1, and verify the compute
  account in the Colab UI separately.
- Existing Linux private receiver/key setup remains owner-controlled. No SSH
  listener or keys were created in this migration. The new Colab session must
  have an approved private route, authorized key and independently verified
  host-key fingerprint, then a NEW full-size canary and capacity receipt.
- Sharing the original folder does not transfer its ownership/storage use to the
  second account. Do not infer that its quota fixes the original disappearance.
  The workflow reads Drive only and transfers new checkpoints to Linux.
- Actual installed CUDA restoration at17000 is mandatory; CPU unit tests and the
  COMPLETE marker alone do not satisfy it. No incomplete Drive18/19/20 is used.
- Recovery after a durable18/19 interruption remains deliberately blocked by the
  current single-invocation17→20 gate; preserve the later checkpoint and request
  a lineage/remaining-update audit, never silently reset or relabel.

The earlier T4 preflight estimated roughly47 minutes for3000 updates, before
archive/transfer overhead. New-account throughput and session limits are unverified.
Measure the new canary; do not substitute old local timings or shorten the run.

## Account-specific assumptions and replacements

| Original assumption | Migration behavior |
|---|---|
| Original account already owns the runtime | Fresh second-account runtime; explicit UI confirmation; actual T4 check later |
| Mount and API share one credential | Authenticate BOTH prompts as second account; API primary-email check; do not claim programmatic compute/mount identity proof |
| Fixed original shortcut ID/path | Owner-selected new shortcut; resolve once; follow API shortcut target; record canonical path and ownership |
| Folder name alone identifies data | API child IDs/checksums and immutable SHA-256 bind all three mounted payloads |
| Shared folder is writable / owns quota | Reader+download is sufficient; no Drive writes; record owner IDs and authenticated-principal quota separately |
| Old policy VLM path is available | Hash-verified 11-file metadata snapshot staged locally; in-memory path override only |
| Saved tokenizer path is available | Tokenizer asset address overridden locally; normalization/device/options preserved |
| Saved output/pretrained paths remain valid | Standard resume uses the copied Linux017000 and overrides output to a new local directory |
| Installation log can be written to Drive | Local versions log included in new account-session provenance |
| Old SSH receipt remains valid | New unused RUN_ID, key/host pin verification, full1.61GB canary bound to this transport config |
| Original notebook setup is reusable unchanged | Separate migration notebook: same pinned stack/payloads, no Drive log write, no deletion, T4 check deferred |

Google references: [shared-file ownership/quota](https://support.google.com/drive/answer/2375057?hl=en),
[Drive shortcut/metadata fields](https://developers.google.com/workspace/drive/api/reference/rest/v3/files).
An API principal check proves only that API identity. The operator must select the
second account for the mount and compute UI. Frozen payload checks protect data
integrity even though the mount's OAuth principal is not programmatically attested.

## Files and owner actions

Use `notebooks/smolvla_account_migration.ipynb` (new notebook; approved training
notebook unchanged). In a browser profile signed into the second account, open
Colab, select File → Upload notebook, and choose that local file. The original
owner must share the original project folder with download permission and add
these NEW small artifacts without overwriting approved files:

- Linux: `/home/tj330/programs/VLA-Project-analytic/runs/account-migration-prep-v1/recovery-tools-v5.zip`
- Linux: `/home/tj330/programs/VLA-Project-analytic/runs/account-migration-prep-v1/recovery-tools-v5-sha256.json`
- Drive destination: the shared `vlarobot_smolvla/` folder, same basenames.

Do not upload/copy the incomplete Drive checkpoints. The unchanged golden input
stays at `/home/tj330/programs/VLA-Project-analytic/runs/durable-recovery-prep-v1/checkpoint-017000.zip`:
1,610,503,770 bytes, SHA-256
`e34c9596d0fdaef579b734829d45a60c257370d9de4e17348055de6734366109`.
It is pulled directly over SSH. Private receiver provisioning/host-key steps
remain in DURABLE_READINESS_REPORT.md and require owner approval; do not expose SSH.

The only owner edits are account/email/share ID/shortcut path, an unused RUN_ID,
UI confirmation, and approved SSH route/fingerprint fields. Do not change the
source/data SHA pins, model settings or training parameters. A runtime switch
requires a fresh run ID and all checks again. No old receipt may be reused.

The exact copy-paste cells below are also the notebook cells. Run in order.
All executable cells stop before an optimizer update. The training command is
commented out and needs an explicit future owner launch. No evaluation cell exists.

# SmolVLA account migration: verified 017000 durable recovery
New account authentication and shared-folder access do not change ownership or fix Drive persistence.
Use a NEW runtime signed into the second account. Original checkpoint files remain untouched.
Run cells in order. Only the final, commented command could start training; Run All cannot train.
Do not run the old notebook's mount/setup/train/export cells in this workflow.


## 1. Owner configuration and authentication
Fill the second account's PRIMARY email, the shared folder (or new shortcut) ID from its Drive URL,
and your new shortcut path. Confirm the Colab UI account separately: API identity does not prove
which account owns the compute session. Share permission must allow downloading approved files.
Before running, the original owner should add recovery-tools-v5.zip to the shared folder.
No Editor role is required for this read-only workflow. SSH keys/network membership are separate
owner-approved prerequisites; none is created by these cells.


```python
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from google.colab import auth, drive

EXPECTED_SECOND_ACCOUNT_EMAIL = "SET_SECOND_ACCOUNT_PRIMARY_EMAIL"
SHARED_FOLDER_ENTRY_ID = "SET_SHARED_FOLDER_OR_SHORTCUT_ID"
PROJECT_ENTRY = Path("/content/drive/MyDrive/vlarobot_smolvla")
RUN_ID = "acct2-durable-v1"  # MUST be unused on Linux and in this runtime
CONFIRMED_SECOND_COMPUTE_ACCOUNT = False  # set True after checking the Colab UI
SSH_ROUTE_APPROVED = False  # set True only after owner-approved private route setup
EXPECTED_HOST_KEY_SHA256 = "SET_FROM_TRUSTED_LINUX_CONSOLE"
SSH_DESTINATION = "tj330@100.86.97.108"  # private candidate; must be revalidated
SSH_PORT = 2222
SOURCE = Path("/content/VLA-Project")
DATASET = Path("/content/vlarobot-data")
STAGING = Path("/content/vla-payloads")
PYTHON = Path("/content/vla-env/bin/python")
assert CONFIRMED_SECOND_COMPUTE_ACCOUNT
assert "SET_" not in EXPECTED_SECOND_ACCOUNT_EMAIL
assert "SET_" not in SHARED_FOLDER_ENTRY_ID
assert RUN_ID.replace("-", "").isalnum()
drive.mount("/content/drive", force_remount=True)
auth.authenticate_user()  # choose the SECOND account here too
print("AUTHENTICATION_COMPLETE: API account still must pass Cell 3")
```

Expected: both Google prompts select the second account. STOP on wrong-account UI or missing share.
Add a shortcut in the second account's My Drive if needed; do not move/duplicate original checkpoints.
The shortcut name/path may differ; edit PROJECT_ENTRY only. No old shortcut ID is assumed.

## 2. Verify operational helper payload; pin this session's canonical root


```python
DRIVE_ROOT = PROJECT_ENTRY.resolve(strict=True)  # resolve the trusted entrypoint ONCE
assert DRIVE_ROOT.is_dir()
HELPERS = Path("/content") / (RUN_ID + "-helpers-v5")
HELPER_SHA256 = "645cc7121449ebc899c19aa52029b535a5e4dc3f3e0e49ed2c65b2851fc86846"
helper_zip = DRIVE_ROOT / "recovery-tools-v5.zip"
assert hashlib.sha256(helper_zip.read_bytes()).hexdigest() == HELPER_SHA256


def safe_extract(archive_path, destination):
    assert not destination.exists(), "Preserve existing files; use a new runtime/run ID"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            assert not name.is_absolute() and ".." not in name.parts
            assert chr(92) not in info.filename and ":" not in info.filename
            assert stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)


safe_extract(helper_zip, HELPERS)
sys.path.insert(0, str(HELPERS))
migration_assets = __import__("account_migration")
migration_assets.verify_processor(HELPERS / "processor")
gold = json.loads((HELPERS / "checkpoint-017000.sha256.json").read_text())
assert gold["step"] == 17000
assert gold["sha256"] == ("e34c9596d0fdaef579b734829d45a60c257370d9de4e17348055de6734366109")
print("HELPERS_AND_PROCESSOR_VERIFIED", DRIVE_ROOT)
```

STOP on any helper/processor hash mismatch, unsafe member, or existing destination.
The canonical root is recorded, not compared with the first account's mount spelling.

## 3. Bind the new API account, shared target, mounted bytes and approved payloads


```python
import google.auth
from account_migration import account_context, stage_payloads
from drive_diagnostics import inspect_drive
from googleapiclient.discovery import build

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"]
)
service = build("drive", "v3", credentials=credentials, cache_discovery=False)
context = account_context(service, SHARED_FOLDER_ENTRY_ID, EXPECTED_SECOND_ACCOUNT_EMAIL)
folder_id = context["target_folder"]["id"]
PINS = {
    "bundle-status.json": "b335fa8a4825688fcfeb539dd7c9539c73dccc35bcf98892c0a94482d79939ff",
    "source.zip": "2d3cb4bd06ac8313b1bb3cdb5aae7f916c2b614dd3884172c4570be9ff36b2f5",
    "dataset-train-v1.zip": "a2bc0842380ebc21440b41494b04b2ccba44961cd329fade9dd86f8976e7577c",
}
context["payload_bindings"] = stage_payloads(service, folder_id, DRIVE_ROOT, STAGING, PINS)
context["canonical_drive_root"] = str(DRIVE_ROOT)
context["operator_confirmed_second_compute_account"] = CONFIRMED_SECOND_COMPUTE_ACCOUNT
context["run_id"] = RUN_ID
DRIVE_CONTEXT = Path("/content") / (RUN_ID + "-drive-context.json")
with DRIVE_CONTEXT.open("x", encoding="utf-8") as stream:
    json.dump(context, stream, indent=2)
diagnostic = inspect_drive(service, folder_id)
(Path("/content") / (RUN_ID + "-drive-diagnostics.json")).write_text(
    json.dumps(diagnostic, indent=2), encoding="utf-8"
)
print("ACCOUNT_AND_SHARED_PAYLOADS_VERIFIED", context["drive_api_principal_permission_id"])
print("Shared target/owners:", context["target_folder"])
```

Expected: new-account permission ID, original shared target/owners, three API-bound payloads.
STOP on wrong API account, inaccessible share, duplicate names, blocked download, or ANY hash mismatch.
The new principal's quota is not proof of the original owner's available storage. No Drive writes occur.

## 4. Recreate the approved local environment and validate the unchanged dataset
CPU-capable setup; no diagnostic optimizer updates. Requirements/uv/Python pins are unchanged.
Installed-package logs are local, so read-only sharing is sufficient.


```python
manifest = json.loads((STAGING / "bundle-status.json").read_text())
assert manifest["status"] == "approved-payload-ready-awaiting-Colab-authentication"
assert manifest["source_zip_sha256"] == PINS["source.zip"]
assert manifest["dataset_zip_sha256"] == PINS["dataset-train-v1.zip"]
safe_extract(STAGING / "source.zip", SOURCE)
safe_extract(STAGING / "dataset-train-v1.zip", DATASET)
subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.13.7"], check=True)
UV = [sys.executable, "-m", "uv"]
if shutil.which("ffmpeg") is None:  # same system dependency as approved notebook
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
assert not PYTHON.parent.parent.exists(), "Use a fresh runtime; do not replace an existing env"
version = (SOURCE / ".python-version").read_text(encoding="utf-8").strip()
subprocess.run([*UV, "venv", "--python", version, str(PYTHON.parent.parent)], check=True)
subprocess.run(
    [
        *UV,
        "pip",
        "install",
        "--python",
        str(PYTHON),
        "-r",
        str(SOURCE / "model/compat/requirements-colab-py311.txt"),
    ],
    check=True,
)
subprocess.run(
    [
        *UV,
        "pip",
        "install",
        "--python",
        str(PYTHON),
        "--no-deps",
        "-e",
        str(SOURCE),
    ],
    check=True,
)
versions = subprocess.check_output(
    [*UV, "pip", "freeze", "--python", str(PYTHON)], text=True, encoding="utf-8"
)
(Path("/content") / (RUN_ID + "-installed.txt")).write_text(versions, encoding="utf-8")
approval_code = """
from pathlib import Path
from vlarobot.vla.training import validate_training_approval
import sys
approval = validate_training_approval(Path(sys.argv[1]))
assert len(approval['episodes']) == 80
print('APPROVED_DATASET_VERIFIED: 80 successful TRAIN episodes')
"""
subprocess.run([str(PYTHON), "-c", approval_code, str(DATASET)], check=True)
```

STOP on setup/version/dataset errors. Do not change dependencies, dataset, model or batch settings.
Changing runtime discards scratch storage: start a fresh session with a new unused RUN_ID and repeat
these verifications, including SSH/canary. Do not silently reuse another account's session receipts.

## 5. Revalidate the private SSH endpoint and full-size transfer from THIS session
Owner must already have approved/provisioned the private route, key and host-key file (see
DURABLE_READINESS_REPORT.md). This cell creates no SSH credentials and starts no Linux service.


```python
assert SSH_ROUTE_APPROVED, "STOP: owner-approved private route required"
assert "SET_" not in EXPECTED_HOST_KEY_SHA256
CONFIG = Path("/content") / (RUN_ID + "-ssh.json")
cfg = {
    "destination": SSH_DESTINATION,
    "port": SSH_PORT,
    "identity_file": "/content/authorized_recovery_key",
    "known_hosts": "/content/verified_known_hosts",
    "remote_python": "/home/tj330/programs/VLA-Project/.venv/bin/python",
    "remote_helper": "/home/tj330/programs/VLA-Project-analytic/tools/recovery/durable_transfer.py",
    "remote_root": "/home/tj330/programs/VLA-Project/runs/independent-recovery-v1",
    "run_id": RUN_ID,
    "timeout_s": 1800,
    "canary_receipt": str(Path("/content") / (RUN_ID + "-canary.json")),
    "canary_sha256": gold["sha256"],
}
assert Path(cfg["identity_file"]).is_file() and Path(cfg["known_hosts"]).is_file()
fingerprints = subprocess.check_output(
    ["ssh-keygen", "-lf", cfg["known_hosts"]], text=True, encoding="utf-8"
).splitlines()
assert len(fingerprints) == 1 and fingerprints[0].split()[1] == EXPECTED_HOST_KEY_SHA256
os.chmod(cfg["identity_file"], 0o600)
with CONFIG.open("x", encoding="utf-8") as stream:
    json.dump(cfg, stream, indent=2)
INPUT_ZIP = Path("/content") / (RUN_ID + "-checkpoint-017000.zip")
TRANSFER = [str(PYTHON), str(HELPERS / "durable_transfer.py")]
REMOTE_INPUT = (
    "/home/tj330/programs/VLA-Project-analytic/runs/durable-recovery-prep-v1/checkpoint-017000.zip"
)
subprocess.run(
    [
        *TRANSFER,
        "pull",
        "--file",
        REMOTE_INPUT,
        "--out",
        str(INPUT_ZIP),
        "--expected-sha256",
        gold["sha256"],
        "--bytes",
        str(gold["bytes"]),
        "--config",
        str(CONFIG),
    ],
    check=True,
)
subprocess.run(
    [
        *TRANSFER,
        "send",
        "--file",
        str(INPUT_ZIP),
        "--config",
        str(CONFIG),
        "--artifact-id",
        RUN_ID + "-canary-017000",
        "--receipt",
        cfg["canary_receipt"],
    ],
    check=True,
)
canary = json.loads(Path(cfg["canary_receipt"]).read_text())
assert canary["receiver_readback_verified"] and canary["file_fsync"]
assert canary["directory_fsync"] and canary["sha256"] == gold["sha256"]
assert canary["bytes"] == gold["bytes"]
SESSION_CONTEXT = Path("/content") / (RUN_ID + "-account-session.json")
session = dict(context)
session["installed_versions"] = versions
session["installed_versions_sha256"] = hashlib.sha256(versions.encode("utf-8")).hexdigest()
session["host_key_fingerprint"] = EXPECTED_HOST_KEY_SHA256
session["ssh_endpoint"] = {"destination": SSH_DESTINATION, "port": SSH_PORT}
session["canary_receipt_sha256"] = hashlib.sha256(
    Path(cfg["canary_receipt"]).read_bytes()
).hexdigest()
with SESSION_CONTEXT.open("x", encoding="utf-8") as stream:
    json.dump(session, stream, indent=2)
print("NEW_SESSION_FULL_SIZE_CANARY_PASSED", canary)
print("Measured end-to-end transfer seconds:", canary["sender_elapsed_s"])

print(
    "Approximate 6 GB upload overhead seconds:",
    6_000_000_000 * canary["sender_elapsed_s"] / canary["bytes"],
)
```

Expected: 1,610,503,770-byte archive, golden SHA, receiver read-back/fsync and current capacity.
STOP if host-key pin, route/auth, capacity, source hash or receipt fails. No GPU update is allowed.

## 6. Stage ONLY the Linux-verified 017000 and prepare a fresh local recovery
No Drive checkpoint is selected. Saved files retain their original bytes; path relocation is
in-memory only and applies to hash-verified processor metadata plus tokenizer asset address.


```python
INPUT = Path("/content") / (RUN_ID + "-input")
ORIGINAL = INPUT / "training-tiago-v1"
RECOVERY = Path("/content") / (RUN_ID + "-recovery")
assert not INPUT.exists() and not RECOVERY.exists()
with zipfile.ZipFile(INPUT_ZIP) as archive:
    assert set(archive.namelist()) == set(gold["files_sha256"]) | {"COMPLETE.json"}
safe_extract(INPUT_ZIP, ORIGINAL / "checkpoints/017000")
shutil.copyfile(STAGING / "source.zip", INPUT / "source.zip")
shutil.copyfile(
    HELPERS / "evidence/original-optimizer-steps.jsonl", ORIGINAL / "optimizer-steps.jsonl"
)  # historical evidence only
COMMAND = [str(PYTHON), str(HELPERS / "recover_training.py")]
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
    str(HELPERS / "processor"),
    "--session-context",
    str(SESSION_CONTEXT),
]
subprocess.run([*COMMAND, "audit", *PATHS], check=True)
subprocess.run([*COMMAND, "prepare", *PATHS], check=True)
print("LINUX_017000_LOCAL_PREPARATION_VERIFIED", RECOVERY)
```

Expected: all 017000 files verified, selected step017000, frozen dataset/config/source checks pass.
018000-020000 Drive checkpoints are never copied/read as resume candidates. STOP on existing paths,
any hash/config mismatch, symlink or failed preparation. Preserve failed attempts.

## 7. Actual T4 restoration, then STOP
This runs the installed SmolVLA initialization/restore path and exits before any optimizer update.


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
assert restored["all_checkpoint_files_verified"] == len(gold["files_sha256"])
assert restored["relocated_processor_directory"] == str(HELPERS / "processor")
print("STOP_READY_FOR_OWNER_REVIEW: actual CUDA restore17000; zero optimizer updates")
```

## 8. Future training command — intentionally commented out
Do not run until all prior gates pass and the owner explicitly launches the continuation.
Frozen target=20000, batch=8, seed=0, checkpoint cadence=1000. Exactly3000 updates in this invocation.
The save callback transfers018000/019000 synchronously; final portable model goes FIRST at020000,
then the full model/optimizer/scheduler/RNG checkpoint. No manual export race or Drive output.


```python
# No training is started by this notebook as delivered.
# Only after explicit owner launch authorization:
# subprocess.run([*COMMAND, "train", *PATHS, "--execute-recovery",
#                 "--durability-config", str(CONFIG)], check=True)
```

Expected after a future authorized run: first17001/restored17000/calls1;
last20000/restored17000/calls3000; receiver-side hashes/fsync receipts for each transfer.
Keep session alive until Linux confirms both final portable and full-checkpoint receipts.
If transfer fails, retries preserve partial artifacts and further updates stop.
If runtime is interrupted after a durable018000/019000, do not restart from017000 or blindly
reuse this notebook: the current single-invocation gate needs a separate lineage/resume review.
No TEST use, no evaluation here. Future local strict import and frozen TRAIN24 evaluation require
separate authorization; unchanged gate is at least10/24 (min(0.40,0.60*23/24)=0.40).


## Persistence and provenance remain mandatory

At the frozen1000-step cadence, local018000/019000 saves are fully verified and
synchronously handed off with size/SHA, fsync/read-back and receipts. At20000,
the portable model/processor/normalization export transfers FIRST, followed by
full model/optimizer/scheduler/RNG state. Three transfer attempts use unique IDs;
exhaustion stops updates, preserves partial/local artifacts and never falls back
to Drive writes. Account session, canonical shared target, immutable payload IDs,
installed versions and processor hashes are recorded in recovery provenance.

Changing only the processor filesystem address does not edit original checkpoint
JSON or normalization tensors. Future saved configs/export naturally record the
new local asset location; portable loading resolves its own included processor.
Actual model tensor equality and optimizer/scheduler/RNG checks still gate updates.
The dataloader cursor is not checkpointed; no bit-for-bit recreation is claimed.

New-account API/route/canary/T4/restoration remain unverified until the owner runs
the cells. Training remains total20000, batch8, seed0, save1000 with approved80
TRAIN episodes. All splits/seeds, five arms, gate10/24, ADR-0029 and December12
remain fixed. No TEST access or evaluation is authorized by this procedure.
