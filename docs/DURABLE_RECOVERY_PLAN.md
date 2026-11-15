# Proposed durable recovery — no training or evaluation executed

The Linux original 017000 backup was independently audited: all 12 COMPLETE
entries match size/SHA-256, including the supplied model and optimizer hashes.
Original files were read only. A separate verified input archive is ready at
`runs/durable-recovery-prep-v1/checkpoint-017000.zip` (1,610,503,770 bytes), SHA-256
`e34c9596d0fdaef579b734829d45a60c257370d9de4e17348055de6734366109`.

The owner reports that a previous recovery completed 3000 updates, then its model,
export and archive disappeared after disconnection while optimizer/log files
survived. This is user-supplied evidence, not proof of quota exhaustion, deletion,
permissions failure, shortcut behavior or Drive write-cache failure. Preserve
those logs and every original/recovery directory. Do not relabel 017000 as 20000.

## Implementation

Use Colab-local `/content` for the new recovery output. Drive remains a read source
for approved setup/data/processor metadata; it is not the checkpoint write target.
The frozen 1000-step save cadence stays unchanged. Each completed 018000/019000
checkpoint is fully verified, archived with provenance/logs, and immediately sent
to an independent Linux filesystem over authenticated SSH. At 020000 the verified
portable inference export is transmitted FIRST, then the full resumable checkpoint.

The Linux receiver creates a new exclusive artifact directory, streams bytes to
a partial file, checks the declared length/SHA-256, flushes and fsyncs, re-reads
the file to rehash it, publishes it, fsyncs the directory and writes a receipt.
The sender requires the matching receipt. Three uniquely named transfer attempts
are allowed; exhausted retries raise and stop further training. Partial artifacts
are preserved. No existing file, checkpoint or recovery evidence is overwritten.
All archives are CRC-checked and checked against every manifest-listed file.

Before training, a full-size golden-input transfer tests the exact endpoint and
records elapsed time and receiver capacity. Its receipt is bound to the transfer
configuration and expected input SHA. The runner refuses missing/mismatched
receipts, Drive-backed output, inadequate capacity or an unverified CUDA restore.
Actual model/AdamW/scheduler/RNG restoration and step17000 are still required.
The original target20000, batch8, seed0, save1000, approved80 episodes, normalization
and model settings are unchanged. No automatic training has been launched.

## Required owner actions before GPU use

1. Supply an already authorized, Colab-reachable SSH route to Linux, with a
   dedicated key and independently verified host key. No SSH endpoint/configuration
   is currently available to this agent. Do not paste credentials into chat.
   If NAT/firewall prevents access, provide an authorized VPN/bastion route or
   choose another independently writable store; this implementation must not be
   represented as tested for an unavailable route. No server/port is started here.
2. Keep Linux online with receiver storage available. This host currently has
   approximately 259 GiB filesystem headroom before the new archives; actual
   capacity is rechecked by each receive. No additional storage purchase is needed
   on the measured host. Use its canonical local disk, not a Drive-mounted folder.
3. Provision the SSH private key and known_hosts file securely in Colab `/content`
   outside the archived trees. The receiver uses existing SSH, strict host-key
   checking and batch authentication; the helper never stores key contents in logs.
4. Run the read-only quota diagnostic and independent transfer preflight on CPU
   before allocating scarce T4 time. Review their outputs before any new training.
   After changing runtimes, re-stage the same verified input and rerun the endpoint
   canary with a new unused run ID. GPU restoration needs T4; canary/diagnostics do not.

## Exact procedure (proposed, not executed)

Use the approved notebook's Drive mount, pinned setup and approved-data validation
cells only. Do not run its old Train, diagnostic optimizer preflight or Export
cells. Keep SOURCE=/content/VLA-Project, DATASET=/content/vlarobot-data and the
original PYTHON=/content/vla-env/bin/python. No approved bundle is replaced.

Upload only the small `recovery-tools-v4.zip` and `recovery-tools-v4-sha256.json`
from this worktree's `runs/durable-recovery-prep-v1/` to the Drive project folder.
The large golden input stays on Linux and is fetched directly over SSH.

### Cell 1: unpack operational helpers only

```python
import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

HELPERS = Path("/content/durable-tools-v4")
PROJECT_DRIVE = Path("/content/drive/MyDrive/vlarobot_smolvla").resolve(strict=True)
assert PROJECT_DRIVE == Path(
    "/content/drive/.shortcut-targets-by-id/1xSPhmxF4hK0yTXPkPXGS8bziahI5LsVK/vlarobot_smolvla"
)
bundle = PROJECT_DRIVE / "recovery-tools-v4.zip"
assert (
    hashlib.sha256(bundle.read_bytes()).hexdigest()
    == "92aa2aa33b58625acce1bc18e91ae2fcee1b66d39bdc6d08a90ee47844509171"
)
with zipfile.ZipFile(bundle) as archive:
    for item in archive.infolist():
        name = Path(item.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    archive.extractall(HELPERS)
sys.path.insert(0, str(HELPERS))
gold = json.loads((HELPERS / "checkpoint-017000.sha256.json").read_text())
assert gold["step"] == 17000
```

### Cell 2: read-only Drive diagnostics (CPU, authentication may prompt)

```python
from google.colab import auth

auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
from drive_diagnostics import inspect_drive

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"]
)
service = build("drive", "v3", credentials=credentials, cache_discovery=False)
report = inspect_drive(service, "1xSPhmxF4hK0yTXPkPXGS8bziahI5LsVK")
Path("/content/drive-persistence-diagnostics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
```

This reads quota/usage/trash usage, principal and project ownership IDs, and model/
optimizer metadata including trashed candidates, parents, sizes and checksums.
It does not delete, restore or modify anything. Account quota may not describe a
shared folder's owner/pooled capacity. API failures remain explicit unknowns.
Preserve the report on Linux before disconnecting. Inspect owner-side Drive
activity/revisions/retention and Trash; mount `df` or a COMPLETE marker alone
cannot establish server persistence. No root cause is asserted by this diagnostic.
References: [Drive about](https://developers.google.com/workspace/drive/api/reference/rest/v3/about),
[Drive files](https://developers.google.com/workspace/drive/api/reference/rest/v3/files).

### Cell 3: configure the authorized endpoint, pull and hand back the golden archive

Only endpoint/key/host-key values require owner edits. Choose an UNUSED run_id for
each new attempt. The Linux remote paths below are already prepared. No receiver
daemon is needed: SSH invokes the standard-library Python receiver per transfer.

```python
CONFIG = Path("/content/durable-ssh.json")
cfg = {
    "destination": "CHANGE_ME@reachable-linux-host",
    "port": 22,
    "identity_file": "/content/authorized_recovery_key",
    "known_hosts": "/content/verified_known_hosts",
    "remote_python": "/home/tj330/programs/VLA-Project/.venv/bin/python",
    "remote_helper": "/home/tj330/programs/VLA-Project-analytic/tools/recovery/durable_transfer.py",
    "remote_root": "/home/tj330/programs/VLA-Project/runs/independent-recovery-v1",
    "run_id": "durable-recovery-v1",
    "timeout_s": 1800,
    "canary_receipt": "/content/durable-canary-receipt-v1.json",
    "canary_sha256": gold["sha256"],
}
assert "CHANGE_ME" not in cfg["destination"]
assert Path(cfg["identity_file"]).is_file() and Path(cfg["known_hosts"]).is_file()
os.chmod(cfg["identity_file"], 0o600)
CONFIG.write_text(json.dumps(cfg, indent=2))
INPUT_ZIP = Path("/content/checkpoint-017000.zip")
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
        cfg["run_id"] + "-canary-017000",
        "--receipt",
        cfg["canary_receipt"],
    ],
    check=True,
)
canary = json.loads(Path(cfg["canary_receipt"]).read_text())
assert canary["receiver_readback_verified"] and canary["directory_fsync"]
print("INDEPENDENT_STORAGE_CANARY_PASSED", canary)
print(
    "Approximate 6 GB handoff overhead seconds:",
    6_000_000_000 * canary["sender_elapsed_s"] / canary["bytes"],
)
```

Allow roughly 47 minutes for 3000 updates based on the earlier preflight, PLUS
measured archive/transfer overhead. This is not a promise about session limits.
If time/capacity is insufficient, stop before training; do not shorten the run or
alter batch/cadence. Canary/receiver capacity is checked before training, and each
transfer checks actual receiver free space. The runner reserves local disk room
for saved states, archives, portable weights and bounded-memory wire files.

### Cell 4: stage original 017000 locally, prepare and verify CUDA restore

```python
INPUT = Path("/content/durable-input-v1")
ORIGINAL = INPUT / "training-tiago-v1"
RECOVERY = Path("/content/durable-recovery-v1")
assert not INPUT.exists() and not RECOVERY.exists()  # preserve all older attempts
saved = ORIGINAL / "checkpoints/017000"
saved.mkdir(parents=True)
with zipfile.ZipFile(INPUT_ZIP) as archive:
    assert set(archive.namelist()) == set(gold["files_sha256"]) | {"COMPLETE.json"}
    archive.extractall(saved)  # exact golden names above, all hash-verified on input
SOURCE_ZIP = STAGING / "source.zip"  # from unchanged approved notebook setup
assert (
    hashlib.sha256(SOURCE_ZIP.read_bytes()).hexdigest()
    == "2d3cb4bd06ac8313b1bb3cdb5aae7f916c2b614dd3884172c4570be9ff36b2f5"
)
shutil.copyfile(SOURCE_ZIP, INPUT / "source.zip")
shutil.copyfile(
    HELPERS / "evidence/original-optimizer-steps.jsonl", ORIGINAL / "optimizer-steps.jsonl"
)  # historical context, not new updates
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
]
subprocess.run([*COMMAND, "audit", *PATHS], check=True)
subprocess.run([*COMMAND, "prepare", *PATHS], check=True)
subprocess.run([*COMMAND, "restore-check", *PATHS], check=True)
print("STOP: review actual CUDA restore, storage canary and capacity before training")
```

Expected: all12 original files verified, restored_step17000, actual SmolVLA on CUDA,
matching model/optimizer/scheduler/RNG, zero optimizer updates. The gate compares
actual loaded tensor values; a CPU test or loaded COMPLETE marker is insufficient.
Dataset/statistics/provenance and batch8/seed0/20000/save1000 must still match.

### Proposed future launch — DO NOT EXECUTE in the current readiness task

```python
# Only after the owner resumes this verified recovery route:
# subprocess.run([*COMMAND, 'train', *PATHS, '--execute-recovery',
#                 '--durability-config', str(CONFIG)], check=True)
```

There is no separate manual export cell to race disconnection. The save callback
hands off 018000 and 019000 checkpoints synchronously; at 020000 it first hands off
the portable export, then the full checkpoint. Expected receipts end in
`receiver_readback_verified:true`, matching SHA/bytes, file_fsync:true and
directory_fsync:true. First actual event must be17001/restored17000/calls1; final
must be20000/restored17000/calls3000. Never substitute the input archive as final.
SSH failure after three attempts aborts further updates and retains local state,
partial remote files and failure metadata. Keep the runtime alive for transfer
repair; do not blindly rerun training after an interruption. Fully transferred
018000/019000 states can support a separately audited future resume; the current
gate intentionally does not silently reinterpret a multi-invocation run.

## Local acceptance after eventual training

Artifacts arrive directly at `runs/independent-recovery-v1/` on Linux with receipts.
Verify receipt SHA/bytes again, extract into a NEW versioned directory with safe
paths, verify export/checkpoint manifests, compare final weights to initialization,
and run the existing strict installed-policy import/real TRAIN inference preflight.
Only then consider the frozen 24 TRAIN VLA-only evaluation and unchanged 10/24
gate. No evaluation has run. No TEST access. December12 and ADR-0029 unchanged.

## Evidence, tests and remaining risks

`docs/evidence/linux-017000-backup-audit.json` records actual local verification.
Tests exercise real receiver subprocesses, corrupt/truncated/trailing transfers,
missing receipts, overwrite refusal, SSH host-key options, canary/config binding,
archive omissions, model-first final ordering, retry exhaustion and preservation.
They are local Linux tests, NOT evidence of a reachable authenticated SSH endpoint
or Colab GPU restoration. The receiver is explicitly Linux; Windows sender use
requires an existing OpenSSH client and remains unverified here.

Independent Linux fsync/read-back reduces dependence on Drive persistence, but
does not guarantee against future Linux disk failure, account compromise or an
unreachable host. Keep the golden original backup and received artifacts intact;
a second independent backup is useful after verified transfer. SSH throughput
and synchronous upload time can consume session budget; measure on CPU first.
LeRobot does not checkpoint the dataloader cursor, so repeated recovery is not
claimed to recreate the lost final model bit-for-bit. Drive quota/ownership and
deletion history remain unresolved until authenticated diagnostic evidence arrives.
