# Proposed recovery from 017000 — training not executed

Correction: the owner reports missing model tensors at 018000–020000 and missing
optimizer tensors at 019000–020000. Do NOT download weights from 020000 unless
physical presence is re-established. Earlier download instructions are superseded.
The owner verified model and optimizer at 017000; all other entries still require
authenticated verification. Original checkpoints, manifests and logs stay untouched.

## Selector/export audit

The approved source.zip selector already checks physical files, sizes and SHA-256.
A local test using that actual implementation created four complete fixtures,
deleted tensors as reported and selected only 017000. No evidence establishes that
the selector accepted missing files at selection time. COMPLETE records historical
writes, not current durability. Drive persistence/deletion or runtime-version
differences remain unresolved. Export copies and hashes actual files before return,
but lacks a final source-to-export/read-back comparison. Discovery also has
filesystem race/error-handling gaps. The recovery guards verify every manifest
entry, safe paths, nonempty sizes, steps, hashes and final source/export equality,
and reject missing required assets even if omitted from an export manifest.
They cannot prevent later Drive deletion. Regression tests cover the reported
missing-tensor pattern, same-size corruption and export omissions.

The downloaded ZIP contains no model and exactly matches the extracted local files.
Extraction did not lose it; upstream cause is not proven. No recovery training or
evaluation has run. The approved source/data bundle and notebook remain unchanged.

## Recover existing weights before considering a replacement continuation

Using the authenticated session, inspect original Drive exports, existing downloaded
ZIP parts, `/content` temporary/training directories and alternate account/shared
drive copies. Inspect Drive Trash, file activity and Manage versions in the web UI.
Copy/download candidates into a separate evidence directory; never overwrite
checkpoint paths. Hash candidates against the original COMPLETE record, not their
names. Do not empty Trash or restore over original evidence. This agent has no
authenticated Drive API session, so those searches remain owner-side verification.
If final weights are recovered and verified, do not start replacement training.

## Resume semantics

Installed LeRobot loads checkpoint model/processors, optimizer, scheduler,
Python/NumPy/Torch/CUDA RNG and step through its resume path. Its loop is
`range(step, cfg.steps)`: 17000 to 20000 means 3000 updates. It does not save the
dataloader cursor or worker state. This is a state-restored replacement
continuation, not bit-for-bit recreation of the missing original final model.
Actual CUDA restoration remains unverified until run in the authorized environment.
The replacement may have a different model hash and must retain separate lineage.

## Exact Colab procedure

New helper source files: `checkpoint_guard.py`, `restore_gate.py` and `recover_training.py` from
`tools/recovery/` in the isolated worktree. They install process-local guards; no
approved `training.py`, `checkpoint_export.py`, YAML, dataset, model configuration
or source.zip is replaced. Hashes are recorded with recovery provenance.

1. Upload these two prepared files to `MyDrive/vlarobot_smolvla/`:
   `runs/recovery-prep-v2/recovery-tools-v2.zip` and
   `runs/recovery-prep-v2/recovery-tools-v2-sha256.json` from the isolated worktree.
2. On a T4 runtime, rerun only Drive mount, pinned environment/bundle setup and
   approved-data validation cells if needed. Do NOT rerun diagnostic preflight,
   the original Train or resume cell, or the original Export cell.
3. Add this new cell. It audits without training:

```python
import hashlib, json, subprocess, zipfile
from pathlib import Path

HELPERS = Path('/content/recovery-tools-v2')
bundle = DRIVE / 'recovery-tools-v2.zip'
expected = json.loads((DRIVE / 'recovery-tools-v2-sha256.json').read_text())['sha256']
assert expected == '958013d958b386774a99745870095f5e841b76478b09e177eb9a0b29fa3ec76d'
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == expected
with zipfile.ZipFile(bundle) as archive:
    assert set(archive.namelist()) == {'checkpoint_guard.py', 'restore_gate.py', 'recover_training.py'}
    archive.extractall(HELPERS)
RECOVERY = DRIVE / 'recovery' / 'training-tiago-step17000-v1'
COMMON = [str(PYTHON), str(HELPERS / 'recover_training.py')]
PATHS = ['--project', str(SOURCE), '--original', str(TRAIN_OUT),
         '--recovery', str(RECOVERY), '--dataset', str(DATASET)]
subprocess.run([*COMMON, 'audit', *PATHS], check=True)
```

Require every 017000 file verified and selected, with 018000–020000 rejected as
reported. Preserve the printed results. If anything differs, stop and investigate.
This audit hashes all entries in COMPLETE, not only model and optimizer tensors.
Expected summary: 17000 `verified: true`; 18000, 19000 and 20000
`verified: false`; `selected` ends with `training-tiago-v1/checkpoints/017000`.
The exact file count comes from the real manifest, not a hardcoded expected count.

4. Once search and audit establish the route, prepare a separate copy, without
   training:

```python
subprocess.run([*COMMON, 'prepare', *PATHS], check=True)
```

Preparation checks the approved dataset, saved batch8/seed0/steps20000/save1000,
normalization hashes and dataset identity; refuses an existing recovery directory;
copies and rehashes 017000; records source hashes and original log hash in
`recovery-provenance.json`. Original logs remain untouched. Check Drive quota:
allow approximately 1.7 GB for each complete checkpoint, three new checkpoints,
the initial recovery copy, inference export/archive and runtime overhead.

5. Mandatory actual Colab restoration test, with no optimizer updates. Run this
   before the training cell:

```python
subprocess.run([*COMMON, 'restore-check', *PATHS], check=True)
restored = json.loads((RECOVERY / 'restore-check.json').read_text())
assert restored['actual_installed_smolvla'] is True
assert restored['device'] == 'cuda'
assert restored['restored_step'] == 17000
assert restored['optimizer_updates_at_restore_gate'] == 0
assert restored['adamw_parameter_groups_verified'] is True
assert restored['scheduler_verified'] is True
assert restored['original_directory_write_protection'] is True
assert restored['model_tensor_values_verified'] > 0
assert restored['optimizer_tensor_values_verified'] > 0
assert restored['rng_tensors_verified'] > 0
print('READY: actual CUDA state restored at 17000; zero optimizer updates')
```

This executes the installed training initialization path, compares every saved
model and optimizer tensor against the live restored values, checks scheduler,
parameter groups and saved Python/NumPy/Torch/CUDA RNG values, writes the report
and intentionally exits before the dataloader/update loop. Expected output:
`RESTORE_GATE_PASSED` with step17000/device cuda and positive tensor counts,
`RESTORE_CHECK_COMPLETE: no optimizer update executed`, then the READY line.
Any mismatch stops. Before configuration is used, output/checkpoint/dataset paths
are checked. Original production/recovery directories must not overlap.
Every checkpoint-save destination is independently confined to recovery.
The train command refuses to start without this report and repeats the same live
restore gate before allowing its first policy update.

6. **Proposed execution only. Do not run before all preceding gates pass.**
   Add a new cell:

```python
subprocess.run([*COMMON, 'train', *PATHS, '--execute-recovery'], check=True)
```

The helper restores the copied 017000 state and overrides only output directory.
It keeps dataset, normalization, model, optimizer/scheduler settings, seed, steps,
batch and checkpoint cadence unchanged. First real event must be step17001,
restored17000, calls1. Last must be step20000, restored17000, calls3000. It refuses
step-zero/pretrained initialization. If interrupted, preserve evidence and audit
invocation counts before retrying; multi-invocation export deliberately fails
pending that audit. Never recover with the original production training cell.
After normal completion, verify actual records in a new cell:

```python
events = [json.loads(s) for s in (RECOVERY / 'optimizer-steps.jsonl').read_text().splitlines()]
first, last = events[0], events[-1]
assert first['optimizer_step'] == 17001
assert first['restored_training_step'] == 17000
assert first['actual_calls_this_invocation'] == 1
assert last['optimizer_step'] == 20000
assert last['restored_training_step'] == 17000
assert last['actual_calls_this_invocation'] == 3000
assert first['actual_post_step_hook'] and last['actual_post_step_hook']
assert last['approved_successful_episodes'] == 80
print('RECOVERY_UPDATES_VERIFIED: 3000 updates; final step 20000')
```

7. After strict final-state/update-count checks, export in another new cell:

```python
RECOVERED_EXPORT = DRIVE / 'recovery' / 'exports' / 'training-tiago-recovery-v1-step-020000'
subprocess.run([*COMMON, 'export', *PATHS, '--export', str(RECOVERED_EXPORT)], check=True)
```

Export requires physically hashed 020000 training state, recovered optimizer
evidence and approved data. It verifies all copied hashes against source and
requires model, processor and normalization assets. Partial exports are preserved
on failure; choose a new version. Recheck durability in a fresh process:

```python
check_code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[2]) / 'src'))
from checkpoint_guard import verify_checkpoint, verify_export
from vlarobot.vla.training import CHECKPOINT_FILES
from vlarobot.vla.checkpoint_export import require_trained_export
verify_checkpoint(Path(sys.argv[3]) / 'checkpoints/020000', CHECKPOINT_FILES)
verify_export(Path(sys.argv[4]), CHECKPOINT_FILES, 20000)
require_trained_export(Path(sys.argv[4]))
print('Fresh-process checkpoint/export read-back passed')
'''
subprocess.run([str(PYTHON), '-c', check_code, str(HELPERS), str(SOURCE),
                str(RECOVERY), str(RECOVERED_EXPORT)], check=True)
```

8. Create and verify an explicit archive rather than asking Drive to ZIP a folder:

```python
import shutil

LOCAL_ZIP = Path('/content/tiago-recovered-export-v1.zip')
ARCHIVE_DIR = DRIVE / 'recovery' / 'archives'
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_ZIP = ARCHIVE_DIR / LOCAL_ZIP.name
assert not LOCAL_ZIP.exists() and not DRIVE_ZIP.exists()
manifest = json.loads((RECOVERED_EXPORT / 'export-manifest.json').read_text())
expected_files = dict(manifest['files_sha256'])
expected_files['export-manifest.json'] = hashlib.sha256(
    (RECOVERED_EXPORT / 'export-manifest.json').read_bytes()).hexdigest()
with zipfile.ZipFile(LOCAL_ZIP, 'x', compression=zipfile.ZIP_STORED) as archive:
    for name in expected_files:
        archive.write(RECOVERED_EXPORT / name, name)
with zipfile.ZipFile(LOCAL_ZIP) as archive:
    assert archive.testzip() is None
    assert archive.getinfo('model.safetensors').file_size > 0
    for name, digest in expected_files.items():
        with archive.open(name) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
with LOCAL_ZIP.open('rb') as stream:
    archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
shutil.copyfile(LOCAL_ZIP, DRIVE_ZIP)
with DRIVE_ZIP.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == archive_hash
transfer = {'archive': DRIVE_ZIP.name, 'sha256': archive_hash,
            'bytes': DRIVE_ZIP.stat().st_size, 'step': 20000,
            'model_sha256': manifest['files_sha256']['model.safetensors']}
(ARCHIVE_DIR / 'recovery-transfer-sha256.json').write_text(json.dumps(transfer, indent=2))
print('TRANSFER_ARCHIVE_VERIFIED', json.dumps(transfer, indent=2))
```

Download that single ZIP and its `recovery-transfer-sha256.json`, not a Drive
folder download. Optimizer state is not included or needed locally. The archive
contains recovery provenance and source/final COMPLETE records. Local transfer
hashes, archive CRC and every extracted export hash must pass, followed by strict
policy loading, full-weight comparison and real TRAIN inference before evaluation.
Keep the Colab runtime available until the downloaded archive is verified locally:
fresh-process reads through the same Drive mount can still use filesystem caches
and do not independently prove remote durability.
No 17000 weights may be labeled 20000. Preserve original and replacement lineages.
TRAIN gate, TEST restrictions, ADR-0029 and deadline stay fixed.
