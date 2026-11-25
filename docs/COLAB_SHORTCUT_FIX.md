# Colab shortcut fix — audit, prepare and restore only

The owner confirmed the project-root shortcut target; COMPLETE.json is a regular
file. The old guard rejected that ancestor. v3 resolves only the trusted project
root to its pinned target before constructing run paths. Every component inside
that canonical tree remains subject to strict symlink, traversal, containment,
size and SHA-256 checks.

Upload `recovery-tools-v3.zip` and `recovery-tools-v3-sha256.json` from
`/home/tj330/programs/VLA-Project-analytic/runs/recovery-prep-v3/` into
`MyDrive/vlarobot_smolvla/`. Leave old helpers, bundles, checkpoints and recovery
directories intact. Existing approved notebook variables PYTHON, SOURCE and DATASET
must still be available; do not rerun the original Train/Export cells.

Run only these three new cells, in order. None starts an optimizer update.

## 1. Verify helper bundle, resolve project root, audit checkpoints

```python
import hashlib, importlib.util, json, subprocess, zipfile
from pathlib import Path

HELPERS = Path("/content/recovery-tools-v3")
SHORTCUT = Path("/content/drive/MyDrive/vlarobot_smolvla")
bundle = SHORTCUT / "recovery-tools-v3.zip"
expected = json.loads((SHORTCUT / "recovery-tools-v3-sha256.json").read_text())["sha256"]
assert expected == "e1fd75546967e299102d231d180f908eac4c6ef81184c2cb69a9f80a972a3c70"
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == expected
with zipfile.ZipFile(bundle) as archive:
    assert set(archive.namelist()) == {
        "checkpoint_guard.py",
        "restore_gate.py",
        "recover_training.py",
    }
    archive.extractall(HELPERS)
spec = importlib.util.spec_from_file_location("recovery_guard_v3", HELPERS / "checkpoint_guard.py")
assert spec is not None and spec.loader is not None
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
EXPECTED_ROOT = Path(
    "/content/drive/.shortcut-targets-by-id/1xSPhmxF4hK0yTXPkPXGS8bziahI5LsVK/vlarobot_smolvla"
)
DRIVE_ROOT = guard.resolve_project_root(SHORTCUT, EXPECTED_ROOT)
ORIGINAL = DRIVE_ROOT / "training-tiago-v1"
RECOVERY = DRIVE_ROOT / "recovery" / "training-tiago-step17000-v2"
guard.require_isolated(ORIGINAL, RECOVERY)
guard.require_canonical_path(ORIGINAL)
guard.require_canonical_path(RECOVERY)
print("CANONICAL_DRIVE_ROOT_VERIFIED", DRIVE_ROOT)
COMMON = [str(PYTHON), str(HELPERS / "recover_training.py")]
PATHS = [
    "--project",
    str(SOURCE),
    "--original",
    str(ORIGINAL),
    "--recovery",
    str(RECOVERY),
    "--dataset",
    str(DATASET),
]
subprocess.run([*COMMON, "audit", *PATHS], check=True)
```

## 2. Prepare an unused isolated recovery directory

```python
assert not RECOVERY.exists(), "Preserve existing recovery; choose a new unused version"
subprocess.run([*COMMON, "prepare", *PATHS], check=True)
```

## 3. Verify actual CUDA restoration, then stop

```python
subprocess.run([*COMMON, "restore-check", *PATHS], check=True)
restored = json.loads((RECOVERY / "restore-check.json").read_text())
assert restored["actual_installed_smolvla"] is True
assert restored["device"] == "cuda"
assert restored["restored_step"] == 17000
assert restored["optimizer_updates_at_restore_gate"] == 0
assert restored["adamw_parameter_groups_verified"] is True
assert restored["scheduler_verified"] is True
assert restored["original_directory_write_protection"] is True
assert restored["model_tensor_values_verified"] > 0
assert restored["optimizer_tensor_values_verified"] > 0
assert restored["rng_tensors_verified"] > 0
print("READY: actual CUDA state restored at 17000; zero optimizer updates")
```

Expected audit output: `CANONICAL_DRIVE_ROOT_VERIFIED` followed by the reported
`.shortcut-targets-by-id/.../vlarobot_smolvla` path; checkpoint 17000 verified;
018000–020000 rejected; selected checkpoint ends in `/checkpoints/017000` under
the canonical target. A fresh full-file audit is still required in Colab.

Prepare creates `recovery/training-tiago-step17000-v2` and prints provenance with
`restored_step: 17000`, `target_step: 20000`, `additional_updates: 3000`. If that
new directory already exists, stop; do not remove it or overwrite its evidence.

Restore-only verification must print `RESTORE_GATE_PASSED` with device cuda,
restored step 17000, positive model/optimizer/RNG tensor counts, verified scheduler
and parameter groups, write protection, and zero optimizer updates. It then prints
`RESTORE_CHECK_COMPLETE: no optimizer update executed` and the READY line.
Any mismatch is a stop condition. Do not launch training with these instructions.

Full later-stage gates remain in TRAINED_EXPORT_RECOVERY.md. No original files,
training settings, frozen criteria or TEST restrictions change.
