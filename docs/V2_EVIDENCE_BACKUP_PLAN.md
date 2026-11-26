# V2 raw evidence external backup plan

**Status:** Plan defined and hash manifest committed. **External copy not executed**
because no external/off-site destination is mounted or identified in this
environment. A Git push alone does not preserve these files.

`runs/` is gitignored (`.gitignore:7`), and the frozen checkpoint export lives
outside the repository. The committed summary in
`docs/V2_NEGATIVE_RESULT.md` and the pushed commit `d62f313` preserve the
accepted conclusions and receipts, not the raw episode logs, frames, logs or
1.6 GB of checkpoint/optimizer tensors.

## Covered sources

The committed manifest `docs/evidence/v2-raw-evidence-manifest.json` has SHA-256
`fdbfea1480ce5bc60a4c302b7b24cae5d4e4850d7f1b2b4427190dd53f323ac3` and covers
132 files / 1,623,929,814 bytes:

- `runs/worldmind/v2-train-recovered-diag-1/` — raw accepted step-021000
  diagnostic: `diagnostic-results.json`, `derived-motion-summary.jsonl`, per-episode
  metadata, actions, steps, latency inputs/logs, frames, `provenance.json` and
  controller logs.
- `runs/worldmind/v2-train-endpoint-1/` — raw step-020000 endpoint-decoder
  baseline used in the accepted comparison.
- `external/v2-transition-warmstart-01/step-021000/` — frozen checkpoint export,
  including `model.safetensors` (manifest SHA
  `0d5059dc5051b12a0d19614cc93f4b6a8ec77887f7e1e97b6e830858399504d6`),
  `recovery-optimizer.pt`, optimizer telemetry, processors, tokenizer,
  `train_config.json`, approval and provenance.

The manifest records each source path, destination-relative path, byte size and
SHA-256. The `external/...` prefix keeps the outside-repository checkpoint
separate from the two `runs/...` campaigns.

## Tooling

`tools/recovery/backup_v2_evidence.py` is a standard-library, cross-platform CLI.
It rejects symlinks, path traversal, missing files, size mismatches and hash
mismatches. It does not invoke training, Webots, the model server or a network
client. Unit coverage is in `tests/test_backup_v2_evidence.py`.

## Owner steps, in order

1. Attach or identify a true external destination. Use a mounted external drive,
   an off-site/cloud-sync directory, or another owner-controlled location
   **outside this repository**. Do not use `runs/`, `Downloads/` as an unsynced
   local folder, or a symlink/junction.
2. Open a terminal in the repository root:
   `cd /home/tj330/programs/VLA-Project-week3`
   (Windows: `cd C:\path\to\VLA-Project-week3`).
3. Copy and verify, replacing the destination with the external path:

   ```text
   uv run python tools/recovery/backup_v2_evidence.py copy --manifest docs/evidence/v2-raw-evidence-manifest.json --destination "<EXTERNAL_PATH>/WorldMind-V2-Evidence-2026-11-26"
   ```

4. The tool writes `V2_EVIDENCE_MANIFEST.json` and `V2_BACKUP_RECEIPT.json` into
   the destination and prints verified file/byte counts. Keep both files with
   the backup.
5. Re-verify from the same repository root:

   ```text
   uv run python tools/recovery/backup_v2_evidence.py verify --manifest docs/evidence/v2-raw-evidence-manifest.json --destination "<EXTERNAL_PATH>/WorldMind-V2-Evidence-2026-11-26"
   ```

6. If the destination is a cloud-sync folder, wait until sync completes and run
   step 5 again from a second terminal/session. Record the final destination
   path and the receipt SHA-256 in the owner's evidence log.
7. Preserve two copies if the media is not redundant: the external copy and the
   original local directories until the external receipt is verified.

## If sources move or the manifest must be rebuilt

The committed manifest contains absolute source paths from this worktree. If the
source roots are relocated, rebuild from the repository root before copying:

```text
uv run python tools/recovery/backup_v2_evidence.py build --output docs/evidence/v2-raw-evidence-manifest.json
```

For a different machine, pass explicit `--source PATH PREFIX` triples for the
two `runs/worldmind/...` directories and the checkpoint export, then commit or
archive the rebuilt manifest with its new SHA-256. Do not overwrite the
committed manifest merely to make a mismatch disappear; investigate the change.

## Current verification status

- Committed manifest and portable copy/verify tool: **present and tested**.
- Raw source files hashed locally: **132 / 132**, no symlinks encountered.
- External/off-site copy: **pending owner destination**.
- Claimed durability: **not yet achieved** until steps 3–6 succeed.
