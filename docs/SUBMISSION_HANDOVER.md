# WorldMind submission handover

## Evidence inventory

| Item | Source to preserve |
|---|---|
| Repository and experiment contract | Week3 worktree, its Git refs, dependency lock, `configs/`, `worlds/`, and [ADRs](DECISIONS.md). The frozen manifest SHA-256 is `d6e5e7e7cc767d8784b208be9ba2f49d292935d8285f2954ae55c15baf027176`. |
| Independent TEST | `runs/worldmind/nov25-frozen-test-arm1-retry`, `nov25-frozen-test-arm3`, `nov25-frozen-test-arm4-resumed`, and `nov25-frozen-test-arm5`. Preserve the original baseline startup directory `nov25-frozen-test-arm1` and the interrupted `nov25-frozen-test-arm4` tree too. |
| Development and earlier evidence | Entire Week3 `runs/` tree, including V1–V5b, failed development profiles, matched VALIDATION, and the corrected arm-3 repair. The canonical scripted VALIDATION baseline is in the analytic worktree at `runs/validation-evidence-v2-canonical`. Preserve the primary worktree's `runs/eval` for baseline TRAIN and the 0/24 VLA gate. |
| Derived analyses | `runs/worldmind/nov25-test-analysis` and `runs/worldmind/nov25-validation-report`, including JSON and the read-only scripts. Results are separate from raw outcomes. |
| Model artifacts | Original step-20000 portable ZIP and receipt in Downloads, analytic export `runs/local-model-validation-v1/export-step-020000`, and the primary TRAIN evidence. The [model manifest](evidence/local-model-validation-step20000.json) identifies the archive and training provenance. |
| Report and presentation | Original submitted `~/Downloads/Project_Report.pdf`, final chapters in `docs/`, vector figures and demo in `docs/figures/`, and the student's final submission PDF once assembled. |
| Integrity | [Submission index](evidence/worldmind-submission.sha256), [figure checksums](figures/SHA256SUMS), [VALIDATION figure sources](figures/provenance.json), and [TEST figure provenance](figures/test-provenance.json). Per-trial hashes remain in the analysis ledger and journal records. |

## Backup and restore check

Use a mounted external destination with capacity comfortably above the more
than 250 GiB measured across the project run trees. Do not write the backup to
the same nearly full filesystem. The following are Linux terminal commands.
Set the destination to your actual external mount. Keep the portable model
archive and raw runs private unless its distribution is approved.

```sh
cd /home/tj330/programs/VLA-Project-week3
BACKUP=/path/to/mounted/external-drive/worldmind-final
mkdir -p "$BACKUP"/{week3,primary,analytic,originals}
git bundle create "$BACKUP/worldmind-history.bundle" --all
git bundle verify "$BACKUP/worldmind-history.bundle"
rsync -aH --exclude='/.git' --exclude='/.venv/' ./ "$BACKUP/week3/"
rsync -aH --exclude='/.git' --exclude='/.venv/' ../VLA-Project/ "$BACKUP/primary/"
rsync -aH --exclude='/.git' --exclude='/.venv/' ../VLA-Project-analytic/ "$BACKUP/analytic/"
rsync -a ~/Downloads/Project_Report.pdf ~/Downloads/acct2-durable-v1-drive-v1-portable-020000-attempt-1.zip ~/Downloads/acct2-durable-v1-drive-v1-portable-020000-attempt-1.receipt.json "$BACKUP/originals/"
cd "$BACKUP/week3"
sha256sum -c docs/evidence/worldmind-submission.sha256
sha256sum -c docs/figures/SHA256SUMS
sha256sum "$BACKUP"/originals/*
```

Confirm the original PDF and model digests against [Phase I conformance](REPORT_CONFORMANCE.md)
and the model manifest. Open the copied PDF figures and demo. For full
source-to-backup content verification, repeat each rsync with `-nci` and the
same exclusions. No reported source file differences should remain.
A copy has not been made to external storage during this review.

## Submission checklist

- [x] All 120 independent TEST trials reconciled to their frozen identities.
- [x] Report distinguishes development, independent evaluation and the VLA TRAIN gate.
- [x] Wilson intervals, paired discordance, exact McNemar and scene-level exact
  sign tests reconciled. Equal totals are distinguished from paired agreement.
- [x] Figure hashes and recorded source provenance checked. The TEST figure
  was regenerated from the reconciled data and its raster matched pixel for pixel.
- [x] Frozen manifest unchanged. Startup and interruption records preserved.
- [ ] Student creates the external backup and verifies it from the destination.
- [ ] Student incorporates the Markdown chapters and measured replacements into
  the institutional submission template, checks captions, numbering, PDF links
  and legibility, and exports the final PDF.
- [ ] Student opens the exported PDF and plays the bundled demo on the presentation
  machine, then submits the PDF and any institution-required supplements.

## Known issues and claim limits

The non-Webots suite's sole known failure is the unrelated notebook-helper SHA
mismatch in `test_v7_current_session_cells_and_helper_manifest`. The notebook
and its expected hash remain intact. Some original figures called for an
external full-robot view or all candidate trajectories. The archive supports
onboard Webots RGB and recorded candidate assessments with a selected trace.
The replacements are explicit about their coverage. Collision reduction,
full-body safety, physical-robot reliability, calibrated uncertainty and
improvement over a competent VLA remain unestablished.
