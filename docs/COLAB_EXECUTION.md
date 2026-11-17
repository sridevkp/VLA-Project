# Exact Colab execution — updated 2026-11-15

Use the current local notebook, not the older copy contained inside frozen source.zip:
`/home/tj330/programs/VLA-Project/notebooks/smolvla_finetune_colab.ipynb`.
The notebook embeds its T4 companion; no fourth payload is required.

## Upload

From `/home/tj330/programs/VLA-Project/runs/colab-prep-v3/`, upload these unmodified files:

| Local filename | Google Drive destination |
|---|---|
| source.zip | My Drive/vlarobot_smolvla/source.zip |
| dataset-train-v1.zip | My Drive/vlarobot_smolvla/dataset-train-v1.zip |
| bundle-status.json | My Drive/vlarobot_smolvla/bundle-status.json |

Do not extract or rename them in Drive. No analytic worktree files are used.
In Colab choose File > Upload notebook and select the local notebook above.
Choose Runtime > Change runtime type > Python 3 / T4 GPU. Save a copy in Drive if prompted.

## Run code cells in order

1. Drive mount: this is the cell that requests Google sign-in/Drive permission.
2. Install isolated stack: checks the pinned manifest and both archive hashes before
   extraction, installs managed Python 3.11.18 and pinned dependencies, checks CUDA/T4.
3. Verify dataset: checks complete 80-success native dataset approval and file hashes.
4. T4 preflight: performs 10 real diagnostic updates, saves, launches a fresh process,
   restores and compares model/optimizer/scheduler/RNG, then performs 10 more updates.
   This cell measures actual GPU memory, loss, throughput, complete checkpoint write cost
   and estimates the unchanged 20,000-step run and 1,000-step checkpoint interval.
   Only the separate diagnostic uses 10-step saves. Stop here on errors; do not change
   batch, model or precision. Review measured time and estimated retained-checkpoint bytes
   against your available session/Drive space before proceeding. Session limits vary.
5. Train or resume: starts/resumes the approved batch-8, seed-0, 20,000-step run with
   1,000-step complete saves. No configuration edits or Hugging Face login are expected
   for the public pinned base assets. The cell normally remains running until completion.
6. Inspect recorded production optimizer evidence: run after the training cell finishes
   or stops. The running training cell itself prints ACTUAL OPTIMIZER UPDATE after the
   first optimizer post-step hook, with optimizer_step > 0 and actual_post_step_hook=true.
   Forward losses and diagnostic preflight steps do not prove production training began.
7. Export for local inference: exports the newest complete hash-verified checkpoint after
   production training stops. It includes processor assets, normalization, training
   approval, optimizer-step evidence and export-manifest hashes. Download the whole folder.

## Paths

Runtime mount: `/content/drive/MyDrive/vlarobot_smolvla`. Source/data are scratch paths
`/content/VLA-Project` and `/content/vlarobot-data`, recreated identically after runtime loss.

Production: `MyDrive/vlarobot_smolvla/training-tiago-v1/`. Numeric checkpoints:
`checkpoints/001000/` through `checkpoints/020000/`; each must have `COMPLETE.json`.
Model/processor assets live in `pretrained_model/`; optimizer/scheduler/RNG/step in
`training_state/`. Evidence: `optimizer-steps.jsonl` and `training-progress.json`.

Partial runs without a usable checkpoint are preserved under their original paths; recovery
may use `training-tiago-v1.restart-NNNN`. The notebook selects the highest valid complete
checkpoint across these paths and refuses duplicate live training processes.

Diagnostic reports: `MyDrive/vlarobot_smolvla/preflight/t4-vNNN/t4-preflight-report.json`.
Diagnostic checkpoints are not production exports or role-gate evidence.

Exports: `MyDrive/vlarobot_smolvla/exports/training-tiago-v1-step-020000/` for the normal
completed run, or the selected restart name and latest completed step.

## Reconnect / resume

If only the browser connection dropped, reconnect and see whether the existing training
cell is still running; do not launch a second job. If the VM was lost, select a new T4
runtime and rerun cells 1–5 with the same paths. Authenticate Drive again when requested.
Setup restores the same scratch paths. A verified completed preflight can be reused;
incomplete diagnostics are preserved and a new version is used. Production resumes from
the newest valid complete checkpoint, restoring model, optimizer, scheduler, RNG and step.
At most the unsaved interval must be repeated. No directories or COMPLETE markers should
be deleted or manually edited. Run inspection/export after training stops.

## Verification limits

All notebook cells and nested programs compile; approved hashes remain unchanged. The
CPU suite and isolated restore checks cover local preparation. Actual fresh Colab package
installation, T4 peak memory/throughput and Drive restoration require the authenticated
runtime; the embedded preflight is the execution test, not a claim these already passed.
Frozen TRAIN evaluation remains 24 pairs, literal gate 40% (at least 10 successes), no TEST
access, December 12 deadline unchanged.
