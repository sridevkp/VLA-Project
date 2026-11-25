# V2 post-training checkpoint import and three-scene diagnostic

This is the executable handoff for the first recovered SmolVLA checkpoint. It
preserves the previous endpoint-decoder evidence and never overwrites it. It is
a TRAIN diagnostic, not the 24-trial admission gate. No V1 TEST outcome is
touched and the sealed 30-identity V2 holdout remains unopened.

Baseline for comparison: `runs/worldmind/v2-train-endpoint-1/` (step-20,000
checkpoint, manifest SHA-256 `5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1`,
three seed-0 TRAIN identities, 0/3, no grasp or lift).

## Owner action required to start training

The local machine has no CUDA device and no authenticated Colab session, so
the GPU run cannot be started from the agent environment. The following is the
only manual action needed. The pinned source archive has already been prepared
locally at `/home/tj330/Downloads/vla-recovery-55c8938.zip` with SHA-256
`dd27b10140fadfba257b7a9696186420b05f02f30894b05938729d6db9e8537e` for commit
`55c89386e5515d2ef391e81d39ed3f42fabb6093`.

1. In a browser signed in to the owner's Google account, open Colab and create
   a Python 3 / T4 GPU notebook. Upload
   `/home/tj330/programs/VLA-Project/notebooks/smolvla_finetune_colab.ipynb`
   with **File > Upload notebook**, then save a copy in Drive.
2. Upload these files into `MyDrive/vlarobot_smolvla/`:
   `runs/colab-prep-v3/source.zip`, `runs/colab-prep-v3/dataset-train-v1.zip`,
   `runs/colab-prep-v3/bundle-status.json` (all from
   `/home/tj330/programs/VLA-Project/`), plus
   `acct2-durable-v1-drive-v1-portable-020000-attempt-1.zip` and its
   `.receipt.json` from `~/Downloads/`, and the prepared
   `vla-recovery-55c8938.zip`.
3. Run only the first three code cells of the notebook (Drive mount, isolated
   stack install, TRAIN dataset verification). Stop before the T4 preflight and
   do not run the V1 training cells.
4. Add one new code cell containing the recovery program in
   [VLA_RECOVERY.md](VLA_RECOVERY.md#train-transition-warm-start-on-colab),
   with `REVISION = "55c89386e5515d2ef391e81d39ed3f42fabb6093"`. Before running
   it, extract the pinned source so no repository credentials are needed:
   `!unzip -q /content/drive/MyDrive/vlarobot_smolvla/vla-recovery-55c8938.zip -d /content/VLA-Project-v2`
5. Run the cell and keep the browser tab attached. It targets 1,000 updates
   (about 20–60 minutes including setup and checkpoint writes). Preserve the
   cell output, `optimizer-steps.jsonl` and all `step-*` directories. If the
   output folder already exists in Drive, use a fresh numbered folder in the
   cell rather than deleting anything; `--resume` applies only to a complete
   checkpoint.
6. Download `step-021000` and `optimizer-steps.jsonl` from
   `MyDrive/vlarobot_smolvla/v2-transition-warmstart-01/` and report the
   export-manifest SHA-256. Then resume section 1 below.

## 1. Receive and freeze the recovered export

From the Colab recovery cell, download the entire `step-021000` directory from
`MyDrive/vlarobot_smolvla/v2-transition-warmstart-01/` into a new local
directory, for example `~/Downloads/v2-transition-warmstart-01/step-021000`.
Retain `optimizer-steps.jsonl`, `recovery-optimizer.pt`, `training-approval.json`
and all processor assets. Do not edit any file in the export.

Verify the export through the same production loader used by inference. Run
from the recovery worktree:

```text
cd /home/tj330/programs/VLA-Project-week3
uv run python -c "from pathlib import Path; from vlarobot.vla.checkpoint_export import require_trained_export; require_trained_export(Path.home() / 'Downloads/v2-transition-warmstart-01/step-021000'); print('export verified')"
uv run python -c "import hashlib,pathlib; p=pathlib.Path.home()/'Downloads/v2-transition-warmstart-01/step-021000/export-manifest.json'; print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Record the printed manifest hash, the source checkpoint hash above, the run
directory and the download date in a new `docs/evidence/` receipt. A failed
verification stops the diagnostic; do not relax the export checks.

## 2. Run the same three diagnostic identities

Use a new output directory every time. The flags below reproduce the verified
endpoint-decoder execution path: learned gripper endpoint decoding, bounded IK
fallback only after a residual miss, verified HOME termination and the
independent SafetyMonitor. The model process loads the export through
`PortableChunkModel`; no scripted proposer is enabled.

```text
cd /home/tj330/programs/VLA-Project-week3
uv run python -m vlarobot.cli eval-vla \
  --export ~/Downloads/v2-transition-warmstart-01/step-021000 \
  --out runs/worldmind/v2-train-recovered-diag-1 \
  --model-python /home/tj330/programs/VLA-Project/.venv-vla/bin/python \
  --webots /home/tj330/.local/opt/webots/webots \
  --recovery-v2 --recovery-ik --endpoint-gripper \
  --diagnostic-index 0 --diagnostic-index 1 --diagnostic-index 2
```

The campaign writes `diagnostic-results.json`, `derived-motion-summary.jsonl`,
per-episode `episode-NN.meta.json`, `.steps.npz`, `.actions.jsonl`,
`.latency.jsonl`, `.latency.motion.jsonl`, the model-server log and the
campaign `provenance.json`. Keep all of them, including failures.

Regenerate the comparison fields with the same calculation used for the
baseline:

```text
uv run python tools/v2_diagnostic_metrics.py runs/worldmind/v2-train-recovered-diag-1
```

## 3. Compare against the endpoint-decoder baseline

For each of `train-clear-i00`, `train-cluttered-i01` and
`train-reach-limited-i02` (all seed 0), record from `derived-motion-summary.jsonl`
and `diagnostic-results.json`:

- model queries and accepted action ticks;
- raw and decoded gripper targets (`raw_min`, `raw_max`, `raw_outside`,
  `gripper_events`);
- closest measured tool-to-CAN distance (`closest_m`) and approach direction;
- gripper close attempts, alignment at each endpoint event and grasp evidence;
- observed CAN XY displacement (`can_xy_m`) and maximum rise (`rise_m`);
- transport, placement, release, HOME completion and independent verification;
- terminal state and reason, plus IK fallback count.

The decisive evidence is an autonomous close, lift and transport confirmed by
the independent task verifier. More accepted ticks or more model queries alone
are not recovery. A 0/3 result with only open-endpoint commands repeats the
known learned-behavior failure and requires a measured training decision, not
another decoder or safety change.

## 4. Gate before the full admission campaign

The admission plan remains all 24 original TRAIN identities with the unchanged
success threshold of at least 10/24. A successful diagnostic does not bypass
that gate. Before admission, preserve a frozen checkpoint receipt and confirm
that the diagnostic execution configuration has not changed. Run the full
plan only if the three scenes show real manipulation progress.
