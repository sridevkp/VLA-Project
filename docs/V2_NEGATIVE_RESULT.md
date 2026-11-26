# V2 recovered checkpoint — negative TRAIN diagnostic result

**Date:** 2026-11-26
**Decision:** Stop the time-boxed V2 side-track. Do **not** run the 24-trial
admission campaign. The approved stop rule applies: the recovered checkpoint is
0/3 on the unchanged three TRAIN diagnostic scenes, with only open-endpoint
gripper commands and no verifier-confirmed autonomous close, lift or transport.

This is not the 24-trial admission gate. V1 results, the frozen `docs/RESULTS.md`
frozen-arm numbers and the sealed 30-identity V2 holdout are untouched. The
threshold remains `>=10/24` TRAIN. No decoder, IK, safety, gate or threshold
setting was changed.

## Checkpoint received and frozen

- Source: `/home/tj330/Downloads/v2-transition-warmstart-01/step-021000`.
- Step `21000`, warm-started from manifest
  `5d31a5d8c1a7728f0e8db7ab379d8d29737d75c18ee1897e12d49153477d92a1`
  (step `020000`).
- Recovered manifest SHA-256:
  `0d5059dc5051b12a0d19614cc93f4b6a8ec77887f7e1e97b6e830858399504d6`.
- Model SHA-256:
  `9aa7631a11704e92d2e820aa44c875024085eae0c178b7d129d506fbd365eac9`.
- Production loader: `require_trained_export`; optimizer state reloaded
  (`update=1000`, `batch=8`).
- Telemetry: 1000 updates, global steps `20001–21000`, final pose loss
  `0.47728`, final gripper loss `0.09334`, final gradient norm `6.25078`.
- Receipt: `docs/evidence/v2-recovered-step21000-receipt.json`.

## Execution configuration did not change

Both runs used `v2_bounded_gripper_verified_home` with `--recovery-v2`,
`--recovery-ik`, `--endpoint-gripper` and diagnostic indices `0,1,2`. The five
controller/adapter hashes are identical between the step-020000 baseline and the
recovered step-021000 run. The only changed input was the model export manifest
hash. The first diagnostic invocation failed only because the default relative
baseline path is absent in this worktree; the corrected invocation adds an
explicit path to the same frozen baseline and changes no policy or control
setting (`docs/evidence/v2-diagnostic-invocation-note.json`).

## Measured comparison

`closest_m` is the minimum measured tool-to-CAN distance. `rise_m` and
`can_xy_m` are the observed CAN maximum rise and XY displacement. The approach
vector is from CAN to the tool at the closest sample, in world coordinates.

| Scene | Baseline closest (m) | Recovered closest (m) | Recovered approach vector (m) | Recovered rise / CAN XY (m) | Gripper endpoint events | Independent verifier |
|---|---:|---:|---|---|---|---|
| train-clear-i00 | 0.7909 | 0.8595 | (-0.7128, 0.3972, 0.2700) | 4.0e-6 / 7.7e-6 | one accepted `0.045` open at t=0.032 s, alignment 0.9921 m | false, no HOME |
| train-cluttered-i01 | 0.8860 | 0.7848 | (-0.6782, 0.2347, 0.3177) | 8.1e-6 / 1.8e-5 | one accepted `0.045` open at t=0.032 s, alignment 0.9930 m | false, no HOME |
| train-reach-limited-i02 | 0.8800 | 0.2581 | (-0.1764, -0.0453, 0.1828) | 1.2e-5 / 2.5e-5 | one accepted `0.045` open at t=0.032 s, alignment 1.0317 m | false, no HOME |

Additional comparable fields:

| Scene | Queries baseline → recovered | Accepted ticks baseline → recovered | IK fallbacks baseline → recovered | Raw opening range recovered (m) | Raw outside recovered | Observed holding recovered | Terminal / reason recovered |
|---|---:|---:|---:|---|---:|---:|---|
| train-clear-i00 | 3 → 2 | 108 → 92 | 13 → 1 | 0.04416–0.04640 | 78 | 0 | SAFETY / VLA IK residual |
| train-cluttered-i01 | 2 → 3 | 92 → 115 | 4 → 10 | 0.04436–0.04674 | 134 | 0 | SAFETY / workspace z |
| train-reach-limited-i02 | 2 → 3 | 92 → 149 | 7 → 3 | 0.04290–0.04554 | 55 | 0 | SAFETY / VLA IK residual |

All 356 accepted gripper endpoint actions in the recovered run (92/115/149)
target `0.045` m; there are zero close targets (`<0.02` m), zero observed
holding ticks, and the CAN moves only micrometres. The single endpoint event per
scene in the table is the de-duplicated first open command, not a second gripper
command. The `train-reach-limited-i02` approach improved from
0.8800 m to 0.2581 m, but 258 mm separation, an open gripper and zero CAN motion
are not grasp, lift or transport. Higher accepted-tick counts (149 versus 92 in
that scene) are not recovery evidence.

Independent verifier results remain `success=false` and `home_reached=false` in
all three scenes. Final target distances are `0.2378`, `0.2487` and `0.2989` m,
effectively unchanged from the baseline. Every episode terminates in `SAFETY`:
two VLA IK residual misses and one workspace-z veto.

## Decision applied

The approved rule "if the three scenes are 0/3 with only open-endpoint gripper
commands, stop and write the V2 negative result" is satisfied. Consequences:

- No 24-trial admission campaign is started before 2026-12-01.
- The V2 holdout remains sealed; no held-out identity, plan or outcome was read.
- V1 TRAIN 0/24 and the frozen TEST campaign are not modified or superseded.
- No further decoder, IK or safety changes are made under this authorization.
- Any future V2 work requires a new owner decision with a measured training
  hypothesis, not another decoder or safety adjustment.

## Evidence

- Machine-readable comparison:
  `docs/evidence/v2-recovered-diagnostic-result.json`.
- Checkpoint receipt: `docs/evidence/v2-recovered-step21000-receipt.json`.
- Invocation note: `docs/evidence/v2-diagnostic-invocation-note.json`.
- Recovered raw campaign (preserved locally, `runs/` is gitignored):
  `runs/worldmind/v2-train-recovered-diag-1/`, including
  `diagnostic-results.json`, `derived-motion-summary.jsonl`, per-episode meta,
  steps, actions, latency, logs, frames and `provenance.json`.
- Baseline raw campaign: `runs/worldmind/v2-train-endpoint-1/`.
