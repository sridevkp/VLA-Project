# Transferred production export: blocked by missing weights

Inspected the direct contents of the primary worktree's `runs/trained-export`
without modifying any transferred file. The existing installed-environment import
validator fails in `require_trained_export` with `FileNotFoundError` for
`model.safetensors`, before model loading or robot commands.

All 19 present manifest-listed files match their hashes. The missing model's
expected SHA-256 is
`107698cfc0274cb2e0fbe2976be0e1e0a198f41c140807ec19bca4c48f04ed58`.
The manifest identifies step 20,000 and the exact final optimizer log record:
restored step 16,000, 4,000 updates in that invocation, eight sampled tensors and
learning rate 2.50000060142902e-06. The stale config `pretrained_path` referring
to 016000 is a resume source reference, not sufficient reason to relabel the
export; actual final weights still require inspection.

The transferred approval has 80 episodes and matches the local approved dataset's
stats, source provenance and integrity fingerprints. Both normalization tensor
files load successfully and contain finite values. Config declares state8,
action7, camera1 RGB 3×480×640 and 50-action chunks. Actual trained inference
compatibility remains unverified.

Of 210 optimizer evidence records, 209 report sampled parameter changes, including
step 19,900. Only the final sample reports unchanged values. The approved logger
compares a small sample across a single update; the existing synthetic regression
demonstrates false sampled equality can coexist with full-tensor changes. The
actual final sample's cause and final-vs-initial weight changes cannot be resolved
without the missing weights. Neither success nor failure is inferred from it.

Complete audit and validator traceback are preserved in the primary worktree at
`runs/import-trained-v1/export-audit.json` and `import-validator.log`.
Transfer the original production export's missing `model.safetensors` into the
same export directory; do not substitute base weights or diagnostic checkpoints.
Then verify its expected hash and rerun import preflight with a new versioned
output directory. No trained inference or physical episodes have run; the gate
is UNMEASURED, not failed. Frozen TRAIN gate is min(0.40, 0.60×23/24)=0.40,
requiring 10/24. No TEST access or experiment changes occurred.
