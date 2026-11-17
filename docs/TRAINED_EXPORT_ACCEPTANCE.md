# Final trained-export acceptance

The owner reports completion at step 20,000, restored from production checkpoint
16,000 with 4,000 updates in the resumed invocation and 80 approved episodes.
Export returned code 0. These are user-supplied records, not locally verified
weights or proof of inference compatibility. No training reset is requested.

The approved logger in the primary worktree hashes only the first 32 scalar
values of at most eight gradient-bearing parameter tensors. It compares values
immediately before and after one selected optimizer call, casting the sample to
float32. Its flag is neither cumulative since initialization nor cumulative since
resume. A false flag can coexist with changes outside that sample. No particular
cause is assigned to the actual final production record before inspecting its
weights, learning rates, sample count and checkpoint state.

`tools/check_optimizer_logging.py` exercises the exact approved logger read-only
with a synthetic SGD update that changes tensor elements outside the sample. It
must report a real optimizer call and false sampled change while the full tensor
changes. This establishes a logging limitation, not production training success.

On receipt of the export, perform these checks in order, retaining all artifacts:

1. Hash the received archive before extraction; reject absolute/traversal paths
   and symlinks. Preserve the original archive and extraction inventory.
2. Verify the export manifest against every file, require final step 20,000,
   production optimizer evidence and the approved 80-episode dataset provenance.
   Check config, processors, tokenizer and normalization state against the approved
   training contract. Export process return code alone is insufficient.
3. Compare all final model tensors against the exact pinned pretrained
   initialization loaded by the approved production path. Verify matching keys,
   shapes and dtypes, finite values, per-tensor differences and aggregate hashes.
   Initialization of newly created tensors must use the original seeded loading
   path; do not substitute a different revision or random initialization.
4. Load the installed SmolVLA policy with portable processors and strict weight
   loading. Use correctly decoded real TRAIN observations to verify camera keys,
   state8/action7 representation, normalization, finite outputs, adapter behavior
   and measured inference latency. A successful export is not this preflight.
5. Only after compatibility is established, run the existing primary-worktree
   `run_vla_campaign` on all frozen TRAIN 8×3 paired scene/seed entries. Preserve
   failed/interrupted attempts separately; no scripted fallback. Every action
   retains live motor safety. Record checkpoint, source and configuration hashes.
6. Report independent can lift and final placement, failure causes, inference
   latency and literal gate: at least 10/24 successes. Preserve the registered
   baseline and frozen tolerances. No TEST access or tuning.

The portable export has not yet been supplied locally. Full-weight comparison,
installed inference and the 24 physical VLA episodes therefore remain pending.
