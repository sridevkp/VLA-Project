# Production training handoff — 2026-11-15

The owner reports a successful authenticated T4 diagnostic: 20 optimizer updates,
1.0675 steps/s, 2.73 GiB peak reserved memory, and complete checkpoint restoration.
These measurements are owner-reported, not independently read from Drive here.
Production was explicitly not started in that earlier report. The subsequent
owner-supplied Drive progress record reports production at step 14,700, restored
from step 0, with 14,700 actual calls, a real post-step hook, changed sampled
parameters and 80 approved episodes. This is production evidence supplied by the
owner, not independently verified local artifacts. The running Colab session must
not be reset or restarted to apply local safeguards.

The approved production configuration remains batch size 8, 20,000 steps and
1,000-step checkpoint intervals. The approved dataset, source bundle, model and
frozen evaluation criteria are unchanged. The measured compute estimate is 5.32
hours before additional checkpoint, startup and session overhead.

The authoritative notebook is in the primary project worktree, not this older
isolated analytic worktree. Its Drive root is
`/content/drive/MyDrive/vlarobot_smolvla`. Diagnostic outputs are under `preflight/`;
production outputs are under `training-tiago-v1` and its owned `.restart-*`
siblings. Do not copy a diagnostic checkpoint into a production directory.

`tools/verify_production_namespace.py` checks the primary worktree's production
source/configuration hashes against the approved source archive, then exercises
directory selection with synthetic complete checkpoint fixtures. A diagnostic
step-20 checkpoint is excluded from fresh production; a diagnostic step-20,000
checkpoint cannot displace a valid production checkpoint. Diagnostic exports are
rejected as production evidence. This is a namespace test, not a GPU restoration
test. Results are preserved in `docs/evidence/production-namespace.json`.

For a fresh production launch, the first real post-update event must identify
`run_kind=smolvla-finetune`, `optimizer_step=1`,
`restored_training_step=0` and `actual_post_step_hook=true`. A resumed production
run must instead name its actual restored production step. A launch message or a
forward pass does not establish training progress.

The launcher selects the highest complete, integrity-checked numeric checkpoint
only within the production namespace. Incomplete checkpoints are ignored and
unrecoverable partial runs are preserved in numbered restarts. The complete
training state includes model, optimizer, scheduler, RNG and global step.

No authenticated Colab/Drive session is available to this local agent. Actual
production optimizer events, production checkpoint hashes, Drive persistence and
cloud resumption remain unverified until their artifacts become accessible.
Production is now reported started by the owner; independent local verification
of the production artifacts remains pending checkpoint delivery.

Independent physical VALIDATION has completed in the isolated worktree. The earlier
18-episode run is retained as a mixed-runtime diagnostic: its first four policy
workers used the canonical robotics environment and the remaining fourteen used
the ML environment. A new 18-episode campaign uses an explicit canonical runtime
guard and completed 15/18, with independent recorded lift/placement checks.
Neither campaign accesses TEST or changes the TRAIN gate of 10/24.
