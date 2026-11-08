# Python helpers

Keep cross-platform project helpers in Python rather than platform-specific shell scripts. The
Webots environment helper is implemented in `src/vlarobot/env.py` and is used by the smoke command.

`audit_can_path.py` runs the deterministic, Webots-free IK continuity and swept-transit audit.
`posture_study.py` owns the candidate search and invokes it only after the full base sweep selects
its best setting. `prepare_original_comparison.py` prepares a clean pre-refactor worktree for
M1-06a fixture-v2 trace capture. The fixture adds opt-in measurements without changing controller
behavior; compare its JSONL output with the brain trace using `cli compare-traces` and retain the
per-run joint diagnostics.

`posture_study.py` performs a deterministic, Webots-free posture/torso/base analysis and runs its
candidate search only after the full base sweep identifies a best setting. The full base grid has
1,960 settings and can take many hours. Use `--skip-base-sweep` to run only studies a-c; it does not
run the dependent candidate search.

`analyze_pregrasp_table.py` replays synchronized PREGRASP interpolation and independent 0.5 rad/s
joint motion through FK against the factory table box. It reports both torso heights, sampled hand
radius, minimum signed clearance, and a deterministic `safe_via` route estimate. It is a kinematic
screen only; Webots contacts and physical clearance remain unverified.
