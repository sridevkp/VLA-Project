# Python helpers

Keep cross-platform project helpers in Python rather than platform-specific shell scripts. The
Webots environment helper is implemented in `src/vlarobot/env.py` and is used by the smoke command.

`audit_can_path.py` runs the deterministic, Webots-free IK continuity, swept-transit, and tabletop
candidate audit. `prepare_original_comparison.py` prepares a temporary pre-refactor worktree for
M1-06a trace capture; compare its JSONL output with the brain trace using `cli compare-traces`.
