# WorldMind controlled VALIDATION pilot readiness report (2026-11-17)

Branch: `work/worldmind-week3`. Accepted milestone: `241b62d`. This report covers only inert,
non-experimental work. No pilot was executed or authorized.

## 1. Scope and authorization boundary

- Resolved the calibration bootstrap, proposed Eq. 3.8 definitions, implemented a conservative
  full-arm swept-collision checker, and reviewed/corrected the scripted waypoint cadence.
- Local code and commits are authorized only. No push, merge, PR update, TEST/TRAIN access,
  Webots execution, real IK, inference, training, actuation, scored evaluation or pilot execution
  was performed.
- No public interface in `core/types.py`, `core/interfaces.py` or `core/wire.py` changed. No
  dependency or version pin changed. `execution/scripted_can.py` was not modified.
- Recommendation is **NO-GO** for the controlled pilot on the current evidence.

## 2. Frozen facts and unchanged artifacts

- Scripted TRAIN `23/24`; scripted VALIDATION `15/18`; VLA-only TRAIN `0/24 FAILED`; scripted
  primary; A-D2 CLOSED/NO-GO.
- Frozen VALIDATION plan: 18 unique episodes, seeds `{0,1,2}`, K=32, H=50, tick `0.032 s`,
  `scripted_can.py` cadence as archived.
- Authoritative report SHA-256 `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`.
- Frozen evidence, Drive canary evidence, scored summaries, splits, seeds, thresholds and gates
  were not altered. `verify_week3_frozen_evidence.py` reports `test_split_accessed: false`.

## 3. Task 1 — non-circular calibration bootstrap

Four strictly separated stages are now encoded:

1. **Uncalibrated diagnostic config.** `configs/worldmind_diagnostic.yaml` has
   `status: diagnostic_uncalibrated`, `diagnostic.mode: evidence_collection_only`,
   `scored_evaluation_allowed: false`, `calibration_stage: 1` and
   `evidence_collection_split: validation`. It retains every hard constraint, frozen split,
   seeds, 18 episodes, K=32, H=50, tick, oracle target source, report SHA, surrogate,
   safety, verification and adapter gate. It contains no Eq. 3.8 weights, ranking direction,
   calibration candidate or calibration evaluator.
2. **VALIDATION-only evidence collection.** `build_calibration_evidence()` writes unapproved
   evidence only. No campaign, score, TEST/TRAIN access or parameter selection occurs.
3. **Owner selection/approval.** `rank_calibration_candidates()` returns a
   `CalibrationProposal` with no `parameters`, `selection` or `approval_record`. The CLI
   `worldmind-calibration-proposal` prints a deterministic selection statistic and never writes
   or approves it. Selection remains `registered_evaluator_max_score_then_candidate_id`.
4. **Subsequent scored evaluation.** Only a separate `ready` config with owner-frozen approved
   parameters can enter `eval-worldmind`. `evaluate_calibration()` labels its output
   `selection_statistic_not_independent_evaluation`, `independent_evaluation: False`,
   `test_split_accessed: False` and `frozen_evidence_modified: False`.

No hard constraint was weakened, no calibration result was invented, post-selection performance
is never presented as independent evaluation, and frozen VALIDATION scenes/seeds are unchanged.

## 4. Task 2 — proposed Eq. 3.8 definitions

All values below are `engineering_choice_not_report_quotation`, `authoritative: false` and
`scored_execution_enabled: false`. They are inert, unregistered and not executable as scoring
inputs.

| Term | Proposed definition | Units / normalization |
|---|---|---|
| `Jg` | `sqrt((||p_ee - p_goal||_2 / s_p)^2 + (angle(q_ee, q_goal) / s_r)^2)` | Dimensionless; `s_p` in m, `s_r` in rad |
| `Jl` | `(sum_k ||p_{k+1} - p_k||_2 / s_l) + lambda_t * (N_steps * dt_s / s_t)` | Dimensionless; `s_l` in m, `s_t` in s, `dt_s` in s |
| `Jr` | `mean_k clip((m - d_min,k) / m, 0, 1)`, with `d_min,k < 0 -> 1` | Dimensionless; `m` in m; lower clearance is worse |
| `Ju` | `mean_k (||e_p,k|| / s_p + e_theta,k / s_r)` | Dimensionless; `s_p` in m, `s_r` in rad |

- Ranking direction: **minimize** all four terms; owner must select weights and confirm the
  aggregation contract. No weight, threshold or risk value is chosen here.
- Required data: goal/tool pose, trajectory link points, clearance samples and prediction-error
  samples from the declared observation path. Missing or non-finite inputs raise
  `MissingObjectiveInput` rather than being imputed.
- Safety precedence: any hard veto raises `HardVetoPrecedesUtility`; utility is never used to
  override a veto.
- `Jr` needs an exact authoritative clearance definition and margin `m` before it can be approved;
  no threshold is inferred from cost, geometry tests or frozen evidence.
- Artifacts: `src/vlarobot/eval/worldmind_objectives.py` and
  `configs/report/objectives.proposal.json`.

## 5. Task 3 — conservative full-arm swept collision check

- `src/vlarobot/world_model/arm_sweep.py` adds `IntermediateSweptArmCheck`, which interpolates
  explicit joint/end-effector substates and calls the conservative `SweptArmCheck` on each
  subinterval. It returns `True` only when every interval has certified coverage and is clear,
  `False` for conservative rejection, and `None` when coverage is incomplete.
- It fails closed on missing/incomplete link geometry, unsupported collision shapes or contacts,
  invalid/non-rigid transforms, non-finite motion, moving scene objects without their own bound,
  and interpolation endpoint mismatch.
- `unverified_urdf_sweep_check` is opt-in and deliberately **not** in
  `WORLD_MIND_BINDING_IDS`. The production registry has no `arm_sweep_checks`, and
  `analytic_surrogate_factory()` still uses `check_transition=None`.
- Therefore `unverified-arm-sweep` remains a hard veto. Unit tests do not certify real TIAGo arm
  geometry; an owner-reviewed full-arm swept-volume certificate is still required.

## 6. Task 4 — scripted waypoint cadence review

- `execution/scripted_can.py` was left unchanged. The production adapter now emits:
  `INIT(open) -> PREGRASP -> DESCEND -> CLOSE(closed) -> LIFT -> TRANSIT -> PLACE_DESCEND ->
  RELEASE(open) -> RETREAT -> HOME(joint target)`.
- Explicit gripper and joint stations, reviewed state-specific position tolerances (`IK_TOL`,
  `IK_TOL_GRASP`, `IK_TOL_TRANSIT`) and per-decision re-derivation from the latest oracle state
  replace the earlier shortened, stale, single-tolerance cadence. Hold steps sum to H=50.
- Fixed holds are only a scripted cadence proxy. They are **not** a claim of measured settling,
  contact equivalence, stall-driven gripper equivalence, tilt-only IK equivalence or physical
  success.
- Remaining physical validation needs are listed in
  `docs/WORLDMIND_WAYPOINT_CADENCE_REVIEW.md`: tool pose/tilt/settling trace comparison, CLOSE and
  RELEASE stall outcomes, corrected-registration tolerance sufficiency, identity-orientation
  grasp/placement safety, alternate-family registration and full-arm certification.

## 7. Verification evidence and exact commands

- `uv run ruff check .` — passed.
- `uv run ruff format --check .` — passed (`296 files already formatted`).
- `uv run ty check src` — passed.
- `uv run python -m compileall -q src tests` — passed.
- `uv run pytest -q tests/test_worldmind_production.py tests/test_worldmind_objectives.py tests/test_worldmind_arm_sweep.py tests/test_worldmind_waypoints.py` —
  `38 passed`.
- `uv run pytest -m "not webots"` — `1 failed, 569 passed, 3 skipped` in `138.26s`. The only
  failure is pre-existing:
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`,
  caused by a `tools/recovery/restore_gate.py` hash mismatch against frozen
  `docs/evidence/drive-canary-v7-bundle.json`. Frozen evidence was not altered.
- `uv run python tools/verify_week3_frozen_evidence.py --locked-source-root /home/tj330/programs/VLA-Project` —
  passed; `report_sha256_ok: true`, `test_split_accessed: false`.
- Exact fail-closed production probe from `241b62d`:
  `uv run python -m vlarobot.cli eval-worldmind --config configs/worldmind_production.yaml --out runs/worldmind/validation-s0-preflight-probe --backend webots --state-source oracle`
  produced `eval-worldmind preflight failed:`, `config: ValueError: worldmind production config is blocked; reviewed status must be 'ready'`, exit code `1`, and no output directory.
- First diagnostic invocation with the shipped blocked template (exit code `1`, no output
  directory):
  ```text
    [FAIL] diagnostic_config: ValueError: backend.declaration is still a blocked template placeholder
  worldmind-diagnostic preflight failed:
    - diagnostic_config: ValueError: backend.declaration is still a blocked template placeholder
  ```

## 8. Owner decisions, blockers and unverified paths

Pilot blockers requiring owner action:

1. Authoritative report PDF and approved definitions matching the frozen report SHA.
2. Approved VALIDATION calibration evidence, evaluator binding and frozen parameter artifact.
3. Four Eq. 3.8 weights, component formulas, units, normalization and ranking direction.
4. Owner-reviewed full-arm swept-volume certificate before `unverified-arm-sweep` can cease to be
   a hard veto.
5. Physical validation/sign-off of waypoint cadence, orientation, gripper timing and IK
   tolerances.
6. Explicit authorization for the controlled scored pilot.

Explicitly unverified: Webots execution, real IK solves, neural inference, training, actuation,
scored VALIDATION results, TEST/TRAIN access, Windows behavior for the new commands, full-arm
geometry certification, waypoint physical equivalence and pilot execution. No success is claimed
for any of these paths.

## 9. First diagnostic command and GO/NO-GO recommendation

Exact first diagnostic run command, from the repository root, after owner-reviewed values have
been inserted into `configs/worldmind_diagnostic.yaml`:

```text
uv run python -m vlarobot.cli worldmind-diagnostic --config configs/worldmind_diagnostic.yaml --out runs/worldmind/validation-s0-diagnostic --backend webots --state-source oracle
```

With the shipped template this command is expected to fail closed, as recorded in section 7. It
does not score, select parameters or access TEST/TRAIN.

**Recommendation: NO-GO.** The implementation and inert checks are ready for controlled evidence
collection only. The pilot must not proceed until every owner decision in section 8 is resolved,
the full-arm veto has a reviewed certificate, physical waypoint validation is signed off, and a
separate scored pilot authorization is issued.
