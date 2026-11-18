# WorldMind Owner-Review Packet (2026-11-18, offline)

**Scope:** `work/worldmind-week3` local implementation/readiness packet. No Webots launch, no robot actuation, no physics steps, no real IK/planning/trajectory execution, no calibration execution, no scored VALIDATION/TEST, no training, no TEST/TRAIN access, no push/merge/PR update, and no frozen-evidence modification were performed while preparing this packet.

**Authoritative frozen report SHA-256:** `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`

## 1. Executive state

- **Task A runtime table inspection v2 (completed, preserved):** `runs/worldmind/validation-s0-webots-table-live-v2` recorded `5/5` table collision boxes `Matched/VERIFIED` on the primary observed-live-base comparison with pinned-initial-base audit preserved (`geometry_export_valid=true`, `geometry_object_count=1572`, `unsupported_shape_count=6`, `no_actuation=true`, `backend_apply_calls=0`, `backend_step_calls=0`, `backend_reset_calls=0`, `scored_results_written=false`, `motion_allowed=false`). Overall diagnostic **FAILED** (`observation_valid=false`, `observation_age_s=13.138615762989502`, `observation_timestamp_s=0.0`, `observation_freshness_limit_s=5.0`, blocker `observation is stale`). Root cause in v2: adapters measured age as client-side RPC/request latency and stamped capture after receipt, conflating transport/startup latency with observation staleness.
- **Task A runtime table inspection v3 (completed once, preserved, no retry):** `runs/worldmind/validation-s0-webots-table-live-v3` (`diagnostic.json` `aca9a869…e4d9a`, `state.json` `a1150d96…37771`, `provenance.json` `db8a1701…886f99`, `geometry.json` `4784afa3…4528`) recorded runtime table geometry **5/5 matched/runtime-grounded** with pinned-initial-base audit preserved, `geometry_export_valid=true` (`geometry_object_count=1572`, `unsupported_shape_count=6`), **capture timestamp validation passed** (consistent/non-future/non-invalid server-stamped capture: `captured_at_utc=2026-11-18T14:18:13.953226+00:00`, `observation_timestamp_s=0.0`). **Observation freshness FAILED** (`observation_age_s≈12.8636s > observation_freshness_limit_s=5.0`, `observation_valid=false`, blocker `observation is stale`), so overall inspection **FAILED** (exit 1). `no_actuation=true`, `backend_apply_calls=0`, `backend_step_calls=0`, `backend_reset_calls=0`, `scored_evaluation_allowed=false`, `scored_results_written=false`, `test_split_accessed=false`, `train_split_accessed=false`, `frozen_campaign_modified=false`, `motion_readiness=NOT_READY`, `motion_allowed=false`. **No inspection retry/Webots launch without new authorization.**
- **v3 freshness diagnosis (evidence + source, not guessed):** `captured_at_utc`/`capture_wall_time_s` are stamped at server-side read-only state capture start in `capture_read_only_state` during `inspect_geometry`; the server then performs full geometry export/table inspection (`capture_read_only_geometry` → `export_bounding_objects`) before RPC return. The client validates/builds the snapshot on receipt (receipt age stored in `snapshot.age_s`) inside `launch_webots_session`; context teardown waits `grace_s=10.0` for self-exit then runs `_terminate_process` (polite/forced, up to `shutdown_timeout_s=30.0`) before CLI finalization recomputes consumption-time `observation_age_s`. v3 capture is valid but consumption age is `~12.8636s` (>5s); the delay therefore spans server export/RPC/transport plus post-receipt shutdown/teardown/finalization. **Exact apportionment cannot be determined from preserved v3 evidence** (no receipt/close/finalization/export-duration timestamps recorded). Freshness threshold remains unchanged (`5.0s`) and all fail-closed validity checks remain in force (consumption-time age determines validity).
- **Missing timing measurement (implemented offline, justified):** `observation_age_s_at_receipt` (validated capture age at client receipt/snapshot, before session teardown) is now reported alongside consumption `observation_age_s` in diagnostic/geometry report and state (`worldmind_webots_diagnostic.py`, `worldmind_webots_geometry_inspection.py`) to separate pre-receipt vs post-receipt delay on any future authorized run; it does not relax the 5s limit or validity semantics.
- **Inert verification (this worktree):** focused tests `tests/test_worldmind_webots_diagnostic.py tests/test_worldmind_webots_geometry_inspection.py` → **88 passed** (receipt-age reporting, timestamp validation, geometry/state-failure separation cases included); focused motion-readiness tests `tests/test_worldmind_motion_readiness_dossier.py` → **4 passed** after swept-bound diagnostic-accounting reconciliation (no proven bounds injected; fail-closed semantics preserved); `ruff check`/`ruff format` passed on touched Python modules/tests.
- **Readiness remains fail-closed:** motion/collision safety is **NOT_READY / not approved / not allowed** (preserved dossier basis and current offline gate accounting). Combined runtime-readiness is **not PASS** due to v3 stale observation at consumption (and unresolved safety/collision/certification gaps). Production scoring path remains blocked (`configs/worldmind_production.yaml` is a structurally completed but blocked/fail-closed template; no approved Eq. 3.8 definitions/weights/ranking, calibration evidence/candidate/approval/bindings, or ready config/passing preflight). No scored VALIDATION is authorized.

## 2. Evidence and artifacts (hashes/paths)

| Artifact | Path | Notes / SHA-256 |
|---|---|---|
| Task A v2 diagnostic | `runs/worldmind/validation-s0-webots-table-live-v2/diagnostic.json` | `aebda07b…ea1db`; FAILED (stale observation), 5/5 tables matched, no actuation |
| Task A v2 geometry | `runs/worldmind/validation-s0-webots-table-live-v2/geometry.json` | `4784afa3…4528` |
| Task A v2 provenance | `runs/worldmind/validation-s0-webots-table-live-v2/provenance.json` | `3808473c…5f56` |
| Task A v2 state | `runs/worldmind/validation-s0-webots-table-live-v2/state.json` | `bd80b8b2…62d6` |
| Task A v2 log | `runs/worldmind/validation-s0-webots-table-live-v2/worldmind-webots-geometry-inspection.log` | `4e9c1a96…4ac6` |
| Task A v3 diagnostic | `runs/worldmind/validation-s0-webots-table-live-v3/diagnostic.json` | `aca9a869…e4d9a`; FAILED (stale observation at consumption, ~12.86s>5s), 5/5 tables matched, capture timestamps valid, geometry export valid, no actuation |
| Task A v3 geometry | `runs/worldmind/validation-s0-webots-table-live-v3/geometry.json` | `4784afa3…4528` |
| Task A v3 provenance | `runs/worldmind/validation-s0-webots-table-live-v3/provenance.json` | `db8a1701…886f99` |
| Task A v3 state | `runs/worldmind/validation-s0-webots-table-live-v3/state.json` | `a1150d96…37771` |
| v1 evidence | `runs/worldmind/validation-s0-webots-table-live-v1/` | preserved read-only, hashes re-verified unchanged |
| Motion readiness dossier | `runs/worldmind/inert-non-scored-2026-11-18/motion_readiness_dossier_current.json` | `a52a4807…6c53` (prior inert artifact), `approved=false`, `overall=FAIL/NOT_READY` |
| Validation execution manifest proposal | `runs/worldmind/inert-non-scored-2026-11-18/validation_execution_manifest.json` | SHA-256 `7d07f2de67f73fb4c3f4692b9a616c2f10393538af258bc124aa1f734a104f5f`, offline/not executed/not scored |
| Eq. 3.8 objective proposal | `configs/report/objectives.proposal.json` | engineering proposal, `review_status=engineering_proposal_not_approved`, `scored_execution_enabled=false` |
| Table 5.1 definitions candidate | `configs/report/definitions.candidate.json` | `review_status=candidate`, owner decision required |
| Production config template | `configs/worldmind_production.yaml` | blocked template (`status=blocked`, placeholders, frozen VALIDATION contract) |
| Arm sweep contract | `configs/worldmind_arm_sweep.yaml` | blocked/review-gated; no owner certificate |
| Frozen-evidence verifier | `tools/verify_week3_frozen_evidence.py` | last run PASS, no TEST split access |

## 3. Workstream A — runtime freshness diagnosis and offline instrumentation

- **v2 root cause (preserved):** `observation_age_s` was computed from client request latency (~13.14 s) rather than the observation capture instant; the observation payload (`t=0.0`) was classified stale despite successful same-RPC geometry export and 5/5 live-base table matches.
- **Implemented capture-boundary fix (inert):** server-side read-only capture records actual capture timestamps (`capture_wall_time_s`/`captured_at_utc` in `capture_read_only_state`), and adapters derive age from capture time rather than request-start latency while keeping geometry export validity and observation/state validity separately reported. Freshness threshold remains `5.0s`; no timestamps synthesized.
- **v3 observed outcome (preserved evidence):** capture timestamps are valid/consistent/non-future/non-invalid, but `observation_age_s` at consumption is ~12.86s (>5s), so validity fails closed. Geometry export and 5/5 runtime table reconciliation succeeded in the same inspection; combined readiness is not PASS.
- **v3 diagnosis (evidence + source):** capture is stamped at server-side state-capture start before full geometry export/RPC return; the client validates on receipt and, after `launch_webots_session` context exit, teardown waits for self-exit then terminates the simulator process before CLI finalization recomputes consumption age. The ~12.86s interval spans server export/RPC/transport plus post-receipt shutdown/teardown/finalization. Exact apportionment is not determinable from preserved evidence because no receipt/close/finalization/export-duration timestamps were recorded.
- **Single missing timing measurement (implemented offline):** `observation_age_s_at_receipt` (validated capture age at client receipt, before session teardown) is now recorded in diagnostic/geometry report and state alongside consumption `observation_age_s`. This separates pre-receipt vs post-receipt delay on future authorized runs without changing the 5s threshold or fail-closed consumption-time validity.
- **Tests:** adapter/capture-path coverage includes delayed-startup fresh capture, truly stale capture, geometry-success/state-failure separation, and receipt-age reporting in both diagnostic and geometry-inspection contexts.

## 4. Workstream B — production readiness owner-review packet

### 4.1 Eq. 3.8 objectives (proposal, NOT approved)

Source: `configs/report/objectives.proposal.json` (`review_status=engineering_proposal_not_approved`, `scored_execution_enabled=false`).

- Verbal form only from report: `J(τi) = wg Jg(τi) + wl Jl(τi) + wr Jr(τi) + wu Ju(τi)`; all formulas, units, normalizations, ranking direction, and weights are engineering proposals requiring owner approval.
- Safety precedence (must remain): **hard veto** via `SafetyMonitor.validate` + `RiskGate` + `unverified-arm-sweep`; vetoed candidates are never ranked and no utility term offsets a veto. `Jr` is not a substitute for hard checks; unverified geometry/missing inputs must fail closed (`MissingObjectiveInput`), never imputed.
- Proposed component definitions (engineering choices): `Jg` goal pose error (normalized m/rad), `Jl` path+time cost proxy, `Jr` clearance-margin risk proxy in `[0,1]` from certified swept-arm clearance, `Ju` prediction-discrepancy proxy (NOT calibrated uncertainty). Selection rule in proposal: `registered_evaluator_max_score_then_candidate_id` on VALIDATION only.
- **Owner decisions required:** approve/revise authoritative formulas, units/scales, normalization margins, ranking direction (`maximize`/`minimize`), weights `(wg,wl,wr,wu)`, any non-safety `Jr` threshold definition, and evaluator binding; record approved calibration candidate and definitions.

### 4.2 Table 5.1 definitions (candidate, NOT approved)

Source: `configs/report/definitions.candidate.json` (`review_status=candidate`, `scored_execution_enabled=false`). All units marked `[PROPOSED; OWNER DECISION]`.

| Metric | Candidate definition | Candidate unit | Evidence source (existing wiring) | Owner decision |
|---|---|---|---|---|
| Task success rate | Successful trials / total trials | ratio (0–1) | `eval/metrics.py` + Wilson interval | confirm unit/definition |
| Prediction error | RMSE/MAE or task-appropriate metric | SI per quantity (m, rad)/steps | `world_model/evaluation.py::prediction_errors` (truthful truncated pairing) | select metric/units |
| Collision rate | Trials with collisions / total trials | ratio (0–1) | recorded displacement/tilt **PROXY** + contact diagnostics (must remain labelled proxy) | confirm definition/unit and proxy labeling |
| Path length | Sum of distances along executed trajectory | m | `eval/derived_metrics.py::end_effector_path_length_m` | confirm |
| Execution time | Simulation or wall-clock duration, clearly identified | s | `eval/runner.py wall_s` and `eval/derived_metrics.py::simulation_duration_s` kept separate | confirm which/labeling |
| Planning latency | Time to generate and evaluate candidates | s | decision sidecar phase timings; full-system latency remains UNVERIFIED | confirm scope/labeling |
| Risk rejection rate | Rejected candidates / evaluated candidates | ratio (0–1) | `eval/decision_metrics.py::risk_rejection_rate` | confirm |
| Reliability | Successful runs and documented failure categories | counts/categories | `failure_states`, `evaluation_failure_category` in summaries | confirm taxonomy/unit |

### 4.3 VALIDATION calibration requirements (non-circular, fail-closed)

Relevant code: `src/vlarobot/eval/worldmind_calibration.py`, `src/vlarobot/eval/worldmind_config.py`, `src/vlarobot/eval/worldmind_preflight.py`.

- **Stages:** (1) uncalibrated diagnostic config (hard checks intact, no calibrated weights), (2) VALIDATION-only evidence collection (18 frozen episodes, hash pinned), (3) draft selection + **owner approval** of non-safety parameters, (4) evaluation with frozen approved parameters. Selection statistic is **not** an independent evaluation of post-calibration performance.
- **Evidence contract:** approved VALIDATION-only evidence artifact, 18 unique episode IDs, explicit source hashes, measurement class, no TEST/TRAIN paths; pinned by SHA-256 in candidate.
- **Candidate/approval contract:** approved candidate must declare `evidence_id/path/sha256`, `component_bindings`, `weights_source`, `parameters` (weights for `Jg,Jl,Jr,Ju` + explicit `maximize` boolean, nonnegative/not-all-zero), `selection`, `approval_record`, provenance; config must be `status=ready` with matching split/seeds/episodes/plan and registered bindings/evaluator.
- **Current state:** `configs/worldmind_production.yaml` is `status=blocked` with placeholders; no approved calibration evidence/candidate/definitions/bindings exist. Preflight and binding validation are fail-closed (`validate_worldmind_bindings`, `run_worldmind_preflight`/diagnostic preflight, `load_worldmind_config`).
- **Deterministic execution plan (offline proposal only):** `runs/worldmind/inert-non-scored-2026-11-18/validation_execution_manifest.json` (18 episodes, seeds `[0,1,2]`, unique scene-seed pairs, `K=32`, `H=50`, `dt=0.032`, `full_system`, `scripted_chunk_adapter`, oracle, `GATE_CALL_POLICY` 33 calls/decision). Not executed, not scored, authorization required.

### 4.4 Production bindings and preflight (blocked, wiring present)

- **Config contract (`worldmind_config.py`):** frozen VALIDATION-only constants (`split=validation`, `base_seed=0`, `seeds=[0,1,2]`, `episodes=18`, `K=32`, `H=50`, `dt=0.032`, oracle, `full_system`, `scripted_chunk_adapter`), explicit component/assessment/calibration/surrogate/safety/verification/adapter/report specs, authoritative report SHA-256 pinned.
- **Bindings (`worldmind_bindings.py`, `backends/worldmind_bindings.py`):** explicit registry keys only (no dynamic import from config); empty shipped registry fails closed. Registered identities required for components, hard checks, surrogates, safety/verification factories, pose solver, waypoint sets/factories, calibration evaluator, and input bindings as declared.
- **Preflight (`worldmind_preflight.py`):** gates split/state/output freshness/config approval, backend declaration, report definitions/candidate review status, bindings, calibration evaluation via registered evaluator, frozen plan integrity, runtime geometry probe completeness, and calibration evidence provenance. No launch/IK/inference/actuation/scoring.
- **Offline wiring status (no owner decisions required):** loader/preflight/binding/calibration/evidence validation and diagnostic-vs-scored separation are implemented and fail-closed; blocked template and proposal/candidate artifacts are in place. **No approved scientific values were added.**

## 5. Workstream C — safety closure (offline analysis, fail-closed)

Source: `runs/worldmind/inert-non-scored-2026-11-18/motion_readiness_dossier_current.json`, `src/vlarobot/eval/worldmind_motion_readiness_dossier.py`, `src/vlarobot/world_model/arm_sweep.py`, `configs/worldmind_arm_sweep.yaml`, `src/vlarobot/world_model/self_collision.py`, `src/vlarobot/world_model/carried_object.py`.

| Gate / unresolved bound | Status | Source | Exact evidence required (to close) |
|---|---|---|---|
| Swept residual: joint displacement/reach bound | UNVERIFIED | dossier `swept_residual.blockers`, arm sweep contract/evaluation (`joint_displacement_bound_rad`, `joint_reach_bound_m`, `max(point_speed*dt/substeps, joint_displacement*reach)` term) | Pinned finite independent maximum joint-displacement bound (rad) and corresponding reach bound (m) consistent with reviewed kinematics/contract; owner-reviewed record binding values to contract SHA/hash |
| Swept residual: geometry approximation error bound | UNVERIFIED | dossier `geometry_approximation_error_bound_m=null` | Pinned finite global geometry-approximation error bound (m) covering capsule/scene approximation used by sweep check, with justification/evidence artifact |
| Swept residual: numerical error bound | UNVERIFIED | dossier `numerical_error_bound_m=null` | Pinned finite numerical-error bound (m) for distance/sweep computations under declared precision |
| Swept residual: transform error bound | UNVERIFIED | dossier `transform_error_bound_m=null` | Pinned finite transform/registration error bound (m) consistent with reviewed registration/URDF/runtime calibration evidence |
| Swept residual inequality | UNVERIFIED (margin required unknown; point allowance 0.019821…, registration allowance 0.004953… known) | `SWEEP_INEQUALITY` in dossier/arm_sweep | All above bounds + reviewed margin/registration inputs such that `residual_m > 0` is certified for the approved contract; reviewed sweep contract/certificate artifacts (`configs/worldmind_arm_sweep.yaml` currently blocked, no certificate) |
| Self-collision / full-body collision coverage | UNVERIFIED | dossier `self_collision`: `contract_coverage_validated=false`, `coverage_proven=false`, 42 uncovered collision bodies (base/casters/wheels/torso/head/Astra/Hokuyo/gripper finger links/body primitives listed) | Full URDF collision-body pair coverage proof (or owner-approved exclusions with justification) beyond pinned capsule-pair checks; explicit reviewed adjacent-pair/grasp-contact decisions (no auto-exemptions); updated coverage evidence binding all required bodies/pairs |
| CAN lifecycle (attachment/release/carried pose) | UNVERIFIED (fail-closed) | dossier `can_lifecycle`: `state=ambiguous`, `carried_geometry_available=false`, `contact_claim_allowed=false` | Observed CAN attachment and release state evidence, proven carried-CAN pose propagation across transition states, and explicit contact-claim policy/evidence; no contact claims without proof |
| Table transform runtime grounding (safety dossier gate) | UNVERIFIED in dossier; runtime-readiness gap remains open | dossier `table_transform`: source-derived, no runtime collision transform (`status=source_derived_unverified`) | **Separate from Task A v2/v3 inspection outcomes:** v2 and v3 provided read-only runtime table-target observation/reconciliation evidence for the five boxes (matched, geometry export valid) but each diagnostic outcome FAILED due to stale observation at consumption; a **fresh, authorized** read-only inspection confirming observation validity/freshness and accepted runtime grounding (or equivalent accepted evidence) is required to close this gate in safety/runtime evidence |
| Motion authorization / arm sweep certification | FAIL (NOT_READY) | dossier blockers `unverified-arm-sweep`, `unresolved_forwarded_geometry`, `motion_authorization` not granted | Independent safety-owner certificate and approval record bound to reviewed contract/cases/hashes; no self-approval by code; all unresolved bounds above closed for declared scope |

**CAN lifecycle and motion-authorization gates remain fail-closed.** No broad certification framework was created.

## 6. Updated critical path to controlled VALIDATION (minimal, conditional)

1. **Close runtime freshness/grounding verification (separately authorized):** run one fresh read-only geometry inspection (single attempt, no auto-retry) and confirm `observation_valid=true`, `observation_age_s <= observation_freshness_max_s` (with receipt-vs-consumption timing captured), 5/5 live-base reconciliation, geometry export valid, and no actuation; archive as accepted runtime-grounding evidence if gates pass.
2. **Owner decisions (blocking):** approve Table 5.1 definitions/units (`configs/report/definitions.candidate.json` prepared for immediate resolution), Eq. 3.8 formulas/units/normalization/weights/ranking direction, calibration evaluator and workflow artifacts, production bindings/definitions files, and safety/collision scope (including swept bounds, coverage exclusions/approvals, CAN contact policy, motion authorization/certificate).
3. **Safety/collision evidence closure (scoped):** provide exact evidence for each unresolved bound in §5 (including global geometry-approximation bound, full URDF collision-body coverage/uncovered bodies/exclusions, CAN lifecycle/contact evidence, and independent safety-owner certification) or approved scope limitations; keep fail-closed until complete.
4. **Offline production readiness:** replace placeholders in a new approved ready config (not by weakening template), register/approve reviewed bindings and definitions/calibration evidence/candidate, and pass `worldmind-preflight` with fresh outputs under the frozen VALIDATION contract.
5. **Controlled VALIDATION (explicitly authorized, single attempts, no auto-retry):** execute only the approved scope/arm/config with frozen plan/manifest constraints, truthful timing/decision evidence, and no TEST/TRAIN access. Any motion/actuation/scoring step requires separate explicit authorization per gate.

**Earliest feasible controlled VALIDATION milestone (conditional):** after (1) a fresh authorized read-only grounding PASS, (2) required owner approvals (Eq. 3.8/Table 5.1, calibration/definitions/bindings), (3) safety certification and required evidence closure or approved scope, and (4) a passing preflight on an approved ready config under the frozen VALIDATION contract. No calendar date is committed here; this is the earliest gate-consistent point.

**Note:** original five-arm comparison is not achievable by 2026-12-12 on current evidence/timeline without gate waivers (not permitted). Scripted-primary WorldMind minimal controlled VALIDATION remains the only near-term candidate path, conditional on the above.

## 7. Owner decisions checklist (explicit)

- [ ] Approve/revise Eq. 3.8 component formulas, units, normalization scales/margins, ranking direction, and weights `(wg, wl, wr, wu)`; approve evaluator binding and selection rule.
- [ ] Approve Table 5.1 metric definitions/units and proxy-labeling policy (especially collision rate, prediction error, execution/planning latency).
- [ ] Approve VALIDATION calibration evidence artifact (stage 2), draft/proposal review, and final approved calibration candidate + approval record (stage 3/4).
- [ ] Approve `configs/worldmind_production.yaml`-equivalent ready config values, `report/definitions` artifact, and all registered production bindings (components, hard checks, surrogate/safety/verification, pose solver, waypoints, calibration evaluator, inputs).
- [ ] Approve swept-residual bounds (joint displacement/reach, geometry approx, numerical, transform/registration) and margin/certificate for declared scope.
- [ ] Approve full-body/self-collision coverage claims and any adjacent/grasp-contact exclusions (or require missing evidence).
- [ ] Approve CAN attachment/release/carried-pose/contact evidence and policy (or keep unapproved).
- [ ] Accept fresh runtime table-grounding/observation evidence for safety gate closure.
- [ ] Grant explicit motion authorization/safety-owner certificate for any physical motion scope (separately authorized).

## 8. Exact read-only verification command (separate authorization required)

**Single attempt only, no auto-retry. Do not run without explicit authorization.** `runs/worldmind/validation-s0-webots-table-live-v3` is preserved completed evidence (FAILED on consumption freshness); a future authorized retry must use a new output path.

```bash
cd /home/tj330/programs/VLA-Project-week3 && WEBOTS_HOME=/home/tj330/.local/opt/webots uv run python -m vlarobot.cli worldmind-webots-geometry-inspection --config configs/worldmind_webots_geometry_inspection.yaml --out runs/worldmind/<new-authorized-readonly-out>
```

**Acceptance criteria for this read-only check:** fresh observation (`observation_valid=true`, `observation_age_s <= observation_freshness_max_s` at consumption, with `observation_age_s_at_receipt` reported for diagnosis), 5/5 live-base table reconciliation matched, geometry export valid, `backend_apply_calls=0`, `backend_step_calls=0`, `backend_reset_calls=0`, `no_actuation=true`, tolerances `0.0001` m/rad as configured, no scored/frozen evidence modification.
