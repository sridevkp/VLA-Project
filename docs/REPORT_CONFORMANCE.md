# REPORT_CONFORMANCE.md — submitted Phase I report → implementation (DRAFT)

Controlling sources: the confirmed submitted Phase I report (SHA-256 `23d5ff15…1522b`)
and ADR-0029 (`docs/DECISIONS.md`, 2026-11-14) plus subsequent owner decisions.

Primary status classes (use one per row, with a scope qualifier where needed):

1. **Implemented and verified** — code plus applicable hash-pinned recorded evidence or
   inert verification; physical evidence is identified explicitly with `MEASURED`.
2. **Implemented but not physically validated** — code and inert tests exist, but no
   authorized physical/campaign evidence supports the physical claim.
3. **Missing** — the required metric or artifact is absent.
4. **Blocked** — authoritative input, instrumentation, or explicit authorization is absent.
5. **Explicitly waived by approved decision** — an approved waiver applies; it is never
   merged into `MEASURED`.

Qualifiers: **MEASURED** (recorded evidence exists), **INFERRED** (structural correspondence
only), **UNVERIFIED**, **PENDING-AUTHORIZATION**, **fake-only**, **nominal-kernel-only**,
**proxy-labelled**, and **replay-only**. A status class and a qualifier are not interchangeable.

## 1. Module conformance (report §3.4)

| Report module | Implementation | Status |
|---|---|---|
| 3.4.1 Simulation environment | `worlds/factory.wbt`, `src/vlarobot/backends/webots/` + `backends/fake/` (deterministic); reset reproducibility `eval/reset_repro.py` | Implemented and verified — `MEASURED` scripted TRAIN 23/24 and VALIDATION 15/18; fake backend is `fake-only` |
| 3.4.2 Perception & state representation | Oracle state with provenance flags (report 4.3 allowance; ADR-0029); `eval/recorder.py` records `state_source` per episode | Implemented and verified — recorded provenance in completed campaigns; oracle use is the documented deviation |
| 3.4.3 Trajectory planning (candidates) | `planning/candidates.py`, `risk/comparison.py` (seeded K draws, five-arm switch) | Implemented but not physically validated — inert/unit evidence only for sampled K; arms 3–5 scored runs are `PENDING-AUTHORIZATION` |
| 3.4.4 World model & prediction | `world_model/` (AnalyticSurrogate, VelocityLimitedKinematics, conditional attachment, fail-closed hard reasons) | Implemented but not physically validated — `MEASURED` read-only replay, `replay-only`; scored-arm prediction `Missing` until instrumentation and runs are authorized |
| 3.4.5 Risk assessment & selection | `risk/gate.py` (hard veto → Eq 3.8 ranking), `planning/verification.py` (verify-retry once), `risk/gate.py::RecoveryState` (reobserve → alternate family → abort) | Implemented but not physically validated — inert/unit-tested; weights not calibrated on VALIDATION |
| 3.4.6 Robot execution | `backends/webots/backend.py`, `planning/execution.py` (SafetyMonitor before every `apply()`), scripted baseline `execution/scripted_can.py` | Implemented and verified — `MEASURED` scripted TRAIN 23/24; VALIDATION 15/18 |
| 3.4.7 Logging & evaluation | `eval/recorder.py` (steps/actions/predicted/contacts/frames), `eval/runner.py`, `eval/metrics.py`, `eval/comparison.py` | Implemented and verified — `MEASURED` campaign artifacts; new supplemental derivations are separate and do not alter the frozen schema |

## 2. Table 5.1 metrics (verbatim rows → current implementation/evidence)

| Report metric | Measurement (report wording) | Implementation | Evidence / status |
|---|---|---|---|
| Task success rate | Successful trials / total trials | `eval/metrics.py` success aggregation + Wilson interval | Implemented and verified — `MEASURED`: scripted TRAIN 23/24; scripted VALIDATION 15/18; VLA-only TRAIN 0/24 (`FAILED` gate) |
| Prediction error | RMSE, MAE or task-appropriate metric | `world_model/evaluation.py::prediction_errors` (quantity-separated, no probabilities) | Implemented but not physically validated — `MEASURED replay-only` (TCP RMSE 4.48–6.69 mm overall across families @1/50 ticks; clear-family subset 4.62/6.69 mm; carried-CAN 14.25–20.84 mm @50 ticks); scored-arm values are `Missing` and `Blocked` on new instrumentation/runs |
| Collision rate | Trials with collisions / total trials | Displacement (>5 mm)/tilt (>5°) **PROXY** + separate FK clearance (ADR-0029), `eval/runner.py` summary `collision_proxy` | Implemented and verified — `MEASURED`, always `proxy-labelled`; this is not contact-only collision detection |
| Path length | Sum of distances along the executed trajectory | `eval/derived_metrics.py::end_effector_path_length_m`: sum of consecutive Euclidean distances of base-frame `ee_xyz` in metres | Implemented but not physically validated — inert tests only. No frozen episode, summary, CSV, or provenance file was modified; no historical value is invented here |
| Execution time | Recorded simulation or wall-clock duration, clearly identified | `eval/runner.py` records `wall_s` per episode + `total_wall_s`; `eval/derived_metrics.py::simulation_duration_s` derives `t[-1] - t[0]` from recorded simulation time | Implemented but not physically validated for the derived metric — `wall_s` is `MEASURED` run bookkeeping, not a Table 5.1 substitution; the derivation is inert-tested and keeps simulation duration separate from wall clock |
| Planning latency | Time to generate and evaluate candidates | Hash-pinned `docs/evidence/planning-optimization-final.json`; no new profiler run | Implemented but not physically validated — `nominal-kernel-only`: K=32, H=50, optimized p50 `0.4677 s`, p95 `0.4803 s`, pre-optimization p50 `2.0996 s`. Full-system and scored per-decision latency are `UNVERIFIED` |
| Risk rejection rate | Rejected candidates / evaluated candidates | Operationalized as `hard_vetoed / (hard_vetoed + safe_unselected + selected)` by `eval/decision_metrics.py::risk_rejection_rate`; invalid/unevaluable excluded and reported separately; zero denominator → `None`. Existing `episode-XX.actions.jsonl` accepted/reason, summary `rejections_total` and VLA adapter vetoes (21) are action/adapter tallies, **not** this rate | `Missing` for the true Table 5.1 rate — the definition and pure aggregation are `INERT-UNIT-TESTED`, but the candidate-level sidecar counts and an authorized run are `Blocked` on instrumentation approval and authorization |
| Reliability | Successful runs and documented failure categories | `failure_states` + `raw_failure_states` + `evaluation_failure_category` in summaries; failure-category derivation | Implemented and verified — `MEASURED` for arms 1–2 (SAFETY 21 / PLACEMENT 3 in frozen VLA campaign) |

### Planning-latency evidence scope

The figures above are from the identical-input nominal analytic kernel in
`docs/evidence/planning-optimization-final.json`: K=32, H=50 (1,600 control steps),
optimized p50 `0.4677 s`, optimized p95 `0.4803 s`, cold batch `0.4746 s`, and
pre-optimization p50 `2.0996 s`. The evidence records zero state/reference difference,
identical vetoes, unchanged safety/trust thresholds, no model inference, and no TEST access.
The uncertified full-arm sweep remains vetoed. This is scope-limited profiling evidence;
it does **not** establish the full-system below-1-second milestone or any scored
per-decision latency result. See `docs/WEEK3_RISKGATE_VALIDATION_PLAN.md` for the
phase-separated measurement protocol.

### Additive decision-observability evidence (inert)

`src/vlarobot/eval/decision_evidence.py`, `decision_metrics.py` and `decision_adapters.py` add
typed sidecar schemas, pure aggregation and optional delegate-only probes for risk-gate
decisions. They are **INERT-UNIT-TESTED** (18 reviewed nodes), write no frozen artifact, and are
not wired into any production path. Providing a real candidate-level risk rejection rate or a
scored-arm prediction error still requires instrumentation approval plus an authorized run, so
those two Table 5.1 cells remain `Missing` / `Blocked`. The Eq. 3.8 component and eight
Table 5.1 definitions/units are a **proposed mapping for owner review** in
`docs/WEEK3_DECISION_OBSERVABILITY.md` §6; no production `definitions.json` is created and no
component meaning is guessed. Status inventory and the exact preflight are in that document
§§7–9 and in `tools/verify_week3_frozen_evidence.py`.

### Supplemental trajectory definitions

`eval/derived_metrics.py` computes:

- **End-effector path length (m)** = Σ‖`ee_xyz[i] − ee_xyz[i−1]`‖₂ over consecutive
  samples. `ee_xyz` is written from `RobotState.ee_pose.position`; the NPZ has no frame or
  unit metadata, so callers must explicitly confirm `frame="base"` and `units="m"`.
  Empty input is undefined (rejected); one sample yields `0.0`; non-finite coordinates are
  rejected.
- **Simulation duration (s)** = `t[-1] − t[0]` with finite, strictly increasing timestamps.
  Empty input is undefined (rejected); one sample yields `0.0`. This is never substituted
  for wall-clock execution time.

These are supplemental read-only derivations; they add no public `summary.json` field and
do not change the runner or recorder contract.

## 3. Documented deviations (ADR-0029 clarifications 7)

1. **Eq 3.2** — propose-and-evaluate instead of policy-internal prediction conditioning
   (SmolVLA cannot consume a predicted state). See `docs/MATH.md`.
2. **Analytic surrogate** instead of the earlier MuJoCo-based plan (M5): no MuJoCo.
3. **Collision metric** is the displacement/tilt proxy plus FK clearance, always labelled
   a proxy — never claimed to be collision detection.

## 4. Experimental obligations

| ADR-0029 item | Status |
|---|---|
| Frozen splits 8/6/10, 3 families, 3 seeds/scene | MEASURED-configured (`configs/eval.yaml`, `eval/settings.py` fail-closed guards; `eval/scenarios.py` deterministic generation) |
| Arm 1 scripted baseline (TRAIN) | MEASURED 23/24 (repaired, archived; 12/12 provenance hashes verified) |
| Arm 2 VLA-only (TRAIN) + gate | MEASURED 0/24 — **gate FAILED**; scripted proposer becomes primary for main comparison; VLA-only remains a separate MEASURED partial arm; never re-run |
| Scripted VALIDATION campaign | MEASURED 15/18 (analytic worktree; supports geometry/error validation, not VLA rescue) |
| Week 3: sampled chunks + surrogate + RiskGate + retry; weights/thresholds on VALIDATION only; full pipeline on validation | **Not executed** — mechanics implemented and inert-tested; `docs/WEEK3_RISKGATE_VALIDATION_PLAN.md` is ready; physical execution is `Blocked` on explicit authorization. Fake-backend 5×18 integration is `fake-only`, not gate evidence |
| Week 4 (days 1–3): frozen TEST campaign across all arms; Wilson; scene-clustered McNemar; Table 5.1 exactly | `PENDING-AUTHORIZATION + SCHEDULED 2026-12-05..12-07` (test split fail-closed until then) |
| Week 4 (days 4–7): MATH.md, REPORT_CONFORMANCE.md, REPORT_FIGURES.md, demo | DRAFTS updated; demo pending |
| Arm-3–5 proposer after failed gate | Scripted proposer is primary by ADR-0029 gate outcome; VLA stays its own arm as a MEASURED partial result. Any change from this allocation requires a new owner decision |

## 5. Non-conformance guard

No result in this tree may be presented as measured unless it traces to a recorded
campaign artifact under `runs/` (hash-pinned) or its committed mirror in `docs/evidence/`.
Fake-backend integration runs, replay conditioning, inert tests and CPU fixtures are
explicitly **not** gate evidence. Nominal-kernel profiling is not full-system latency.
The submitted report remains a hash-matched candidate until the owner's confirmation is
recorded in `docs/DECISIONS.md`.

### Frozen VALIDATION coverage check

`eval/worldmind_validation_coverage.py::load_canonical_baseline_coverage` checks
the reviewed canonical summary's SHA-256 and reconciles its records to the
exact frozen 18 trial keys. It preserves raw failure-state labels and reports
missing keys separately. Its structural reconciliation helper grants no
scientific eligibility to arbitrary input. Diagnostic and fixture records
cannot substitute for the pinned baseline or for another arm's outcomes.
