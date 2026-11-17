# REPORT_CONFORMANCE.md — submitted Phase I report → implementation (DRAFT)

Controlling sources: the confirmed submitted Phase I report (SHA-256 `23d5ff15…1522b`)
and ADR-0029 (`docs/DECISIONS.md`, 2026-11-14) plus subsequent owner decisions.
Labels: **MEASURED** (recorded evidence exists), **INFERRED**, **UNVERIFIED**,
**PENDING-AUTHORIZATION** (requires owner approval per ADR-0029 protocol).

## 1. Module conformance (report §3.4)

| Report module | Implementation | Status |
|---|---|---|
| 3.4.1 Simulation environment | `worlds/factory.wbt`, `src/vlarobot/backends/webots/` + `backends/fake/` (deterministic); reset reproducibility `eval/reset_repro.py` | INFERRED (M1 validation report; scripted TRAIN 23/24 MEASURED) |
| 3.4.2 Perception & state representation | Oracle state with provenance flags (report 4.3 allowance; ADR-0029); `eval/recorder.py` records `state_source` per episode | INFERRED |
| 3.4.3 Trajectory planning (candidates) | `planning/candidates.py`, `risk/comparison.py` (seeded K draws, five-arm switch) | INFERRED (unit-tested; not physically scored beyond arms 1–2) |
| 3.4.4 World model & prediction | `world_model/` (AnalyticSurrogate, VelocityLimitedKinematics, conditional attachment, fail-closed hard reasons) | MEASURED replay evidence; scored-arm prediction pending |
| 3.4.5 Risk assessment & selection | `risk/gate.py` (hard veto → Eq 3.8 ranking), `planning/verification.py` (verify-retry once), `risk/gate.py::RecoveryState` (reobserve → alternate family → abort) | INFERRED (unit-tested; **weights not yet chosen on VALIDATION**) |
| 3.4.6 Robot execution | `backends/webots/backend.py`, `planning/execution.py` (SafetyMonitor before every `apply()`), scripted baseline `execution/scripted_can.py` | MEASURED (scripted TRAIN 23/24; VALIDATION 15/18) |
| 3.4.7 Logging & evaluation | `eval/recorder.py` (steps/actions/predicted/contacts/frames), `eval/runner.py`, `eval/metrics.py`, `eval/comparison.py` | INFERRED (recorder contract tested; campaign evidence exists) |

## 2. Table 5.1 metrics (verbatim rows → current implementation/evidence)

| Report metric | Measurement (report wording) | Implementation | Evidence / status |
|---|---|---|---|
| Task success rate | Successful trials / total trials | `eval/metrics.py` success aggregation + Wilson interval | MEASURED: scripted TRAIN 23/24; scripted VALIDATION 15/18; VLA-only TRAIN 0/24 (gate FAILED) |
| Prediction error | RMSE, MAE or task-appropriate metric | `world_model/evaluation.py::prediction_errors` (quantity-separated, no probabilities) | MEASURED read-only replay (TCP RMSE 4.48–6.69 mm overall across families @1/50 ticks; clear-family subset 4.62/6.69 mm; carried-CAN 14.25–20.84 mm @50 ticks); scored-arm values PENDING week 3/4 |
| Collision rate | Trials with collisions / total trials | Displacement (>5 mm)/tilt (>5°) **PROXY** + separate FK clearance (ADR-0029), `eval/runner.py` summary `collision_proxy` | MEASURED on completed campaigns (proxy, labelled as such) |
| Path length | Sum of distances along the executed trajectory | **NOT implemented as an explicit metric**; derivable from recorded `episode-XX.steps.npz` EE/base streams | MISSING for week 4 — must be computed from existing recordings, no new runs needed |
| Execution time | Recorded simulation or wall-clock duration, clearly identified | `eval/runner.py` records `wall_s` per episode + `total_wall_s` (currently kept out of the metric report) | PARTIAL — surface into report metrics for week 4 |
| Planning latency | Time to generate and evaluate candidates | Per-episode `episode-XX.latency.jsonl` in the frozen VLA campaign; analytic probes (`docs/WORLDMIND_DEVELOPMENT_CHECKPOINT.md`: 32×50 Jacobian path p50 3.895 s — **<1 s/32-candidate milestone NOT met**) | PARTIAL — per-decision scored latency pending week 3 |
| Risk rejection rate | Rejected candidates / evaluated candidates | `episode-XX.actions.jsonl` accepted/reason; summary `rejections_total`; adapter vetoes in frozen VLA campaign (21) | MEASURED for arms 1–2; arms 3–5 PENDING |
| Reliability | Successful runs and documented failure categories | `failure_states` + `raw_failure_states` + `evaluation_failure_category` in summaries; failure-category derivation | MEASURED for arms 1–2 (SAFETY 21 / PLACEMENT 3 in frozen VLA campaign) |

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
| Arm 2 VLA-only (TRAIN) + gate | MEASURED 0/24 — **gate FAILED**; scripted-primary retained; never re-run |
| Scripted VALIDATION campaign | MEASURED 15/18 (isolated worktree; supports geometry/error validation, not VLA rescue) |
| Week 3: sampled chunks + surrogate + RiskGate + retry; weights/thresholds on VALIDATION only; full pipeline on validation | **NOT DONE** — mechanics implemented and unit-tested (fake-backend 5×18 integration is explicitly simulated, not gate evidence); physical execution requires authorization |
| Week 4 (days 1–3): frozen TEST campaign across all arms; Wilson; scene-clustered McNemar; Table 5.1 exactly | **PENDING-AUTHORIZATION + SCHEDULED 2026-12-05..12-07** (test split fail-closed until then) |
| Week 4 (days 4–7): MATH.md, REPORT_CONFORMANCE.md, REPORT_FIGURES.md, demo | DRAFTS started (these documents); demo pending |
| Arm-3–5 proposer after failed gate | Per ADR-0029: scripted proposer is primary for the main comparison; VLA stays its own arm as a MEASURED partial result. Exact proposer semantics for arms 3–5 in the main comparison flagged for owner confirmation |

## 5. Non-conformance guard

No result in this tree may be presented as measured unless it traces to a recorded
campaign artifact under `runs/` (hash-pinned) or its committed mirror in `docs/evidence/`.
Fake-backend integration runs and CPU fixtures are explicitly **not** gate evidence.
The submitted report remains a hash-matched candidate until the owner's confirmation is
recorded in `docs/DECISIONS.md`.
