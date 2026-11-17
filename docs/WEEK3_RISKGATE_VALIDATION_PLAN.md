# WEEK3_RISKGATE_VALIDATION_PLAN.md — bounded validation design (2026-11-17)

Status: **design only — no Week 3 evaluation has been launched**. This document specifies
the protocol that would be used only after explicit owner authorization. It does not run
Webots, real IK, model inference, training, actuation, scored evaluation, or any TEST-split
activity.

Controlling sources:

- submitted Phase I report, SHA-256
  `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b` (Eq. 3.8 and
  Table 5.1);
- ADR-0029 in `docs/DECISIONS.md` (four-week campaign, frozen sizes, five arms, gate,
  Week 3 and Week 4 boundaries);
- ADR-0031 recorder contract and subsequent append-only decisions;
- `docs/REPORT_CONFORMANCE.md` for current status classification;
- `docs/WEEK3_DECISION_OBSERVABILITY.md` for the additive typed evidence schema, the risk
  rejection-rate definition, the prediction-pairing and latency contracts, the proposed
  Table 5.1 mapping, and the exact read-only validation preflight.

ADR-0029 deadline remains **2026-12-12**. Week 4 TEST access is not permitted before the
authorized window **2026-12-05..12-07**. A-D2 remains CLOSED / NO-GO and is not part of
this plan.

## 1. Evidence classes used below

These classes are disjoint and must not be merged in reports:

| Class | Meaning |
|---|---|
| **MEASURED** | Hash-pinned recorded campaign or replay artifact exists. State whether it is physical, replay-only, or nominal-kernel-only. |
| **INERT-UNIT-TESTED** | Pure-Python/numeric unit evidence only; no Webots, real IK, inference, actuation, or scored run. |
| **NOMINAL-KERNEL-PROFILED** | Identical-input analytic-kernel timing evidence only (`docs/evidence/planning-optimization-final.json`); not full-system latency. |
| **UNVERIFIED PHYSICAL** | Physical behavior has no recorded evidence for the claim. |
| **BLOCKED / PENDING-AUTHORIZATION** | Requires authoritative input, instrumentation, a run window, or an owner decision. |
| **EXPLICITLY WAIVED** | Only an approved waiver such as M1-06a; never counted as MEASURED. |

## 2. Frozen baseline and proposer allocation

Frozen facts are not reopened:

- scripted TRAIN: **23/24** — MEASURED;
- scripted VALIDATION: **15/18** — MEASURED;
- VLA-only TRAIN: **0/24 — FAILED** — MEASURED gate result;
- scripted proposer is primary for the main comparison after the failed gate;
- VLA-only remains a separate **MEASURED partial arm**, not a gate rescue and not an input
  to re-tuning;
- K=32 candidates per decision and H=50 prediction steps remain fixed;
- the control tick is 32 ms (`configs/robot.yaml::time_step`);
- existing safety constraints and thresholds are unchanged.

The frozen five arm labels from ADR-0029 remain:
`scripted_baseline`, `vla_only`, `vla_risk_veto`, `vla_world_model`, `full_system`.
For the main-comparison arms (`vla_risk_veto`, `vla_world_model`, `full_system`), the
proposer source is **scripted** under the post-gate decision. The historical `vla` in those
arm labels denotes the ablation slot, not permission to re-promote the failed VLA policy.
The `vla_only` arm retains the VLA proposer and its frozen 0/24 result as a separate arm.

## 3. Candidate sampling

If and when a run is authorized, each decision must use `planning/candidates.py::
SeededCandidateGenerator`:

- exactly **K=32** action chunks per decision;
- each chunk has exactly **H=50** control actions;
- decision seed = `SeedSequence([generator_seed, decision_index]).generate_state(1)[0]`;
- draw seeds = `np.random.default_rng(decision_seed).integers(0, 2**31, size=32)`;
- every draw seed, decision index, generator seed, and proposer source is logged;
- `chunk.source` is checked against the declared proposer source; a mismatch fails closed;
- no unlogged perturbation, re-seeding after a bad result, or silent candidate substitution
  is permitted.

Candidate generation is INERT-UNIT-TESTED; no candidate-sampling run has been performed for
Week 3.

## 4. Surrogate prediction

`world_model/rollout.py::AnalyticSurrogate.predict_trajectory` predicts each candidate over
H=50 steps. The surrogate is the documented analytic geometric-kinematic model (no
MuJoCo). It emits fail-closed hard reasons for incomplete or invalid predictions and does
not manufacture grasp, contact, or success probabilities.

Arm behavior is fixed by `risk/comparison.py`:

- `vla_risk_veto`: no surrogate prediction; heuristic/no-prediction veto ablation;
- `vla_world_model`: surrogate prediction and J ranking, but no predictive veto ablation;
- `full_system`: surrogate prediction, predictive hard checks (`prediction_complete`,
  `prediction_constraints`), hard-veto precedence, then J ranking;
- motor safety via `execution/safety_monitor.py` remains authoritative in every arm;
  “no predictive veto” never means no motor safety.

The analytic surrogate and its replay error evidence are MEASURED **replay-only**;
scored-arm prediction error is MISSING / BLOCKED until recorded predicted-vs-observed
trajectories exist under an authorized run.

## 5. Hard safety constraints and fail-closed behavior

Before any action reaches a backend:

1. every required hard-check name must be present exactly once;
2. missing, duplicated, unknown (`None`), failed, or non-finite hard checks are rejected;
3. non-finite J components or a non-finite score are rejected;
4. `RiskGate.select` ranks only candidates that pass all hard checks;
5. an empty safe set produces a recorded reason and enters the bounded recovery ladder;
6. `SafetyMonitor.validate()` still vetoes immediately before every `backend.apply()`.

Frozen safety limits in `configs/safety.yaml`, verification tolerances, and trust-region
limits are not calibration parameters and are not changed here. The uncertified full-arm
sweep remains vetoed; unit geometry examples do not establish full-robot collision
acceptance.

## 6. RiskGate objective J and calibration parameters

The only permitted objective is the submitted report Eq. 3.8:

`J(τ_i) = wg·Jg(τ_i) + wl·Jl(τ_i) + wr·Jr(τ_i) + wu·Ju(τ_i)`

Component meanings remain exactly those in the report: `Jg` goal error, `Jl` trajectory
length or execution cost, `Jr` forecast risk, `Ju` uncertainty or prediction-related
penalty; `wg, wl, wr, wu` are normally configuration parameters. Hard constraints have
precedence; a low combined cost cannot explain a mandatory-restriction violation.

`eval/report_contract.py` must receive a reviewed production `definitions.json` whose
`report_sha256` matches the submitted PDF and whose four component definitions and eight
Table 5.1 names/definitions/units are explicit. Until then, scoring remains BLOCKED and
no component meaning is guessed in production code.

Calibration parameters subject to VALIDATION-only selection:

- the four weights `(wg, wl, wr, wu)`;
- the ranking direction (`maximize` or `minimize`), declared explicitly;
- any **non-safety** Jr decision threshold that is first given an exact authoritative
  definition in the reviewed bindings.

Not calibration parameters: frozen safety thresholds, verification tolerances, K, H,
seeds, splits, gate criteria, or arm definitions. No weight, threshold, or direction is
selected by this plan.

## 7. Validation-only calibration protocol

1. Use only the frozen VALIDATION split (6 scenes, 3 seeds/scene, 18 episodes).
2. Do not access TEST before 2026-12-05; do not rerun or alter TRAIN campaigns.
3. Before evaluating candidate weights, record the calibration grid/rule, objective,
   component bindings, source hashes, and exact VALIDATION episode list.
4. Because VALIDATION is the only permitted calibration source, any internal calibration/
   post-calibration partition must be declared before use. If no such partition is
   approved, report calibration and post-calibration results as the same VALIDATION
   evidence set and do not claim an independent holdout.
5. Select parameters only from recorded VALIDATION evidence; record the selected values,
   command, configuration hash, rationale, and all rejected candidates.
6. A post hoc parameter change requires a new append-only decision and a fresh reported
   analysis; it is never applied silently to historical outputs.
7. The scripted-primary allocation is not revisited using VALIDATION outcomes.

Calibration is currently BLOCKED / PENDING-AUTHORIZATION; no physical VALIDATION run is
part of this bounded phase.

## 8. Retry, recovery, and abort semantics

`risk/comparison.py::ComparisonPlanner.with_recovery` defines the bounded ladder:

1. run the primary K=32 scripted-proposer set;
2. if the safe set is empty, reobserve and rerun the primary set;
3. if still empty, request an alternate candidate family (for example, base standoff);
4. if still empty, abort the decision with a recorded reason;
5. verification via `planning/verification.py` may retry **at most once**; otherwise abort.

Every stage records the recovery event, candidate count, veto reasons, observation hash,
chosen index (or `null` with reason), and final verification outcome. A verification
success must be based on observed state, never on a predicted placement or HOME alone.
Failure to reobserve, a missing alternate family, or missing feedback fails closed.

## 9. Measured, inert, nominal, and unverified status inventory

| Item | Status |
|---|---|
| Scripted TRAIN and VALIDATION campaign outcomes | MEASURED; frozen; no rerun |
| VLA-only TRAIN 0/24 gate failure | MEASURED partial arm; FAILED; no rescue |
| Candidate sampling, RiskGate veto ranking, recovery ladder, verification retry | INERT-UNIT-TESTED only |
| Supplemental path length and simulation-duration helpers | INERT-UNIT-TESTED only; no frozen artifact changed |
| K=32/H=50 nominal analytic-kernel profile | NOMINAL-KERNEL-PROFILED: p50 `0.4677 s`, p95 `0.4803 s`, pre-optimization p50 `2.0996 s`; not full-system latency |
| Physical behavior of arms 3–5 with scripted-primary proposer | UNVERIFIED PHYSICAL; no Week 3 campaign |
| Full-system and scored per-decision latency | UNVERIFIED; not claimed below 1 s |
| True risk rejection rate from candidate-level counts | Definition and pure aggregation INERT-UNIT-TESTED (`decision_metrics.py`); measurement MISSING, BLOCKED on instrumentation plus authorized run |
| Scored-arm prediction-error trajectories | Pairing contract INERT-UNIT-TESTED (`pair_executed_prediction`); measurement MISSING, BLOCKED on recorder instrumentation plus authorized run |
| Additive decision-evidence schema and delegate-only probes | INERT-UNIT-TESTED only (`decision_evidence.py`, `decision_adapters.py`); not wired into production, no frozen artifact changed |
| Week 4 frozen TEST campaign and statistics | BLOCKED / PENDING-AUTHORIZATION until 2026-12-05 |
| M1-06a 5+5 repetition waiver | EXPLICITLY WAIVED by approved decision; never reported as MEASURED |

## 10. Exact metrics and evidence outputs for an authorized run

Per decision, emit a JSONL record containing: decision index, simulation tick, source
state hash, proposer source, generator seed, all 32 draw seeds, H=50, prediction count,
hard-check names/results, J components, weights, ranking direction, score, selected index,
rationale built from those numbers, all rejection reasons, no-safe/recovery events, and
phase-separated timing fields. Missing values must be `null` with a reason, never invented.

Per episode, emit: outcome and success flag, failure category, collision-proxy and contact/
FK-clearance channels, end-effector path length (base-frame metres), simulation duration
(seconds), wall-clock execution time (seconds, clearly separated), prediction error by
quantity when trajectories are recorded, evaluated/rejected candidate counts when
instrumentation exists, veto/no-safe/retry counts, and verification result.

Per campaign, emit: Wilson 95% success intervals, per-scene rates/CIs/seed lists,
failure-category counts, phase-separated latency p50/p95 with K and H, calibration
provenance, selected parameters, and a manifest with SHA-256 for source commit, configs,
splits, report bindings, campaign artifacts, and recorder provenance. Week 4 adds paired
`(scene, seed)` comparisons with scene-level clustering, not naive episode independence.

Supplemental path length is `Σ‖ee_xyz[i] − ee_xyz[i−1]‖₂` in base-frame metres; simulation
duration is `t[-1] − t[0]` with finite strictly increasing timestamps. Wall-clock time is
non-deterministic run bookkeeping and is never substituted for either.

The per-decision record above is realized by the additive typed schema in
`src/vlarobot/eval/decision_evidence.py` (`DecisionEvidence`, `CandidateEvidence`,
`CostComponents`, `PhaseTimings`, `PredictionPairingEvidence`, `ExecutionEvidence`,
`VerificationEvidence`). The risk rejection-rate denominator, prediction-pairing boundaries
and phase-separated latency summary are implemented in `src/vlarobot/eval/decision_metrics.py`.
The optional delegate-only probes in `src/vlarobot/eval/decision_adapters.py` are inert and
must not be wired into production without the §12 approval. Exact contracts, node IDs and the
read-only preflight commands are in `docs/WEEK3_DECISION_OBSERVABILITY.md` §§2–8; the frozen
hash preflight is `tools/verify_week3_frozen_evidence.py`.

## 11. Phase-separated latency protocol

Authorized runs must time and report separately:

1. candidate generation (K=32 draws);
2. surrogate prediction (H=50 per candidate, where applicable);
3. RiskGate hard checks and J scoring;
4. verification and recovery;
5. model inference where applicable and separately authorized.

Report p50/p95 per phase and per decision with K=32 and H=50 explicit, plus backend,
hardware, and EMA/operating-point notes. The existing nominal-kernel evidence is retained
as a separate scope-limited result. No K or H reduction, safety-check removal, or
single-run result may be used to claim the below-1-second target.

## 12. Authorization boundaries and human gates

This phase authorizes only this document, the supplemental metric helpers/tests/tool,
and documentation corrections. It does **not** authorize:

- Webots, real IK, model inference, training, checkpoint downloads, or actuation;
- any Week 3 physical or simulated scored evaluation;
- TEST split access before the authorized Week 4 window;
- new safety weights or thresholds without validation evidence;
- changes to frozen gates, seeds, splits, configs, recorder/runner schemas, provenance,
  A-D2, PR #2, `main`, or archived evidence.

Required human gates before execution:

1. approve the exact run command, backend, scene/seed list, and output destination;
2. supply/review the authoritative `definitions.json` bound to the report SHA-256;
3. approve candidate instrumentation changes if candidate-level rejection rate or scored
   prediction error is required;
4. approve VALIDATION-only calibration parameters before any recurrence;
5. separately authorize the Week 4 TEST campaign within 2026-12-05..12-07.

Until those gates are passed, all arms 3–5 results remain UNVERIFIED PHYSICAL and all
execution-dependent activities remain PENDING-AUTHORIZATION.
