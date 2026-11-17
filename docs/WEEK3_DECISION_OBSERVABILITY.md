# WEEK3_DECISION_OBSERVABILITY.md — risk-gate evidence and validation readiness (2026-11-17)

Status: **additive, sidecar-only, inert — no production path is wired and no evaluation is
launched.** This document specifies typed evidence schemas, pure metric aggregation, optional
nonintrusive probes, an exact Table 5.1 mapping for owner review, and the exact validation
preflight. It does not run Webots, real IK, neural inference, training, actuation, a scored
evaluation, or any TEST-split activity.

Controlling sources:

- submitted Phase I report SHA-256 `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`
  (Eq. 3.8 and Table 5.1, transcribed verbatim below);
- ADR-0029 (four-week campaign, frozen sizes, five arms, gate, Week 3/4 boundaries) and
  subsequent append-only decisions, ending with ADR-0046 in `docs/DECISIONS.md`;
- `docs/WEEK3_RISKGATE_VALIDATION_PLAN.md` (execution protocol), `docs/REPORT_CONFORMANCE.md`
  (status classes), `docs/evidence/safe-ci-test-inventory-v1.json` (38-node allowlist).

## 1. Execution-path integration map (inspected before any edit)

No production `brain/main/controller` function composes the full WorldMind chain:

- `src/vlarobot/cli.py:218` `brain()` only connects, resets and steps a backend.
- `src/vlarobot/cli.py:321` `eval()` runs the scripted baseline only.
- The full chain is assembled only in `tests/test_rollout_comparison.py:124,373` and
  `tools/independent_candidate_validation.py:205`.

Call chain and safe observability seams:

| Stage | Anchor | Safe sidecar seam |
|---|---|---|
| Candidate generation | `planning/candidates.py::SeededCandidateGenerator.propose` (`:30`) | snapshot `draw_log[-1]` immediately **after** the delegated call |
| Surrogate prediction | `world_model/rollout.py::AnalyticSurrogate.predict_trajectory` (`:118`), called from `risk/comparison.py::ComparisonPlanner.select` (`:58`) | wrap the **injected** `predict` callable at planner construction |
| Assessment / hard checks | `risk/gate.py::CandidateAssessment`, `risk/comparison.py` extra checks | wrap the **injected** `assess` callable; classify from returned `hard_checks` + public gate config |
| Ranking / selection | `risk/gate.py::RiskGate.select` (`:38`) | do **not** wrap `select`: `ComparisonPlanner.select` calls it `N+1` times (`:104-105`); derive rankability from `CandidateAssessment` and public weights instead |
| Pre-apply veto / execution | `planning/execution.py::execute_first` (`:11`), `execution/safety_monitor.py::SafetyMonitor.validate` (`:79`) | wrap the **whole** `execute_first` call and/or delegate-only monitor/backend probes |
| Verification / recovery | `planning/verification.py::PlacementVerifier.check/next_action` (`:43,73`) | wrap whole verifier calls; record observed result and next transition |
| Recording / metrics | `eval/recorder.py:154-212,312-341` | write new sidecars only; never touch frozen `steps.npz`, `meta.json`, `summary.json`, `episodes.csv`, `provenance.json`, `plan.json`, `timings.json` |

Timing hazard anchors (probe whole phases only, opt-in `perf_counter_ns()`):
`planning/execution.py:17-22`, the rollout H-loop, `risk/comparison.py:70-82,104-105`,
recorder I/O, and backend/ZMQ paths.

Data that cannot be recovered without **behavior-sensitive changes** (STOP and seek approval
before attempting): executed-candidate↔prediction IDs spanning recorder files, typed hard-check
codes, per-candidate J components and trajectories, pre-apply veto details, decision IDs across
sidecars, and phase-separated timing inside the planner. The probes in this phase capture only
what is returned by or injected into the existing objects and write sidecars; they cannot
retrofit those fields into archived scored evidence.

## 2. Candidate evidence schema (typed, fail-closed)

Source: `src/vlarobot/eval/decision_evidence.py` (`SCHEMA_VERSION = 1`). All records are
dataclasses with `from_dict`/`to_dict`; `allow_nan=False` serialization is enforced in tests.
Every absent value is `None` plus an explicit reason; no timing, cost or pairing value is
invented.

Enums:

- `CandidateDisposition`: `selected`, `safe_unselected`, `hard_vetoed`,
  `invalid_or_unevaluable`. The first three are the disjoint **evaluated** classes; the
  union constant is `EVALUATED_DISPOSITIONS`.
- `HardCheckResult`: `pass`, `fail`, `unknown` (gate saw `None`), `not_run`, `duplicate`.
- `MissingCostReason`: `not_applicable_arm`, `not_computed`, `not_observed`,
  `nonfinite_component`, `nonfinite_total`, `assessment_error`.
- `PairingStatus`: `paired`, `missing_observations`, `truncated`, `recovery_boundary`,
  `abort_boundary`, `executed_candidate_mismatch`, `underspecified`.

`CostComponents` — `Jg, Jl, Jr, Ju, total_j, weights(wg,wl,wr,wu), maximize, missing_reason`.
Nonfinite components are converted to absence with a reason; `total_j` is the weighted sum
only when all four components are finite. A missing reason is mandatory iff any component is
absent, and forbidden otherwise.

`CandidateEvidence` — `episode_id, decision_id, candidate_id, candidate_index, sampling_seed,
proposer_source, candidate_count, disposition, hard_check_result, hard_veto_reasons,
unevaluable_reasons, hard_checks[(name,result)], safe, rankable, costs, selected_candidate_id,
no_safe_set, recovery_transition, prediction_available, prediction_horizon_steps,
prediction_hard_reasons, trajectory_state_count`. Validation enforces: safe/rankable for
selected and safe-unselected, unsafe/unrankable for hard-vetoed, unknown safety and unrankable
for invalid/unevaluable, exactly one selected candidate per decision, unique candidate IDs,
and no selected candidate in a no-safe-set decision.

`DecisionEvidence` — `episode_id, decision_id, arm, proposer_source, decision_index,
candidate_count, draw_seed, draw_seeds, candidates, selected_candidate_id, no_safe_set,
recovery_events, timings, execution, verification, prediction_pairing, provenance,
schema_version`. `candidate_count` must equal the number of candidates; `draw_seeds` length
must match when present.

Supporting records: `PhaseTimings`, `PredictionPairingEvidence`, `ExecutionEvidence`,
`VerificationEvidence`. A worked synthetic example is
`tests/fixtures/decision_observability/decision-evidence-example.json` (explicitly
non-measured, non-scored).

## 3. Risk rejection rate definition

Source: `src/vlarobot/eval/decision_metrics.py::risk_rejection_rate`.

```
evaluated = hard_vetoed + safe_unselected + selected
risk_rejection_rate = hard_vetoed / evaluated
```

- Invalid/unevaluable candidates are **excluded from the denominator** and reported as a
  separate `invalid_or_unevaluable` count; they are never silently dropped.
- If `evaluated == 0`, the rate is `None` with `undefined_reason = "no evaluated candidates"`
  — never `0.0`.
- `RiskRejectionRate` records `rejected`, `evaluated`, `invalid_or_unevaluable`, `rate`,
  `denominator_definition`, `undefined_reason`.
- `risk_rejection_aggregate` reports a pooled rate plus an optional **macro** rate (unweighted
  mean of defined per-decision rates), with counts of defined and undefined decisions.
  `candidate_disposition_counts` reports every class plus the `evaluated` union.

Report Table 5.1 wording is "Rejected candidates divided by evaluated candidates". The exact
instrumented numerator/denominator above is the concrete, owner-reviewable operationalization;
existing action-level/adapter veto tallies (21 in the frozen VLA campaign) are **not** this
rate and are not substituted for it.

## 4. Prediction pairing contract

Source: `pair_executed_prediction` in `decision_metrics.py`; record type
`PredictionPairingEvidence`.

- **Identity:** `executed_candidate_id == prediction_candidate_id` is required. A rejected or
  unselected candidate cannot be scored as the executed prediction (`ValueError`:
  "rejected candidates cannot be scored as executed predictions").
- **Frame:** `observed_reference_frame` must be explicitly `"base"`; no world-frame or
  unspecified-frame inference.
- **Alignment:** one timestamp per observed state, finite, nonnegative, strictly increasing,
  with `observed_timestamp_s == observed_timestamps_s[0]`; positive finite `prediction_dt_s`
  (defaults to `prediction.dt_s`).
- **Boundaries:** explicit `recovery_boundary` / `abort_boundary` strings stop scoring at the
  boundary; abort yields `ABORT_BOUNDARY` with no error metrics. Missing observations yield
  `MISSING_OBSERVATIONS` (no invented error). `allow_truncated=True` yields `TRUNCATED` with
  errors computed over the common prefix. Fewer than two common states yields
  `UNDERSPECIFIED`.
- **Errors:** `prediction_errors()` output is retained under `error_metrics_si` with an
  explicit `error_units` map: metres (robot/object position), radians (joint/object
  orientation), steps (`prediction_steps`). SI only; no probabilities.
- Pairing is executed-candidate-only; the reported scored-arm prediction error remains
  `MISSING / BLOCKED` until a real sidecar plus an authorized run exists.

## 5. Latency contract

Source: `PhaseTimings`, `summarize_phase_latencies`. All durations use `time.perf_counter_ns()`
and are whole-phase, opt-in windows. Fields (seconds, `None` + `missing_reasons` when absent):
`candidate_generation_s`, `surrogate_prediction_s`, `hard_safety_checks_s`,
`riskgate_ranking_s`, `selection_s`, `planning_end_to_end_s`, `model_inference_s`,
`verification_s`, `recovery_s`.

- Summary uses deterministic linear-interpolated p50/p95 (`numpy.percentile`) over explicit
  samples; each phase is summarized independently and components are **not additive**
  (e.g. `planning_end_to_end_s` overlaps its sub-phases).
- Model inference is a separate phase; the analytic surrogate has no neural inference.
- Scope only: nominal analytic-kernel profile at K=32, H=50 — optimized p50 `0.4677 s`,
  p95 `0.4803 s`, pre-optimization p50 `2.0996 s`
  (`docs/evidence/planning-optimization-final.json`). Full-system and scored per-decision
  latency are **UNVERIFIED**; the below-1-second milestone is not claimed.

## 6. Report-contract mapping (proposed, for owner review — no production definitions.json)

Eq. 3.8 (verbatim report p.):
`J(τi) = wg Jg(τi) + wl Jl(τi) + wr Jr(τi) + wu Ju(τi)`, where "Jg measures goal error",
"Jl length or execution cost", "Jr risk", "Ju uncertainty or prediction-related penalties";
"the weights wg, wl, wr, wu are normally configuration parameters". Hard safety constraints
have precedence. `CostComponents` carries exactly these four symbols, the four weights, the
ranking direction and an explicit absence reason; component meanings are supplied by the
reviewed bindings, never guessed in code.

Table 5.1 verbatim rows (report p.44) and the proposed evidence/unit mapping. **Units are
proposed from the measurement wording for owner review only**; `eval/report_contract.py`
requires an authoritative `definitions.json` with `report_sha256` matching the PDF and exactly
eight named `{definition, unit}` entries. No such production file is created in this phase.

| Table 5.1 metric (name) | Verbatim measurement | Proposed unit | Evidence source / status |
|---|---|---|---|
| Task success rate | Successful trials divided by total trials | ratio (0–1) | `eval/metrics.py` + Wilson interval — MEASURED: scripted TRAIN 23/24, VALIDATION 15/18, VLA-only 0/24 FAILED |
| Prediction error | RMSE, MAE or task-appropriate metric | SI per quantity (m, rad) / steps | `world_model/evaluation.py::prediction_errors` + §4 pairing — replay-only MEASURED; scored-arm MISSING/BLOCKED |
| Collision rate | Trials with collisions divided by total trials | ratio (0–1) | displacement/tilt **PROXY** + FK clearance — MEASURED, always proxy-labelled |
| Path length | Sum of distances along the executed trajectory | m | `eval/derived_metrics.py::end_effector_path_length_m` — inert-tested; no frozen artifact changed |
| Execution time | Recorded simulation or wall-clock duration, clearly identified | s | `eval/runner.py` wall_s (MEASURED bookkeeping) + `simulation_duration_s` derivation — kept separate |
| Planning latency | Time required to generate and evaluate candidates | s | §5 phase timings; nominal-kernel-only profile K=32/H=50 — full-system UNVERIFIED |
| Risk rejection rate | Rejected candidates divided by evaluated candidates | ratio (0–1) | §3 definition + `risk_rejection_rate` — MISSING until sidecar instrumentation + authorized run |
| Reliability | Successful runs and documented failure categories | counts / categories | `failure_states` + `evaluation_failure_category` in summaries — MEASURED for arms 1–2 |

The proposed `definitions.json` shape (do **not** create until owner-reviewed) is:
`{"report_sha256": "23d5ff15…1522b", "components": {Jg,Jl,Jr,Ju: "<verbatim definition>"},
"table51": {<exact eight names>: {"definition": "<verbatim measurement>", "unit": "<approved unit>"}}}`.
Names/units above require owner confirmation before `ReportDefinitions.load` is used in any
scored path.

## 7. Exact VALIDATION readiness preflight (all read-only, no campaign run)

Run from the `VLA-Project-week3` worktree. These commands do not access TEST, Webots, real IK,
inference, training or actuation; they lint, type-check, compile, run the reviewed inert nodes,
verify the frozen baseline, and confirm frozen-evidence hashes.

```text
cd /home/tj330/programs/VLA-Project-week3
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv run python -m compileall -q src/vlarobot/eval/decision_evidence.py src/vlarobot/eval/decision_metrics.py src/vlarobot/eval/decision_adapters.py tools/verify_week3_frozen_evidence.py
uv run python tools/verify_safe_ci_selection.py
uv run python -m pytest -q <38 reviewed safe-CI node ids from docs/evidence/safe-ci-test-inventory-v1.json>
uv run python -m pytest -q <18 reviewed inert node ids listed in §8>
uv run python tools/verify_week3_frozen_evidence.py --locked-source-root /home/tj330/programs/VLA-Project
```

The frozen-evidence checker recomputes: report SHA-256; baseline provenance SHA-256 and its
12 locked source files against `--locked-source-root`; the 225-entry VLA campaign manifest plus
its provenance/summary SHA-256; and the planning-evidence JSON SHA-256 with its 12 pinned
sources and 12 frozen sources. It prints JSON with `test_split_accessed: false` and writes
nothing.

Execution prerequisites that remain **human gates** (from the validation plan §12): approve
the exact run command/backend/scene-seed list/output; supply and review the authoritative
`definitions.json`; approve candidate instrumentation if candidate-level rejection rate or
scored prediction error is required; approve VALIDATION-only calibration parameters; and
separately authorize the Week 4 TEST window (2026-12-05..12-07). None are satisfied here.

## 8. Reviewed inert test nodes and rationale

Files: `tests/test_eval_decision_evidence.py` (5), `tests/test_eval_decision_metrics.py` (8),
`tests/test_eval_decision_adapters.py` (5). All are pure-Python with synthetic fixtures; no
Webots, IK, inference, actuation, scored run or TEST access. Transitive imports are confined to
`vlarobot.eval.{decision_evidence,decision_metrics,decision_adapters}`, `vlarobot.core.types`,
`vlarobot.risk.{comparison,gate}`, `vlarobot.world_model.{rollout,evaluation}` and NumPy.

Exact node IDs (also the §7 inert selection):

```text
tests/test_eval_decision_evidence.py::test_synthetic_fixture_round_trips_and_serializes_without_nan
tests/test_eval_decision_evidence.py::test_missing_cost_components_require_an_explicit_reason
tests/test_eval_decision_evidence.py::test_selected_candidate_must_match_decision_level_identifier
tests/test_eval_decision_evidence.py::test_no_safe_set_cannot_carry_a_selected_candidate
tests/test_eval_decision_evidence.py::test_unavailable_timings_require_a_reason_for_every_phase
tests/test_eval_decision_metrics.py::test_risk_rejection_rate_uses_explicit_evaluated_denominator
tests/test_eval_decision_metrics.py::test_empty_evaluated_denominator_is_undefined_not_zero
tests/test_eval_decision_metrics.py::test_aggregate_counts_undefined_decisions_and_reports_macro_separately
tests/test_eval_decision_metrics.py::test_latency_summary_uses_deterministic_linear_percentiles
tests/test_eval_decision_metrics.py::test_prediction_pairing_rejects_rejected_candidate_identity
tests/test_eval_decision_metrics.py::test_prediction_pairing_full_alignment_reports_si_errors
tests/test_eval_decision_metrics.py::test_prediction_pairing_missing_observations_are_not_invented
tests/test_eval_decision_metrics.py::test_prediction_pairing_abort_boundary_prevents_scoring
tests/test_eval_decision_adapters.py::test_generation_probe_delegates_once_and_snapshots_existing_draw_log
tests/test_eval_decision_adapters.py::test_prediction_and_assessment_wrappers_return_original_objects
tests/test_eval_decision_adapters.py::test_builder_classifies_evaluated_and_invalid_candidates
tests/test_eval_decision_adapters.py::test_execution_wrapper_records_delegated_result_without_changing_it
tests/test_eval_decision_adapters.py::test_probe_wrappers_delegate_verdicts_and_verification
```

Result at `6e4c86e` + this additive work: `18 passed`.

## 9. Constraints unchanged

No change to candidate generation/sampling, RiskGate objective/ranking, hard-veto logic,
SafetyMonitor, recovery policy, frozen thresholds/seeds/splits/gates, archived evidence, scored
summaries, A-D2 infrastructure, recorder/runner schemas, PR #2, or `main`. No public interface
in `core/types.py`, `core/interfaces.py` or `core/wire.py` changed. No production
`definitions.json` created. No TEST-split access. Full status mapping is in
`docs/REPORT_CONFORMANCE.md`; the execution protocol is in
`docs/WEEK3_RISKGATE_VALIDATION_PLAN.md`.

## 10. Additive production composition inert nodes (reviewed)

File: `tests/test_worldmind_production.py` (14). All use synthetic config/report/calibration
fixtures, injected reviewed-style bindings and `FakeBackend`; no Webots, real IK, neural
inference, training, actuation, scored campaign or TEST split. The production constant
`AUTHORITATIVE_REPORT_SHA256` is monkeypatched only inside these tests to a synthetic fixture
report hash; the shipped production constant is unchanged.

```text
tests/test_worldmind_production.py::test_scripted_adapter_is_seeded_bounded_and_scripted_only
tests/test_worldmind_production.py::test_jitter_never_clips_or_silently_accepts_out_of_bound_draws
tests/test_worldmind_production.py::test_full_composition_records_k32_nplus1_gate_execution_pairing_and_recording
tests/test_worldmind_production.py::test_generation_is_reproducible_from_explicit_seed
tests/test_worldmind_production.py::test_hard_veto_takes_precedence_and_prevents_apply
tests/test_worldmind_production.py::test_no_safe_set_uses_bounded_recovery_ladder
tests/test_worldmind_production.py::test_live_safety_veto_precedes_backend_apply
tests/test_worldmind_production.py::test_verification_failure_switches_to_alternate_candidate_family
tests/test_worldmind_production.py::test_selected_prediction_pairing_uses_the_correct_decision_trajectory
tests/test_worldmind_production.py::test_sidecars_are_created_exclusively_and_never_overwritten
tests/test_worldmind_production.py::test_blocked_template_and_missing_reviewed_inputs_fail_closed
tests/test_worldmind_production.py::test_unregistered_bindings_fail_closed
tests/test_worldmind_production.py::test_plan_preflight_rejects_test_training_wrong_state_and_existing_output
tests/test_worldmind_production.py::test_cli_exposes_new_command_preserves_existing_commands_and_fails_closed
```

Result in this worktree: `14 passed`. These nodes are additive to the §8 18-node decision
observability selection and the 38-node safe-CI selection; none of the three selections was
expanded to untested or simulator-dependent files.
