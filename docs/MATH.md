# MATH.md — report equations mapped to implementation (DRAFT)

Status labels follow ADR-0029: **MEASURED** (recorded evidence exists), **INFERRED**
(structural correspondence argued from code), **UNVERIFIED** (no evidence yet).

The controlling source is the confirmed submitted Phase I report
("WorldMind: World Model-Based Robot Simulation", SHA-256
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`) and ADR-0029
(`docs/DECISIONS.md`). Equation wording below is quoted from the report text extraction.

## Equations of the submitted report (3.1–3.8)

| Eq | Report statement | Implementation | Status |
|---|---|---|---|
| 3.1 | `ŝ_{t+1:t+H} = W(s_t, a_{t:t+H−1})` — world model predicts future states from current state and an action sequence | `src/vlarobot/world_model/rollout.py::AnalyticSurrogate.predict_trajectory` (per-chunk state trajectory; velocity-limited tracking, conditional attachment, release/settling, fail-closed hard reasons) | INFERRED (mechanism complete; scored-arm prediction error pending week 3/4) |
| 3.2 | `a_{t:t+H−1} = πθ(I_t, L, ŝ_{t+1:t+H}, c_t)` — VLA policy consumes instruction, observation, *predicted future state*, context | **Documented deviation (ADR-0029):** SmolVLA cannot consume a predicted state; the architecture is propose-and-evaluate. Proposer: `src/vlarobot/vla/action_adapter.py` + `src/vlarobot/vla/model_service.py` via `planning/candidates.py`; evaluation of each proposed chunk replaces policy-internal conditioning | MEASURED deviation (frozen VLA-only campaign, 0/24) |
| 3.3 | `u_t = C(a_t, q_t)` — control mapping of selected action to robot commands | `src/vlarobot/backends/webots/kinematics.py` (registered IK), `execution/safety_monitor.py` validation, `planning/execution.py::execute_first` (one control tick, re-validated before `backend.apply()`) | INFERRED |
| 3.4 | `e_t = d(s_{t+1}, ŝ_{t+1})` — prediction error vs actual outcome | `src/vlarobot/world_model/evaluation.py::prediction_errors` (quantity-separated TCP/joint/object errors; never probabilities) + recorded replay evidence (`docs/WORLDMIND_DEVELOPMENT_CHECKPOINT.md`, TCP RMSE 4.48–6.69 mm overall across families at 1/50 ticks (clear-family subset: 4.62/6.69 mm)) | MEASURED (read-only replay conditioning; not autonomous-task claims) |
| 3.5 | `s_t = P(o_t)` — perception/state representation | Oracle state with provenance flags (report 4.3 allowance; ADR-0029): `backends` observations + `get_ground_truth()` restricted to eval/debug/oracle mode; per-episode `state_source` provenance in `eval/recorder.py` meta | INFERRED |
| 3.6 | `τ_i = {s^{(i)}_{t+1}, …, s^{(i)}_{t+H}}` — candidate trajectories | `src/vlarobot/planning/candidates.py::SeededCandidateGenerator` (seeded draws, all draw seeds logged) + `risk/comparison.py::sampled_candidates` | INFERRED |
| 3.7 | `ŝ^{(i)}_{t+1:t+H} = W(s_t, τ_i)` — per-candidate prediction | `predict` callable injected into `risk/comparison.py::ComparisonPlanner` (comparison.py:51), backed by the analytic surrogate | INFERRED |
| 3.8 | `J(τ_i) = wg·Jg + wl·Jl + wr·Jr + wu·Ju` with hard-constraint precedence: "The picked candidate should fulfill hard safety constraints … A low combined cost must not be applied for the explanation of a trajectory that violates a mandatory restriction." | `src/vlarobot/risk/gate.py::RiskGate.select` — hard checks fail closed *before* any J ranking; nonfinite/missing/duplicated checks rejected; rationale embeds actual components and weights | INFERRED (mechanism complete + unit-tested; **weights not yet calibrated on VALIDATION — ADR-0029 week 3 pending; no weights are claimed**) |

## Component definitions (report wording)

- `Jg` — goal error.
- `Jl` — trajectory length or execution cost.
- `Jr` — forecast risk.
- `Ju` — uncertainty or prediction-related penalties.
- `wg, wl, wr, wu` — "normally configuration parameters".

`configs/world_model.yaml` is packaged but has **no code consumer yet**; it is the intended
home for the Eq 3.8 weights once they are chosen on VALIDATION only (ADR-0029 week 3).
No weights are invented or claimed in this draft.

## Discrepancy: equations 3.9–3.12, 4.1, 5.1

ADR-0029's week-4 documentation item references "`docs/MATH.md` (Eqs 3.1-3.12, 4.1, 5.1 →
code)". The confirmed submitted report contains **only Eqs 3.1–3.8** (plus Tables 2.1, 3.1,
3.2, 5.1, 5.2, 6.1). Equation numbers 3.9–3.12, 4.1 and 5.1 do not exist in this PDF.
**This is recorded as a discrepancy**: those numbers must originate in the Steering
Directive (or a later report revision). Pending owner/steering-directive input, this file
maps exactly what the confirmed report contains. UNVERIFIED — not resolvable from the
repository at this time.

## Report Tables

- **Table 5.1** (planned evaluation criteria) → `docs/REPORT_CONFORMANCE.md`.
- **Table 5.2** (results template) → week-4 deliverable; actual measurements only.
- **Table 6.1** (twelve-week schedule) → superseded operationally by ADR-0029's
  four-week campaign; the report's own milestones (§6.2) remain the conformance targets.
