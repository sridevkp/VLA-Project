# WorldMind Owner Decision Readiness (2026-11-18, offline supplement)

**Purpose:** prepare the existing owner-review packet for actual decisions without rewriting it. This supplement distinguishes items that can be approved/confirmed from existing evidence versus items requiring independent review. No approvals are granted here.

**Primary packet:** `docs/WORLDMIND_OWNER_REVIEW_PACKET_2026-11-18.md` (`be8f22a`).

## 1. Eq. 3.8 objectives (`configs/report/objectives.proposal.json`)

| Item | Existing evidence/status | Decision readiness |
|---|---|---|
| Verbal component meanings (goal error, length/cost, risk, uncertainty/prediction penalties) | Consistent with report verbal intent; proposal records meanings only | **Can be confirmed as intent** if owners accept report wording; formulas remain separate |
| Hard safety precedence (veto/ranking exclusion, no utility offsets veto, `MissingObjectiveInput` fail-closed) | Implemented/required by system contracts and proposal | **Can be approved as policy** from existing architecture/contracts (independent of numeric formulas) |
| Component formulas (`Jg,Jl,Jr,Ju`) | Engineering proposals only, explicitly not report quotations | **Requires independent review/approval** (scientific choice) |
| Units, normalization scales/margins, ranking direction, weights `(wg,wl,wr,wu)` | Proposed values/structure only; no approved calibration outcome | **Requires independent review/approval** + VALIDATION calibration/selection evidence |
| Evaluator binding/selection rule | Proposed/registered mechanism exists; no approved evaluator artifact | **Requires independent review/approval** (binding + artifact) |
| Any non-safety `Jr` threshold | Not defined/approved | **Requires independent review/approval** if introduced (must not replace hard vetoes) |

## 2. Table 5.1 metric conventions (`configs/report/definitions.candidate.json`)

| Metric | Existing evidence/status | Decision readiness |
|---|---|---|
| Task success rate, risk rejection rate | Definitions align with existing aggregation/disposition code; units proposed as ratios | **Largely confirmable** subject to owner sign-off on exact denominator/inclusion rules and ratio unit convention |
| Path length | Derived metric implementation exists (`end_effector_path_length_m`) | **Largely confirmable** (m, executed EE/tool trajectory) subject to owner sign-off |
| Execution time / planning latency | Sources exist but semantics split (sim vs wall; phase timings; full-system latency UNVERIFIED) | **Requires independent review** to approve canonical definition, unit labeling, and which source is authoritative/reportable |
| Prediction error | Truthful truncated pairing implemented; exact task metric (RMSE/MAE/per-quantity/steps) not owner-selected | **Requires independent review** to select metric, quantities, and units |
| Collision rate | Recorded as displacement/tilt **PROXY** + contact diagnostics; must remain labeled proxy | **Requires independent review** (proxy acceptability/labeling or stronger evidence definition) |
| Reliability | Failure states/categories recorded in summaries; taxonomy granularity owner-facing | **Requires independent review** to approve category taxonomy and reporting convention |

## 3. Calibration rules and artifacts (`worldmind_calibration.py`, manifest/proposals)

| Rule/artifact | Existing evidence/status | Decision readiness |
|---|---|---|
| VALIDATION-only, frozen 18-episode/seed plan, hash-pinned evidence, 4-stage non-circular workflow, selection statistic ≠ independent evaluation | Implemented and fail-closed in code/config contracts; manifest proposal prepared offline | **Can be approved as process/policy** from existing contracts (subject to legal/experimental governance) |
| Evidence content (measurements, episode outcomes, source hashes) for stage 2 | Not collected; no approved evidence artifact | **Requires independent review/collection authorization** (and valid runtime/safety preconditions) |
| Draft candidate grid, selected parameters, approval record, approved candidate/evidence artifacts | Not produced/approved | **Requires independent review/approval** after evidence exists |
| No TEST/TRAIN use, no post-hoc tuning on TEST | Enforced by policy/contracts | **Can be approved as invariant** |

## 4. Production bindings (`configs/worldmind_production.yaml`, `worldmind_config.py`, `backends/worldmind_bindings.py`, preflight)

| Binding/config element | Existing evidence/status | Decision readiness |
|---|---|---|
| Frozen deterministic contract (split/seeds/episodes/K/H/dt/oracle/full_system/scripted_chunk_adapter, gate-call policy) | Implemented and verified in config/preflight code and manifest proposal | **Can be approved as execution contract** if owners accept frozen plan assumptions |
| Fail-closed loading/validation/preflight structure, explicit registry (no dynamic config imports), diagnostic-vs-scored separation | Implemented and tested | **Can be approved as wiring/policy** |
| Concrete binding identities/implementations for components, hard checks, surrogate/safety/verification, pose solver, waypoints/factories, calibration evaluator, inputs | Placeholders/empty registries in shipped production path; no reviewed registrations | **Requires independent review/approval** (and in some cases implementation/evidence) |
| Approved `definitions.json`, ready production config values, runtime geometry probe evidence, calibration linkage | Not available | **Requires independent review/approval** with supporting evidence |

## 5. Safety/collision coverage (offline, finite, fail-closed)

**Existing evidence usable now:** motion readiness dossier (`a52a4807…6c53`) and arm-sweep contract/config identify finite unresolved bounds and covered areas (e.g. frame-corrected static gripper envelope fits pinned radius in dossier evidence; pinned capsule-pair checks exist but full-body coverage unproven).

| Area | Can be closed by owner approval from existing evidence now? | Requires independent evidence/review |
|---|---|---|
| Swept residual terms (joint displacement/reach, geometry approx, numerical, transform/registration, strict positive residual) | No | **Yes** — pinned finite bounds, margin/certificate, and reviewed contract for scope |
| Full-body/self-collision coverage and exclusions | No (42 uncovered bodies; exclusions not approved) | **Yes** — coverage proof or approved exclusions/adjacency/contact decisions |
| CAN lifecycle/contact claims | No (ambiguous, fail-closed) | **Yes** — attachment/release/carried-pose evidence and contact policy |
| Motion authorization/certification | No | **Yes** — independent safety-owner certificate/approval (external to code) |
| Runtime table-transform grounding gate | Partially evidenced (v2/v3 matched 5/5 runtime table geometry, geometry export valid) but not acceptable as closure due to stale observation at consumption in each inspection | **Yes** — fresh authorized read-only verification with valid/fresh observation and accepted grounding evidence |

**Offline implementation direction (finite, no new broad framework):** complete explicit bound accounting in sweep/coverage artifacts, enumerate every uncovered body/pair and required bound term with pass/fail criteria, and keep runtime checks/certificates fail-closed until owner-supplied evidence is linked.

## 6. Arm-level execution gate matrix (audited 2026-11-18 after ADR-0060)

The safety gate is not uniform across the five frozen arms. Sources:
`risk/comparison.py::ComparisonPlanner.select` (returns at the first branch for arms 1–2; adds
`prediction_constraints` only for `full_system`) and `world_model/rollout.py` (`check_transition
is None ⇒ hard_reason "unverified-arm-sweep"`).

| Frozen arm | Predicts? | Hard-checks prediction validity | Needs Eq. 3.8 components | Needs sweep certificate | Executable after authorization alone |
|---|---|---|---|---|---|
| `scripted_baseline` | no | no | no | no | **yes** (already MEASURED 23/24 TRAIN, 15/18 VALIDATION) |
| `vla_only` | no | no | no | no | **yes** (already MEASURED 0/24 TRAIN — gate FAILED) |
| `vla_risk_veto` | no | no | **yes** | no | no — blocked on Eq. 3.8 definitions/weights |
| `vla_world_model` | yes | **no** (explicit no-veto ablation) | **yes** | no | no — blocked on Eq. 3.8 definitions/weights |
| `full_system` | yes | **yes** (`prediction_complete`, `prediction_constraints`) | **yes** | **yes** | no — blocked on Eq. 3.8 **and** sweep scope |

Consequences for the reviewer question "is full-body swept certification a mandatory prerequisite?":

- With `check_transition=None`, every predicted trajectory carries `unverified-arm-sweep`, so
  `full_system` vetoes all K=32 candidates, reaches an empty safe set and aborts with a recorded
  `NO_SAFE_SET` rationale. This is verified behaviour, not a guess:
  `tests/test_rollout_comparison.py::test_close_command_never_fabricates_grasp_and_unknown_sweep_fails_closed`
  and `tests/test_worldmind_arm_sweep.py` (module docstring) assert it.
- Arms 1–4 do not consult that veto. Therefore the certified swept volume is **not** a global
  prerequisite for simulated execution; it gates **only the `full_system` arm**.
- The Phase I report's collision requirement is a measurement (displacement/tilt proxy plus FK
  clearance, ADR-0029 DESIGN bullet), not a certified swept-volume guarantee. `ARCHITECTURE.md`
  §7 requires a veto when *predicted minimum clearance < margin*. The residual-certificate
  inequality (`arm_sweep.py`, `SWEEP_INEQUALITY`, four finite bounds + margin) is therefore a
  **team-imposed assurance objective that is stricter than the report text**, adopted under
  AGENTS hard rule 6.
- Relaxing it for `full_system` is a **safety-scope owner decision**, not an engineering fix.
  The smallest legitimate version of that decision is either (a) owner supplies the four finite
  residual bounds and a reviewed certificate for the declared simulation scope, or (b) owner
  approves a declared scope limitation that `full_system` reports the sweep-veto ablation and is
  reported as BLOCKED. Both are one-signature decisions; no new framework is needed.

## 7. Measured throughput budget (OBSERVED + INFERRED)

OBSERVED — `runs/validation-evidence-v2-canonical/parallel-progress.json`
(`/home/tj330/programs/VLA-Project-analytic`), matching artifact timestamps 2026-11-15
10:54 → 11:47 local:

- `wall_s_parallel = 3167.826…` s = **52.8 min** for 18 episodes with 3 workers;
- **20.46 episodes/hour aggregate**, **6.82 episodes/hour per worker**, **528 s serial per
  episode**;
- `completed=18`, `planned=18`, `successes=15`, `test_accessed=false`.

INFERRED (nominal, not full-system measured) — `planning/execution.py::execute_first` applies
exactly one control action per decision and re-observes, and the scripted episodes averaged
24.44 s simulated at `dt=0.032` (≈764 control steps). At the hash-pinned nominal kernel p50 of
`0.4677 s` per K=32/H=50 decision batch (`docs/evidence/planning-optimization-final.json`),
full-system planning adds ≈6 min per episode:

- per full-system episode ≈ 885 s → 3 workers ≈ 12 episodes/hour;
- frozen VALIDATION, five arms (90 episodes) ≈ **4.4 h**;
- frozen TEST, five arms (150 episodes) ≈ **12.3 h**.

Conclusion: on the measured harness the campaign fits inside the remaining calendar. **Time is
not the binding constraint; the owner decisions in §1–§5 are.** Actual decision count depends on
the owner-set `bounds.max_decisions`/`max_steps`, and CPU contention across 3 workers is not
measured, so the full-system figure stays INFERRED until a real run exists.
