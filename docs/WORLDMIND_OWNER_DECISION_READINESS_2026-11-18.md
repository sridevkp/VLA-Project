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
