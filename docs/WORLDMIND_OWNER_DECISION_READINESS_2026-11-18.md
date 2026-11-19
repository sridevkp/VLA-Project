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

## 8. Final decision packet (2026-11-19) — five items, one sitting

Only unresolved choices that need an owner signature appear here. No new framework, no new
evidence class, no reopened decision. Sources: files in this tree plus the measured canonical
campaign at `/home/tj330/programs/VLA-Project-analytic/runs/validation-evidence-v2-canonical/`.

### 8.0 What changed since §6–§7

| Change | Evidence | Effect on these decisions |
|---|---|---|
| Freshness measured at point of use (`38b7572`) | unit tests pass; v3 evidence stays byte-identical and `FAILED` (`age ≈12.86 s > 5.0`) | **C** is now attemptable once, with a new output path |
| Full-chain FK for the arm sweep (`d83920f`) | `Kinematics.forward` returns one 4×4 matrix, but both sweep builders indexed it with link indices 4–13 and swallowed the resulting `IndexError` into `None`. Consequence: **an approved certificate could never pass a single transition.** Fixed via `_link_transforms`; the unit-test fake now mirrors the real `forward` / `chain.forward_kinematics` split | **B is now the only structural blocker** for `full_system`; C and D are unaffected |
| Shipped sweep scope vs the measured expert | 18 episodes / **13,766 control steps**; corrected FK reproduces recorded `ee_xyz` to <0.1 mm (n=117) | **B is unavoidable**: a certificate alone can never admit the expert |

Measured scope result at the shipped contract (`margin 0.02`, 62 substeps, pad `q = 0.03981`).
"Allowed q" is the largest pad that still reports clear; a step passes iff `q < allowed`.

| Check as currently implemented | Allowed q min / median | Pass at `q=0.03981` | Cause |
|---|---|---|---|
| 19 arm capsules vs table box | +0.1481 / +0.1481 | **100 %** | none — 3.7× headroom |
| arm capsules vs two floor half-spaces | clear on every step | **100 %** | none |
| gripper envelope (r=0.10 m) vs table box | −0.0969 / +0.0221 | 29.7 % | the 0.1 m radial inflation, not motion: worst raw capsule AABB is only +0.0031 m above the table top |
| gripper envelope vs CAN envelope | −0.1221 / −0.0425 | 26.8 % | grasping requires contact |
| self-pairs, mechanically non-adjacent | **−0.0574** / −0.0508 | **0.0 %** | URDF capsule inflation at coincident joint frames (18 of 19 radii ≥ 0.0441 m), e.g. `arm_5_link:2 \| wrist_ft_tool_link:0` = −0.0412 |

So: `full_system` needs **one** decision (B), not a rebuild. Table AABB in base frame is
`x[0.3005,1.9397] y[−1.9412,2.6204] z[−0.1544,0.72215]`; measured minimum fingertip height is
`0.7699 m`, i.e. **≈47.7 mm above the tabletop plane**.

### 8.1 Packet A — Eq. 3.8 and Table 5.1 (scientific contract)

Frozen and needing only confirmation, because they are already implemented and already approved in
substance: the verbal Eq. 3.8 form, hard-veto precedence (no utility term offsets a veto), the
eight Table 5.1 metric **names and definitions**, and "collision rate" as a labelled proxy.

Needing approval now: the four component formulas and their units. These are engineering choices,
not report quotations — `configs/report/objectives.proposal.json` says so explicitly.

| Component | Formula (as proposed) | Units | Direction |
|---|---|---|---|
| `Jg` | `sqrt((‖p_ee − p_goal‖₂ / s_p)² + (angle(q_ee, q_goal) / s_r)²)` | dimensionless | minimize |
| `Jl` | `(Σ_k ‖p_{k+1} − p_k‖₂ / s_l) + λ_t · (N_steps · dt_s / s_t)` | dimensionless | minimize |
| `Jr` | `mean_k clip((m − d_min,k) / m, 0, 1)`; `d_min,k < 0 → 1` | dimensionless ∈ [0,1] | minimize |
| `Ju` | `mean_k (‖e_p,k‖ / s_p + e_θ,k / s_r)` | dimensionless | minimize |

Missing any input raises `MissingObjectiveInput`; nothing is imputed. `Jr` never substitutes for a
hard check. Symbols `s_p (m)`, `s_r (rad)`, `s_l (m)`, `s_t (s)`, `λ_t`, `m (m)` and the weights
`(wg, wl, wr, wu)` stay **VALIDATION-selected through the registered evaluator**, exactly as the
proposal and §7 of the plan already state; approving A does not fix their numeric values.

Table 5.1: only unit/taxonomy choices remain — ratios for success/collision/rejection, `m` for path
length, `s` for execution time and planning latency (sim vs wall kept separate), SI-per-quantity for
prediction error, counts+categories for reliability.

```
Owner signature (A):
  [ ] approve the four formulas + dimensionless units + minimize direction as written
  [ ] approve, with the amendments written here: ______________________________________
  [ ] reject (state which component and why): _________________________________________
  [ ] Table 5.1 units: [ ] accept the proposed units  [ ] substitute: __________________
  Calibration evaluator binding `registered_calibration_evaluator_v1`: [ ] authorize
  Name: ______________________  Date: __________
```

### 8.1a Packet A decision-time amendment — pending (2026-11-19)

Packet A **as written is not approved**: its `Jr` input requires certified full-arm clearance that
the shipped surrogate does not compute, and its `Ju` formula needs future observed states. The
following minimal amendment is in `configs/report/objectives.proposal.json` and `src/vlarobot/eval/worldmind_objectives.py`; `Jg` and `Jl` are unchanged.

- `Jr_soft = mean_k clip((m - tilde_d_k) / m, 0, 1)`, with `tilde_d_k < 0 -> 1`. Here
  `tilde_d_k = min_o segmentBoxClearance(p_{k-1}, p_k, AABB_o(current), r_tool)`: the minimum exact
  tool-sphere segment/AABB clearance over current observed object boxes and the predicted tool path.
  It is an **uncertified geometric proxy**, not full-arm clearance and not a safety certificate.
  Hard vetoes (`SafetyMonitor.validate`, `RiskGate`, `unverified-arm-sweep`) stay separate and
  precedent; the soft score fails closed on incomplete scenes, missing geometry or truncated
  predictions.
- `Ju_consistency = mean_{i in C} r_i`, where `r_i` is the normalized command-vs-prediction residual
  for EE position/orientation/gripper, `JointTarget` joints, or `GripperCmd` opening. Inputs are only
  the current observation (identity alignment), the candidate `ActionChunk` and the predicted
  `Trajectory`. No future observed state, correction residual, fabricated uncertainty or ensemble
  is accepted; this is not calibrated uncertainty.
- Nothing is activated: scientific registries stay empty, no weight/scale/threshold is selected,
  `full_system` stays `blocked`, scored evaluation stays disabled, and approved VALIDATION
  calibration still precedes any weight choice. ADR-0064 records this pending contract change.

Owner signature (A amendment):

```
  [ ] approve the decision-time Jr/Ju amendment above
  [ ] reject (state which component and why): ______________________________________
  Name: ______________________  Date: __________
```

### 8.2 Packet B — the one safety-scope amendment (`full_system` only)

> **Superseded as a proposal by §9 (2026-11-19).** The measured revision keeps most of the
> geometric checks that this section removed and corrects the table top from `0.72215 m` (AABB) to
> `0.65889 m` (exact). This section is retained for the audit trail only.

This is an **applicability** amendment, not a weakening of the veto. `unverified-arm-sweep` stays
hard and fail-closed for every claim outside the list below, and for every transition where the
catalog, base pose, or required object is missing.

| Element | Amended scope |
|---|---|
| checked bodies | the 19 arm-link capsules from the pinned URDF (unchanged) |
| checked statics | table box + two floor half-spaces (unchanged) |
| gripper vs table box | **exempt**; replaced by the tool-point support predicate below |
| self-pairs (arm–arm, arm–gripper, gripper–gripper) | **exempt**, all pairs |
| CAN | `permitted_contact_objects: ["CAN"]` — required for grasping. Note this exempts CAN for *every* checked capsule, not just the fingertips |
| tool-point support predicate | fingertip point (gripper capsule end, `front + R·(0,0,−0.21875)`) must satisfy `p_tip.z − table_top_z ≥ margin_m` on every substep; `table_top_z` derived from the pinned catalog (0.72215 m), `margin_m` = the existing contract value |
| base-pose gate | unchanged (`0.003 m`, `0.005 rad`) |
| certificate bounds | `max_joint_displacement_bound_rad = 2.35 × 0.032 = 0.0752` (URDF joint limit), `joint_reach_bound_m = 1.13913` (URDF chain length), `geometry_error_bound_m = 0.0`, `registration_uncertainty_bound_m = 0.005`, `numerical_error_bound_m = 1e-6`, `transform_error_bound_m = 1e-4` |

Residual risk carried by B, to appear in the paper verbatim: (1) gripper finger/body geometry other
than the tip may approach the tabletop without being checked; (2) arm-link contact with CAN is
exempt; (3) no self-collision claim is made for any configuration; (4) 42 non-arm collision bodies
remain uncovered. The measured mitigation for (1) is the ≥47.7 mm tip clearance over 13,766 expert
steps.

What B costs: one bounded code change (~1 commit, 1 test file). It cannot be expressed in the current
YAML because `permitted_contact_objects` only exempts scene objects and `adjacent_pairs` only exempts
capsule pairs, so both new elements need named contract fields; I will add ADR-0061 and keep the
shipped config `status: blocked` until the signature exists.

Acceptance test for the implementation, offline and Webots-free: the amended checker must return
`True` on all **13,766** recorded expert transitions, and must still return `False` when the tip is
lowered below `table_top + margin` or when an arm capsule is moved into the table box. That converts
B from a promise into a measurable gate.

```
Owner signature (B):
  [ ] approve the amended scope exactly as tabled above (arm+floor+tool-point+CAN-contact)
  [ ] approve only these subsets: [ ] self-pair exemption  [ ] gripper/table exemption
                                  [ ] CAN contact  [ ] certificate bounds
  [ ] reject (state the required alternative): ________________________________________
  Name: ______________________  Date: __________
```

### 8.3 Packet C — one read-only, runtime-grounding inspection (v4)

Single attempt, no retry, no actuation, no stepping, no reset, no IK, no scoring. Launches a
read-only Webots session only to export `boundingObject` geometry and re-check table grounding with
the repaired freshness measurement.

```
cd /home/tj330/programs/VLA-Project-week3
WEBOTS_HOME=/home/tj330/.local/opt/webots uv run python -m vlarobot.cli \
  worldmind-webots-geometry-inspection \
  --config configs/worldmind_webots_geometry_inspection.yaml \
  --out runs/worldmind/geometry-inspection-v4-2026-11-19
```

Acceptance (all required): `outcome=PASS`; `observation_valid=true`;
`observation_age_s <= 5.0` measured at consumption; 5/5 table targets within `0.0001 m / 0.0001 rad`;
`backend_apply_calls=0`; no `step`, `reset`, IK, or actuation recorded. If it fails, the evidence is
kept byte-identical and nothing is re-run.

```
Owner signature (C):  [ ] authorize one attempt at the command above, output path as written
  Name: ______________________  Date: __________
```

### 8.4 Packet D — scored VALIDATION run (after A, B, C)

```
uv run python -m vlarobot.cli eval-worldmind \
  --config <ready-config> --out runs/worldmind/<new-authorized-dir> \
  --backend webots --state-source oracle --split validation
```

Frozen contract it must satisfy: 18 episodes, seeds `[0,1,2]`, unique (scene, seed) pairs,
`K=32`, `H=50`, `dt=0.032`, `scripted_chunk_adapter`, five arms. It connects to an **already
running** `sim-server`; it never launches Webots. TEST and TRAIN stay sealed.

Preconditions, all fail-closed in `run_worldmind_preflight` (§4): `split`, `state_source`,
`fresh_output`, `config` (`status: ready`), `backend_declaration`, `report_definitions`,
`bindings` (currently `components={}` and `calibration_evaluators={}`), `calibration`,
`frozen_plan`, `runtime_geometry`, `calibration_evidence`. Today the first blocked gate is `config`.

Stop conditions: any unhandled exception, any episode ending `NO_SAFE_SET` where the frozen contract
requires a decision, or any output collision aborts the run with no retry.

Budget (INFERRED): 5 arms × 18 = 90 episodes; ≈885 s/episode full-system → ≈4.4 h with 3 workers,
≈22 h serial. OBSERVED serial equivalent is 528 s/episode without prediction work.

```
Owner signature (D):  [ ] authorize the VALIDATION run above once preflight is green
  Arms: [ ] all five   [ ] only: ____________________________________________
  Name: ______________________  Date: __________
```

### 8.5 Packet E — earliest realistic schedule

| Day | Work | Gate |
|---|---|---|
| Thu 11-19 | FK defect repaired (`d83920f`); this packet issued | done |
| Fri 11-20 | owner signs A, B, C | signatures |
| Fri 11-20 → Mon 11-23 | implement + test B (amended checker, 13,766-transition acceptance), ADR-0061, no activation | B signed |
| Mon 11-23 | run C once; if PASS, register the amended contract as a *candidate*, still `blocked` pending the certificate record | C signed + PASS |
| Mon 11-23 → Wed 11-25 | owner reviews the amended contract + certificate record; I wire A's evaluator and the ready config | A signed + evidence |
| Wed 11-25 → Thu 11-26 | preflight green; scored VALIDATION (≈4.4 h, INFERRED) | D signed |
| Thu 11-26 → Fri 11-27 | calibration selection under the 4-stage workflow; freeze the approved candidate | calibration approval |
| Fri 11-27 → Wed 12-02 | analysis, tables, figures from measured artifacts | — |
| 12-05 → 12-07 | TEST window, ≈12.3 h for 150 episodes (INFERRED) | separate authorization, sealed until then |
| 12-08 → 12-11 | write-up, buffer | — |

Slack is 4 working days before 2026-12-12 if A/B/C are signed by 11-20. Every later signature day
consumes buffer one-for-one; if B slips past 11-27, the defensible fallback is to report the four
non-`full_system` arms as measured and `full_system` as `unverified-arm-sweep`-blocked — which is a
result, not a plan.

### 8.6 What is measured, what is inferred, what is unknown

| Quantity | Status |
|---|---|
| Control/physics steps per episode (166–1059, mean 764.8, n=18) | OBSERVED |
| WorldMind decisions per episode | MEASURED from decision sidecars: `worldmind-sidecars/instrumentation.json` (`decisions_total`, `decisions_per_episode`, `candidate_evaluations_total`, `gate_select_calls`) is surfaced as `summary.json.worldmind_instrumentation`; no count inferred from control steps |
| Candidate-evaluation latency, K=32/H=50 | NOMINAL-KERNEL-PROFILED: p50 0.4677 s; not full-system |
| Per-episode wall time without prediction work | OBSERVED: 528 s serial, 20.46 ep/h at 3 workers |
| Full-system wall time, incl. 3-worker contention | UNKNOWN — no run exists |
| Expert admission under the amended scope | INFERRED from 13,766 measured steps; becomes OBSERVED when the B acceptance test runs |
| Windows execution of any of the above | UNKNOWN — never run |

### 8.7 What happens on receipt

A signature on A + B + C produces, in one working session: the amended checker and its
13,766-transition test, ADR-0061, one v4 inspection attempt, and a ready-to-run preflight report.
Nothing in this packet removes a hard veto, changes a frozen threshold, touches TEST/TRAIN, or
authorizes actuation. The shipped sweep config stays `status: blocked` in every case.

## 9. Narrowed Packet B (2026-11-19) — measured revision of §8.2

§8.2 proposed replacing the geometric checks with a fingertip-only support predicate. Measured
against the recorded expert campaign, that proposal removed checks that the recorded data does not
support removing, and it mis-stated the table clearance because the shipped table box is an AABB
of a world-axis-aligned table. This section replaces §8.2 as the Packet-B candidate for approval.
It is narrower in what it removes, not weaker in what it keeps.

**Status of the implementation:** offline and diagnostic. `ExactSweptArmCheck` and
`packet_b_narrowed_sweep_check` are not reachable from `production_worldmind_bindings()`, the
shipped `configs/worldmind_arm_sweep.yaml` stays `status: blocked`, `components={}` and
`calibration_evaluators={}` stay empty, and `full_system` still vetoes every prediction with
`unverified-arm-sweep`. ADR-0061 records the numerics.

### 9.1 What is retained, modified and excluded

| Element | §8.2 proposal | Narrowed revision (this section) |
|---|---|---|
| checked bodies | 19 arm capsules | **retained**: 19 arm capsules, `109` checked pairs (details below) |
| checked statics | table box + 2 floor half-spaces | **retained**, but the table is now the exact oriented box (`top 0.658885820 m`, not the AABB `0.722153468 m`) |
| arm–arm self-pairs | exempt, all pairs | **modified**: `62` structurally impossible pairs removed (`15` same-rigid-body, `39` one-joint adjacency, `8` URDF volumes that already intersect), `109` retained and checked |
| arm–gripper self-pairs | exempt, all pairs | **modified**: only `16` measured-unsatisfiable `front` housing pairs removed; `9` `front × wrist` pairs and every finger pair stay checked |
| gripper vs table | exempt, fingertip predicate replaces it | **excluded**: the `0.1 m` gripper envelope is not used; the exact URDF gripper boxes are checked against the exact table box at the same `0.039807 m` separation |
| CAN | exempt for every capsule | **modified**: finger primitives only, and only while a caller-declared phase predicate permits it; `front` and all arm capsules stay vetoed |
| base-pose gate | `0.003 m` / `0.005 rad` | unchanged |
| certificate | new displacement/reach/speed terms | **not proposed**: no certificate is asserted; the pinned speed value is labelled underived |

Pair accounting: `171` enumerated capsule pairs = `15` + `39` + `8` + `109`; gripper boxes add
`5 × 19 = 95` box/capsule pairs (`76` removed, `19` retained). The manifest is a pinned partition
and the loader fails closed on any tampering, unknown key, duplicate row, missing witness or
inconsistent count.

### 9.2 Evidence for each exclusion

| Reason | Count | Evidence (measured, not assumed) |
|---|---|---|
| `same_rigid_body` | 15 capsule + 16 box | both primitives belong to one URDF link; rigid self-contact cannot be a collision |
| `directly_jointed` | 39 capsule + 35 box | exactly one URDF joint between the links, including fixed joints |
| `urdf_volumes_intersect` | 8 capsule | the two URDF collision volumes intersect in the zero pose |
| `parallel_linkage` | 9 box | both fingers are children of `front` on parallel prismatic axes; closed gap `0.0004 m` |
| `gripper_mount` | 16 box | maximum recorded clearance over all `13,766` steps stays below `0.039807 m`, with the argmax witness pinned per row |

The `gripper_mount` rule is deliberately per pair. `front:0 × arm_5_link:1` is removed at a measured
maximum of `0.033725 m`; `front:{1..4} × arm_5_link:1` reach `0.040725`–`0.043886 m` and are **not**
removed, and all five `front × arm_5_link:0` pairs reach `0.085725`–`0.095834 m` and stay checked.
Their campaign minima are `0.016501`–`0.032522 m`, so they still veto recorded configurations.

### 9.3 Positive and negative test results

Positive (accept) path:

- The only observed accept path is an explicitly labelled control clone that removes exactly the
  pairs a given state violates. The strict checker rejects that same state. No shipped code path
  contains that relaxation.
- Recomputing each of the `16` pinned `gripper_mount` witnesses from the recorded NPZ reproduces
  the pinned maximum to better than `1e-6 m`.

Negative (reject) path, all asserted in `tests/test_worldmind_packet_b_scope.py`:

- tool/gripper below the permitted table clearance — raising the table until the tightest fingertip
  sits `0.02 m` above it (touching nothing) still rejects, with every arm capsule left above
  `0.039807 m`; the recorded campaign itself measures `0.053530 m` at the tightest
  gripper-box/table pair (`episode-11:253`) and an AABB lower bound of at least `0.148097 m` for
  every capsule/table pair (`episode-06:306`);
- arm capsule intersecting the table — the raised-table capsule case is rejected;
- unintended CAN contact at a `1 mm` gap from a fingertip is rejected, and the same state is
  accepted only when the phase predicate permits fingertips;
- joint-limit and joint-velocity violations are rejected by `SafetyMonitor` with the effective
  URDF limits;
- missing or non-finite input returns `None` (fail-closed), never `True`;
- a genuine non-adjacent capsule violation is represented by the `8` retained pairs that reject
  recorded steps, e.g. `arm_6_link:0 × wrist_ft_tool_link:0` at `-0.030749 m` slack.

Not available: a recorded **passing** expert transition. With the pinned allowance, `0 / 13,766`
steps pass and `0` of them are true penetrations, and no passing configuration exists in the
`15,710`-candidate searched neighbourhood. The strict checker's accept path is therefore exercised
only by the labelled control clone. This is a limitation of the evidence, not a property of the
checker, and it is why this remains a diagnostic proposal rather than a certificate.

### 9.4 Scientific definitions still needing approval

1. The Eq. 3.8 component formulas, units and normalization (verbal form approved; the numeric
   contract is not).
2. The weights and ranking direction for those components.
3. The calibration evaluator and the evidence it must emit.
4. Table 5.1 units and metric taxonomy.
5. The derivation of `speed_bound_m_s`: the pinned `28.493167 m/s` is `≈10.6×` larger than the
   repo-URDF bound `2.35 rad/s × 1.13913 m ≈ 2.677 m/s`. Approval is needed for either keeping the
   pinned value as a deliberately conservative bound or replacing it with a derived one.
6. Whether `full_system` execution requires an approval of this narrowed scope at all, or whether
   `unverified-arm-sweep` should stay a hard veto for the whole arm.

Items 1–4 now have implementation-ready proposals in `configs/report/objectives.proposal.json`
(and `configs/report/definitions.candidate.json`): formulas, units, normalization, calibration
evaluator inputs/outputs/evidence, Table 5.1 denominators and failure taxonomy. They are ready
for approval, not approved; no weight or scale is selected here. Item 5 is **not** being changed:
the pinned `28.49316700274912 m/s` is retained and the derived candidates stay unapplied (§9.7).
Item 6 resolves to `full_system` remaining `blocked` pending the safety decision (§9.7).

Explicitly **not** claimed: a calibrated uncertainty model (the implemented quantity is a
prediction-discrepancy proxy), zero geometric error, a certified displacement bound, or full-body
safety certification (`42` of `61` URDF collision bodies have no arm capsule and are unchecked).

### 9.5 Remaining prerequisites for controlled VALIDATION

1. Owner signs Packet A (in substance) and resolves items 1–4 of §9.4.
2. Owner signs this narrowed Packet B, or signs the narrower option of keeping every pair checked
   and reporting `full_system` as blocked.
3. Packet C read-only inspection runs once and returns `PASS` with
   `observation_age_s <= 5.0` at consumption, so runtime table grounding is measured rather than
   inferred.
4. A `status: ready` config with populated `components` and `calibration_evaluators`, plus the
   calibration evidence and approved candidate.
5. `run_worldmind_preflight` green: `split`, `state_source`, `fresh_output`, `config`,
   `backend_declaration`, `report_definitions`, `bindings`, `calibration`, `frozen_plan`,
   `runtime_geometry`.
6. Explicit run authorization for the frozen five-arm VALIDATION (`18` episodes, seeds `[0,1,2]`,
   `K=32`, `H=50`, `dt=0.032`), with a running `sim-server`.
7. Separate authorization for the sealed TEST window (`2026-12-05..07`); it is not implied by any
   of the above.

### 9.6 Earliest realistic first scored episode

Given the measured schedule: if A and B are signed on `2026-11-20`, B's implementation is already
in place and testable offline, C needs one authorized attempt, and the ready config plus
calibration evidence need owner review, the earliest realistic *first scored episode* is
`2026-11-25`. The two non-predicting arms (`scripted_baseline`, `vla_only`) need only items 2–6;
`vla_risk_veto` and `vla_world_model` additionally need the Eq. 3.8 numeric contract (item 1) and
stay at `2026-11-25` only if that is signed on `2026-11-20`, otherwise later. `full_system` stays
gated by `unverified-arm-sweep` and has no earliest date until Packet B or the narrower fallback is
signed. The full five-arm VALIDATION block needs the approved config; the TEST window is
`2026-12-05..07` and remains sealed.

### 9.7 Geometry investigation hard stop and `full_system` status (2026-11-19)

The geometric investigation is **closed**. Commit `77cefd4` and its eight-pair classification,
speed-bound verdict and check-verdict are accepted as **diagnostic evidence only**, not as
authorization to activate the proposed checker. In particular:

- `configs/worldmind_arm_sweep.yaml` stays `status: blocked`; no `workspace_separated` exclusions
  are activated and the pinned `speed_bound_m_s = 28.49316700274912 m/s` is not replaced. The
  derived full-chain candidate (`≈5.37 m/s`) and the repo-URDF bound (`≈2.677 m/s`) remain
  unapplied proposals.
- A sampled positive separation is **not** proof of separation throughout the joint workspace.
  §9.3 records `0 / 13,766` passing expert steps, no positive control in the recorded data, `8`
  retained narrow-clearance pairs that still veto recorded configurations, and `42 of 61` URDF
  collision bodies with no arm capsule. An operating-domain contract cannot be justified from
  this evidence, so no limited-domain boundary is asserted here and no exploratory search is
  started.
- **`full_system` is `blocked` pending the safety decision in this packet** (Packet B / §9).
  `production_worldmind_bindings()` keeps `components={}` and `calibration_evaluators={}`, and
  `full_system` continues to veto every predicted candidate with `unverified-arm-sweep` and record
  `NO_SAFE_SET`. This does not block engineering on the other four arms.

### 9.8 Scientific contract — implementation-ready proposal (not approved)

The scientific contract is now concrete and ready for owner approval, without fixing any numeric
parameter and without modifying a frozen threshold:

- `configs/report/objectives.proposal.json` gives Eq. 3.8 component formulas, units,
  normalization, ranking direction (`minimize`) and the `missing_data` / `safety_precedence`
  contracts for `Jg`, `Jl`, `Jr`, `Ju`. `Ju` is a **prediction-discrepancy proxy**, explicitly not
  calibrated uncertainty. Every component is `authoritative: false`; the file is
  `engineering_proposal_not_approved` with `scored_execution_enabled: false`. A new
  `calibration_evaluator` block specifies inputs (draft + hash-pinned approved VALIDATION
  evidence + registered callable), outputs (finite per-candidate scores, recommendation under
  `registered_evaluator_max_score_then_candidate_id`, labelled
  `selection_statistic_not_independent_evaluation`) and fail-closed conditions.
- `configs/report/definitions.candidate.json` gives all eight Table 5.1 metrics with explicit
  **denominators**, units marked `[PROPOSED; OWNER DECISION]`, a `failure_taxonomy` derived from
  `eval/metrics.py::derive_evaluation_failure_category`, and a `measurement_instrumentation`
  pointer to the sidecar fields. Proxy hits, contacts and unverified-arm-sweep remain separate
  channels; they are not merged into one risk number.
- Weights, scales/margins and ranking direction remain **VALIDATION-selected by the owner** and
  are not chosen here. The calibration proposal is a selection statistic, not an independent
  evaluation of post-calibration performance.

### 9.9 Bounded VALIDATION run plan and authorization scope

Frozen plan (unchanged): `validation` split, `base_seed=0`, seeds `[0,1,2]`, 18 unique `(scene,
seed)` episodes, `K=32`, `H=50`, `dt=0.032`, `scripted_chunk_adapter`, five arms. Bounds
`max_decisions`/`max_steps` stay owner-set placeholders in the ready config; the actual effective
bounds and per-decision timing enable flag are now recorded in `worldmind-sidecars/provenance.json`.

Run command (only after the ready config, green preflight and Packet D signature exist; connects
to an already-running `sim-server`, never launches Webots):

```
cd /home/tj330/programs/VLA-Project-week3
uv run python -m vlarobot.cli eval-worldmind \
  --config configs/worldmind_production.yaml --out runs/worldmind/<new-authorized-dir> \
  --backend webots --state-source oracle --split validation
```

Per-episode decisions, candidate evaluations, K+1 gate calls and phase latencies are measured by
`build_instrumentation()` and written to `worldmind-sidecars/instrumentation.json` and to
`summary.json.worldmind_instrumentation`; `measurement_class` is `measured` only for a real Webots
backend and `synthetic` otherwise. This makes the five-arm throughput figures checkable against a
real run instead of the nominal-kernel inference in §7. Stop conditions, proxy labelling and
fail-closed preflight are unchanged (§8.4).

Authorization: **ADR-0030 is scenario design only and grants no scored-run authorization.** No
scored run is authorized by any existing signature. The only execution currently requestable is
the bounded read-only v4 inspection of §8.3 (Packet C), with its exact command and acceptance
criteria. Any scored VALIDATION requires a new, explicit Packet D signature (all five arms or a
named subset). A partial campaign requires its own explicit approval and **must be reported as a
partial campaign**, preserving the five-arm design as the reference protocol. TEST stays sealed
until `2026-12-05..07` and requires a separate authorization; no combination of A/B/C/D grants it.
