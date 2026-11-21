# WorldMind V5b postmortem correction implementation plan addendum v1

- **Document version:** 1.0 (2026-11-21).
- **Authority:** separately owner-approved, offline-only implementation of the read-only V5b
  postmortem findings A-E. This addendum is diagnostic bookkeeping only; it is not a new campaign
  authorization, a scientific result, a performance claim or a change to the frozen Stage 1 plan.
- **Decision record:** `docs/DECISIONS.md` ADR-0079.
- **Scope boundary:** no Webots process, live campaign, V5c execution, scoring, calibration
  selection, `full_system`, Packet B/D, TRAIN/TEST access, geometry change, push or merge. V1-V5b
  evidence directories are read-only and are never resumed or rewritten.

## 1. Preserved contract

Every value below remains byte-identical and behaviourally unchanged:

- K=32 candidates, H=50 prediction horizon, `control_dt_s=0.032`, `max_steps=1059`;
- frozen scene/seed plan and frozen plan provenance hash
  `013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2`;
- candidate generation, selected candidate identity, immutable selected action payloads and
  scripted waypoint order/cadence;
- IK tolerance semantics, the reviewed `0.15 rad` execution bound, `SafetyMonitor`,
  `PlacementVerifier`, terminal verification and `retry_once → abort`;
- all scoring/calibration/report definitions and V1-V5b evidence trees;
- the non-comparability of observed-state feedback with the immutable H=50 prediction.

No new numerical parameter is introduced. Gripper arrival tolerance and timeout steps are derived
only from existing reviewed values in `configs/robot.yaml` and `configs/eval.yaml`:
`finger_speed=0.03 m/s`, `control_dt_s=0.032 s`, `gripper_open_timeout_s=4.0 s`,
`gripper_opening_limits=[0.0, 0.045] m`, giving `0.00096 m` arrival tolerance and `125` steps.

## 2. Correction A — deterministic reachability precheck

ADR-0080 supersedes this section. The original V5b archive records the exception and observed
state available in schema 1. Its failed draw target and numerical residual were not independently
reproduced from that archive. The residual magnitudes in the offline regression are injected
fixture values.

- Run before candidate generation for the explicit feedback mode only, using the already
  re-anchored observed waypoints and fixed solver seed `0`.
- Reuse the adapter's reviewed per-waypoint IK tolerances (or adapter-level tolerances when a
  waypoint omits them) exactly. Do not clamp, substitute, repair or silently modify an endpoint.
- A finite residual over tolerance emits a versioned, structured `unreachable_observed_target`
  event in `diagnostic-events.jsonl` with episode/decision/waypoint identifiers, measured residual,
  exact tolerances, `candidate_generation_performed=false` and `backend_actions_applied=0`; the
  campaign then reports the episode as `UNREACHABLE_OBSERVED_TARGET`.
- A solver exception or non-finite residual remains a true exception and still fails closed.

## 3. Correction B — configured HOME completion phase

- The selected chunk remains immutable. Selected HOME actions keep their original payloads and the
  reviewed `configs/robot.yaml:home_settle_tolerance_rad` error accounting.
- After selected-chunk convergence only, a separately labelled `diagnostic_home_completion` phase
  drives the configured `configs/worldmind_diagnostic.approved.yaml:verification.home_joints`
  vector with the approved `home_tolerance_rad=0.02 rad`.
- Every intermediate HOME command is projected from observation, passed through the unchanged
  `SafetyMonitor`, applied only when accepted, and counted in the same global 1059-step ledger.
  A distinct diagnostic event records its source, initial/final error, terminal reason and step
  consumption. Its errors are never mixed with selected HOME action errors.
- A HOME completion failure maps to the episode abort state distinctly; no prediction pairing is
  created for this phase.

## 4. Correction C — explicit convergence classification

- Preserve each final selected-action error exactly; do not rewrite it to a tolerance pass.
- Classify actions as `target_converged`, `global_step_ceiling`, `joint_move_timeout`,
  `joint_stall`, `safety_rejected`, `backend_apply_rejected` or `execution_exception`.
- Persist a convergence summary: converged indices/count and max converged error are computed only
  from converged actions; non-converged indices, reason counts and final errors remain separate.
- Episode termination still preserves the precedence of safety/backend failure, global ceiling,
  applicable timeout, stall and convergence.

## 5. Correction D — proxy attribution and observed gripper completion

- Add version-1 `collision-proxy-attribution.jsonl` with per-object max/final displacement and
  tilt, per-object threshold hits, the threshold source and backend-recorded contact steps. The
  displacement/tilt proxy is never promoted to a verified contact, grasp, release or swept-arm
  certificate.
- Gripper completion uses the reviewed finger speed, opening limits, control tick and timeout.
  The opt-in rate-limited FakeBackend models observable opening progression; default FakeBackend
  behaviour is unchanged.
- Every gripper apply is safety validated and apply validated. The completion error, unit, target,
  terminal reason and arrival/timeout budgets are recorded explicitly. No grasp or release claim
  is made.

## 6. Correction E — full observed base pose evidence

- Every attempt snapshot records the complete observed `base_pose_world` position and quaternion;
  a missing transform is explicitly `null`, never invented.
- The feedback schema is bumped from 1 to 2. Schema 1 rows remain readable verbatim; schema 2 adds
  the HOME completion phase, convergence summary, explicit completion fields and full pose. No
  frozen scored metric, report definition or comparison with the H=50 prediction is changed.

## 7. Versioned diagnostic evidence

| Channel | File | Schema | Scored? |
| --- | --- | --- | --- |
| Feedback actions/attempts | `observed-state-feedback.jsonl` | 2 (schema 1 readable) | No |
| Explicit diagnostic events | `diagnostic-events.jsonl` | 1 | No |
| Per-object collision proxy | `collision-proxy-attribution.jsonl` | 1 | No |

All three are emitted only for the explicit `diagnostic_observed_state_feedback_chunk` mode, are
non-comparable with the frozen prediction, and emit no `prediction_pairing.jsonl` record.

## 8. Verification plan and acceptance

Offline verification must include at least 11 deterministic A-E regressions, the existing feedback
regressions, Ruff check/format, `git diff --check`, the full `not webots` suite and a fresh
recomputation of all preserved V1-V5b tree hashes and frozen configuration/source hashes.
Exact acceptance output is recorded in ADR-0079. Windows, Webots timing and physical response are
not verified by this addendum.

## 9. GO / NO-GO

- **NO-GO** for a V5c live or scored campaign, scoring, calibration selection, `full_system`,
  retrying/resuming V5 or V5b, production use, or any claim of physical task success.
- **CONDITIONAL GO**, only after this addendum's acceptance passes, for a separately authorized,
  non-scored VALIDATION diagnostic that explicitly selects `scripted_proposer` plus
  `diagnostic_observed_state_feedback_chunk`, writes to a fresh versioned output directory and
  treats every feedback row as diagnostic and non-comparable.
