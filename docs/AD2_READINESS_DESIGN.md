# A-D2 readiness design — preparation only

**Decision status: NO-GO for implementation or execution.** This document is a design,
not an executable harness. No Webots/model server, inference, IK solve, actuator command,
training or scored evaluation was run. Only existing TRAIN evidence and source were read.
The frozen **0/24, gate FAILED, scripted-primary** decision remains final. The repaired
23/24 baseline, checkpoint, processor, dataset, splits, seeds, five arms and December12
deadline remain unchanged. A-D2 would be a separately authorized privileged diagnostic,
not a gate retake or an additional evaluation arm.

Labels below distinguish **FACT** (source or saved measurement), **HYPOTHESIS** (untested
explanation), and **PROPOSAL** (requires the approvals in section9).

## 1. A-D1 evidence and immutable inputs

**FACT.** [A-D1 report](AD1_NUMERICAL_REPORT.md), commit `6db9573`, used exactly two CPU
IK solves on matched episode0/frame0, source attempt0000, `train-clear-i00-s100`, seed100.
A is the recorded expert hold reference; B is the saved IK result for its registered FK
pose; C is the saved IK result for learned action0. B returned exactly A. C requested
0.670555mm translation and0.005184768rad rotation, yielding0.2122484414rad joint L2 change.
Its maximum joint change is0.15rad, with another joint nearly at that trust bound.
Task/null components are0.206236/0.050159rad under the saved local Jacobian analysis.
C passed the existing residual and safety checks. These are numerical results, not
measurements of motor response or causes of the failed HOME episodes.

A was an initialized hold reference, not an independently logged frame0 JointTarget apply.
The reset motor targets are different from A; do not mistake either for measured q0.
A and B are now identical commands, so their physical comparison is a negative control
for fresh-session variability, not a new test of an IK implementation.

**PROPOSAL.** Planned identity `AD2-train-clear-i00-s100-frame0-v1`; new output root
`runs/diagnostics/ad2-train-clear-i00-s100-frame0-v1/` in the isolated worktree. It must
not pre-exist. No output run directory has been created. Do not overwrite evidence.

Load full-precision arm maps from A-D1 `result.json:cases.A.joints`,
`case-B.json:joints`, `case-C.json:joints`, respectively. Require B/C to equal their
copies in `result.json`; no solver, model call, adapter decode, target reconstruction,
rounding, projection or recomputation of these maps. Human-readable values:

| Joint | A = B (rad) | C (rad) |
|---|---:|---:|
| arm_1_joint |0.07103975613359034 |0.07000000000000002 |
| arm_2_joint |−0.000056330103023860545 |−0.0015776230243764139 |
| arm_3_joint |−0.00002289653672216385 |−0.15002289653672213 |
| arm_4_joint |0.000021903720995360823 |−0.002329658044912425 |
| arm_5_joint |−0.00000550120551103984 |−0.15000548310097536 |
| arm_6_joint |0.0000018481290345948952 |0.0061974754183294345 |
| arm_7_joint |−0.000001680523213603215 |−0.0014491773677773425 |

Evidence roots (existing; read-only):

- A-D1: `runs/ad1-train-clear-i00-s100-frame0-v1/` in the analytic worktree.
- Original source: `/home/tj330/programs/VLA-Project/runs/demos/train-s100-registered-v1/attempts/attempt-0000/`.
- Scene: that attempt's `episode-00.meta.json:scenario`, verbatim; seed100, clear TRAIN,
  parked APP/HAM. This is the approved demonstration snapshot, not evaluation seed0.
- Source `episode-00.steps.npz` row0, `policy.jsonl` line1 and
  `episode-00.physical.jsonl` line1 identify the reference instant.
- Saved model action0 and matching observation/export provenance stay in A-D1; only the
  saved arm solution C would be applied. Neither the expert open-gripper action nor C's
  legal0.04490866512m gripper prediction is applied in this arm-only diagnostic.

The [design source inventory](evidence/ad2-design-source-inventory-v1.json) pins sizes,
SHA-256s, exact maps and selected reference records. A future harness must also verify
A-D1's complete input provenance and original frozen campaign inventory before/after;
this design inventory does not replace either. The report PDF and inspected primary
runtime source are pinned separately there. Any drift is a stop, not permission to repair.

## 2. Recovered authoritative report and conformance

**FACT: the missing source was located.**
[Project_Report.pdf](/home/tj330/Downloads/Project_Report.pdf),70 PDF pages,
24,118,576bytes, SHA-256
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`.
Cover: *WorldMind: A World Model-Based Framework for Intelligent Robot Simulation*,
October2026; certificate explicitly identifies the Project Phase I report. Authors:
Rahul Ajith, Sridev K, Titus James, Varun Sadanand. Local report/download directories and
project references were searched; there were no connected document sessions. The similarly
named seminar submission is a different document. No PDF was copied into or altered in
this repository. Earlier statements that the source was unavailable are historical.

The identity matches the requested report, but there is no previously frozen submitted-PDF
hash to independently certify this exact revision. Owner confirmation of this recovered
revision is needed before claiming final contractual conformance. Page numbers below are
**PDF pages**, not the report's printed numbering.

| Authoritative provision | Comparison with ADR-0029 / A-D2 |
|---|---|
| §3.1.1–3.1.2, pp25–26: feasible control, monitored outcomes, repeatability, safety, reliability and traceability | Supports isolated diagnostics with recorded settings and actual outcomes; forbids treating invalid observations/rejection as success. Supplies no reset tolerances. |
| §3.3.3, p30; §3.6.1.5, p37; §4.7, p48: supported controller interface, movement-limit checks, measured execution/deviation | Saved JointTarget replay through the existing safety/backend path fits this purpose. Does not prove one tick safe or physically attainable. |
| §4.3, p46: distinguish direct simulator state from visually estimated perception | Permits prototype simulator-state provenance; A-D2 truth is explicitly PRIVILEGED DIAGNOSTIC only. It is not new authorization for oracle policy inputs. |
| §4.8–4.9, pp49–50: comparable state variables/units, initial state, chosen trajectory, outcome and timing records; integration tests do not prove performance | Requires frame/unit-labelled pre/post states, command/verdict/ack, and limitations. No scored success claim from three pulses. |
| §5.4–5.6, pp54–55; Table5.1, p55 (printed44) | Eight metrics: task success, prediction error, collision rate, path length, execution time, planning latency, risk rejection rate, reliability. A-D2's motor response is supplementary diagnostic evidence, not replacement task metrics or a gate result. |
| §6.1, p57: original12-week schedule | ADR-0029 explicitly supersedes the timeline with four weeks and December12. A-D2 does not change it. |
| Eq3.2 / Eq3.8 and ADR-0029 | Existing declared propose-and-evaluate deviation and hard-veto-before-ranking contract stay unchanged. No proposer, risk ranking, surrogate or retry loop is introduced by this bench. |

**Conformance boundary.** No relevant report section specifies numerical reset match bounds,
32ms pulse authorization, gripper endpoint changes, IK trust changes or a snapshot restore
method. The report supports the diagnostic purpose, but does not fill these decisions.
Exact submitted-revision identity, numerical matching criteria, reference-state omissions
and physical collision coverage cannot yet be certified. No report silence is treated as
approval for behavior changes. Architecture and ADR-0029 remain controlling implementation
constraints; no public interface change is proposed here.

## 3. Exact proposed protocol — three sessions, one measurement tick each

**PROPOSAL; not an execution instruction.** After separate implementation review and launch
approval, use fixed order A, B, C, one fresh Webots process/session each. No model server,
scripted state machine, evaluation runner, candidate generator or automatic retry.

1. Offline preflight: verify the sealed A-D1/snapshot/source/config/URDF/registration hashes;
   assert A=B, target names/values exact, saved acceptance/residual evidence present, scenario
   TRAIN identity exact, output absent. Pin Python/dependencies, Webots executable/version,
   world and all included PROTO/assets/controller imports, effective environment and settings.
   Reject unexpected startup options, simulator randomization, control clients or plugins.
2. Start one session via `launch_webots_session` with a unique port, PID and nonce, ordinary
   rendering, original `worlds/factory.wbt` and the approved runtime. Reset once with the
   saved ScenarioSpec and seed100. Seed100 does not itself certify physics determinism.
   Record **every preparatory simulator tick**, including constructor/reset settling.
3. Treat the returned reset observation as the candidate pre-command frame0. Capture a
   same-instant diagnostic snapshot without stepping or mutating velocity history. Require
   all approved reference and between-case matching gates, complete required fields and
   collision preflight to pass. No extra settling, teleportation, conditioning command,
   resampling or reset retry to manufacture a match. Abort on mismatch.
4. Read A/B/C's saved arm map. Run unchanged `SafetyMonitor.validate(JointTarget, state)`
   with actual current state and pinned effective limits. Preserve existing saved pose/
   gripper/residual checks as provenance; the offline IK acceptance does not replace live
   checks. Independently confirm the saved pose's applicable workspace checks without
   sending any EEPoseTarget. Never use EEPoseTarget/EEDelta apply: those invoke another IK.
   Record live backend validation and full ApplyResult. No gripper/torso/base/head command.
5. Only after accepted apply, send **one** `step(1)`. Confirm acknowledged simulator-time
   increment32ms and exactly one underlying physics tick. Capture returned q1/velocity and
   an atomic same-instant diagnostic snapshot. No second tick, convergence wait, HOME,
   release, hold extension or success evaluation. Do not infer tick count from wall time.
6. Persist all evidence, mark completion/rejection/ambiguity, then stop that simulator and
   prove process-tree cleanup before starting the next case. Shutdown must not create an
   additional observed/commanded tick; record any unavoidable shutdown progression separately
   as a protocol deviation. Freeze and hash artifacts; do not append to completed cases.

**Tick interpretation needs approval.** Existing initialization/reset necessarily advances
many ticks. This proposal means **one post-reset measurement tick**, not one total lifetime
simulator tick. If the latter was intended, the existing reset is incompatible: stop rather
than alter it. No diagnostic pulse occurs on a failed preflight. Safety rejection is logged
without stepping or shrinking the target; stop the sequence for review. The maximum budget
is three fresh sessions, not three guaranteed accepted pulses. No replacement sessions.

Timeout/disconnect after apply or step is ambiguous execution, never a license to resend.
`ZmqBackendClient._request` resets its poisoned socket and raises; it does not resend the
operation. Preserve nonce/sequence/last ack, stop and report. No resume into a used session.

## 4. Reset matching: recorded evidence versus missing decisions

**FACT.** `WebotsBackend.reset` restores startup joint fields/motor targets, base/CAN and
scenario objects, clears physics and settles torso/head before returning an observation.
It does not restore arbitrary snapshot positions and velocities as a complete state.
Its fingerprint is taken **before the final reset tick**; frame0 is after that tick.
`physics_step_count_at_reset=0` therefore does not mean zero preparation ticks.

Reference torso is **0.3497446130613471m**, not its0.35m target. Frame0 arm velocities are
nonzero: arm1 about0.00464767054rad/s and arm3 about−0.00271683520rad/s (full vector remains
in NPZ). Zero velocity must not be substituted. Frame0 base speed norms are
0.00002611530867254974m/s and0.000036844188467267185rad/s. Earlier fingerprint vectors
are not the frame0 twist. Individual frame0 finger sensor values are
−0.0000010779306241058846m and0.0000017950385083258884m; measured opening is
0.0000014364845662158865m. These are feedback, not requested gripper targets.

| Quantity / proposed metric | Reference and required live evidence | Missing approval or data |
|---|---|---|
| Arm q0 and qdot0 | NPZ row0 in named order; per-joint absolute differences and vector L2, frame0 timestamp | Per-joint/vector bounds; preserve nonzero reference velocities. |
| Torso position/velocity | NPZ row0, same prismatic SI units; separately log0.35m motor target | Position/velocity bounds. Existing0.002m torso-settle tolerance is relative to configuration, not an A-D2 match bound. |
| Base SE(3) / twist | Policy line1 full xyz/quaternion; compare translation and relative-rotation angle, not only xy/yaw. Live Supervisor full twist plus reference speed norms | Pose/norm bounds; frame0 full velocity vectors unavailable. Earlier fingerprint twist cannot substitute. |
| CAN/APP/HAM SE(3) / twist | NPZ row0 gt_xyz/gt_quat and gt_names, scenario identity; live world-frame poses and linear/angular velocities | Pose/velocity bounds; no frame0 object velocities in inspected recordings. Physics reset does not prove zero after settling. |
| Fingers and hold | Physical line1 unrounded sensors/pad positions and measured opening; live left/right sensors separately, motor target/mode/speed/force and contact state | Bounds; force/PID reference incomplete. Earlier reset motor targets must be checked as earlier metadata, not asserted frame0 measurements. |
| Other motors/settings | Earlier reset fingerprint: head/torso/wheel targets/modes/speeds; frame0 wheel diagnostics | Exact-vs-numeric matching rules and missing force/PID/velocity reference fields. No silently changed wheel hold or head conditioning. |
| Reference vs across cases | Each case against recorded frame0; B/C also against all previous case starts | Approve both bounds: separate reference comparisons alone can permit twice the tolerance across cases. |

All required numerical criteria remain **UNSET**, hence fail closed. Required categorical
fields (scene, seed, names, registration/hash, settings, hold mode, session uniqueness,
target maps) must match exactly. No generic `allclose` default or tolerance borrowed from
HOME, residual acceptance, float serialization or old reset-settle settings.

**PROPOSAL / owner choice.** The existing records cannot certify a full dynamic-state
match. Either accept an explicitly weaker, bounded-observability diagnostic (same source
snapshot, measured positions/nonzero joint velocities, available base speed norms, complete
live cross-case twists, and disclosed missing historic twists/settings), or keep NO-GO.
Do not invent missing reference data, capture a fourth reference session, silently choose
another frame, or claim exact simulator-state restoration. New snapshot generation or
reference-conditioning would be a separately scoped protocol, not this A-D2.

## 5. Safety and privilege assessment

**FACT.** Live JointTarget validation checks finite/known joints, motor position limits and
`abs(q_target−q_current)/command_horizon` against velocity limits. Its existing horizon is
**1s**, not32ms. Preserve it. A0.15rad target difference is not a promised one-tick motion.
Log configured/actual motor velocity caps, force/torque caps, acceleration, PID/control
mode where available, and sensor tracking; do not increase them to reach the setpoint.

Gripper limits remain0–0.045m and force cap100N; individual measured prismatic sensor values
are not commands. Retain reset's gripper hold motor targets/modes unchanged. Confirm them
before and after each pulse, with no command replay of the expert open endpoint. Legacy
`*_rad` field names for torso/fingers must be labelled metres in diagnostic records.

JointTarget safety does **not** apply the EEPoseTarget workspace branch or certify obstacle,
self-collision, swept volume or table clearance. The saved residual/workspace acceptance
and a legal joint vector do not establish physical safety. Current displacement/tilt and
sparse FK clearance diagnostics are proxies, not geometry certification. A-D2 must not
inherit a general collision-safe claim from prior unit tests or partial link coverage.

**PROPOSAL.** Require a separately reviewed collision preflight for this exact starting
geometry and all three immutable target motions: actual Webots/URDF link/obstacle coverage,
self-collision exclusions, table/fingers/object bounds and conservative swept-volume
coverage under unchanged motor limits. Record geometry provenance, distances/contact pairs
and coverage gaps; missing required geometry fails closed. Required clearance criterion
and acceptable coverage are unresolved, not invented here. Detect unexpected contact,
penetration, non-finite sensor data, motor rejection or target mismatch; record and stop.
Post-tick contact evidence cannot retroactively make a risky command safe. No RiskGate,
heuristic fallback or world-model policy is inserted to rescue a case.

Ground-truth base/object poses/velocities, per-link/pad poses and contact information are
**PRIVILEGED DIAGNOSTIC** reset/safety/measurement evidence only. Commands come exclusively
from the three sealed joint maps. Truth must not produce, repair or rank commands. A safety
stop is permitted; an oracle target adjustment is prohibited. No model observation is built.

## 6. Existing APIs and isolated implementation plan (not implemented)

The inventory pins the actual primary-runtime source below; line references are review
anchors, not instructions to execute anything now.

| Existing component | Reuse / limitation |
|---|---|
| [episode_runner.py:650](/home/tj330/programs/VLA-Project/src/vlarobot/execution/episode_runner.py:650), `launch_webots_session` | Unique port/process/logs, startup readiness and finally cleanup. Use launcher only, not scored EpisodeRunner. Review inherited controller environment and absence of autonomous stepping. |
| [backend.py:129](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/backend.py:129), `WebotsBackend.reset` | Soft reset in each fresh process; torso/head settling and finalization ticks. Do not enable new settling settings or use kinematics export/probes. |
| [robot_io.py:507](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/robot_io.py:507), `set_joint_positions`; `reset_fingerprint`:368 | Reset writes fields AND motor targets; fingerprint has raw sensors and hold settings, but timing predates frame0. Not a complete snapshot restore API. |
| [backend.py:262](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/backend.py:262), `apply(JointTarget)` | Existing safety check then seven motor setPosition calls, no IK/no tick. Never apply pose/delta commands. |
| [backend.py:251](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/backend.py:251), `step(1)`; [visualization.py:53](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/visualization.py:53) | One underlying robot.step; inspect active bridge callbacks for side effects before launch approval. |
| [backend.py:635](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/backend.py:635), `_robot_state` | Registered FK, joint finite-difference velocities, diagnostics. Repeated calls at the same timestamp overwrite velocity history and can report zeros. Preserve returned observation; extra instrumentation must not call this again. |
| [backend.py:393](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/backend.py:393), `get_ground_truth` | Object poses only; no complete object velocities or geometry/contact certification. No new fields in frozen core GroundTruth. |
| [safety_monitor.py:88](/home/tj330/programs/VLA-Project/src/vlarobot/execution/safety_monitor.py:88), `validate` | Unchanged validators/effective limits plus backend's own actual-current-state check; log both. |
| [client.py:174](/home/tj330/programs/VLA-Project/src/vlarobot/backends/zmq/client.py:174), `_request`; server dispatch | Existing nonce/sequence, acknowledgement and timeout behavior; no automatic resend. Treat ambiguous operations as terminal. |

Registration stays `configs/calibration/urdf-base-train-v1.yaml` SHA
`dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0`;
URDF SHA `1a6de9920b241f8d2fe6ca675e73137cd4a3daced88b67764669825aaa1b4e1b`.
Frozen time_step32ms, IK trust0.15rad and saved residual limits0.005m /0.0872664626rad
remain evidence, not parameters to tune. Use actual effective frozen motor/safety settings,
not default SafetyLimits constructed without their runtime configuration.

**PROPOSAL.** Implement later on an isolated diagnostic branch/worktree: pure-Python
artifact loader and admission checks, fixed A/B/C coordinator, plus debug-only Webots-side
instrumentation under `backends/` for same-instant velocity/contact/motor snapshots and raw
tick counting. Keep Supervisor imports there. If the existing RPC cannot expose required
fields, use a separate namespaced diagnostic endpoint/wrapper, not changed public core
interfaces or frozen controller/server/evaluation behavior. Read-only instrumentation must
not call step, solve, motor setters, reset or state-cache-mutating observation methods.
Its exact design needs review; public APIs alone are insufficient today. No automatic
calibration, probing, model initialization or solver calls may occur during startup.

Before physical authorization, future CPU-only tests should prove immutable-target loading,
provenance rejection, no retry on ambiguous apply/step, exactly one accepted step request,
no step on rejected commands, abort on unavailable/mismatched fields, unchanged gripper
state, session isolation and cleanup. Test fakes are not collision or physics certification.
Review tests/instrumentation statically for any implicit solve or inference. Such code and
tests are deliberately **not implemented or executed** in this preparation task.

## 7. Required artifacts, measurements and limitations

Planned files: top-level `protocol.json` (approved criteria/version), `provenance.json`,
`preflight.json`, target maps and source hashes; per-case `session.json`, `before.json`,
`command.json`, `safety.json`, `apply-result.json`, `after.json`, wire/event/tick ledger,
Webots/controller stdout/stderr, timing and cleanup records; aggregate `measurements.json`,
`limitations.md`, SHA-256 inventory. Frame0 source values and earlier fingerprint must be
separate records. Raw non-finite/unavailable values require explicit tags, not default zero.
Preserve artifacts even if startup, reset, validation, pulse, logging or cleanup fails.

Before/after: absolute/relative simulation time, raw tick count, PID/nonce/sequence; all
joint sensors and named velocity estimates with timestamp/estimation interval; base SE(3)
and twist; CAN/APP/HAM SE(3)/twists; individual fingers/pads and held motor settings; arm
motor target/caps/feedback; contact identities/geometry status. Retain source target and
saved pose matrices verbatim. Record registration and frame transforms for every FK/truth
comparison. FK evaluation for measured q1 would be permitted in later authorized
instrumentation; it is not an IK solve and is not performed in this design task.

Report per-joint `q1−q0`, `q_target−q0`, `q_target−q1`, L2/max norms, signed changes and
`(q1−q0)/0.032s` separately from backend velocity estimates. Compare registered FK(q1) to
saved target/achieved FK and privileged endpoint motion with matching tool references;
arm7 endpoint and pad midpoint are different references. Label translation metres and
rotation-angle radians; do not compare mismatched frames or call target residual motor
error. Reuse A-D1 Jacobian metrics as historical numerical evidence, not a new Jacobian/IK
calculation or a physical null-space classification after one tick.

**HYPOTHESES tested only narrowly:** learned-target sensitivity may create a different
initial arm motion; motor limits/inertia may suppress or redirect it; session variation may
mask it. A/B's identical command exposes inconsistency, but two sessions do not estimate a
reproducibility distribution. C differs from B in learned target AND resulting joint
command; one tick cannot attribute learning quality separately from IK branch selection.
No settling/convergence, grasp, placement, HOME-causation, reliability or performance
improvement claim follows. An inconclusive result ends this protocol; longer holds,
replication or further measurements require separate authorization.

## 8. GO / NO-GO checklist

| Gate | Status now |
|---|---|
| A-D1 completed, immutable saved targets and TRAIN snapshot identified | PASS, source inspection only |
| Relevant October2026 Phase I report recovered and read | PASS; exact submitted revision requires confirmation |
| Frozen result/gate/scripted-primary and all limits preserved | PASS |
| Reference dynamic-state completeness | BLOCKED: historic object twists/full base twist and some motor settings absent |
| Numerical reference and pairwise reset criteria approved | BLOCKED: none supplied; no substitutes proposed as approved |
| One measurement tick versus preparatory ticks explicitly accepted | BLOCKED pending owner decision |
| Exact-scene collision/swept-volume coverage and preflight criterion | BLOCKED; no full certification established |
| Non-mutating instrumentation, bridge/tick review and isolated harness verified | NOT IMPLEMENTED, intentionally |
| Separate A-D2 implementation and physical execution authorization | NOT GRANTED |
| Webots/model/IK/actuator/evaluation executed in this task | NONE |

**Overall NO-GO.** The design is ready for an owner decision, not ready to launch.
Full project lint/test workflows were not run: this documentation-only task permits only
non-executing checks, and solver/simulator-capable tests would exceed its scope. Markdown
whitespace, saved target equality, file hashes and PDF identity were checked without
importing project runtime. Windows/Linux runtime behavior remains untested for A-D2.

## 9. Minimal owner decisions before any next stage

1. Confirm the recovered PDF/hash is the controlling submitted Phase I revision.
2. Approve explicit matching metrics **and numerical bounds** for each row in section4,
   including reference and pairwise comparisons; decide whether the documented missing
   historic twists/settings make this NO-GO or permit a weaker diagnostic with those limits.
   No new reference session or inferred zero-state is implicitly authorized.
3. Confirm one32ms **post-reset measurement tick** per case, with existing startup/reset
   ticks recorded but excluded; approve the exact-scene collision coverage/clearance
   preflight requirements. Missing coverage must remain a stop, not a safety waiver.
4. Separately authorize isolated harness/instrumentation implementation and review. Only
   after its evidence is accepted, authorize execution of this named three-session protocol
   with immutable A/B/C, identical gripper hold and the stated abort/no-retry rules.

No action is requested to change the trained policy, thresholds, dataset or frozen campaign.
Preparation stops here awaiting the owner's decision.
