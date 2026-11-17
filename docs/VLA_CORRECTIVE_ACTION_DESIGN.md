# Corrective-action assessment — design only, awaiting owner decisions

No proposal below is implemented or authorized for execution. No model forward, numerical
IK experiment, simulator, training, projection, new evaluation or withheld-split access was
performed. The recovered checkpoint, processor, approved dataset and original campaign
remain immutable inputs. **0/24, gate FAILED and scripted-primary remain final.**

The Phase I report is absent. Under [ADR-0029](/home/tj330/programs/VLA-Project-analytic/docs/DECISIONS.md:1305),
its contents cannot be inferred from implementation. Any affected behavioral/model/termination
change is **conformance-blocked until the report is available and reviewed**, in addition to
requiring explicit owner approval. Design permission is not implementation, training, compute,
physical-experiment or evaluation permission. No proposal automatically becomes a sixth arm,
a replacement frozen VLA-only arm, a gate retake or a change to the scripted-primary decision.

## Evidence and uncertainty carried forward

Accepted source: [85f744d postmortem](VLA_TRAIN_POSTMORTEM.md) and its
[numeric evidence](evidence/vla-train-postmortem-v1.json).

| ID | Observed evidence | Supported conclusion | Still hypothesis/unverified |
|---|---|---|---|
| E1 |761/1200 proposals exceed the upper bound;21 executed-prefix vetoes; labels only0/0.045m; processor statistics match |Current unconstrained proposal path does not enforce the physical bound; no demonstrated unit/processor mismatch |Which learning/sampling effects cause the exact errors; performance of any proposed bound |
| E2 |HOME after4/7 actions at0.02rad departure/return crossings; EE movement below0.334mm; no manipulation success |Current termination can stop after a small excursion unrelated to task completion; final scoring correctly fails |How much learned error, IK branch choice and motor tracking each contribute |
| E3 |All80 first expert pose labels are exactly zero; predicted first pose residuals are0.104–0.863mm and0.000562–0.006238rad |Initial hold is imperfectly imitated; inverse pose-delta convention remains consistent |Whether IK amplifies those small errors, or tracking dominates |
| E4 |All4000 initial50 training labels keep gripper open; first close205–531 frames; all evaluated episodes stop before a second chunk |The recorded chunks do not test later grasp/lift/transport/release competence |Global undertraining or failure to learn closure cannot be concluded |
| E5 |Initial state nearly matches training starts; finite scene-coverage gaps exist |No observed gross initial proprioceptive shift; support coverage is incomplete |Causal visual/domain shift and convergence explanations are unproven |

Keep original observations, algebraically inferred normalized coordinates and hypotheses
separate. Original raw normalized inference tensors were not recorded; no design assumes
those hidden values are available. Proposed mechanisms below imply no measured success gain.

Version terminology: **P** = scientific policy/control/termination protocol; **M** = weights/
architecture; **R** = processor/runtime policy bundle; **D-view** = derived training targets,
not rewritten native data; **D** = dataset content/schema; **Run** = new measurement output.
A P version is not automatically a core wire-version change. If a design changes public core
interfaces, the architecture/DECISIONS update and interface tests remain mandatory.

## 1. Bounded gripper-output decision matrix

Invariant external action remains `[dx,dy,dz,drx,dry,drz,g]`, with **g an absolute
per-finger opening in metres**, G=0.045m. SafetyMonitor stays unchanged; every candidate
still passes finite/shape, opening, workspace, IK, joint and motor checks. The existing exact
float32 endpoint canonicalization is not broadened. Nonfinite proposals are errors, not
values to saturate. Pose coordinates and state normalization are not silently changed.

| Proposal | Expected mechanism and compatibility | Retraining / semantic change | Required versions and approval |
|---|---|---|---|
| G-C: learned bounded continuous g |A separately designed gripper generator could emit `g=G·sigmoid(u)` or another bounded distribution, then assemble physical7-D actions. Original physical labels can be reused, but they are not logits. Six pose outputs retain their meaning; processor must explicitly bridge mixed physical/normalized outputs |New training/fine-tuning is required for a learned parameterization. Not drop-in reuse of existing weights. External units stay the same; internal generation/loss changes |New P,M,R and training-config identity; D-view if targets transformed; native D need not change. New Run for any later measurement. Approval A-GC plus separately approved training plan/compute and evaluation |
| G-D: discrete close/open prediction |Predict class logits per action, map selected class to0 orG in physical metres. Existing labels are exactly binary apart from known float32 representation, so a versioned deterministic class-target view can use the same immutable dataset. Pose action remains continuous |New classification head/loss and training required. Restricts effective gripper command set; changes proposal semantics even though external7-D schema remains. Loses intermediate-width expressiveness |New P,M,R,D-view; native D unchanged unless materialized as a new dataset. Explicit class mapping, tie/nonfinite behavior and any temporal rule require A-GD; then separate training/compute/evaluation approval |
| G-P: deterministic physical projection |After the existing inverse normalization and finite checks, an explicitly versioned transform could compute `g'=min(G,max(0,g))`, leaving all other coordinates unchanged. Existing labels, weights and statistics can remain byte-identical |No retraining inherently required. **This is clipping and a changed effective policy.** Raw invalid outputs would no longer produce the original SAFETY outcome, so it conflicts with current steering despite unchanged limits |New P and R/wrapper manifest; no new weight M or D required. Any new Run remains a variant, never the original campaign. A-GP must explicitly rescind the no-clipping/output-rejection requirement for that named variant after conformance review |

**E1 supports considering all three; it does not choose a winner.** Current normalized
open coordinate is approximately0.75687036, not1.0. Projecting normalized values to[0,1]
would therefore be a representation error. A new physical head cannot pass g through the
old inverse normalization a second time. Keeping a7-element array alone does not establish
processor compatibility.

The existing SmolVLA linear head predicts **flow velocity**, trained against velocity MSE:
[modeling_smolvla.py:790](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py:790).
Putting a sigmoid on that head does not simply bound the final action and would alter the
flow dynamics. G-C needs a complete bounded generative/head design, not a one-line head patch.
A finite-logit sigmoid also does not attain the exact endpoint labels0/G. An endpoint-aware
likelihood or declared target transform would be a new training convention; no epsilon,
loss weight or target adjustment is selected here. Final floating-point physical values,
including0 andG, must still be tested against the unchanged predicate.

G-D could use a predeclared argmax with ambiguous/tied or nonfinite logits rejected, rather
than silently introducing a tuned probability cutoff. Any cutoff, smoothing, hysteresis,
minimum dwell or previous-state hold is another policy decision needing approval. Majority
open labels and very late/brief closure phases could bias a classifier; state prediction is
not grasp confirmation. Intermediate-width control may be useful beyond this dataset, so
binary support here does not prove discrete control is generally sufficient.

G-P is the lowest training cost, but changes the most immediate interpretation of a raw
violation. It may conceal poor conditional predictions, cause saturation/chatter, and does
not fix early HOME termination or prove grasp competence. A future approved wrapper would
have to preserve original normalized and physical outputs, projected outputs, modification
counts and post-transform safety vetoes separately. It must not report old invalid proposals
as originally valid. Bounds alone certify neither collision safety nor task success.

**Exact approval gates:**

- **A-GC:** after Phase I review, approve a named bounded-continuous architecture, endpoint
  treatment, processor bridge, target/loss conventions and separate model/training plan.
  No authorization to resume or overwrite step20000 is implied.
- **A-GD:** after Phase I review, approve a named binary-head/class-target design, decoding
  and ambiguity rules, processor bridge and separate model/training plan.
- **A-GP:** after Phase I review, explicitly authorize physical output projection for a
  named new policy bundle, overriding current no-clipping semantics **only for that variant**.
  Approval must specify raw-output preservation and unchanged downstream safety.
- None grants Webots evaluation, a new comparison arm, additional data collection or access
  to validation/TEST. Such execution requires its own scope and Run authorization.

## 2. Task-aware termination decision matrix

Current mechanism: `departed |= max|q-q0|>0.02`; stop on any later `max|q-q0|≤0.02`.
E2 shows why this can fire before manipulation. It is a termination weakness, not evidence
that the success scorer falsely passed an episode. Changing the predicate changes episode
exposure and potentially model calls/actions even if all physical success thresholds remain
fixed. Leaving the numeric0.02rad threshold unchanged does not make a new stop rule the same
frozen protocol. No detector or threshold is changed here.

| Proposal | Predicate concept / expected mechanism | Inputs and privilege | Risks / confounding | Versions and exact approval |
|---|---|---|---|---|
| T-B: existing budget termination |Stop on safety/model failure or existing horizon1140; record early HOME as an annotation instead of ending there. Evaluate final physical criteria at the declared end |Robot observations, action count; no privileged state needed for stopping. Existing ground-truth success scoring stays evaluation-only |Longer exposure and later veto/contact opportunities; the three existing future chunks already contain invalid commands. Final HOME-at-end versus earlier HOME-history meaning must be settled, not silently changed |New P and Run; M/R/D unchanged if only controller stop policy changes. A-TB must approve stop logic, end-of-run HOME definition and assessment scope after report review |
| T-H: robot-history-qualified HOME |Stop only when HOME-now plus predeclared close/reopen history and any declared persistence/progress condition hold |Joint state, measured opening and model-command history are permitted observations; FK EE motion may be derived from joints. No object oracle required |Empty close/reopen can satisfy history with no object; noisy/stalled aperture, false stage flags, added state-machine assistance. A monitoring rule is not allowed to steer or inject actions |New P and Run. No M/D inherently; any new event thresholds/timers are explicit protocol parameters, not borrowed safety tolerances. A-TH must approve all predicates/parameters and confirm VLA-only arm eligibility |
| T-V: visual progress-qualified HOME |Require evidence of object lift, transport/placement and release, then HOME, using a specified visual estimator |RGB and permitted robot/calibration inputs; no simulator truth. Source is nonprivileged, but adds a perception/termination component |Occlusion, monocular depth ambiguity, tracker drift and false completion. Adds possible heuristic assistance; estimated success must not replace frozen physical scoring |New P,R/perception version and Run; perception M/D-view/D only if newly trained/labeled. A-TV must approve estimator, uncertainty/event definitions, arm assignment and conformance |
| T-O: ground-truth-qualified stopping |Use measured simulator object progress/placement to suppress premature HOME stops in an explicitly privileged diagnostic |`get_ground_truth()` object poses: **privileged**. Current VLA-only policy explicitly excludes oracle task termination |Leaks task state through the stop condition even if never fed to SmolVLA; cannot be called unchanged VLA-only. May hide controller defects |New **PRIVILEGED DIAGNOSTIC** P and Run; M/D unchanged. A-TO requires report review and explicit oracle-control exception/label. Never count it as the frozen learned-only arm |
| T-L: learned completion signal |Stop on a separately specified learned completion/EOS output subject to declared rules, failures and budget |Inference can use only permitted RGB/state. Training completion labels may use evaluation truth with provenance; that does not make inference input an oracle |Self-reported completion can be wrong, especially at initial hold; class imbalance/calibration; premature/late stops remain possible |New P,M,R/service schema and completion-label D-view (D if new fields materialized), plus training and Run. A-TL must approve output contract, labels, decode rule and any API changes |

All alternatives retain immediate safety rejection and a hard action budget. T-B is the
simplest nonprivileged diagnostic alternative, not a proven performance repair. T-H may
mistake command history for grasp. T-V uses permitted data but is not automatically free of
arm-semantic changes. T-O is allowed only as a separately approved oracle diagnostic, not
under the current VLA-only contract. T-L is larger model/data work and not the first step.
A persistence timer or classifier threshold cannot be selected from the failed traces and
quietly made a frozen rule; parameters and selection procedure require owner/report review.

**Approval A-TB/A-TH/A-TV/A-TO/A-TL** must name exactly one predicate, all new parameters,
permitted/privileged inputs, preserved physical scoring, assigned diagnostic/experimental
arm, output/version directory and bounded execution scope. The missing Phase I report
blocks conformance approval for these affected changes. No such approval currently exists.

## 3. Command-to-motion realization

| Stage | Expert path | VLA path | What a comparison must preserve |
|---|---|---|---|
| Proposal |Scripted geometric policy solves/updates joint targets, then sends original joint commands |Learned6-D pose residual plus absolute gripper command |Same recorded starting state and physical target semantics, not merely same scene name |
| Dataset boundary |Exporter stores `pose_delta(observed_EE,FK(active_joint_target))`; drops the redundant joint choice from the label |Decoder adds base-frame translation and left-multiplies rotation delta |Same registered FK/frame and time alignment. No division/multiplication by dt |
| Realization |Original expert joint target reaches backend safety/motors |Full-orientation IK warm-started at measured joints, inside existing trust region, then pose/joint checks |Same torso/base pose, IK seed, limits, target orientation and residual checks |
| Physical response |Joint goal drives finite-speed motors over lockstep ticks |A new decoded goal may arrive every32ms |Log commanded q, achieved q/qdot, velocity/acceleration/force settings, dwell and step count |

References: [dataset export](/home/tj330/programs/VLA-Project/src/vlarobot/vla/dataset_export.py:21),
[expert joint submission](/home/tj330/programs/VLA-Project/src/vlarobot/execution/scripted_can.py:1727),
[VLA bounded full-pose solve](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/pose_solver.py:34),
[decode/safety/apply](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:105).

A6-D target can correspond to multiple7-joint configurations. Different joint targets with
similar FK are not automatically an IK error; they can still affect joint-space HOME detection
and transient motion. A shortfall between commanded pose residual and one-tick displacement
is not automatically a units bug: targets and measured per-step motion differ by definition.
E3 motivates separating these mechanisms before concluding that retraining solves transfer.

### Smallest proposed staged diagnostic

**D1: one matched TRAIN snapshot, CPU numerical audit only.** Use the already-preserved
local validation input: approved native episode0/frame0 / source attempt0000,
`train-clear-i00-s100`, and its saved step20000 inference chunk. This is an actual same-input
expert label and learned prediction pair, avoiding a new policy call or mismatched images.
The first saved gripper output is already legal; do not clip or choose a different sample.
Snapshot data, source command history and provenance must be pinned before coding.

Inputs:
- `runs/datasets/tiago-train-v1/data/chunk-000/file-000.parquet`, first row;
- `runs/demos/train-s100-registered-v1/attempts/attempt-0000/episode-00.steps.npz`
  and matching `.actions.jsonl` to recover q0 and active q-star;
- `runs/local-model-validation-v1/existing-import/physical-action-chunk.npy`, action0,
  with source-state/image hashes in `processor-action-audit.json`.

Compute three cases, without sending a command:

| Case | Target / computation | Measurement isolated |
|---|---|---|
| A |Original recorded expert joint target q-star |Reference joint choice, no new solve |
| B |Expert label reconstructed target X-star=FK(q-star), then current IK(X-star; q0) |Loss of original joint choice/IK branch at the same intended expert pose |
| C |Saved learned target X-M=X0 composed with saved residual, then the same IK(X-M; q0) |Learned target difference plus the decoder's response |

This requires **two deterministic IK solves**, B and C; A is the original reference.
Use the exact current registration, limits, trust region and safety/residual tests. Record
workspace/joint/gripper vetoes rather than adapting a rejected input to make it fit.
Log pose-error vectors, orientation errors, q-B minus q-star, q-C minus q-B, bounds and
residuals, and Jacobian singular values/task-versus-null-space components where numerically
well-defined. Report spectra without inventing a calibrated singularity cutoff. No ground
truth or new neural inference is needed. At first expert hold q-star is the recorded hold;
that makes this a particularly small initial-precision/branch test, not a grasp test.

Interpretation: if B differs in joint space while FK agrees, decoder redundancy is observed;
if C introduces the target difference, learned residual error is observed; large changes in
q-C for a small pose difference support sensitivity as a hypothesis. These effects can
coexist. Linearized null-space decomposition is local, not a proof about finite motions.
D1 cannot measure motor response or claim physical causation.

**Exact approval A-D1:** authorize implementation and execution of only this one-snapshot,
two-IK-solve CPU artifact audit on an isolated branch/output directory, with no policy
forward, Webots, projection, actuator commands, dataset/model change or scored evaluation.
It is a new numerical diagnostic and is **not** run under today's design permission.
Because it does not change deployed behavior or score an arm, it need not claim Phase I
conformance; its results cannot authorize a correction in the report's absence.

**D2: the smallest physical screen for all three explanations.** Only after D1, propose
three matched fresh-session **arm-only one-tick** replays of A, B and C from the same TRAIN
scene/reset state, one existing32ms tick per case, with no new model inference. Hold gripper
motors identically in their existing reset condition; this isolates arm realization rather
than executing a manipulated7-D policy. It is an instrumented bench, **not unchanged
VLA-only evaluation**, and the original model chunk is never modified or relabeled.

Match/log q0, qdot0, torso, base pose, object poses and velocities before each pulse. The
reset-matching criteria must be preregistered and explicitly approved; no new numerical
matching tolerance is invented here. Record each q command, safety verdict, measured q1/qdot1,
registered FK(q1), motor settings, step timestamp and termination/cleanup. If independent EE
truth is used to validate FK, label it PRIVILEGED DIAGNOSTIC and keep it out of commands.
Every actual command must pass the unchanged SafetyMonitor and backend checks.
Any simulator truth used for object/reset matching is likewise debug-only, explicitly
privileged evidence, not a policy input. Preserving existing checks does not certify full
collision geometry or swept-volume coverage; short diagnostic pulses are not assumed safe
merely because they satisfy joint limits. That remaining physical-bench risk belongs in
the separately reviewed launch protocol.

The comparison can locate the stage introducing an **initial one-tick** discrepancy:
A versus B tests the extra IK path, B versus C includes learned-target error, q-command
versus q1 measures immediate motor tracking. It cannot establish settling/convergence,
repeatability or full-task causation from one pulse per condition. One-tick lag may be
normal under motor limits. If differences are smaller than reset variation or coexist,
report ambiguity and stop, rather than declaring a winner. A longer fixed-target hold or
replication is a separate follow-up with a newly approved count/duration; no automatic
extension or seed resampling is allowed.

**Exact approval A-D2:** after Phase I conformance review and D1 report, authorize a named
new diagnostic P/Run: exactly three fresh TRAIN sessions, scene `train-clear-i00-s100`,
A/B/C arm-only setpoints, one32ms pulse each, specified reset-match criteria, unchanged
registration/limits, identical gripper hold, logging and stop gates. Abort a provenance,
reset-match or infrastructure failure; record any safety rejection without shrinking the
command, replacing the sample or rerunning it to obtain an accepted result. No task-success
claim, frozen campaign rerun, oracle proposer, policy modification or other split access.
The missing report and unspecified match criteria currently block a launch-ready D2 plan.

D1/D2 need new diagnostic protocol/run identities, **not new model or native dataset
versions**. A temporary harness/instrumentation version is required; it must not modify the
frozen execution path. Broader conclusions would need separately authorized replication.

## 4. Recommended order and approval boundary

1. **Now:** accept this design assessment only. Preserve all existing artifacts and final role.
   Obtain the authoritative Phase I report before approving affected behavioral changes.
2. **First requested measurement, if the owner chooses:** A-D1, the one-snapshot numerical
   audit. It is cheaper and less confounded than retraining or a new task campaign.
3. **Then, only if D1 leaves the physical response unresolved:** finalize reset-matching and
   conformance, seek A-D2 for the three one-tick physical screens. If insufficient, stop with
   uncertainty and propose the smallest additional hold/replication scope separately.
4. **Choose one gripper variant after report review.** E1 makes bounded generation a high-value
   design target. G-C/G-D preserve an intrinsically bounded proposal at training cost; G-P is
   cheaper but explicitly violates current no-clipping semantics unless formally amended.
   No option is selected as an approved policy here.
5. **Assess one termination variant separately.** T-B is the simplest diagnostic exposure
   change; T-H/T-V/T-L add assumptions and T-O is privileged. Never combine gripper and
   termination changes in a first causal comparison: attribution would be confounded.
6. **Only later:** seek a separate training or evaluation authorization for a defined model/
   protocol variant, with preregistered scope and new directories. That approval must say how
   the result relates to the frozen study. No new result replaces0/24, retakes the final gate,
   changes scripted-primary, adds an arm or opens validation/TEST by implication.

**Execution status: NO-GO for all experiments and corrections.** The only recommendation
ready to request in isolation is A-D1's bounded artifact audit. No routine implementation
permission is inferred. Await explicit owner authorization; do not implement this document.
