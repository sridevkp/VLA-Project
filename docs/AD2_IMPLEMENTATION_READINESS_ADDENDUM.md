# A-D2 implementation-design addendum — no physical authorization

This supplements [the readiness design](AD2_READINESS_DESIGN.md) at `34294c9`.
**Only design and CPU-only validation planning are approved.** No harness or fixture test
has been implemented; no Webots, model server, inference, IK, actuator command or scored
experiment has run. Original **0/24, gate FAILED, scripted-primary** remains final.
All frozen parameters, limits, scene splits, five arms and December12 deadline remain unchanged.

**Conclusion:** a bounded, *partially observed* one-tick comparison is feasible in principle.
An exact replay of the complete historical dynamic state is not established. Physical
execution remains **NO-GO** until the reset interpretation, numerical criteria, missing-data
qualification and collision preflight are approved and subsequently verified. This is not
an authorization to gather the missing measurements.

Labels: **Observed** means an existing record/source; **Derived** means arithmetic on those
records; **Proposed** means an unapproved protocol choice; **Unverified** marks missing proof.
The [read-only evidence inventory](evidence/ad2-preparation-evidence-v2.json) pins30 files,
cached PROTO locations, PDF hash and the reproducibility calculations below. Local paths
are evidence locators, not portable execution paths; a future CLI must accept explicit roots.

## 1. Candidate Phase I report

**Observed.** `/home/tj330/Downloads/Project_Report.pdf` still has SHA-256
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`,
24,118,576bytes. Cover/certificate identify the four-author WorldMind report, October2026,
Project Phase I. This verifies identity against the recorded local candidate, **not the
independently authenticated submitted revision**. Owner confirmation remains outstanding.

Read relevant PDF pages25–26 (§3.1: supported interfaces, repeatability, safety, traceability),
46 (§4.3: simulator-state provenance),48 (§4.7: simple controller tests and execution errors)
and49 (§4.8: comparable units/state and logged initial conditions). The relevant requirements
support isolated controller diagnostics with checked safety and disclosed limitations.
They provide no reset tolerance, one-tick safety certificate or authority to change policy.
The earlier design maps these provisions, Table5.1 and ADR-0029; that mapping remains
conditional on the candidate revision. No report content is substituted by implementation.

## 2. Additional historical evidence and its limits

Source root for all historical references:
`/home/tj330/programs/VLA-Project/runs/demos/train-s100-registered-v1/attempts/attempt-0000/`.

| Missing question | Additional evidence found | What remains unverified |
|---|---|---|
| Object velocities | `episode-00.steps.npz` stores983 times, object xyz/quaternions for APP/CAN/HAM, but no twists. `policy.jsonl` line1 stores CAN pose; physical line1 stores fingers/pads only. Backend reset clears object dynamics immediately before its final tick. | ResetPhysics does not establish zero velocity **after** gravity/contact integration. Later frame differences are interval averages after commands, not frame0 velocity. No historical object twist recovered. |
| Full base twist | Earlier reset fingerprint contains six world velocity components; frame0 policy contains full pose and **speed norms**0.00002611530867254974m/s and0.000036844188467267185rad/s. | Fingerprint precedes finalization. Cannot assign its direction/components to frame0, even though time separation is one tick. Row0→row1 pose differences do not recover an instantaneous twist. |
| Force / torque / PID | Cached PROTOs and installed node defaults give nominal caps/gains; packaged runtime has no setControlPID/setAvailableTorque call in the inspected backend path. Finger force setters occur in open/close and explicit gripper apply paths. | Historical effective available force/torque, PID integrator/controller internal state and external overrides were not logged. Source absence is evidence against a known override, not a measurement of runtime state. |
| Reset timing | `episode.log:12–20` records reset seq1, wall duration2.9317s, return relative sim time0.032s, followed by get_gt calls with unchanged time; first apply seq5, first step seq6 to0.064s. | Exact historical torso iteration count/absolute simulation timestamp and every startup tick were not logged. Wall duration cannot infer tick count. |
| Finger hold | Policy's earlier reset fingerprint stores position-mode target0m left and0.0000007171095937807197m right, speed0.03m/s; physical line1 gives actual frame0 sensors. No gripper apply precedes frame0 in the log. | This supports persistence of earlier targets through finalization; it is not a fresh read of those targets/force/PID at frame0. Scalar mean aperture cannot certify symmetry. |

**Observed source timing:** `RobotIO.__init__` enables sensors and makes two raw ticks;
`initialize_can` makes another. `WebotsBackend.reset` has an initial tick, variable torso
settle loop, head settle `int(2000/32)=62` ticks, and a final object-finalization tick after
fingerprint capture. `physics_step_count` is reset before that last raw tick and excludes
it. A zero reset counter is not zero elapsed physics. Counting only `backend.step` misses
preparation. The returned qdot uses finite differences over its recorded interval, not a
hardware velocity sensor. Preserve qdot and interval; never replace the nonzero reference
with zero. **Derived additional identity:** using the earlier fingerprint's eight named
joint positions as the previous endpoint, `(q_frame0−q_fingerprint)/0.03200000000000003`
reproduces every saved frame0 joint velocity with **maximum absolute error0.0**. The
inventory preserves both endpoints and the complete calculation. This supports the
finite-difference estimator/interval interpretation, not instantaneous velocity or historic
object twists. A future reader could capture the pre-finalization endpoint separately,
without relabelling it as frame0; how to observe that internal reset boundary without
changing the frozen reset remains an implementation-review requirement.

Backend reset seed100 is not an explicit physics RNG reseed; WorldInfo's cached
schema defaults randomSeed0/basicTimeStep32ms, but historical effective values need pinning.

## 3. Reset protocol and proposed numerical criteria

### Three different comparisons

**Proposed exact categorical checks:** source identity/frame/seed100/scene JSON; A/B/C
maps and hashes; A=B exact; runtime/URDF/registration/world and dependency identity; sensor
names/types/SI units; control modes, motor target/cap settings, simulator parameters;
32ms basic step; fresh PID/nonce/port and no unexpected clients. Reject absent/non-finite
required fields. Floating-point motor configuration values read from the same definition
are exact configuration matches, not a blanket tolerance for measured motion.

**Proposed reference-to-case checks:** compare each case start to **actual row0/line1**,
not the earlier fingerprint. Use named arm/torso position and qdot, full base translation
and relative quaternion angle, available base speed norms, each object's xyz/rotation,
individual finger positions and aperture. Historic unavailable fields remain unavailable.

**Proposed pairwise checks:** compare every new case against all preceding admitted starts,
using the same metrics plus complete live base/object twists and motor settings. Do not
allow pairwise drift of twice a reference tolerance. Record both residuals independently.
No post-hoc widening if B/C fails, and no case rerun to obtain matching starts.

### Measured scale and candidate bounds — ALL UNAPPROVED

PositionSensor's installed schema has noise0/resolution−1; cached TIAGo sensors do not
specify a resolution override. This does **not** provide a positive physical quantization
step or a calibrated accuracy claim. JSON/NPZ float64 precision is not physical repeatability.
Native float32 model-state ULPs must not set raw-sensor reset tolerance.

Existing `runs/repeat-fresh-01.jsonl` through `03.jsonl` first INIT records are seed0,
relative time0, with the old reset/FK context, **not A-D2's seed100/frame0 at0.032s**.
Three samples cannot certify a tail bound or be silently pooled with the approved snapshot.
They provide only an explicitly limited scale estimate. Ranges below are maximum pairwise
observed differences; no new simulation or forward/IK calculation was used.

| Quantity / metric | Existing scale | Proposed reference AND pairwise ceiling | Rationale and approval condition |
|---|---:|---:|---|
| Each arm position, absolute rad | Largest old range9.7542772286e−8 | **1e−7rad** | Round that observed maximum upward to one significant digit; an exploratory ceiling, not a confidence bound. Wrong timing/context means applicability is unverified. |
| Torso position, absolute m | Old range6.1716776689e−9 | **7e−9m** | Round upward to next1e−9m. Same applicability limitation; do not replace with existing2mm settle criterion. |
| Base translation, Euclidean m | Old pairwise maximum0.0001322225514 | **0.00014m** | Round upward to1e−5m. This is about21% of C's requested0.670555mm, so passing does not remove task-space confounding. |
| Base rotation, relative angle rad | Old maximum0.0012512376487 | **0.0013rad** | Round upward to1e−4rad; about25% of C's requested rotation. Do not call this precise enough for a causal EE comparison. |
| Mean finger opening, absolute m | Old range1.6218097331e−9 | **2e−9m** | Round upward to1e−9m. Mean-only check insufficient: individual fingers still required. |
| Arm finite-difference qdot | No same-frame repeat qdot series | **6.25e−6rad/s**, conditional | Arithmetic position-error budget `2×1e−7/0.032`; applies only if BOTH differenced endpoint position errors meet the proposed position ceiling and intervals match. Not an independently measured velocity tolerance. |
| Torso finite-difference velocity | Same missing endpoint guarantee | **4.375e−7m/s**, conditional | `2×7e−9/0.032`; same endpoint/interval condition. |
| CAN/APP/HAM position and rotation | Old CAN translation range0; no full matched object-state repeatability | **UNSET**, not zero by default | Zero observed variation in three unmatched records is not a measured resolution or complete scene bound. Need applicable existing repeat evidence or separately authorized measurement. |
| Individual fingers; full live base/object twist; reference speed norms | No applicable repeatability/resolution estimate | **UNSET** | Do not borrow arm bounds, assume velocity direction or fill missing values. |

These numbers are review candidates only, **not admissibility settings**. Choosing an upward
rounding convention adds no statistical assurance. No factor-of-ten “safety margin” is
hidden. Owner can reject the weak-context numbers and retain NO-GO; a relaxation needs a
stated scientific rationale, not convenience. Conditional velocity bounds cannot be activated
without their endpoint premise. The old reset fingerprint and frame0 are distinct instants,
not repeated measurements of one state.

**Preferred decision:** retain NO-GO until suitable bound evidence/approval exists for all
required variables. If owner accepts partial observability, name each waived *historical
measurement requirement*, keep complete live pairwise measurement, and limit inference to
an initial command-versus-motion comparison conditional on admitted starts. This is not a
waiver of motor/safety limits. No claim of exact initial-state equivalence is then permitted.
A/B's identical target offers one negative-control discrepancy, not a variance estimate.

Tick accounting is categorical: exactly one32ms post-reset integration per accepted case,
with all preparation ticks separately counted. For double timestamp subtraction only,
propose arithmetic consistency allowance `2*(ulp(t0)+ulp(t1)+ulp(0.032))` seconds, alongside
an exact underlying tick delta1 and unchanged basicTimeStep32. This allowance covers the
rounding in stored times/subtraction, not an extra tick or variable-duration integration.
It too remains unapproved. Compare quaternion signs invariantly; do not use acos near1
without a documented numerical treatment or silently reuse numerical IK residual floors.

## 4. Exact-scene geometry and command path

**Observed assets:** frozen `worlds/factory.wbt:59–65` instantiates TIAGo; `:133–155`
contains HAM/CAN, Table and APP. The saved ScenarioSpec moves HAM/APP and offsets CAN;
all other factory geometry remains, including other robots, tools and moving valve
controllers. “Clear” does not mean an empty world. Do not disable those controllers or
assume all untracked objects are immobile without changing the protocol.

Cached PROTO files are content-hashed in the inventory. They are local cached source,
not proof that every resolved asset was identical during historical recording. Important
specifics:

- Tiago→TiagoBase/Body/FrontArm/Gripper plus Astra/Hokuyo composition is explicit.
  Tiago `selfCollision FALSE` is propagated; factory world does not override it.
- URDF and source audit enumerate collision owners for base/casters/wheels, torso/head,
  camera, shoulder/arm1–7, wrist connector, gripper body and both fingers. Existing source
  audit maps27 owners and reports no primitive-count differences in those mappings;
  base fragments and Hokuyo need explicit association. Count agreement is not containment.
- `runs/live-geometry-v3/live-bounding-objects.json` provides already-recorded expanded
  primitives and owner transforms. It is a generic factory probe, **not the approved
  source snapshot**. No new probe was executed. Earlier frame audits found nontrivial
  link-frame disagreements; static registration of the tool does not certify all links.
- Can's cached boundingObject is a cylinder radius0.03175m, height0.1222m; Apple a sphere
  radius0.05×scale; Hammer uses four boxes plus a cylinder. Their saved scene transforms
  and actual dimensions/scale must be accounted for, not substituted by point positions.
- Table size is4.5×1.5×0.75m at world(−0.23882837,−4.8402928,0). Table PROTO uses child
  SolidBox top and four feet. **The existing live export's owner list omits the table/top/
  feet**: enumeration of top-level objects with boundingObject misses nested solids.
  Treating that export as complete would be unsafe. The two floor Planes are also marked
  unsupported by the existing primitive exporter.
- Existing `contact_nodes` (`robot_io.py:135–175`) filters robot descendants and deduplicates
  other node contacts. Thus an empty list cannot establish absence of self-collision or
  per-link/contact-pair coverage. Robot-floor support contacts are expected; a blanket
  “no contacts” rule would be wrong. No enclosure/carried-object assumption is justified
  merely by a closed-looking gripper; no carried object is commanded in this bench.

**Required acceptable preflight, not yet established:** recursive geometry closure for all
nearby world solids (including table children/floor), all potentially moving robot links
and their ancestors, exact transforms at admitted start, supported conservative primitives,
self-pair adjacency/fixed-link exclusions reviewed individually, allowed existing support
contacts, and margins justified by geometry/registration/reset uncertainty. Unknown shape,
transform, relevant object/controller motion or exclusion is fail-closed.

A straight interpolation from q0 to target is not necessarily the physical motor trajectory.
For a conservative32ms swept bound, use each motor's maximum physically justified excursion
and initial velocity, with force/acceleration constraints only when actually certified;
include base/torso/passive joints and possible overshoot, then propagate those intervals
through complete link geometry. Merely sampling endpoints or the nominal target segment
is insufficient. No such volume has been computed here. Missing acceleration/physics
bounds may force a larger envelope or NO-GO; no invented torque-to-motion certificate.
A positive clearance threshold has not been approved or measured. Existing5mm/5° proxy
collision thresholds are evaluation definitions, not a pre-command clearance certificate.

### Motor path and limits

`RobotIO.__init__:73–114` sets arm requested velocity0.5rad/s, fingers0.03m/s and enables
position sensors at32ms. `backend.apply:262–278` reads current state, invokes unchanged
SafetyMonitor, then seven setPosition calls for JointTarget; it makes no step or IK solve.
Pose/delta apply would invoke IK and is prohibited. Live limits come from actual motors
(`backend._safety_limits:1015`), with existing1s safety command horizon. No change to it.

Nominal cached FrontArm max velocities are1.95,1.95,2.35,2.35,1.95,1.76,1.76rad/s;
max torques43,43,26,26,3,6.6,6.6Nm. Requested0.5rad/s implies a nominal0.016rad/tick
setpoint-speed scale, **not a proved physical displacement bound**. Torso max velocity0.07m/s,
maxForce2000N; gripper max velocity0.05m/s/maxForce16N per finger. Installed Motor schemas
supply controlPID(10,0,0), acceleration−1 when not overridden. These nominal caps/gains
must not be relabelled as historically measured effective settings. Safety's100N upper
command allowance does not replace each actual16N motor cap. Preserve0–0.045m gripper limits.

## 5. Same-instant snapshots without state mutation

**Feasible by source design; not proven in a live session.** A dedicated Webots-side reader
can call already-enabled position sensor `getValue`, Supervisor getTime/getPosition/
getOrientation/getVelocity, existing contact getters, and motor target/mode/velocity/cap/
available-force getters. Bracket with identical simulation time and raw tick ledger; no
concurrent step/client. Use backend-returned qdot unchanged, with its actual time interval.
Raw getters must not invoke `_robot_state`/`_observation`: those overwrite `_last_joint_pos`
and `_last_time`. Actual backend.apply still performs its existing state read; log that
cache transition rather than modifying it. Same-instant means simulator-time coherent,
not every getter returns a fresh independently sampled sensor value.

Installed Python Motor exposes acceleration/available-force/torque getters but **no
getControlPID**. Reading an expanded node's controlPID field can establish its declared
field, not necessarily a runtime setControlPID override or hidden PID integrator. Separate
those concepts. Enabling force/torque feedback or pose tracking now is a mutation, not a
read-only snapshot; unavailable feedback remains unavailable. Do not introduce a sampling
warm-up tick. Existing contact tracking is enabled at startup; expand raw diagnostic pair
capture only if the installed API supports it without enabling a new tracker.

`Visualization.step:53` makes one robot.step, then UI tick and marker updates. Source
`ui_bridge.tick` polls messages/encodes images but contains no motor/step call. Marker
updates write visual fields; the snapshot reader must not call them. The ordinary apply/
step path remains intact; queued UI commands must have no consumer in the isolated bench.
Other world controllers and shutdown progression need ledger accounting; a stopped/timeout
RPC must never be retried as if it were read-only. Static review cannot prove Supervisor
getter behavior, sensor freshness or process scheduling in a live runtime.

## 6. Minimal future isolated implementation (not implemented)

1. **Pure artifact reader:** explicit roots; reject pre-existing output, path escape,
   missing/mismatched hashes, incorrect snapshot/seed, missing criteria and changed maps.
   Read exactly A from result and B/C from saved cases, cross-check copies; assert A=B.
   No solver/model imports or creation. Historical matrices are labels/evidence only.
2. **Admission/comparison functions:** typed immutable records; separate missing/invalid,
   configuration mismatch, numeric mismatch and safety rejection; every approved criterion
   carries units, provenance, rationale and protocol version. No default tolerance.
3. **Diagnostic snapshot adapter under backends/webots:** only raw getter allowlist and
   backend-cache-free serialization. A separate diagnostic wrapper/endpoint, not changes
   to frozen core interfaces/controller/eval. Future implementation review must determine
   how to count raw startup/reset steps without changing physics or frozen control behavior.
4. **Coordinator:** fixed A→B→C, fresh launcher process/port/nonce each, one reset each,
   unchanged SafetyMonitor/backend apply JointTarget, one step only after acceptance, no
   inference/IK/fallback. No subprocess executable command is supplied before approval.
5. **Ledger/report:** source and effective-state hashes; before/apply/after/cleanup records,
   RPC seq/nonce, tick phases, all rejection/timeout paths. Preserve incomplete artifacts.
   Stop the sequence on provenance/reset/geometry/command rejection or infrastructure
   ambiguity; never allocate a fourth session or silently resume a partially used one.

Full planned artifacts and measurements remain in sections6–7 of the base design. Keep
instrumentation code entirely separate from frozen execution sources and training artifacts.
No new dependency or public interface is needed to design the CPU fixtures.

## 7. CPU-only fixture-test specifications — not executed

Use plain immutable JSON/NPZ-derived fixtures and fake transport/process/snapshot objects.
Imports must not load Webots, Torch/model code, IKPy or a real backend constructor. Patch
process/network/actuator constructors to raise if reached. No simulator executable lookup
that executes it, real socket, sleep-driven scheduler, actuator command or FK/IK calculation.

| Specification | Stimulus | Required assertions |
|---|---|---|
| Immutable artifact loading | Saved maps; reordered keys; one-bit target change; missing B/C copy | A=B exact; C retained; no normalization; mutation/hash mismatch rejected; inputs unchanged after test. |
| Provenance rejection | Wrong seed/frame/scene/hash, swapped earlier fingerprint, missing units/criteria, absent twist with strict mode | Reject before any fake apply/step; partial-observability mode requires explicit approved named omission, never zero substitution. |
| Safety rejection | Inject rejected ApplyResult; actual pure SafetyMonitor with synthetic out-of-limit/NaN/unknown-joint JointTarget | No step/fallback/retry, record reason, cleanup and stop sequence. Real saved C is not altered to make this fixture. |
| Ambiguous RPC | Ack loss after fake apply or fake tick; wrong nonce/seq; duplicate response | At most one dispatch; terminal ambiguous status; no resend/new session; keep ledger even if ack absent. |
| Session isolation | Reused PID/nonce, port held by prior fake session, stale output, surviving child | Admission fails; no fourth case/replacement; no attach to existing simulator. |
| Tick accounting | Raw preparation counts, reset counter0 with final raw tick, accepted measurement delta1, delta0/2, wall delay, time-rounding edge | Preparation separate; require exactly one measurement tick and approved timestamp consistency; no wall-time-based decisions. |
| Non-mutating snapshot | Getter-only fake objects and cache sentinels | No setters/enables/step/observation/state helper; unchanged velocity history, targets, timestamps, feedback mode; missing getter explicit. |
| Cleanup and storage failure | Exceptions at startup/reset/preflight/apply/step/write/stop; hung fake child | Finally cleanup once with recorded escalation; failure never converted to completion; no extra tick to flush; no deletion/overwrite. |
| Pairwise admission | Both B/C within reference bound but outside pairwise bound; opposing quaternion signs | Reject pairwise drift; accept equivalent quaternion signs via documented metric; no implicit doubling of tolerance. |

Fixture pass would verify coordinator logic only, not collision containment, historical state,
actual controller API behavior, physics determinism, motor tracking or safety certification.
Implementing/running these tests is a future separately approved code task, not performed here.

## 8. Decision matrix and remaining owner decisions

| Item | Existing evidence establishes | Needed next | Status |
|---|---|---|---|
| Candidate report | Recorded bytes/title/hash match | Owner confirms submitted revision | CONDITIONAL |
| A/B/C and matching snapshot | Immutable maps, A=B, source identity and q/qdot | Future loader fixture verification; no new solve | DESIGN READY |
| Historical object/full base twist, effective force/PID | Some norms/configuration and earlier records only | Accept explicitly weaker historical match or remain blocked; no fabricated reconstruction | BLOCKED |
| Numeric reset ceilings | Exploratory older reproducibility scales and conditional arithmetic bounds | Owner approval of bounds/premises; applicable measurements for unset fields need separate execution authority | UNAPPROVED |
| Geometry / contacts | Source primitives, partial mapping, table/Plane/self-contact gaps identified | Complete recursive geometry/transform/exclusion/sweep preflight and approved margin | BLOCKED |
| Same-instant reader | Getter-based design possible; no public complete snapshot today | Isolated implementation review/CPU fixtures, then live validation only when authorized | NOT IMPLEMENTED |
| Three-session one-tick interpretation | Existing reset cannot be one lifetime tick | Explicit preparation-versus-measurement approval | UNAPPROVED |
| Physical execution | No authority granted | Separate explicit launch authorization after previous gates | NO-GO |

**Minimal decisions:** (1) confirm candidate report revision; (2) choose strict NO-GO versus
named partial historical observability, approve numeric metrics/bounds including unresolved
fields and the one post-reset tick interpretation; (3) approve exact-scene collision
coverage/exclusions/margin requirements without relaxing existing safety; (4) separately
authorize isolated code/CPU fixtures and review before any later physical authorization.

A bounded diagnostic can answer “what initial motion followed each saved joint target under
these admitted, measured starts?” It cannot establish an exact historical replay, a new
success rate or causation of the HOME failures. The frozen failed result is unaffected.
Preparation stops after documentation and non-executing checks, awaiting owner decisions.
