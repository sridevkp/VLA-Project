# V3 alternatives assessment — learned manipulation routes under the V2 stop rule

**Status:** V3 is a separate assessment and implementation arm under the
2026-11-28 owner addendum. No training, GPU use, model substitution, gate
change, decoder/IK/safety change or holdout access has occurred. V1 remains
immutable; V2 step-021000 remains documented at `0/3`; the `>=10/24` TRAIN gate
remains unchanged. The Dec 5 hard stop applies to all V3 experiments.

**Evidence files:** `docs/evidence/v3-dataset-measurements.json` (SHA-256
`d8c6310ff737dc6c6a773c44c011af20db43989334bcbd20650f4453bbfb3856`),
`docs/evidence/v3-feature-timing.json` (SHA-256
`ce4a6b8391f48c23650e53966cfa3fd5dc1e6117b7b76d0475439a1a8e6b9c5d`) and
`docs/evidence/v3-demo-plan-80.json` (SHA-256
`f9e9d706f49db4e3041cb032d1ff133c6a7beb5452c6c552b883dbc0c75cb80e`). Every
numeric statement below is tagged **MEASURED** (directly observed/derived) or
**INFERRED** (engineering estimate from measured anchors). Dates and file paths
are protocol facts, not performance figures. The 80-demo plan evidence is
**MEASURED plan output**, not collected data. Daily five-line status is recorded
in `docs/V3_DAILY_REPORT.md`.

### Owner addendum — 2026-11-28

- **A. In-sample admission.** TRAIN has only 8 base scene identities. Before any TRAIN
  admission pass is reported as a working VLA, run an object-position
  intervention test (move the CAN, hold other inputs fixed, and require
  waypoints to shift in the right direction) and a scene-identity memorization
  check such as leave-one-base-scene-out offline error. If the intervention
  test fails, report the result as **“memorized TRAIN scenes”**, not as a
  working VLA.
- **B. Data priority.** Use up to 80 additional demonstrations primarily for
  randomized CAN position/yaw inside the reachable envelope, starting now in
  parallel with model code. Language-varying demos only if time remains.
  Language conditioning stays **UNMEASURED** without instruction-dependent
  destinations.
- **C. Option B context.** Avoid privileged object pose in the prediction
  context where possible. If used, label it **oracle-derived** in every table
  and include zeroed/shuffled-prediction ablations.
- **D. Separate V3 arm.** Frozen V1/V2 results are never modified. A V3
  evaluation on VALIDATION/TEST requires explicit owner approval after the V3
  configuration is frozen by commit.
- **E. Hard stop and reporting.** V3 experiments stop on Dec 5. Report daily in
  five lines.

Recorded as ADR-0087 in `docs/DECISIONS.md`.

## 1. SPEC — exact Phase I passages and conformity

The source PDF is `Project_Report.pdf`, SHA-256
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`
**MEASURED**. Page numbers below are PDF page indices from `pdftotext -layout`.

### Fig. 3.1 and the world-model/planner block

PDF p. 28, §3.3.1, exact caption and passage:

> Figure 3.1: Overall Architecture of the Proposed World Model-Based Vision-Language-Action (VLA)
> Robotic Manipulation System.

> The predicted future state can be represented as:
>
> ŝt+1:t+H = W (st , at:t+H−1 )                              (3.1)

PDF p. 29, §3.3.1, exact bullet:

> • Sends the predicted results and risk information to the VLA Policy to help select suitable
>    actions.

This makes prediction an input to policy selection, not only a post-hoc ranking
signal **INFERRED**.

### SmolVLA, the VLA block and Eq. 3.2

PDF p. 29, §3.3.2, exact passages:

> The Vision-Language-Action (VLA) Policy module helps the robot understand user instructions
> and decide what actions to perform. It connects the instructions given by the user with the
> visual information from the environment and the predictions provided by the World Model and
> Planner. A pretrained VLA model, such as SmolVLA, is considered as a possible model for
> implementing this module.

> The action-generation process can be represented as:
>
> at:t+H−1 = πθ (It , L, ŝt+1:t+H , ct )                         (3.2)

> After generating the required actions, the VLA Policy sends them to the Robot Controller,
> which converts them into commands that the robot can perform in the Webots simulation. The
> actual model, input format, and action commands will depend on the selected pretrained VLA
> model and its compatibility with the robot.

### Fig. 3.3 and the interaction diagram

PDF p. 35, §3.6.1, exact caption:

> Figure 3.3: Placeholder for a module interaction diagram showing data exchanged between perception,
> planning, prediction, risk assessment and control.

PDF p. 35, §3.6.1.1 adds:

> The exact state fields and perception method will be completed after the Webots sensor and
> scene interfaces are confirmed.

### Conformance decision

- A **different pretrained VLA conforms** as an implementation choice **INFERRED**: the report
  says “such as SmolVLA” and that the actual model depends on compatibility. It must still be
  learned, consume imagery/language/context, produce robot actions and preserve prediction and
  risk selection.
- A **compact learned waypoint policy conforms functionally but deviates from the “pretrained
  VLA” wording** unless it reuses a pretrained vision-language encoder **INFERRED**. It may
  satisfy the Eq. 3.2 functional contract only if predicted state is a real policy input.
- A **scripted-primary hybrid does not conform to VLA-primary control** **INFERRED**. It is an
  explicitly labelled engineering fallback; scripted task-level action selection is not
  equivalent to a learned VLA policy.

## 2. DATASET — measured TRAIN inventory

All figures in this section are derived from the frozen TRAIN dataset
`runs/datasets/tiago-train-v1` and its recorded demo phase traces. No TEST,
V2 holdout or sealed identity was accessed **MEASURED**.

| Quantity | Value | Status |
|---|---:|---|
| Demonstrations | 80 | MEASURED |
| Frames | 63,081 | MEASURED |
| Capture rate | 31.25 Hz | MEASURED |
| Frames/demo | min 538, mean 788.5125, median 782.5, max 1,141 | MEASURED |
| Camera views | 1: `observation.images.camera1`, 480×640×3 | MEASURED |
| Distinct instruction strings | 1: “Pick up the can and place it at the target location.” | MEASURED |
| Base scene identities | 8 (clear 3, cluttered 3, reach-limited 2) | MEASURED |
| Demos per base scene | 7–12 | MEASURED |
| CAN xy offset x | −0.0665 to 0.0998 m | MEASURED |
| CAN xy offset y | −0.0654 to 0.0701 m | MEASURED |
| CAN yaw offset | −3.1133 to 3.1379 rad | MEASURED |
| Recorded phase-tagged policy rows | 8,101 across 12 phase names | MEASURED |
| Waypoint target changes | 2,677 total; mean 33.4625/demo, min 30, max 36 | MEASURED |
| Gripper close transitions in recorded commands | 80 | MEASURED |
| Gripper release transitions in recorded commands | 80 | MEASURED |

**In-sample warning:** these 80 demos cover only 8 base scene identities
**MEASURED**. A TRAIN admission pass is therefore an in-sample result and does
not establish scene generalization **INFERRED**. The required admission
evidence is:

1. **Object-position intervention:** move the CAN to a new position, keep the
   instruction, robot state, camera and other scene inputs fixed apart from the
   resulting image change, and require the learned waypoints to shift in the
   correct direction **INFERRED protocol**.
2. **Scene-identity memorization check:** leave one base scene identity out at a
   time and report offline waypoint/gripper error on the held-out identity
   **INFERRED protocol**. A large train-to-held-out gap means the policy learned
   scene identity, not the task **INFERRED**.

If the intervention test fails, the result must be reported as **“memorized
TRAIN scenes”** and must not be called a working VLA **INFERRED consequence of
the owner addendum**.

Phase-tagged row counts are `INIT 400`, `HOME 1,108`, `PREGRASP 1,374`,
`PREGRASP_REALIGN 900`, `DESCEND 629`, `CLOSE 320`, `LIFT 435`, `TRANSIT 733`,
`RETREAT 400`, `PLACE_DESCEND 400`, `RELEASE 1,322`, `DONE 80` **MEASURED**.
These are recorded scripted-expert phase labels used only to derive offline
supervision; they are not runtime inputs **MEASURED**.

### Low-level action distribution — measured before blaming regression collapse

| Action field | Min | Mean | Median | Std | Max | Status |
|---|---:|---:|---:|---:|---:|---|
| dx (m) | −0.0575 | −0.000585 | −0.000018 | 0.01277 | 0.1015 | MEASURED |
| dy (m) | −0.1102 | −0.000089 | −0.000006 | 0.02144 | 0.1525 | MEASURED |
| dz (m) | −0.0492 | −0.000319 | 0.000025 | 0.00903 | 0.0541 | MEASURED |
| drx (rad) | −0.2321 | −0.000384 | −0.000026 | 0.04508 | 0.2389 | MEASURED |
| dry (rad) | −0.3241 | 0.010154 | −0.000024 | 0.04754 | 0.3228 | MEASURED |
| drz (rad) | −0.3282 | −0.001696 | 0.000000 | 0.07827 | 0.3502 | MEASURED |
| gripper target (m) | 0.0 | 0.028610 | 0.045 | 0.02165 | 0.045 | MEASURED |

Near-zero translation actions (`||dx,dy,dz|| < 0.0001 m`) are 30.01% of frames
**MEASURED**. Translation means are near zero but standard deviations are
non-trivial and gripper labels are binary endpoints **MEASURED**. Therefore
“regression mean collapse” remains a hypothesis, not an established cause of
SmolVLA failure **INFERRED**.

### Ablatable modalities

- Image: real RGB vs grayscale/shuffled/replaced **MEASURED available**.
- State: state8 vs zeroed/shuffled/per-joint permuted **MEASURED available**.
- Language: currently untestable because 1 distinct string exists; requires new
  TRAIN demonstrations with at least 2 strings **MEASURED**.
- Temporal context: 1 frame vs last 4 / last 8 recorded observations **INFERRED**.
- Prediction context: none vs analytic prediction vs learned prediction vs
  counterfactual prediction intervention **INFERRED**.
- Gripper representation: binary endpoint target vs explicit close/hold/release
  waypoint decision **MEASURED available from recorded command transitions**.

## 3. OPTIONS A–D

### Shared measured timing anchor

CPU-only feature timing used the frozen step-021000 preprocessor on a real
TRAIN JPEG/state/task sample, 2 CPU threads, 20 timed iterations after 3 warmup
iterations **MEASURED**:

| Operation | Mean | p95 | Status |
|---|---:|---:|---|
| JPEG decode | 3.5627 ms | 5.6618 ms | MEASURED |
| Decode + tensorize | 9.1708 ms | 12.8618 ms | MEASURED |
| SmolVLA preprocessor alone | 12.0196 ms | 15.0187 ms | MEASURED |
| Full feature pipeline | 14.8016 ms | 18.5992 ms | MEASURED |

Extrapolations **INFERRED**: caching features for all 63,081 frames costs about
15.6 min on this CPU if serial; caching only the 2,677 waypoint-change samples
costs about 40 s. Prior measured SmolVLA training throughput was 1.0675 steps/s
on the approved T4 session, about 5.32 h for 20,000 steps **MEASURED**. Compact
policy T4 estimates below are scaled from this anchor and are not promises.

### Option comparison

| Option | Model size | Input → output | Training cost | Integration points | Expected failure modes |
|---|---|---|---|---|---|
| **A. Compact multimodal waypoint policy executed by existing controller** | 2–5 M parameters: small image encoder + state/token MLP + temporal block + waypoint/gripper/phase heads **INFERRED** | RGB 480×640 downsampled to 224×224, state8, task tokens → target waypoint pose, gripper endpoint, next-phase/progress class **INFERRED** | Feature cache 40 s for waypoint samples; T4 15–45 min for bounded epochs **INFERRED** | `src/vlarobot/vla/model_service.py` for input contract; `src/vlarobot/planning/candidates.py` for source-tagged proposer; `src/vlarobot/execution/worldmind_pipeline.py` for episode wiring; existing IK/interpolation/settling/command path and `src/vlarobot/execution/safety_monitor.py` unchanged | Waypoint error accumulates; phase class learned from one instruction may be degenerate; controller retries can mask policy error; image/state distribution shift **INFERRED** |
| **B. A plus Eq. 3.2 conditioning from world-model-predicted features** | A + 0.1–0.5 M context/prediction encoder **INFERRED** | Same as A plus predicted context vector (distance, clearance, progress, gripper, validity) → same waypoint output **INFERRED**; learned predictor head preferred over numeric-only prepared head for full claim | Feature cache + predictor fit; T4 20–60 min if trained with A **INFERRED** | `src/vlarobot/world_model/rollout.py` and `src/vlarobot/world_model/analytic.py` produce context; `src/vlarobot/planning/prediction_conditioned.py` for controlled selector; `src/vlarobot/risk/gate.py` remains before execution; `src/vlarobot/planning/verification.py` unchanged | Analytic prediction error can mislead; prepared head is numeric only, so full Eq. 3.2 claim needs learned representation; intervention may change selection without improving success **INFERRED** |
| **C. Compact low-level chunk policy** | 3–8 M parameters: A encoder + temporal decoder for H actions **INFERRED** | RGB/state/task (+history) → 7-D action chunk, H = 50 to match frozen chunk shape **INFERRED** | Feature cache 15.6 min full-data or 40 s sampled; T4 45–120 min **INFERRED** | `src/vlarobot/vla/action_adapter.py`, `src/vlarobot/execution/vla_episode.py`, `src/vlarobot/vla/model_service.py`; RiskGate and motor safety unchanged | Repeats SmolVLA closed-loop precision failure; binary gripper regression; chunk horizon mismatch; IK residual vetoes; no clear phase semantics **INFERRED** |
| **D. Staged hybrid learned/scripted** | Selector 0.01–1 M parameters; learned waypoint policy optional 2–5 M if A is used **INFERRED** | Source-tagged scripted and learned candidate waypoints → selected candidate index **INFERRED** | Selector CPU minutes; add A cost only if learned candidates are enabled **INFERRED** | `src/vlarobot/planning/candidates.py` source tags; `src/vlarobot/risk/gate.py` veto first; `src/vlarobot/planning/prediction_conditioned.py` selector after veto; `src/vlarobot/planning/execution.py` motor safety; verifier unchanged | Scripted completion attributed to VLA; duplicate candidates; prediction head sensitivity mistaken for learned competence; documented proposal deviation **INFERRED** |

**Option B context constraint (owner addendum C):** avoid privileged object pose
in the prediction context where possible **INFERRED design requirement**. If a
CAN pose or any other oracle-derived field is used, every table must label it
**oracle-derived** and the experiment must include zeroed and shuffled
prediction-context ablations **INFERRED protocol**. A model that changes output
only when oracle pose is present has not demonstrated learned
prediction-conditioned selection **INFERRED**.

### Responsibility split

- Learned policy: choose the next waypoint, gripper close/release decision and
  task progression signal from observations **INFERRED**.
- Existing motion controller: IK, interpolation, joint target settling, command
  execution, retries and independent SafetyMonitor enforcement stay unchanged
  **MEASURED current implementation**.
- Reusing IK/interpolation/settling/execution/safety is acceptable **INFERRED**.
- Scripted expert selecting task-level actions is not VLA-primary control
  **INFERRED**; D must be labelled as hybrid, not recovery.

### Waypoint-level supervision without runtime oracle

Offline derivation is feasible **MEASURED**: recorded policy traces contain
2,677 waypoint target changes, 80 close transitions, 80 release transitions and
12 named phases across 80 demos. A supervised sample can be assembled from
`(RGB, state8, task string, optional K-frame history) → (waypoint target,
gripper endpoint, progress class)` using only recorded observations and the
recorded controller command that was actually accepted **INFERRED**.

Runtime must not receive expert phase labels, CAN ground-truth pose, object
boxes or verifier outcomes **MEASURED constraint from current input contract**.
The model must infer phase from observation history; a single frame is likely
insufficient to separate approach, grasp, transport and release **INFERRED**.
Use last 4–8 observations as design range **INFERRED**.

### Additional demonstration priority (owner addendum B)

Use the additional demonstration budget (up to 80) primarily for randomized CAN
position/yaw inside the reachable envelope **INFERRED target from owner
addendum**. The existing TRAIN generator already draws CAN home-offset radius
and yaw per seed block **MEASURED implementation**, so a fresh TRAIN seed range
provides new CAN poses over the same 8 base scene identities. Collection must
start in parallel with model code and use the existing accepted-demo audit
**MEASURED pipeline constraint**.

Language-varying demos are secondary and only if time remains **INFERRED
sequencing from owner addendum**. Language conditioning stays **UNMEASURED**
until demonstrations pair distinct instructions with distinct destinations
**MEASURED current dataset limitation**.

The intended measured collection command is:

```text
uv run python -m vlarobot.cli collect-demos \
  --n 80 --seed 200 --split train --backend webots \
  --out runs/demos/v3-can-random-s200-n80
```

This command was **not run** because this host has no Webots executable or
`WEBOTS_HOME` **MEASURED environment limitation**. The same command on a Webots
host uses the existing accepted-demo audit and does not touch the sealed
holdout **MEASURED pipeline constraint**.

A read-only plan check for `n=80, seed=200` built 80 entries: 10 per base scene
identity, 10 seed blocks, CAN x offset −0.094084 to 0.085783 m, y offset
−0.088809 to 0.069048 m, yaw −3.135123 to 3.133950 rad and CAN home-offset
radius 0.001531 to 0.098406 m **MEASURED plan output, not collected**. This
radius is the horizontal displacement `sqrt(dx²+dy²)` from the nominal CAN home,
not the physical can radius; the physical can radius is 0.03175 m **MEASURED
from `configs/robot.yaml`**. The 20 reach-limited entries
are retained only if the accepted-demo audit admits them; failures are
quarantined, so the target is up to 80 accepted demos, not 80 attempts
**MEASURED pipeline constraint**.

### Minimum additional demonstrations for instruction dependence

Current instruction count is 1 **MEASURED**, so language dependence is
unmeasurable. Hard minimum for a two-instruction paired test is 48 additional
TRAIN demos: 2 strings × 8 base scene identities × 3 seeds **INFERRED**.
Recommended target is 80 additional demos: 2 strings × 8 base scene identities ×
5 seeds **INFERRED**, collected with the same accepted demo audit and no holdout.
Both estimates assume one new instruction variant or goal phrasing; three
variants would require proportionally more **INFERRED**. This is a secondary
budget: CAN randomization takes priority under addendum B **INFERRED**.

## 4. ANTI-DISGUISE TESTS

1. **Oracle exclusion.** Static scan and runtime schema check must show model
   inputs are only RGB, state8, task string and explicitly allowed prediction
   context **MEASURED current contract**. Mutating oracle CAN pose/phase fields
   in a replay must leave policy output unchanged **INFERRED test**.
2. **Input hash logging.** SHA-256 of image bytes, state vector, task string,
   history frame hashes and prediction context must be written to every decision
   sidecar before execution **INFERRED requirement**. Replay with identical
   hashes must reproduce the same policy output **INFERRED test**.
3. **Image ablation.** Shuffle or gray-scale image while keeping state/task fixed;
   learned waypoint error should increase **INFERRED expected effect**. No change
   means the policy ignored vision.
4. **State ablation.** Zero or permute state8 while keeping image/task fixed;
   waypoint error should increase **INFERRED expected effect**.
5. **Language ablation.** After new demonstrations exist, swap instruction
   strings on identical observations; a genuinely instruction-dependent policy
   must change output **INFERRED expected effect**. Until then the result is
   `UNTESTABLE`, not a pass **MEASURED**.
6. **Learned-vs-expert waypoint error.** On held-out scene groups, report
   position error in m, orientation error in rad and gripper decision accuracy;
   compare against scripted-jitter and no-image/no-state ablations **INFERRED
   protocol**. A compact policy that does not beat the scripted-jitter baseline
   has no learned-competence claim **INFERRED**.
7. **Temporal ablation.** Compare 1-frame, 4-frame and 8-frame history on the
   same scene-grouped split; phase-class accuracy and waypoint error should
   improve with history if the single-frame hypothesis is correct **INFERRED**.
8. **Low-level distribution check.** Repeat the measured action-distribution
   table before any regression-collapse explanation is accepted **MEASURED
   anchor**. Current near-zero translation fraction is 30.01% **MEASURED**.
9. **Object-position intervention.** Move the CAN, hold instruction/state/camera
   setup and other non-CAN scene inputs fixed apart from the resulting image,
   and require the waypoint change to have the correct direction and sign
   **INFERRED expected effect**. If it does not, report **“memorized TRAIN
   scenes”** and stop calling the policy a working VLA **INFERRED gate**.
10. **Scene-identity memorization.** Run leave-one-base-scene-out offline
    evaluation across the 8 base identities and report the train-to-held-out
    error gap **INFERRED protocol**. High held-out error means memorization, not
    task generalization **INFERRED interpretation**.

### Scene-grouped offline split

Use base scene identity as the grouping unit, not random frame splits **INFERRED
protocol**. There are 8 base groups with 7–12 demos each **MEASURED**. Proposed
development split: keep one clear, one cluttered and one reach-limited base
identity for offline validation; train on the remaining 5 groups **INFERRED**.
No frame from a demo may appear in both sides of the split **INFERRED**. The
admission memorization check is stricter: run 8-fold leave-one-base-scene-out
evaluation over all base identities, separate from the 5/3 development split
**INFERRED owner addendum A**. Sealed holdout remains untouched **MEASURED
constraint**.

### Eq. 3.2 controlled path

1. Fit a small learned context predictor from TRAIN trajectories to produce the
   predicted context used by the selector **INFERRED**.
2. Hold image, state, language and candidate actions fixed; vary only predicted
   context **INFERRED**. Compare no-context, analytic-context, learned-context
   and counterfactual-context arms **INFERRED**. Prefer non-privileged predicted
   fields (distance, clearance, progress, gripper, validity) over CAN pose
   **INFERRED owner addendum C**.
3. Counterfactual interventions: set distance/clearance/progress/gripper fields
   to zero, shuffle across candidates, and apply bounded perturbations
   **INFERRED**. If oracle-derived CAN pose is present, label it in every table
   and include zeroed/shuffled-prediction arms **INFERRED owner addendum C**.
4. Require selection changes attributable only to prediction context before any
   Eq. 3.2 claim **INFERRED**. Oracle-context runs are diagnostic only and never
   enter the real pipeline **MEASURED constraint**.
5. Keep RiskGate hard veto before selection and independent verifier after
   execution **MEASURED current architecture**.

## 5. RANKED RECOMMENDATION AND HARD STOP

1. **A — compact waypoint policy** is the fastest path to a real closed-loop
   prototype **INFERRED**, because it reuses the existing controller and avoids
   another low-level chunk/IK intervention.
2. **B — A plus Eq. 3.2 conditioning** is the strongest conformant learned route
   **INFERRED**, but only after A shows non-trivial waypoint competence.
3. **D — staged hybrid** is the safest engineering fallback **INFERRED**, but it
   is a documented deviation and must not be reported as VLA-primary recovery.
4. **C — low-level chunk policy** is lowest priority **INFERRED**: it shares the
   failure mode already observed in SmolVLA and adds no new task-level semantics.

### Day-by-day plan to hard stop Dec 5

| Date | Work | Gate / stop |
|---|---|---|
| Nov 28 | Record addendum; define object-position intervention and LOSO protocol; start up to 80 randomized CAN position/yaw TRAIN demos in parallel with CPU model code | No training/GPU; measured collection needs Webots |
| Nov 29 | Finish or continue accepted-demo collection; freeze V3 data manifest and scene-grouped split | No holdout; language demos only if time |
| Nov 30 | Implement A offline trainer and anti-disguise tests; run object-position intervention and LOSO memorization check | If intervention fails, report **“memorized TRAIN scenes”** |
| Dec 1 | T4 train A only after CPU review and approval; measure learned-vs-expert waypoint error | Stop A if waypoint error does not beat scripted jitter |
| Dec 2 | TRAIN-only closed-loop A prototype; verifier checks close/lift/transport if competent | No full admission, no holdout |
| Dec 3 | Fit B non-privileged context; run zeroed/shuffled-prediction ablations; label oracle-derived fields | No Eq. 3.2 claim without intervention evidence |
| Dec 4 | C comparator only if A/B fail; D labelled hybrid fallback if no learned waypoint competence; freeze V3 config and evidence | Never label D as VLA-primary; no new training |
| Dec 5 | **Hard stop**: report ranked outcome in five lines; any VALIDATION/TEST evaluation needs explicit owner approval after the frozen commit | Stop regardless of result |

### Owner decisions and remaining approvals

**Directed by the 2026-11-28 addendum:**

- V3 remains a separate arm; frozen V1/V2 results are never modified.
- Up to 80 additional TRAIN demos are prioritized for randomized CAN
  position/yaw inside the reachable envelope, starting in parallel with model
  code; language-varying demos are secondary.
- TRAIN admission is in-sample over 8 base scenes and requires the
  object-position intervention and leave-one-base-scene-out memorization check.
- Option B must avoid privileged object pose where possible and label any
  oracle-derived field with zeroed/shuffled-prediction ablations.
- V3 experiments stop on Dec 5 and report daily in five lines.

**Still required before the corresponding action:**

1. Confirm whether V3 code/data work continues on `work/vla-recovery` or moves
   to a new `work/v3-feasibility` branch from commit `4c668e2`.
2. Approve the Webots collection host or start command for the 80-demo
   randomized-CAN campaign. This environment has no Webots executable or
   `WEBOTS_HOME`, so measured collection cannot start here **MEASURED
   environment limitation**.
3. Approve the approved Colab T4 session for bounded A/B training after CPU
   implementation review; no local GPU and no new dependency without a
   `docs/DECISIONS.md` licence note.
4. Approve TRAIN-only diagnostic prototypes on the existing three diagnostic
   identities; no 24-trial admission and no threshold change.
5. Approve activation of the prepared Eq. 3.2 head only under B's controlled
   prediction interventions and only as a TRAIN diagnostic.
6. Give explicit approval for any V3 evaluation on VALIDATION/TEST after the V3
   configuration is frozen by commit. Before that approval, only TRAIN
   diagnostics are permitted.
7. Confirm that V1, V2, frozen `docs/RESULTS.md`, decoder, IK, SafetyMonitor
   thresholds, the `>=10/24` gate and the sealed holdout remain immutable.

Without these remaining approvals, this document and the daily report are the
complete deliverable and no corresponding action will be implemented **MEASURED
commitment**.
