# V3 alternatives assessment — learned manipulation routes under the V2 stop rule

**Status:** Two-hour, read-only feasibility assessment. No training, GPU use,
model substitution, gate change, decoder/IK/safety change or holdout access.
V1 remains immutable; V2 step-021000 remains documented at `0/3`; the
`>=10/24` TRAIN gate remains unchanged.

**Evidence files:** `docs/evidence/v3-dataset-measurements.json` (SHA-256
`d8c6310ff737dc6c6a773c44c011af20db43989334bcbd20650f4453bbfb3856`) and
`docs/evidence/v3-feature-timing.json` (SHA-256
`ce4a6b8391f48c23650e53966cfa3fd5dc1e6117b7b76d0475439a1a8e6b9c5d`). Every
numeric statement below is tagged **MEASURED** (directly observed/derived) or
**INFERRED** (engineering estimate from measured anchors). Dates and file paths
are protocol facts, not performance figures.

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

### Minimum additional demonstrations for instruction dependence

Current instruction count is 1 **MEASURED**, so language dependence is
unmeasurable. Hard minimum for a two-instruction paired test is 48 additional
TRAIN demos: 2 strings × 8 base scene identities × 3 seeds **INFERRED**.
Recommended target is 80 additional demos: 2 strings × 8 base scene identities ×
5 seeds **INFERRED**, collected with the same accepted demo audit and no holdout.
Both estimates assume one new instruction variant or goal phrasing; three
variants would require proportionally more **INFERRED**.

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

### Scene-grouped offline split

Use base scene identity as the grouping unit, not random frame splits **INFERRED
protocol**. There are 8 base groups with 7–12 demos each **MEASURED**. Proposed
development split: keep one clear, one cluttered and one reach-limited base
identity for offline validation; train on the remaining 5 groups **INFERRED**.
No frame from a demo may appear in both sides of the split **INFERRED**. Sealed
holdout remains untouched **MEASURED constraint**.

### Eq. 3.2 controlled path

1. Fit a small learned context predictor from TRAIN trajectories to produce the
   predicted context used by the selector **INFERRED**.
2. Hold image, state, language and candidate actions fixed; vary only predicted
   context **INFERRED**. Compare no-context, analytic-context, learned-context
   and counterfactual-context arms **INFERRED**.
3. Counterfactual interventions: set distance/clearance/progress/gripper fields
   to zero, shuffle across candidates, and apply bounded perturbations **INFERRED**.
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
| Nov 26 | Owner approval only; no implementation starts without it | No training, no GPU |
| Nov 27 | Freeze V3 protocol; build scene-grouped split; export waypoint supervision and input hashes | Offline only |
| Nov 28 | Implement A trainer and anti-disguise tests in code, CPU only | No GPU; review before T4 |
| Nov 29 | T4 train A if approved; measure learned-vs-expert waypoint error and ablations | Stop A if waypoint error does not beat scripted jitter |
| Nov 30 | TRAIN-only closed-loop prototype with A; verifier confirms close/lift/transport if competent | No full admission, no holdout |
| Dec 1 | Fit B context/predictor if A passes; run controlled prediction interventions | No Eq. 3.2 claim without intervention evidence |
| Dec 2 | C comparator only if A/B fail diagnostic competence; otherwise skip | No decoder/IK/safety changes |
| Dec 3 | D labelled hybrid fallback if no learned waypoint competence | Never label as VLA-primary |
| Dec 4 | Freeze evidence, write final report, preserve raw artifacts | No new training |
| Dec 5 | **Hard stop**: report ranked outcome; any admission/holdout work needs a new owner decision | Stop regardless of result |

### Exact approval needed from owner

Approve all of the following before any V3 implementation begins:

1. A new `work/v3-feasibility` branch from `d62f313` for protocol/code work only.
2. Permission to build offline waypoint supervision from TRAIN demo logs and to
   use scene-grouped offline splits; no sealed holdout access.
3. Permission to collect 48–80 additional TRAIN demos **INFERRED** with at least
   2 instruction strings **INFERRED**, if the first A prototype shows instruction
   sensitivity is needed.
4. Permission to use the approved Colab T4 session for bounded A/B training after
   CPU implementation review; no local GPU and no new dependency without a
   `docs/DECISIONS.md` licence note.
5. Permission to run TRAIN-only diagnostic prototypes on the existing three
   diagnostic identities; no 24-trial admission and no threshold change.
6. Permission to activate the prepared Eq. 3.2 head only under B's controlled
   prediction interventions and only as a TRAIN diagnostic.
7. Confirmation that V1, V2, frozen `docs/RESULTS.md`, decoder, IK, SafetyMonitor
   thresholds, the `>=10/24` gate and the sealed holdout remain immutable.

Without item 1–7, this document is the complete deliverable and nothing further
will be implemented **MEASURED commitment**.
