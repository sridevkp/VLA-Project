# Failed TRAIN VLA-only campaign: read-only postmortem

**The final result remains0/24, frozen gate FAILED, scripted-primary under ADR-0029.**
This investigation does not correct, reinterpret or replace any outcome. It uses only
preserved TRAIN records, approved TRAIN demonstrations, saved configurations and installed
source. No inference, simulation/rollout, new evaluation, training, threshold change,
projection or policy modification was performed. All225 original campaign files and all21
export members were checked against their existing hashes before analysis.

## 1. Gripper error originates in an unconstrained prediction path

**Observed.** All24 saved50-action chunks contain an upper-bound violation. Across all
1,200 proposals, **761 (63.4167%)** exceed the existing endpoint-aware limit; none is negative.
Physical targets range **0.043113556–0.045792781m**, mean0.045058982m. These whole-chunk
figures include unexecuted predictions. The21 actual SAFETY vetoes are the first rejected
commands in those episodes, not761 physical applications. Of1,200 proposals,85 were accepted,
21 rejected before backend submission, and1,094 were never reached after episode termination.
No output was rewritten or skipped to improve results.

The pre/postprocessor statistic tensors match each other and approved metadata exactly.
Action coordinate6 uses mean **0.02861039899289608m**, std **0.021654436364769936m**.
The saved pipeline is STATE/ACTION=MEAN_STD, VISUAL=IDENTITY. The installed code applies

`z=(x-mean)/(std+1e-8)` for training; `x=z*std+mean` for inference postprocessing.

No clipping, unit conversion, second inverse normalization, finger-gap doubling or gripper
sign inversion appears on this path. Model server calls the postprocessor once:
[model_service.py:86](/home/tj330/programs/VLA-Project/src/vlarobot/vla/model_service.py:86),
[normalization formulas](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/processor/normalize_processor.py:325).
The earlier installed-processor round-trip audit reported0.0 error; this investigation
checks saved statistics/configuration without loading a policy or repeating inference.

**Inference from the observed affine mapping.** The physical boundary corresponds to
postprocessor-input coordinate **0.7568703582**. Saved outputs imply coordinates
**0.6697545261–0.7934809078**: some predictions necessarily lie beyond the normalized
open-label endpoint if this recorded pipeline is applied. Thus an affine inverse exposes
rather than creates a physical range constraint. Its eps round-trip effect at the open
label is nanometres and cannot explain the maximum0.792781mm whole-chunk excess.

The installed model has a linear output head and returns the integrated flow sample
without a physical bounding operation:
[linear head](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py:575),
[flow update/return](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py:826).
The model's configuration disables ALOHA adaptation. Action padding is32 internally and
sliced to the saved7-D interface; gripper remains coordinate6.

**Unverified.** Original pre-postprocessor tensors were not saved. The equivalent
normalized values above are algebraic inversions, not captured hidden intermediates.
The artifacts support unconstrained prediction overshoot; they do not isolate whether
finite training, sampling noise, a specific learned bias or another learning dynamic
produced the exact errors. No causal ablation was run.

## 2. The0.045m boundary and physical meaning are consistent

**Observed.** Prior full replay verified80 episodes/63,081 labels against their accepted
source commands: only0 and0.045m physically, zero out-of-limit command labels. Open labels
number40,106; closed labels22,975. Native float32 stores the open label as
**0.04500000178813934m**, a1.788nm representational excess. The exact endpoint canonicalization
already handles that value; it is not a general tolerance:
[ActionAdapter:113](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:113).
The next float32 value0.04500000551342964 is rejected. Safety is finite and inclusive
`0≤opening≤0.045`, with no isclose allowance:
[SafetyMonitor:185](/home/tj330/programs/VLA-Project/src/vlarobot/execution/safety_monitor.py:185).

The model has no physical limit guarantee simply because labels are bounded. Its gripper
output is nevertheless interpreted consistently as **absolute per-finger opening in metres**.
Observed finger/mean-aperture feedback is a separate state feature and is not the label:
[dataset_export:21–53](/home/tj330/programs/VLA-Project/src/vlarobot/vla/dataset_export.py:21),
[schema](/home/tj330/programs/VLA-Project/src/vlarobot/vla/schema.py:11).
The three inverse pose-delta rotations are also consistent: labels use `q_target*q_source^-1`,
while execution uses `q_delta*q_current`; translation is target-minus-source, then added to
current position. They are desired controller-target residuals, **not measured velocity or
promised one-step displacement**:
[label pose_delta](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:62),
[decode](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:123).
No scale/frame/action-schema mismatch was demonstrated.

**Hypothesis.** Residual EE targets may be realized differently by the VLA's bounded full-pose
IK than the original expert's joint targets. A redundant7-joint arm does not have a unique
inverse for a6-D target. This is a possible control-realization difference, not an observed
metres/radians or absolute/delta mismatch.

**Unverified.** No expert-target replay through this adapter, Jacobian/null-space analysis
or alternative controller test was performed. Actual motor lag is not itself proof of a
representation defect. For the85 accepted proposals, translation-target norms were
0.0688–0.8077mm; observed one-step EE movement was0.0155–0.2979mm.

## 3. Chunk plausibility: initial approach, not an observed full task

**Observed.** All24 episodes requested only their **first50-action chunk**, approximately1.6s
at32ms/action. Maximum accepted prefix was16 actions (0.512s); total accepted actions85.
All predicted gripper values stayed at least43.114mm open. There is no close/grasp command
in these chunks and no recorded object lift. All six pose coordinates lie within the
respective global TRAIN label ranges; this establishes marginal scale consistency, not
conditional correctness or safe future execution.

Temporal shape is not flat: average translation norm grows from **0.297mm** over the first
five proposals to **25.999mm** over the last five. Last-five mean dx is+21.411mm and dz is
−9.231mm; maximum translation norm is62.769mm. These are **approach-like proposals** only.
Summing command residuals would not yield a valid predicted physical trajectory, and no
hypothetical rollout was constructed.

Crucially, **all4,000 first50 labels across80 demonstrations are also open**. First close
occurs at frames205–531, median311.5: approximately **6.56–16.99s**, median9.968s.
Thus absence of a close in an initial1.6s proposal is consistent with the training prefix.
No episode survived long enough to query the policy at a later grasp/lift/transport/release
state. Positive dz alone would not prove lift, especially with an open gripper.

There is a narrower observed imitation error: all80 expert first actions have
exactly zero translational/rotational pose residual while opening the gripper.
The24 predicted first actions instead have translation norms0.104–0.863mm and
rotation norms0.000562–0.006238rad. These include vetoed predictions, not24 executed
motions. They show imperfect initial holding precision; whether bounded IK amplifies
those small errors into joint excursions is unverified.

**Hypothesis.** The chunks reflect some learned initial dwell/approach structure, with
inadequate boundary accuracy and possibly weak conditional control.

**Unverified.** Successful grasp, lift, transport or release capability beyond these early
prefixes is neither demonstrated nor ruled out by the saved chunks. Claiming the model
never learned closure would exceed this evidence. The entire deployed campaign nevertheless
failed, and its gate remains failed.

## 4. Exact cause of the three HOME labels

**Observed.** The executor defines `e(t)=max_j|q_j(t)-q_j(reset)|`. Any e>0.02rad sets a
persistent departed flag; any subsequent e≤0.02rad immediately ends the episode as HOME.
It does not require prior grasp, lift or transport before that control termination:
[vla_episode.py:126–131](/home/tj330/programs/VLA-Project/src/vlarobot/execution/vla_episode.py:126).
The final physical scorer correctly rejects the resulting placements; HOME alone is not success.

| Preserved pair | Accepted actions / elapsed simulation | Threshold crossing then return (rad) | Maximum EE movement | Final CAN target error |
|---|---|---|---|---|
| train-clear-i00-s1, episode08 |7 /0.224s |0.03220099→0.01620164, arm3 dominates departure |0.3330mm |194.262mm |
| train-clear-i03-s1, episode11 |4 /0.128s |0.03211904→0.01611956, arm7 dominates departure |0.2343mm |171.709mm |
| train-clear-i06-s1, episode14 |7 /0.224s |0.03208616→0.01608682, arm3 dominates departure |0.2037mm |210.013mm |

[episode08 metadata](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/episode-08.meta.json),
[episode11 metadata](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/episode-11.meta.json),
[episode14 metadata](/home/tj330/programs/VLA-Project/runs/eval/vla-train-s0-step20000-recovered-v1/episode-14.meta.json).
Calculations use each matching `.steps.npz`: `joint_names/joint_pos`, `ee_xyz`, `gt_names/gt_xyz`.
Executed gripper commands remained43.270–44.994mm open. Can rise was only0.162–0.284µm.
The last joint targets change sign around these excursions; this is not evidence of a
completed learned pick-and-place cycle.

These episodes are **without executed vetoes**, not veto-free chunks. Their first future
unsafe values are at zero-based indices12,4,12 respectively; execution ended after7,4,7
accepted actions. No unsafe action was skipped to continue the task: the existing HOME
termination ended each failed episode before those proposals were reached.

**Hypothesis.** Short joint motion/oscillation and the task-agnostic departure/return detector
produce premature termination. The numerical threshold crossing directly explains why the
code stopped; the relative contributions of learned commands, IK redundancy and motor
tracking to the crossing remain unresolved.

**Unverified.** No threshold, termination logic, IK branch or command was altered, and no
continued-execution counterfactual was simulated. Removing this stop would not establish
success: the saved future chunks already contain invalid gripper values.

## 5. Distribution shift and inadequate learning: evidence limits

**Observed.** Reset joint/gripper states are extremely close to demonstration starts:
nearest initial-state distance, standardized by approved TRAIN std, is **9.41e-9–1.31e-8**;
no input coordinate lies outside its global TRAIN range. Camera inputs have the approved
RGB480×640 shape and recorded pixel hashes match the saved PNGs. Mean channel ranges are
similar (TRAIN R101.30–103.19/G69.85–71.41/B46.43–47.10; evaluation
R101.68–102.93/G69.79–71.43/B46.43–47.02 on0–255 scale). This argues against a gross reset,
black-image or channel-scale problem, not against every possible visual shift.

The successful demonstration sample is35 clear/31 cluttered/14 reach-limited episodes.
Within-family coverage is finite: one evaluation clear-x offset is0.074mm below its observed
TRAIN minimum; one reach-limited-y offset is **0.09010m vs TRAIN maximum0.07014m**, about20mm
beyond that sample. Two yaw coordinates lie outside raw empirical ranges; angles wrap, so
those counts alone are not meaningful OOD evidence. The scene generator/splits themselves
are unchanged. Such empirical support gaps do not prove a changed underlying distribution.

Seed blocks differ: upper violations among400 outputs are308 (seed0),139 (seed1),314 (seed2).
Mean gripper target is45.129/44.871/45.177mm respectively. Model noise and scene geometry both
use the paired seed, so these differences **do not isolate sampling noise** or justify seed
selection/resampling. Similar across-scene predictions also do not prove the vision encoder
is ignored when initial robot states and early task phase are similar.

**Hypotheses.** Inadequate learned boundary precision is directly consistent with unsafe
continuous predictions on endpoint-rich labels. Conditional motion fidelity, sparse successful
coverage, sampling variability and controller-realization differences could contribute to
transfer failure. JPEG training inputs versus raw-RGB runtime inputs are an encoding difference,
not a demonstrated decoder/unit error.

**Unverified.** The artifacts do not establish global undertraining, convergence failure,
weak vision conditioning, a particular optimizer fault or a causal domain-shift explanation.
No new policy forward, loss study, matched-observation seed test or extra demonstration was run.

## Ranked possible corrective actions — proposals only, none authorized or implemented

1. **Review an explicitly bounded gripper-proposal contract for a separately versioned policy.**
   Strongest support:21 immediate safety failures and761 unsafe proposals. A bounded/discrete
   gripper head or deterministic projection are possible future designs, not serialization
   repairs. They change model/output behavior and may require retraining; each requires explicit
   approval and Phase I conformance review. Keep current limits and vetoes; never widen0.045m
   or silently clamp this final campaign. No predicted improvement is claimed.
2. **Review task-aware termination semantics for a new protocol version.** Exact evidence: the
   three4–7-action HOME crossings. Any new termination predicate changes episode exposure and
   potentially arm semantics even if final physical thresholds stay fixed. It requires explicit
   approval; do not add scripted task assistance to VLA-only, use oracle state in the proposer,
   rescore this campaign or assume later execution would succeed.
3. **Isolate command-to-motion realization before blaming the representation or retraining.**
   A future authorized matched expert-target/adapter replay and held-observation study could
   separate IK/tracking effects from learned prediction error. New inference, simulated replay
   or counterfactual execution is a new measurement requiring approval; none was performed here.
4. **Only then assess conditional learning and coverage in a new approved training plan.**
   Use TRAIN-only diagnostics of later task phases and deliberate successful coverage if
   authorized. New data/model/loss/batch/step settings require approval and versioned provenance;
   do not modify the approved80-episode dataset or tune on withheld splits.

The authoritative Phase I report remains unavailable in the project. No proposed projection,
termination revision or representation change is certified as contract-compliant from an absent
report. Existing safety rejection and preserving/reporting these failures are compliant now.
No proposed action can retroactively change0/24, the failed40% gate or scripted-primary selection.

## Reproducible evidence and preservation

[Machine-readable calculations](evidence/vla-train-postmortem-v1.json) include all24 chunk
summaries, exact HOME error sequences, training-prefix/closure counts, scene/input comparisons
and explicit uncertainty flags. The saved read-only inspection script and log are in
`/home/tj330/programs/VLA-Project-analytic/runs/vla-train-postmortem-v1/`.
Use the existing `.venv-vla/bin/python analyze.py` only from a new copied/versioned evidence
directory if repetition is needed; its inputs are fixed and it contains no policy or simulator
imports. Derived coordinate calculations are arithmetic on saved physical actions, not
new network predictions. Full authoritative original50×7 chunks remain at
`runs/eval/vla-train-s0-step20000-recovered-v1/episode-NN.latency.jsonl`, one record at line1.
Failed gate, proposer-role and repaired23/24 baseline remain intact and separate.
