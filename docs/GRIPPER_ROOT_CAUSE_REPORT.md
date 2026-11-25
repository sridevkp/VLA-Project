# Gripper overshoot: read-only root-cause report

**Finding:** the saved chunk crosses the physical boundary through an unconstrained
learned output. No unit/semantic mismatch or gripper-label corruption was found.
The existing adapter and safety veto behave as specified. No inference, training,
Webots, evaluation, clipping, limit change or behavior-changing correction was run.
This explains the observed interface failure; it does not establish why the learned
approximation has this particular bias, or a task success rate.

## Quantitative evidence

Read all **80 approved TRAIN episodes / 63,081 frames**. Verified every source
command-log and numeric-step SHA against the approved source provenance. Replayed
the held gripper command at every frame using the exporter's timestamp rule:
**63,081/63,081 native gripper labels match exactly after float32 serialization**.
All240 recorded opening commands were accepted and ordered.

| Quantity | Measurement |
|---|---|
| Original physical commands and held labels | Only0 and0.045m; zero above0.045m |
| Closed labels |22,975 (36.4214%) |
| Open labels |40,106 (63.5786%) |
| Native float32 min/max, promoted to float64 |0 /0.04500000178813934m |
| Literal native values above decimal0.045 |40,106; exclusively the float32 representation of the valid endpoint |
| Values above the existing endpoint-aware limit |0 |
| Physical labels within1µm,0.1mm,0.5mm or1mm of upper bound |40,106 in each band; no intermediate command levels |
| Saved TRAIN gripper mean/std |0.02861039899289608 /0.021654436364769936m |
| Recomputed global label mean/population std |0.02861035924787363 /0.021654412045152234m |
| Saved50 predictions: min/max |0.04416652396321297 /0.045476075261831284m |
| Saved prediction mean/std |0.04507563881576061 /0.00027709324330209176m |
| Predicted violations |35/50 above both0.045 and the serialized label maximum |
| Maximum excess |0.4760753mm, equivalent to0.0219851 TRAIN standard deviations |
| Matched episode0/frame0–49 labels |All0.045m; prediction MAE0.2249173mm, RMSE0.2872314mm |

The tiny saved-versus-recomputed statistic differences are0.000039745mm in mean
and0.000024320mm in std; neither is a demonstrated explanation for the0.476mm excess.
Saved statistics match the approved dataset metadata and both processor files.
Training normalizes the open label to approximately0.756870091 (including eps1e-8).
The saved physical predictions imply postprocessor-input coordinates0.718380507 to
0.778855473. These are affine inversions of recorded outputs, not newly captured
model intermediates. The preceding processor audit's round-trip error was0.0.

## Complete signal path and meaning

1. **Expert:** sends `GripperCmd(0.0)` for close and `GripperCmd(GRIPPER_OPEN)` for
   open. These are commanded half-apertures, not measured finger joint positions:
   [scripted_can.py](/home/tj330/programs/VLA-Project/src/vlarobot/execution/scripted_can.py:491).
   The backend maps that opening to both slider targets using its closed/open
   coordinates and full stroke:
   [backend.py](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/backend.py:1002),
   [robot_io.py](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/robot_io.py:103).
2. **Recording:** records the actual command plus acceptance verdict after backend
   apply. Observed gripper feedback is a separate stream:
   [recorder.py](/home/tj330/programs/VLA-Project/src/vlarobot/eval/recorder.py:338),
   [record_action](/home/tj330/programs/VLA-Project/src/vlarobot/eval/recorder.py:205).
3. **Labels/storage:** `derive_command_labels` holds the most recent accepted
   `opening` value and stores it as action coordinate6, converted to float32.
   It explicitly excludes future measured/stall opening as a label. Observation
   state coordinate7 instead stores measured mean per-finger opening:
   [dataset_export.py](/home/tj330/programs/VLA-Project/src/vlarobot/vla/dataset_export.py:21),
   [state construction](/home/tj330/programs/VLA-Project/src/vlarobot/vla/dataset_export.py:135),
   [schema.py](/home/tj330/programs/VLA-Project/src/vlarobot/vla/schema.py:11).
   Observed state openings reach0.0451321714m (31,146 frames above0.045), but the
   exact command replay proves these excursions did not contaminate target labels.
   Feedback uses mean absolute displacement from each slider's closed position,
   not the sum/full jaw gap:
   [gripper.py](/home/tj330/programs/VLA-Project/src/vlarobot/backends/webots/gripper.py:17).
4. **Training normalization:** approved metadata supplies float32 statistics to
   both processors. MEAN_STD is `(x-mean)/(std+eps)`; inverse is `z*std+mean`:
   [training.py](/home/tj330/programs/VLA-Project/src/vlarobot/vla/training.py:332),
   [installed normalize_processor.py](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/processor/normalize_processor.py:325).
   No metres/mm, factor-two, joint-coordinate or absolute/delta conversion occurs
   for the gripper. Only the six pose coordinates are deltas.
5. **Learned output:** action head is linear; training uses flow velocity MSE;
   inference starts from noise and integrates `x_t += dt*v_t`, returning `x_t`.
   No physical clamp/sigmoid/tanh bounds this path. Frozen ALOHA adaptation is false:
   [linear head](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py:575),
   [loss](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py:790),
   [flow sampler](/home/tj330/programs/VLA-Project/.venv-vla/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py:808).
6. **Inference denormalization:** the project passes the chunk through the saved
   postprocessor once, then returns finite physical50x7 values:
   [model_service.py](/home/tj330/programs/VLA-Project/src/vlarobot/vla/model_service.py:86).
7. **Adapter/safety:** action coordinate6 is an absolute `GripperCmd` opening in
   metres. Safety validation precedes IK/motor application:
   [action_adapter.py](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:105).
   All stages agree on absolute per-finger opening; there is no demonstrated semantic mismatch.

## Exact endpoint rule

[ActionAdapter lines113–118](/home/tj330/programs/VLA-Project/src/vlarobot/vla/action_adapter.py:113)
canonicalize **only exact equality** to each configured endpoint's float32
representation.0.045 becomes0.04500000178813934 (excess1.788nm). The next float32
value,0.04500000551342964, is already rejected. This is not an adjustable closeness
window or a tolerance of one ULP above the stored endpoint.

[SafetyMonitor](/home/tj330/programs/VLA-Project/src/vlarobot/execution/safety_monitor.py:185)
requires a finite value and rejects `opening < lower or opening > upper`, with
bounds `[0,0.045]`. There is no `isclose` allowance. The scripted2mm opening/stall
observation tolerance is a different predicate and grants no command allowance.
All35 predictions exceed the endpoint representation exception; widening that
exception would change behavior.

## Contract assessment and next action

[ADR-0029](/home/tj330/programs/VLA-Project-analytic/docs/DECISIONS.md:1305)
requires invalid-output rejection through existing IK/safety; the
[architecture](/home/tj330/programs/VLA-Project-analytic/docs/ARCHITECTURE.md:344)
requires rejection of injected out-of-limit actions. Neither located text grants
general action projection. The actual frozen VLA-only implementation records a
`SAFETY` failure and stops on a rejected action, with no fallback:
[vla_episode.py](/home/tj330/programs/VLA-Project/src/vlarobot/execution/vla_episode.py:119).
The Phase I report PDF remains unavailable in the searched project/report paths;
ADR-0029 explicitly identifies it as missing. Its detailed projection permissions
cannot be certified or invented. Steering directives prohibit clipping and limit changes.

| Option | Status under current instructions |
|---|---|
| Preserve outputs and reject out-of-limit commands; report veto causes | Contract-compliant existing behavior; no correction needed to the meaning/mapping |
| Later evaluate unchanged VLA-only with mandatory motor safety and failures recorded | Planned contract-compliant measurement, **but currently prohibited until owner releases the evaluation stop** |
| General clamp, near-boundary snap, bounded output mapping or resampling/skipping unsafe actions | Behavior/policy change; requires explicit approval and report conformance review; not authorized |
| Widen limits, change thresholds, relabel dataset, retrain, alter weights/loss/model | Prohibited; would require explicit authorization/formal contract review |
| Scripted fallback inside VLA-only or heuristic/world-model assistance | Changes frozen arm semantics; not permissible as a silent repair |

**Recommendation:** retain the verified checkpoint and exact rejection behavior.
The demonstrated issue is unconstrained prediction overshoot near an endpoint-rich
label distribution, not a defective unit conversion needing an adapter fix. Do not
introduce a projection as a purported serialization correction. The next decision
is whether to authorize the already planned unchanged VLA-only measurement with
safety rejections recorded, or explicitly review a separately versioned output
policy against the authoritative report. Neither is executed here. A single chunk
cannot decide the literal24-episode10/24 gate or trigger its scripted-primary fallback.

Evidence and reproducible read-only inspection:
`runs/gripper-root-cause-v1/{inspect_labels.py,label-analysis.json,label-analysis.log}`.
Run the saved inspection script with the existing `.venv-vla/bin/python` only into
a new copied/versioned evidence directory if repetition is needed; no model import
or inference is performed. [Committed numeric evidence](evidence/gripper-root-cause-v1.json)
includes source verification, all50 saved values and per-episode coverage. The
portable model SHA remains7531991e27719e6fee402344aadbe43b2fc4ac858a87ae5ec77809c8efe385d8.
