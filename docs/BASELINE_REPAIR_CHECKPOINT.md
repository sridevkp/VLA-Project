# Fresh-v1 expert repair checkpoint — 2026-11-14

**MEASURED where linked below.** ADR-0029, the December 12 deadline, frozen splits,
scene geometry, seeds and literal gate formula are unchanged. Fresh-v1 remains a completed
failed diagnostic baseline (0/24), not successful demonstrations or a zero-success VLA pass.

## Fresh-session reproducibility

Evidence: `runs/eval/repro-train-clear-i03-s0-fresh-v4/`, including `measured-comparison.json`,
`comparison.json`, three `summary.json` files, step/action recordings, logs and analysis source.
Protocol: three independent Webots launches, fixed `train-clear-i03-s0`, seed 0, rendering on;
a capture reset followed by the policy reset in each process. The comparison below uses the
**policy reset**, with the extra capture reset disclosed; the grasp comparison uses only the
policy reset. These protocols are not pooled as identical experiments.

| Measurement | Run 0 vs 1 | Run 0 vs 2 |
|---|---:|---:|
| Initial CAN world pose | identical | identical |
| Max initial arm joint difference | 3.25e-14 rad | 0.149 mrad |
| Max initial EE coordinate difference | 7.22e-15 m | 0.150 mm |
| First joint command difference >1 mrad (zero-based index) | 23 | 1 |
| Max joint command coordinate difference | 0.1915 rad | 0.6851 rad |
| Max matched-step CAN coordinate difference | 41.2 mm | 78.6 mm |

Step rows: 349 / 360 / 455; final distances: 0.29638 / 0.26526 / 0.29301 m.
Outcomes: **0/3 success**. Runs 0/1 fail CLOSE (`misaligned at close`); run 2 fails TRANSIT
(trust-region IK best error 18.886 mm > 15 mm). Maximum CAN rises are only 5.06 / 7.75 /
7.38 mm. Thus categorical zero success is consistent, while trajectories and failure states
are **not reproducible**. Tiny initial floating-point differences amplify; this evidence does
not isolate their source or prove deterministic physics/IK.

Earlier recorder failures/interrupted runs are retained in v1/v2/v3 directories and excluded.
The prepared recorder serialized joint dictionary keys as floats; that harness bug was fixed
without changing policy commands. Both completed comparison recordings are numeric traces,
not RGB videos. Repetition attempts 1/2 were interrupted by SIGTERM during PREGRASP and
are excluded from outcome counts, with `INTERRUPTED.json` records preserved.

## Fixed-scene instrumented comparison

Evidence root: `runs/eval/grasp-train-clear-i03-s0-v2/`. `gt_pad` is **PRIVILEGED DIAGNOSTIC**,
using Supervisor geometry to align; it is not deployable perception. Both modes retain
rendering and the existing safety path. Individual finger positions, pad midpoint, can pose,
closure feedback, contacts and all policy boundaries are retained.

- **none:** failed DESCEND, arm-6 tracking timeout (44 mrad residual), before closure.
  PREGRASP pad XY error 42.48 mm; final pad XY error 43.00 mm, pad z 0.913812 m versus
  can z 0.810613 m. Fingers stay open at approximately 45.01/45.00 mm. Maximum can rise
  0.012 mm; no grasp, lift, transport or placement. Final target distance 234.92 mm.
- **gt_pad — PRIVILEGED DIAGNOSTIC:** physically successful. PREGRASP exit pad XY error
  **0.886 mm**, CLOSE entry **0.866 mm**. CLOSE exit finger positions **28.95/32.28 mm**,
  average opening **30.61 mm**, pad XY error **3.34 mm**. LIFT exit can rise **83.28 mm**;
  maximum rise **99.71 mm**. At TRANSIT exit can pose is
  `(-1.04066, -4.29188, 0.900104) m`, with pad XY error **2.96 mm**. Final can pose
  `(-1.02387, -4.25187, 0.811072) m`; final distance **54.63 mm**, height error **0.012 mm**,
  tilt **0.068 degrees**, all four frozen final criteria true. Both raw comparison
  recordings and separate physical reports are retained; this is one comparison pair,
  not a population-level success-rate estimate. Individual sensor readings also expose an unresolved
  carry-phase detail: at TRANSIT exit gt_pad reports **1.97/59.25 mm** and the first corrected
  run **61.31/0.016 mm**, exceeding the configured 45 mm nominal stroke on one side.
  The numerical records are preserved; no gripper tuning or data approval is inferred from
  physical task success. Resolve this signal before approving demonstration semantics.

`physical-report.json` records the physical samples at state boundaries. Centre-between-fingers
is explicitly a geometry proxy, not evidence of contact or lift. Empty `contact_nodes` does not
prove absence of contact (the successful corrected run also reports an empty list).

## Smallest measured geometry correction

A constant tool-frame translation is unsupported: the measured pad-minus-FK vector varies with
pose in that frame. Fitting a rigid **URDF-root to robot-base** registration on the completed
`none` recording instead gives mean residual 0.336 mm and max 1.250 mm. Holding the fit fixed
on the independently recorded **open-gripper** gt_pad path gives **0.735 mm RMS**,
mean **0.729 mm**, max **1.250 mm** over **3465** held-out samples; final statistics and
counts are retained with the evidence. Closed,
asymmetric finger positions move the physical midpoint relative to this nominal TCP and are
measured separately; this fit is not a claim of submillimetre closed-gripper midpoint accuracy.

ADR-0034 implements one fixed chain element for consistent FK/IK/scoring, with validated,
provenanced YAML in `configs/calibration/urdf-base-train-v1.yaml`. No runtime ground-truth
input enters the correction. Default remains disabled for legacy runs; the corrected campaign environment selects
it explicitly for both policy and backend. No target, safety, solver, or placement tolerance
was changed. Regression tests include held-out recorded geometry, inverse-target consistency,
unchanged joint bounds, invalid-transform rejection and OFF equivalence.

**First corrected none trial:** physically successful. `runs/eval/grasp-train-clear-i03-s0-registered-v1/none/`
contains the result, physical report and raw recordings. Maximum can rise **139.64 mm**;
CLOSE exit finger feedback **34.78/26.65 mm**, opening **30.72 mm**, pad XY error **2.09 mm**.
At LIFT exit can rise is **121.07 mm**; at TRANSIT exit the can is at
`(-1.02867, -4.31405, 0.939405) m`, showing transport while lifted. After release/HOME, can pose
is `(-1.00667, -4.30717, 0.811071) m`; final distance **13.95 mm**, height error **0.011 mm**,
tilt **0.065 degrees**, all frozen placement checks true. Success is supported by this motion,
not HOME alone. Two unchanged fresh-session repetitions also succeeded. Completed acceptance:

| Run | Maximum can rise | Final placement error | Outcome |
|---|---:|---:|---|
| First | 139.64 mm | 13.95 mm | physical success |
| Repeat 3 | 136.98 mm | 15.26 mm | physical success |
| Repeat 4 | 139.06 mm | 14.52 mm | physical success |

**3/3 completed fixed-scene successes**, with the two interrupted attempts reported separately.
Every completed run has physical recordings and a report; the latter two also have RGB frames.
This establishes repeated fixed-scene success, not success on all frozen scenes. Calibration
coefficients were fixed before all three trials. Acquisition/decoder provenance is distinct:
first trial used the old decoder, repeats 3/4 used raw BGRA decoding for the diagnostic check
(their server starts precede the final array-decoder edit). The final baseline retains
`getImageArray()` acquisition with the verified linear-order reshape; no faster camera path
is adopted without the separate required equivalence campaign.

## VLA preparation and remaining acceptance

The Colab notebook rejects absent/unapproved successful TRAIN data and prepares numbered Drive
checkpoint resumption. Downloaded checkpoint config dimensions and candidate API provenance are
recorded in `VLA_NOTES.md`; no installed-policy forward pass or training is claimed. A real RGB
frame inspection exposed a row-layout bug in the existing Webots array decoder; the linear-array
reshape fix is separate from the geometry correction. Its final production acquisition path
is verified live in `runs/eval/rgb-layout-check-v1/camera.png` (640x480 RGB). Existing
captured images from the old decoder are diagnostic artifacts, not approved demonstrations.

## Repaired frozen baseline and verification

The frozen 24-episode train rerun is **RUNNING**, not completed, in
`runs/eval/baseline-train-s0-fresh-v2-registered/`. A preflight compared every generated
scene/object/offset/seed entry against fresh-v1's plan and required exact equality (24/24).
Rendering and RGB acquisition remain on, RGB frames are recorded, `grasp_servo=none`, and
both backend and policy use the frozen static registration. Each outcome records the
registration path/SHA-256; the campaign directory also fingerprints config and source files.
The separate supervisor retains exit status, an actual-can-rise audit from recorded NPZ data,
and the literal gate threshold after completion. VLA-only is unmeasured; no VLA gate pass is
claimed. If the corrected rate is zero it flags an explicit owner decision; it never changes
the formula. No failed run is approved as a demonstration.

Acceptance/check output:

```text
ruff check .: All checks passed!
ruff format --check .: 108 files already formatted
pytest -m "not webots": 260 passed, 516 warnings in 159.59s
registered fixed TRAIN physical acceptance: 3/3 completed successes
none: success=false; failure_state=DESCEND; final_target_distance=0.2349160404388185
gt_pad [PRIVILEGED DIAGNOSTIC]: success=true; final_target_distance=0.05463311925979174
```

Evidence: `runs/eval/repair-required-checks.log` (the exact required uv command chain),
plus `repair-final-checks-v2.log` and its status JSON. Live Linux Webots runs
above are measured. Windows live execution, installed LeRobot forward pass, Colab resumption,
and the new 24-episode result remain unverified. Optional repository-wide `ty` checking has
42 diagnostics in pre-existing scenario/eval paths; it is not claimed clean.
