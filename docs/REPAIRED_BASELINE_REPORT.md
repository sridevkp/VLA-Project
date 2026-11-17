# Repaired frozen TRAIN baseline and demonstration readiness

**MEASURED: 23/24 successes (95.8%), Wilson 95% interval [79.8%, 99.3%].**
Eight TRAIN scenes × seeds 0, 1, 2. Repeated seeds within scenes are correlated; these are descriptive binomial intervals, not a claim about held-out scenes or general deployment reliability.

## Preserved evidence

- Repaired campaign: `runs/eval/baseline-train-s0-fresh-v2-registered/`.
- Full 24 outcomes: `episodes.csv`; aggregate and per-scene intervals: `summary.json`; complete scene/seed geometry: `plan.json`.
- Independent CAN trajectory/placement measurements and recording SHA-256 hashes: `physical-audit.json`.
- Original failed diagnostic baseline remains separate at `runs/eval/baseline-train-s0-fresh-v1/` (0/24). No recordings or statistics were pooled.
- The full scene/seed/geometry plan matches fresh-v1 exactly; frozen configuration, policy, backend, calibration and world hashes match retained provenance. Rendering and production `getImageArray` RGB acquisition remained enabled.

## Per-scene results

| TRAIN scene | Family | Success | Rate | Wilson 95% interval |
|---|---|---:|---:|---|
| train-clear-i00 | clear | 3/3 | 100.0% | [43.9%, 100.0%] |
| train-cluttered-i01 | cluttered | 3/3 | 100.0% | [43.9%, 100.0%] |
| train-reach-limited-i02 | reach-limited | 2/3 | 66.7% | [20.8%, 93.9%] |
| train-clear-i03 | clear | 3/3 | 100.0% | [43.9%, 100.0%] |
| train-cluttered-i04 | cluttered | 3/3 | 100.0% | [43.9%, 100.0%] |
| train-reach-limited-i05 | reach-limited | 3/3 | 100.0% | [43.9%, 100.0%] |
| train-clear-i06 | clear | 3/3 | 100.0% | [43.9%, 100.0%] |
| train-cluttered-i07 | cluttered | 3/3 | 100.0% | [43.9%, 100.0%] |

## Physical outcomes and remaining failures

All 23 successful outcomes have independently recorded can rise ≥40 mm. Successful-run maximum rise spans 128.47–140.63 mm. Independent final placement disagrees with 0 successful outcomes.
HOME alone is not success. The frozen distance/height/tilt criteria remain unchanged; final CAN coordinates and orientation are independently read from recorded ground truth.

Failure categories: `{"TRANSIT": 1}`. CLOSE, DESCEND, LIFT and completed-but-unsuccessful placement counts are CLOSE=0, DESCEND=0, LIFT=0, completed-but-unsuccessful-placement=0.

- Episode 18, `train-reach-limited-i02-s2`: **TRANSIT**, IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.016387 m, tilt=0.411 deg. Actual max rise 136.92 mm; final height above initial rest 130.84 mm; target distance 128.37 mm. Preserved as a failure and excluded from demonstrations.

At the failed stop, the accepted gripper target remained zero while measured mean finger displacement was 30.65 mm. The last accepted joint target differed from feedback by at most 29.04 mrad (arm_3), within the existing 30 mrad intermediate settle tolerance. This identifies the immediate bounded-IK stop without proving global unreachability or assigning an unmeasured cause. Detailed numbers are retained in `remaining-failure-18.json`; new demonstration traces also preserve rejected Cartesian IK targets.

The observed remaining mechanism is reported without changing trust regions, tolerances or solver parameters. The evidence-backed repair remains the static TRAIN-calibrated URDF-to-base registration (ADR-0034); no further policy correction or broad parameter tuning was performed during this frozen campaign.

The baseline retains CAN and nominal-TCP trajectories. Individual pad/finger measurements are in the preserved instrumented none-vs-gt_pad comparison and in the new demonstration physical streams; they are not fabricated for these 24 baseline episodes. `gt_pad` remains **PRIVILEGED DIAGNOSTIC**, never a learned-policy input.

### Every physical episode

| Episode | Scene | Success | Max can rise (mm) | Elevated travel (mm) | Final distance (mm) | Final height change (mm) |
|---:|---|---|---:|---:|---:|---:|
| 00 | train-clear-i00-s0 | True | 135.39 | 235.02 | 13.68 | 0.009 |
| 01 | train-cluttered-i01-s0 | True | 140.40 | 268.50 | 13.83 | 0.010 |
| 02 | train-reach-limited-i02-s0 | True | 140.63 | 319.16 | 14.41 | 0.012 |
| 03 | train-clear-i03-s0 | True | 138.85 | 230.47 | 15.44 | 0.011 |
| 04 | train-cluttered-i04-s0 | True | 137.42 | 279.25 | 13.94 | 0.007 |
| 05 | train-reach-limited-i05-s0 | True | 139.94 | 291.10 | 14.50 | 0.008 |
| 06 | train-clear-i06-s0 | True | 132.70 | 252.20 | 14.40 | 0.012 |
| 07 | train-cluttered-i07-s0 | True | 135.14 | 239.70 | 13.06 | 0.011 |
| 08 | train-clear-i00-s1 | True | 135.64 | 225.01 | 14.19 | 0.012 |
| 09 | train-cluttered-i01-s1 | True | 139.02 | 273.63 | 14.09 | -0.002 |
| 10 | train-reach-limited-i02-s1 | True | 136.71 | 289.77 | 14.87 | -0.022 |
| 11 | train-clear-i03-s1 | True | 136.19 | 162.64 | 14.55 | 0.012 |
| 12 | train-cluttered-i04-s1 | True | 131.63 | 222.03 | 14.88 | -0.021 |
| 13 | train-reach-limited-i05-s1 | True | 137.96 | 242.25 | 15.13 | 0.009 |
| 14 | train-clear-i06-s1 | True | 136.16 | 202.90 | 13.85 | 0.009 |
| 15 | train-cluttered-i07-s1 | True | 132.67 | 237.01 | 15.21 | 0.010 |
| 16 | train-clear-i00-s2 | True | 137.79 | 249.34 | 13.99 | -0.024 |
| 17 | train-cluttered-i01-s2 | True | 132.64 | 253.15 | 15.35 | -0.001 |
| 18 | train-reach-limited-i02-s2 | False | 136.92 | 213.11 | 128.37 | 130.840 |
| 19 | train-clear-i03-s2 | True | 134.33 | 274.44 | 14.66 | 0.011 |
| 20 | train-cluttered-i04-s2 | True | 132.80 | 238.06 | 14.11 | -0.013 |
| 21 | train-reach-limited-i05-s2 | True | 137.53 | 141.38 | 13.88 | 0.009 |
| 22 | train-clear-i06-s2 | True | 128.47 | 260.63 | 14.71 | 0.009 |
| 23 | train-cluttered-i07-s2 | True | 136.89 | 266.78 | 13.51 | -0.026 |

## Runtime and provenance

Total measured campaign wall time, including serial phase and handoff: **8132.00 s (135.53 min)**. Parallel-resume phase: 3296.15 s. Compatibility preparation ran concurrently, so this is shared-host campaign runtime, not an exclusive policy throughput benchmark.
The first 11 episodes ran serially. Scheduling changed only at a completed episode boundary to two disjoint fresh-session workers for the remaining 13; no physical episode was interrupted. `orchestration-handoff.json`, `parallel-runtime.json`, worker logs and `campaign-status.json` preserve timing scopes. Final `timings.json` measures aggregation of resumed recordings and must not be quoted as simulation runtime. Earlier serial per-episode timings were overwritten by the original CLI; total serial runtime is retained, not reconstructed.
Registration SHA-256: `dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0`. Full frozen-source fingerprints: `provenance.json`; recording fingerprints: `physical-audit.json`; numerical environment checkpoint: `numerical-environment-checkpoint.json`.

Collision displacement/tilt PROXY positive in 24/24 episodes; it tracks intended CAN movement as well as other objects and is **not collision detection**. Recorded contact steps: 0; FK clearance is unmeasured. No collision-free claim is made from these channels.

## Demonstration readiness and installed compatibility

Successful TRAIN demonstration collection can begin at the measured reliability above; perfection is not required. The versioned fresh collection is `runs/demos/train-s100-registered-v1/`, target 80 independently admitted successes, maximum 240 attempts, TRAIN seed blocks starting at 100. Failed and interrupted attempts remain quarantined. It requires full-step 31.25 Hz RGB, closure/pad measurements, grasp enclosure, ≥40 mm lift, carried motion and final placement; every action retains the safety veto. No validation/test scene is used.
The gripper contract is measured: opposing slider positions in metres, absolute symmetric per-finger command 0–45 mm, soft motor limits without hard stops. Sensor excursions above 45 mm are feedback and are never substituted for commands. The empty probe and 6,920-sample pad-geometry validation are retained in `runs/gripper-semantics-v1/`.
Installed strict pretrained inference, registered IK/safety decoding, native image dataset round trip, exact training-config parsing, saved-processor reload, actual DataLoader-collated training forward and padded-tail masking passed. See `VLA_NOTES.md` and evidence directories. The actual successful-data preview and final dataset repeat native reload/normalization/forward checks; synthetic fixtures never count toward the target.
Resumable Colab workflow is prepared with isolated pinned dependencies and complete numeric model/processor/optimizer/scheduler/RNG/step checkpoints, without symlinks. Training approval requires approximately 80 physically verified TRAIN successes and exact dataset provenance/statistics hashes. No fine-tuning has started. Colab CUDA throughput and Windows execution remain unverified.

## Literal VLA gate and deadline

The literal rule is `VLA_rate >= min(0.40, 0.60 * repaired_scripted_rate)`, giving **40.0%**. No VLA-only evaluation exists, so the gate result is **UNMEASURED**. The failed fresh-v1 diagnostic baseline cannot be used to claim a zero-success VLA passes. Frozen evaluation criteria, splits, seeds and the **December 12** deadline remain unchanged.

## First successful-data admission checkpoint

The first two fresh TRAIN attempts (seed 100, clear-i00 and cluttered-i01) both pass the
independent physical admission audit. Max can rise is 139.78 / 138.42 mm, enclosed carried
horizontal motion 270.10 / 233.08 mm, final distance 14.27 / 15.20 mm. Both have all four
physical checks, all frozen final success flags, full-step RGB and aligned individual
finger/pad telemetry, with zero safety rejections. Individual sensors reach 61.44 / 60.79 mm,
consistent with the verified soft-limit semantics; action labels retain commanded targets.

Collection is running toward 80 in `runs/demos/train-s100-registered-v1/`. Current admission
counts are in `collection-status.json`; failed/partial episodes remain quarantined.
A separate two-success preview is exported and checked before final 80-success export.
Neither an early preview nor compatibility fixtures approve training.

The two-success native preview **passed**: 1,907 frames at 31.25 Hz, camera1/state8/action7,
exact first/last RGB/state/action round trip, real dataset MEAN_STD arithmetic, strict
pretrained inference `(1,50,7)` and actual DataLoader-collated training forward with tail
padding masked. Training loss 0.0336075 is a finite compatibility check, not a trained-policy
performance result; optimizer steps remain zero. Evidence:
`runs/datasets/tiago-train-preview-v1/{source-provenance.json,installed-forward.json}`.
Original JPEG embedding reduced Parquet storage from 464,816,674 to 99,428,761 bytes without
image re-encoding. The early subset cannot approve training: no `training_approval.json`
is issued until the configured approximately 80 successful episodes pass final validation.

Required checks: `uv run ruff check .` and `uv run ruff format --check .` pass;
`uv run pytest -m "not webots"` reports **291 passed, 1 skipped** (optional PyArrow test).
That optional image integration test separately passes in the installed VLA environment.
Changed modules pass targeted type checking; full `src tests` type checking still reports
42 pre-existing diagnostics. Windows execution and Colab GPU execution remain unverified.
