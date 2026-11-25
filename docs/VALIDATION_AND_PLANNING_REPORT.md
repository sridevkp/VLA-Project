# Analytic/RiskGate measured checkpoint — 2026-11-15

Canonical physical VALIDATION completed 18 fresh sessions: **15/18**, 83.33%,
descriptive Wilson 95% interval 60.78–94.16%. Clear achieved 6/6 (i00/i03 each
3/3), cluttered 6/6 (i01/i04 each 3/3), reach-limited 3/6 (i02 1/3, i05 2/3).
This is the registered scripted controller, not a trained-policy result or TRAIN gate.

Independent recording checks confirmed all recorded successes: can rise
131.01–140.44 mm, final placement distance 13.85–15.79 mm, absolute final height
error at most 0.750 mm, tilt at most 0.069 degrees and HOME joint error at most
0.01933 rad. HOME alone was not counted as physical success.

Failures: i02/seed0 PREGRASP (7.944 mm against 5 mm tolerance), i02/seed1 TRANSIT
(15.586 mm against 15 mm), i05/seed2 TRANSIT (16.731 mm against 15 mm).
The latter two lifted the can 134.66 and 135.99 mm before transport IK failed.
No thresholds or trust regions were changed.

Three workers completed in 3,167.83 seconds, using Python 3.11.18, NumPy 2.4.6,
IKPy 4.4.4 and SciPy 1.17.1. Exact-pair repeated trajectory reproducibility is not
established. Shared-host scheduling and clustered scene families limit inference.
The earlier mixed-runtime 15/18 remains separate diagnostic evidence, never pooled.
Raw recordings: `runs/validation-evidence-v2-canonical`. Independent checks and
hashes: `docs/evidence/validation-v2-canonical-physical.json`; runtime sidecar:
`docs/evidence/validation-v2-runtime.json`.

The committed identical-input 32×50 optimization measured median 0.468 seconds,
p95 0.480 seconds, versus previous median 2.100 seconds. All 1,600 predicted
ticks, vetoes and 100 FK comparisons were numerically identical. This measures
the analytic kernel, excluding learned inference and certified full-arm collision
computation; loaded-host timings vary.

Live geometry probing reached 17/30 URDF collision-bearing links, recording 110
primitives across 57 objects including two unsupported planes. Collision coverage
is **NOT CERTIFIED**. Passive/base geometry, self-collision exclusions, swept
volumes and carried-object coverage remain unresolved. Inspected Webots
selfCollision was disabled. Missing geometry remains fail-closed. A source-derived
diagnostic wrist collision-origin correction reduced a sampled centre discrepancy
from approximately 54.7 mm to 0.66 mm; this is not a safety certificate or a change
to the accepted arm registration/controller.

Four independently sampled physical diagnostic chunks executed 200 total ticks
across free motion, closure, conditional carrying and release. TCP RMSE was
0.66–1.61 mm; conditional carried-object position RMSE was 7.55 mm; release
object RMSE was 38.35 mm; closure finger-feedback RMSE was 16.42 mm. These expose
contact/attachment limitations, not validated grasp or placement prediction.
Four chunks are not 200 independently sampled candidates. None is a successful
demonstration.

Five-arm fixtures exercise distinct selections while retaining motor safety.
Real learned-policy comparisons await the production checkpoint. Missing report
J/Table 5.1 definitions have parameterized fail-closed interfaces, with no invented
weights or thresholds.

The owner reports production step 14,700; its running session is left untouched.
Local validation follows delivery of the final export. The TRAIN gate remains
10/24, frozen criteria and splits remain unchanged, TEST remains untouched before
Week 4, and the December 12 deadline is unchanged.
