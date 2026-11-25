# Results

## VALIDATION development evidence

The matched comparison uses 18 scene-seed identities per arm, with six trials
per family. Arm 1 is the scripted expert and arms 3–5 use the scripted chunk
adapter. Both arm-5 weight profiles scored 14/18. The approved deterministic
tie-break selected equal with weights (1, 1, 1, 0). These rates reuse the
selection set. [Freeze record](../configs/worldmind_validation_freeze.json)

| Arm | Success | Wilson 95% | Clear | Cluttered | Reach-limited |
|---|---:|---:|---:|---:|---:|
| 1 · Scripted expert | 15/18 | 60.8–94.2% | 6/6 | 6/6 | 3/6 |
| 3 · Heuristic prefilter | 14/18 | 54.8–91.0% | 6/6 | 6/6 | 2/6 |
| 4 · Analytic ranking, equal | 15/18 | 60.8–94.2% | 6/6 | 6/6 | 3/6 |
| 5 · WorldMind, frozen equal | 14/18 | 54.8–91.0% | 6/6 | 6/6 | 2/6 |
| 5 · Goal-heavy development profile | 14/18 | 54.8–91.0% | 5/6 | 6/6 | 3/6 |

Arm 5 equal had no successes missed by baseline and one baseline-only success.
Arm 3 and arm 5 equal agreed on all 18 identities, including all four failures.
Arm 4 and baseline traded one success each. The exact two-sided McNemar
p-values for these comparisons are 1.00. Wilson intervals are at the trial
level. Repeated seeds share scene geometries. Sparse discordant observations
limit inference and cannot establish equivalence. The arm-3 seed-0 repair completed at tick 804.
The two earlier starts lacked the approved definitions file and remain
preserved as unreliable attempts.
[Matched scene ledger and metrics](evidence/worldmind-validation-tables.md)

### Failures and predictive intervention

The frozen arm 5 failed on reach-limited-i02-s0, i02-s1, i05-s1, and i05-s2.
The expert IK solver rejected each failing waypoint before a candidate pool
existed. The surrogate had no prediction for these waypoints and did not
anticipate the failures. Reachability constrained all profiles, although the
failing seeds varied and arm 4 also exhausted one episode budget. Separately,
the goal-heavy profile rejected 64 RETREAT candidates across two pools on a
clear scene. Its earlier 0.52 mm predicted placement distance described tool
position at an earlier phase.

Three LIFT exceptions recovered on the first observed tick at 88.26–88.35 mm
above the table, exceeding the unchanged 80.17 mm floor. Candidate hard-veto
rates were 73/218,976 (0.033%) for equal and 116/220,512 (0.053%) for goal-heavy.
Arm 3 recorded 0/34,016 at its prefilter. Arm 4 has no candidate veto.
[Risk and recovery evidence](../runs/worldmind/nov24-matched-analysis/completed.json)

### Prediction, cost and collision evidence

Frozen arm-5 one-tick object-position RMSE was 65.4 mm in clear, 64.7 mm in
cluttered, and 67.5 mm in reach-limited, with MAE 61.9, 61.0 and 63.8 mm.
The error aggregates matched CAN/HAM/APP identities. The approximately 13 mm
per-side grasp clearance gives context for the scale. The matched 5-, 10-,
25- and 50-tick bins had no complete aligned executable prefixes. The
separate three-tick illustrated trace is fully paired for its short prefix.
[Prediction evidence](../runs/worldmind/nov25-validation-report/analysis.json)

| Arm | Decision p50 / p95 | Mean path, all / success (m) | Simulated time, all / success (s) | Final CAN error, all / success (mm) |
|---|---:|---:|---:|---:|
| 1 | Unavailable | 2.505 / 2.718 | 24.44 / 27.40 | 41.95 / 14.65 |
| 3 | 14.0 / 18.8 ms | 2.451 / 2.722 | 21.18 / 23.33 | 49.14 / 14.94 |
| 4 | 40.6 / 975.3 ms | 2.551 / 2.703 | 24.55 / 25.82 | 36.10 / 14.98 |
| 5 equal | 44.2 / 1,093.7 ms | 2.457 / 2.729 | 21.80 / 24.82 | 47.88 / 14.80 |
| 5 goal-heavy | 28.5 / 759.9 ms | 2.454 / 2.702 | 22.17 / 24.45 | 42.02 / 15.12 |

Predictive decision times have substantial tails. Baseline inner wall time
averaged 498.4 s. Comparison-arm attempt times averaged 472.0, 636.6 and
590.8 s for arms 3, 4 and 5 equal. The timer boundaries differ. Shorter
all-trial paths can reflect early aborts.

The backend contact channel recorded zero nonrobot contacts. The HAM/APP
displacement or rotation proxy was positive in every eligible episode, including
baseline. Large APP motion also occurs in the baseline, and the proxy cannot
attribute its cause. **Collision reduction was not measured.**

## TEST independent evaluation

**Results pending.** No TEST scene was accessed. Launch is authorized from
November 30, 2026. The frozen 30-trial plan applies to arms 1, 3 and 5, and to
arm 4 if throughput permits. Arm 2 is excluded.

| Arm | Eligible trials | Successes | Table 5.1 measurements |
|---|---:|---:|---|
| 1 · Scripted baseline | Pending | Pending | Pending |
| 3 · Heuristic prefilter | Pending | Pending | Pending |
| 5 · Frozen WorldMind equal | Pending | Pending | Pending |
| 4 · Analytic ranking | Conditional | Pending | Pending |

The independent report will use frozen matched identities, Wilson intervals,
paired comparisons, reported unreliable attempts, and metric applicability.
