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

The frozen TEST plan assigns 30 trials to each of four arms. All four campaigns used
the same 10 scenes and three seeds. The policy remained fixed throughout TEST.

| Arm | Successes / eligible | Wilson 95% | Planned | Eligible episodes/hour | Unreliable attempts |
|---|---:|---:|---:|---:|---:|
| Arm 1 baseline | 24/30 | 62.7%–90.5% | 30 | 20.27 | 0 |
| Arm 3 heuristic | 23/30 | 59.1%–88.2% | 30 | 21.30 | 0 |
| Arm 4 ranking | 23/30 | 59.1%–88.2% | 30 | 15.37 | 3 |
| Arm 5 WorldMind | 23/30 | 59.1%–88.2% | 30 | 17.17 | 0 |

The superseded baseline launch stopped before controller startup because its
detached worktree used system Python without NumPy. That launch is retained separately.
Its same-seed replacement explicitly used the pinned project interpreter. Arm 4
retained three eligible pre-reboot outcomes. Its 27 remaining identities completed
after resumption, including three interrupted attempts with unknown wall time. The
reported arm-4 throughput is for newly eligible trials after resumption.

| Contrast | Matched trials | First-only / second-only successes | Descriptive exact McNemar p | Scene-level exact sign p |
|---|---:|---:|---:|---:|
| arm3 vs arm1 | 30 | 0 / 1 | 1.0000 | 1.0000 |
| arm4 vs arm1 | 30 | 0 / 1 | 1.0000 | 1.0000 |
| arm4 vs arm3 | 30 | 1 / 1 | 1.0000 | 1.0000 |
| arm5 vs arm1 | 30 | 1 / 2 | 1.0000 | 1.0000 |
| arm5 vs arm3 | 30 | 2 / 2 | 1.0000 | 1.0000 |
| arm5 vs arm4 | 30 | 2 / 2 | 1.0000 | 1.0000 |

Identities are paired only when both outcomes are eligible. Repeated seeds share
scene geometry. The scene-level sign test is cluster-aware.

| Arm | Clear | Cluttered | Reach-limited |
|---|---:|---:|---:|
| Arm 1 baseline | 12/12 | 8/9 | 4/9 |
| Arm 3 heuristic | 12/12 | 8/9 | 3/9 |
| Arm 4 ranking | 12/12 | 7/9 | 4/9 |
| Arm 5 WorldMind | 11/12 | 8/9 | 4/9 |

Arms 3, 4 and 5 tie at 23/30. Arm 5 has two unique successes and two unique
failures against each of arms 3 and 4. Their paired outcomes differ.

| Arm | Decision p50 / p95 ms | Hard-vetoed / evaluated | Valid predicted steps | Backend contact episodes |
|---|---:|---:|---:|---:|
| Arm 1 baseline | Unavailable | Not applicable | 0 | 0 |
| Arm 3 heuristic | 14.2 / 21.8 | 0 / 57312 | 0 | 0 |
| Arm 4 ranking | 44.8 / 1093.8 | Not applicable | 17901 | 0 |
| Arm 5 WorldMind | 47.1 / 1166.2 | 196 / 347616 | 16282 | 0 |

| Arm | TCP RMSE / MAE (mm) | Joint RMSE / MAE (rad) | Object-position RMSE (mm) | Object-angle RMSE (rad) |
|---|---:|---:|---:|---:|
| Arm 4 ranking | 20.51 / 11.33 | 0.02714 / 0.01416 | 68.64 | 0.25734 |
| Arm 5 WorldMind | 21.19 / 11.93 | 0.02757 / 0.01466 | 66.85 | 0.22662 |

TEST errors pool aligned steps from complete executable prefixes. They are
separate from the VALIDATION one-tick estimates. Arm 4 excluded 85 prefixes and
arm 5 excluded 81. Prediction error is inapplicable in arms 1 and 3.
The confirmed TEST LIFT exception occurred once, in cluttered-i01-s0.
The initial, first predicted and first observed heights above the table were
78.98, 108.71 and 87.74 mm, respectively. The observation cleared the 80.17 mm
floor at tick 270. Those data are separate from the three VALIDATION recoveries.

| Arm | Path all / success (m) | Simulated all / success (s) | Final CAN error all / success (mm) | Attempt wall time (s) | Displacement/tilt proxy |
|---|---:|---:|---:|---:|---:|
| Arm 1 baseline | 2.40 / 2.75 | 22.35 / 25.34 | 50.58 / 14.77 | 498.30 | 30/30 observed |
| Arm 3 heuristic | 2.43 / 2.76 | 21.40 / 23.52 | 47.56 / 14.70 | 484.62 | 30/30 observed |
| Arm 4 ranking | 2.44 / 2.75 | 22.86 / 25.35 | 47.87 / 14.72 | 676.94 | 30/30 observed |
| Arm 5 WorldMind | 2.35 / 2.74 | 21.05 / 24.27 | 43.56 / 14.61 | 621.15 | 30/30 observed |

The displacement/tilt proxy is a separate geometric measurement. Backend contacts
cannot establish complete collision coverage. Prediction-error values, coverage and
all terminal categories are recorded in the linked ledger. All 30 episodes per arm
exceeded the displacement/tilt proxy threshold. The contact channel logged zero
nonrobot contacts across 82,305 recorded contact-channel samples. Collision reduction was not measured.


[TEST outcome figure](figures/test-family-success.pdf) · [Per-episode evidence and metric ledger](../runs/worldmind/nov25-test-analysis/analysis.json)
