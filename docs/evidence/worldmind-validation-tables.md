# Matched VALIDATION development evidence

Arm 5 equal recorded 14/18 successes, compared with 15/18 for the scripted baseline.
The approved tie-break selected equal (both arm-5 profiles recorded 14/18).
The predictive hard veto rejected 0.033% and 0.053% of evaluated candidates in
the equal and goal-heavy profiles respectively (about 0.03–0.05%).
Reach-limited geometry constrained all arms, although individual seeds differed.
The scripted-primary comparison does not establish improvement over a competent VLA.
The VLA-only TRAIN result of 0/24 stands.

## Scene and seed outcomes

| Identity | Arm 1 | Arm 3 | Arm 4 equal | Arm 5 equal | Arm 5 goal_x2 |
|---|---|---|---|---|---|
| validation-clear-i00-s0 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-clear-i00-s1 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-clear-i00-s2 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-clear-i03-s0 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | NO_SAFE_SET |
| validation-clear-i03-s1 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-clear-i03-s2 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-cluttered-i01-s0 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-cluttered-i01-s1 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-cluttered-i01-s2 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-cluttered-i04-s0 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-cluttered-i04-s1 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-cluttered-i04-s2 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-reach-limited-i02-s0 | PREGRASP | PLACEMENT_FAILED | PLACEMENT_FAILED | PLACEMENT_FAILED | PLACEMENT_FAILED |
| validation-reach-limited-i02-s1 | TRANSIT | PLACEMENT_FAILED | COMPLETE | PLACEMENT_FAILED | COMPLETE |
| validation-reach-limited-i02-s2 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-reach-limited-i05-s0 | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| validation-reach-limited-i05-s1 | COMPLETE | PLACEMENT_FAILED | WORLDMIND_BOUND_EXHAUSTED | PLACEMENT_FAILED | PLACEMENT_FAILED |
| validation-reach-limited-i05-s2 | TRANSIT | PLACEMENT_FAILED | PLACEMENT_FAILED | PLACEMENT_FAILED | PLACEMENT_FAILED |

## Families and Wilson 95% intervals

| Arm | Family | Successes / eligible | Wilson 95% |
|---|---|---:|---:|
| Arm 1 | all | 15/18 | 60.8%–94.2% |
| Arm 1 | clear | 6/6 | 61.0%–100.0% |
| Arm 1 | cluttered | 6/6 | 61.0%–100.0% |
| Arm 1 | reach-limited | 3/6 | 18.8%–81.2% |
| Arm 3 | all | 14/18 | 54.8%–91.0% |
| Arm 3 | clear | 6/6 | 61.0%–100.0% |
| Arm 3 | cluttered | 6/6 | 61.0%–100.0% |
| Arm 3 | reach-limited | 2/6 | 9.7%–70.0% |
| Arm 4 equal | all | 15/18 | 60.8%–94.2% |
| Arm 4 equal | clear | 6/6 | 61.0%–100.0% |
| Arm 4 equal | cluttered | 6/6 | 61.0%–100.0% |
| Arm 4 equal | reach-limited | 3/6 | 18.8%–81.2% |
| Arm 5 equal | all | 14/18 | 54.8%–91.0% |
| Arm 5 equal | clear | 6/6 | 61.0%–100.0% |
| Arm 5 equal | cluttered | 6/6 | 61.0%–100.0% |
| Arm 5 equal | reach-limited | 2/6 | 9.7%–70.0% |
| Arm 5 goal_x2 | all | 14/18 | 54.8%–91.0% |
| Arm 5 goal_x2 | clear | 5/6 | 43.6%–97.0% |
| Arm 5 goal_x2 | cluttered | 6/6 | 61.0%–100.0% |
| Arm 5 goal_x2 | reach-limited | 3/6 | 18.8%–81.2% |

## Paired exact McNemar tests

| Arm | Comparator | Matched | Arm-only success | Comparator-only success | Exact p |
|---|---|---:|---:|---:|---:|
| Arm 1 | arm3 | 18 | 1 | 0 | 1.0000 |
| Arm 3 | baseline | 18 | 0 | 1 | 1.0000 |
| Arm 4 equal | arm3 | 18 | 1 | 0 | 1.0000 |
| Arm 4 equal | baseline | 18 | 1 | 1 | 1.0000 |
| Arm 5 equal | arm3 | 18 | 0 | 0 | 1.0000 |
| Arm 5 equal | baseline | 18 | 0 | 1 | 1.0000 |
| Arm 5 goal_x2 | arm3 | 18 | 1 | 1 | 1.0000 |
| Arm 5 goal_x2 | baseline | 18 | 1 | 2 | 1.0000 |

These exploratory p-values reuse the development split and are unadjusted.
The split was also used for selection, so these are not held-out effect estimates.

## Prediction, veto and execution record

The accompanying JSON records pre-execution placement availability, flags, margins,
candidate choices and replacement rounds for failed predictive trials. A later waypoint
that failed IK before a pool existed cannot be counted as an anticipated failure.
The horizon tables use only confirmed complete executable prefixes, grouped by family.
The non-target geometry comparison is a tool-envelope segment proxy in metres.
Placement-prediction error is tool-to-tool-goal distance and is separate from observed final CAN placement.
Object-position RMSE around 0.065–0.066 m exceeds the approximately 13 mm per-side grasp clearance.
Clearance proxies and recorded backend contacts are different measurements.

| Arm | Proxy observed episodes | Proxy positive episodes | Backend contact episodes |
|---|---:|---:|---:|
| Arm 1 | 18 | 18 | 0 |
| Arm 3 | 18 | 18 | 0 |
| Arm 4 equal | 18 | 18 | 0 |
| Arm 5 equal | 18 | 18 | 0 |
| Arm 5 goal_x2 | 18 | 18 | 0 |

## Execution and planning

| Arm | Planning p50 / p95 ms | Inner baseline / attempt wall s | Mean path m, all / successes | Simulated s, all / successes | Final CAN error mm, all / successes |
|---|---:|---:|---:|---:|---:|
| Arm 1 | Unavailable | 498.4 | 2.505 / 2.718 | 24.44 / 27.40 | 41.95 / 14.65 |
| Arm 3 | 14.0 / 18.8 | 472.0 | 2.451 / 2.722 | 21.18 / 23.33 | 49.14 / 14.94 |
| Arm 4 equal | 40.6 / 975.3 | 636.6 | 2.551 / 2.703 | 24.55 / 25.82 | 36.10 / 14.98 |
| Arm 5 equal | 44.2 / 1093.7 | 590.8 | 2.457 / 2.729 | 21.80 / 24.82 | 47.88 / 14.80 |
| Arm 5 goal_x2 | 28.5 / 759.9 | 483.8 | 2.454 / 2.702 | 22.17 / 24.45 | 42.02 / 15.12 |

Planning is candidate generation and assessment wall time.
Arm 3 hard prefilter vetoes: 0 / 34,016 evaluated candidates, including the repair. Arm 4 has no predictive veto.
Arm 5 equal: 73 / 218,976. Arm 5 goal-heavy: 116 / 220,512. Execution motor vetoes are separate.
Baseline inner execution wall time and comparison-arm attempt wall time have different scopes.

## Timing-aligned geometric and placement predictions

| Arm | Quantity | Complete pairs | Mean predicted mm | Mean observed mm | RMSE mm | MAE mm |
|---|---|---:|---:|---:|---:|---:|
| Arm 4 equal | object_minimum_clearance_pairs_m | 7643 | 598.53 | 601.10 | 8.70 | 2.64 |
| Arm 4 equal | placement_tool_distance_pairs_m | 111 | 1.45 | 5.98 | 7.30 | 4.58 |
| Arm 4 equal | table_minimum_clearance_pairs_m | 7643 | 173.93 | 175.41 | 4.13 | 1.56 |
| Arm 5 equal | object_minimum_clearance_pairs_m | 6793 | 597.96 | 601.30 | 10.29 | 3.41 |
| Arm 5 equal | placement_tool_distance_pairs_m | 98 | 1.48 | 6.63 | 7.89 | 5.22 |
| Arm 5 equal | table_minimum_clearance_pairs_m | 6793 | 175.61 | 177.53 | 4.88 | 2.00 |
| Arm 5 goal_x2 | object_minimum_clearance_pairs_m | 6838 | 616.41 | 619.55 | 9.83 | 3.19 |
| Arm 5 goal_x2 | placement_tool_distance_pairs_m | 102 | 1.22 | 5.33 | 6.86 | 4.21 |
| Arm 5 goal_x2 | table_minimum_clearance_pairs_m | 6838 | 172.58 | 174.37 | 4.64 | 1.87 |

## Horizon-specific errors

| Arm | Family | Horizon ticks | Quantity and unit | Count | RMSE | MAE |
|---|---|---:|---|---:|---:|---:|
| Arm 4 equal | clear | 1 | joint_rad | 20848 | 0.018256 | 0.010008 |
| Arm 4 equal | clear | 1 | object_m | 2606 | 0.065162 | 0.061463 |
| Arm 4 equal | clear | 1 | object_rad | 2606 | 0.107597 | 0.074373 |
| Arm 4 equal | clear | 1 | tcp_m | 2606 | 0.016946 | 0.009142 |
| Arm 4 equal | cluttered | 1 | joint_rad | 21632 | 0.017608 | 0.009441 |
| Arm 4 equal | cluttered | 1 | object_m | 2704 | 0.065619 | 0.062101 |
| Arm 4 equal | cluttered | 1 | object_rad | 2704 | 0.074481 | 0.040769 |
| Arm 4 equal | cluttered | 1 | tcp_m | 2704 | 0.015715 | 0.008049 |
| Arm 4 equal | reach-limited | 1 | joint_rad | 18664 | 0.017541 | 0.009328 |
| Arm 4 equal | reach-limited | 1 | object_m | 2333 | 0.066808 | 0.063521 |
| Arm 4 equal | reach-limited | 1 | object_rad | 2333 | 0.153738 | 0.114820 |
| Arm 4 equal | reach-limited | 1 | tcp_m | 2333 | 0.014948 | 0.007755 |
| Arm 5 equal | clear | 1 | joint_rad | 20832 | 0.018235 | 0.009960 |
| Arm 5 equal | clear | 1 | object_m | 2604 | 0.065430 | 0.061859 |
| Arm 5 equal | clear | 1 | object_rad | 2604 | 0.110708 | 0.079955 |
| Arm 5 equal | clear | 1 | tcp_m | 2604 | 0.017007 | 0.009196 |
| Arm 5 equal | cluttered | 1 | joint_rad | 19888 | 0.018388 | 0.010223 |
| Arm 5 equal | cluttered | 1 | object_m | 2486 | 0.064717 | 0.060994 |
| Arm 5 equal | cluttered | 1 | object_rad | 2486 | 0.076921 | 0.044194 |
| Arm 5 equal | cluttered | 1 | tcp_m | 2486 | 0.016757 | 0.008955 |
| Arm 5 equal | reach-limited | 1 | joint_rad | 13624 | 0.018669 | 0.010444 |
| Arm 5 equal | reach-limited | 1 | object_m | 1703 | 0.067548 | 0.063823 |
| Arm 5 equal | reach-limited | 1 | object_rad | 1703 | 0.139972 | 0.106928 |
| Arm 5 equal | reach-limited | 1 | tcp_m | 1703 | 0.017580 | 0.010314 |
| Arm 5 goal_x2 | clear | 1 | joint_rad | 19816 | 0.018063 | 0.009836 |
| Arm 5 goal_x2 | clear | 1 | object_m | 2477 | 0.064681 | 0.060973 |
| Arm 5 goal_x2 | clear | 1 | object_rad | 2477 | 0.110191 | 0.078021 |
| Arm 5 goal_x2 | clear | 1 | tcp_m | 2477 | 0.016653 | 0.008803 |
| Arm 5 goal_x2 | cluttered | 1 | joint_rad | 19360 | 0.018534 | 0.010417 |
| Arm 5 goal_x2 | cluttered | 1 | object_m | 2420 | 0.064829 | 0.061039 |
| Arm 5 goal_x2 | cluttered | 1 | object_rad | 2420 | 0.072581 | 0.039554 |
| Arm 5 goal_x2 | cluttered | 1 | tcp_m | 2420 | 0.016957 | 0.009196 |
| Arm 5 goal_x2 | reach-limited | 1 | joint_rad | 15528 | 0.018204 | 0.010058 |
| Arm 5 goal_x2 | reach-limited | 1 | object_m | 1941 | 0.066830 | 0.063324 |
| Arm 5 goal_x2 | reach-limited | 1 | object_rad | 1941 | 0.129850 | 0.092734 |
| Arm 5 goal_x2 | reach-limited | 1 | tcp_m | 1941 | 0.016516 | 0.009321 |

## Failure attribution from pre-execution records

The placement column gives the last measured predicted tool-goal distance in the trial.
Earlier chunks can legitimately lack a placement prediction.
The JSON retains each saved decision rather than replacing unobserved waypoint predictions.

| Arm | Failed identity | Terminal | Last selected phase | Last measured placement | Actual IK rejection | Failed hard checks |
|---|---|---|---|---:|---|---|
| Arm 1 | validation-reach-limited-i02-s0 | PREGRASP | — | — | — | — |
| Arm 1 | validation-reach-limited-i02-s1 | TRANSIT | — | — | — | — |
| Arm 1 | validation-reach-limited-i05-s2 | TRANSIT | — | — | — | — |
| Arm 3 | validation-reach-limited-i02-s0 | PLACEMENT_FAILED | — | — | PREGRASP_REALIGN: IK rejected: error=0.007946m (limit 0.005000m), tilt=0.195deg (limit 15.000deg) | — |
| Arm 3 | validation-reach-limited-i02-s1 | PLACEMENT_FAILED | — | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.017017 m, tilt=0.428 deg | — |
| Arm 3 | validation-reach-limited-i05-s1 | PLACEMENT_FAILED | — | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.015443 m, tilt=0.386 deg | — |
| Arm 3 | validation-reach-limited-i05-s2 | PLACEMENT_FAILED | — | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.016293 m, tilt=0.408 deg | — |
| Arm 4 equal | validation-reach-limited-i02-s0 | PLACEMENT_FAILED | PREGRASP | — | PREGRASP_REALIGN: IK rejected: error=0.007948m (limit 0.005000m), tilt=0.195deg (limit 15.000deg) | — |
| Arm 4 equal | validation-reach-limited-i05-s1 | WORLDMIND_BOUND_EXHAUSTED | HOME | 1.32 mm | — | — |
| Arm 4 equal | validation-reach-limited-i05-s2 | PLACEMENT_FAILED | TRANSIT | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.016716 m, tilt=0.419 deg | — |
| Arm 5 equal | validation-reach-limited-i02-s0 | PLACEMENT_FAILED | PREGRASP | — | PREGRASP_REALIGN: IK rejected: error=0.007948m (limit 0.005000m), tilt=0.195deg (limit 15.000deg) | — |
| Arm 5 equal | validation-reach-limited-i02-s1 | PLACEMENT_FAILED | TRANSIT | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.016227 m, tilt=0.407 deg | — |
| Arm 5 equal | validation-reach-limited-i05-s1 | PLACEMENT_FAILED | TRANSIT | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.015227 m, tilt=0.380 deg | — |
| Arm 5 equal | validation-reach-limited-i05-s2 | PLACEMENT_FAILED | TRANSIT | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.016533 m, tilt=0.415 deg | — |
| Arm 5 goal_x2 | validation-clear-i03-s0 | NO_SAFE_SET | RELEASE | 0.52 mm | — | RETREAT/table_clearance=64 |
| Arm 5 goal_x2 | validation-reach-limited-i02-s0 | PLACEMENT_FAILED | PREGRASP | — | PREGRASP_REALIGN: IK rejected: error=0.007948m (limit 0.005000m), tilt=0.195deg (limit 15.000deg) | — |
| Arm 5 goal_x2 | validation-reach-limited-i05-s1 | PLACEMENT_FAILED | TRANSIT | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.015227 m, tilt=0.380 deg | — |
| Arm 5 goal_x2 | validation-reach-limited-i05-s2 | PLACEMENT_FAILED | TRANSIT | — | TRANSIT: IK solve failed: no solution within 0.150000 rad trust region met position tolerance 0.015000 m; best error=0.016648 m, tilt=0.418 deg | — |

Displacement or rotation of HAM/APP above 5 mm or 5° is a collision proxy.
The simple threshold is saturated across arms. Large APP motions also appear in the baseline.
No actual collision reduction was measured. No collision or safety improvement is claimed.
