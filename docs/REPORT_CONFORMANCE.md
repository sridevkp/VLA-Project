# Phase I report conformance

Reference: submitted *WorldMind: World Model-Based Robot Simulation*,
SHA-256 `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`.
The report's Chapters 3–6 define the comparison, model, planned evaluation and
milestones. ADR-0029 and its later owner decisions bound the evaluated design.
Terms used here are **OBSERVED** for recorded values, **INFERRED** for conclusions
supported by implementation or trace structure, and **UNKNOWN** for unmeasured
claims. TEST is pending.

## Chapter 3 architecture and Chapter 4 implementation

| Requirement | Disposition and evidence |
|---|---|
| §3.4.1, §4.2 reproducible simulation and control | **Implemented and verified in Webots.** The pinned R2025a TIAGo factory world ran all 18 planned identities under each corrected VALIDATION profile. One arm-3 identity required an approved repair after two staging failures. Both failed starts remain in the record. |
| §3.4.2, §4.3 task state from perception | **Implemented with approved oracle state.** Pinned task and scene identities, full base pose and recorded CAN/HAM/APP states supply evaluation input. Learned scene perception and robustness to its errors are **unmeasured**. |
| §3.4.3, Eq. 3.2 VLA policy and candidate proposal | **Deviation.** SmolVLA cannot accept the predicted state as a native input. Prediction is external to the proposer. The VLA-only promotion gate failed at 0/24 TRAIN, so arms 3–5 use the scripted primary. End-to-end competent-VLA improvement is **unmeasured**. |
| §3.4.4, §4.5 future-state model | **Implemented and measured on aligned simulated prefixes.** The analytic model replaced the earlier MuJoCo plan. No learned residual, calibrated outcome-probability model or contact dynamics are claimed. |
| §3.4.5, §4.6 risk and selection | **Implemented and measured within the approved geometric scope.** Arm 5 applies predictive hard checks ahead of Eq. 3.8. Arm 4 removes that veto. The ADR-0084 arm 3 uses a prediction-free prefilter and draw-order selection. |
| §3.4.6, §4.7 control | **Implemented and verified in recorded simulation.** Stateful phase progress, bounded verification recovery, independent command admission and configured HOME verification. Full-body sweep certification and real-robot transfer are **unmeasured**. |
| §3.4.7, §§4.8–4.9 feedback and integration | **Implemented and measured.** Decisions, commands, simulator receipts, observations, timing, failures and verification are recorded. Some joint finger coordinates read around 10⁻⁶ m outside the motor range. Logging classifies the numerical excursions under the approved 10⁻⁴ m logging allowance. Motor, task and safety thresholds retain their original meaning. |

## Chapter 5 and Table 5.1

| Metric | Frozen measurement and status |
|---|---|
| Task success | Complete placements / eligible trials. Every task failure remains in the denominator. A software failure is unreliable and its attempts are retained. VALIDATION arms 1/3/4/5 equal recorded 15/14/15/14 successes out of 18 each. |
| Prediction error | Quantity-specific RMSE and MAE on confirmed matched prefixes, by family and horizon. Arms 1 and 3 have no applicable predictions. Arm 3 has a null value, reason `arm_has_no_prediction`, and zero valid pairs. |
| Collision rate | Actual collision reduction **not measured**. The backend contact channel recorded zero nonrobot contacts and is limited in coverage of physical interaction. The separate HAM/APP displacement or tilt proxy (strictly above 5 mm or 5°) was positive in every eligible episode, including the baseline. |
| Path length | Observed sum of successive base-frame TCP distances in metres. All eligible and successful-only populations are reported separately. |
| Execution time | Observed simulated seconds from timestamps. Baseline inner wall time and comparison-arm attempt wall time are reported separately with their timer boundaries. |
| Planning latency | Observed decision time to generate and evaluate the pool, summarized with p50/p95. Baseline has no comparable pool timing. The earlier nominal-kernel benchmark is separate evidence. |
| Risk rejection | Hard vetoes / (hard vetoes + safe unselected + selected), excluding invalid and unevaluable rows. A zero evaluated denominator is undefined. Arm 3 measures its prefilter, arm 4 has no candidate veto, and execution motor vetoes are separate. |
| Reliability | Eligible task outcomes and terminal categories alongside unreliable attempts. One repaired VALIDATION startup identity is retained as an explicit exception history, with its eventual eligible outcome. |

The §5.5 supplemental minimum-clearance and no-acceptable-candidate measures appear
in the [VALIDATION evidence report](RESULTS.md). The success, metric and paired
results are split-specific. Independent TEST observations are pending. A zero
observation count or unsupported quantity carries its missingness explicitly.

## Chapter 6 milestones and limitations

The reproducible world, task-state module, scripted baseline, measurable world-model
predictions, integrated selection, logged closed-loop execution and development
comparison are evidenced. The independent final comparison remains pending. The
Phase I twelve-week Gantt was replaced operationally by the owner schedule, with
frozen TEST authorized from November 30 and report completion by December 12.

Figures 4.1–4.7 are remapped to real recorded images, traces or evidence summaries
in [REPORT_FIGURES.md](REPORT_FIGURES.md). Equations and operating limits are in
[MATH.md](MATH.md). Phase I proposed a general world-model-based VLA framework.
The actual experiment tests a scripted-primary comparison in a single simulated
task. Its evidence cannot establish improved performance over a competent VLA.
