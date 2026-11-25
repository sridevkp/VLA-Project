# Phase I figures and measured replacements

Experiment figures use **VALIDATION development evidence**. The
[provenance ledger](figures/provenance.json) contains SHA-256 digests of the
source packets and analysis tables. These replacements leave the submitted
Phase I PDF unchanged.

| Phase I figure | Final asset | Evidence and caption |
|---|---|---|
| 3.1–3.4 · architecture, interactions and data flow | [Architecture PDF](figures/architecture.pdf) | Evaluated Webots oracle and scripted-primary execution loop. The general architecture supports a model service outside the evaluated scripted path. Arm 4 omits the predictive veto and arm 3 uses a prediction-free prefilter. |
| 4.1 · environment | [Recorded RGB frame](figures/webots-reset.png) | Actual onboard R2025a TIAGo view at reset of arm 5 equal, validation-clear-i00-s0. The archive has no external full-robot view for this trial. |
| 4.2 · prediction | [Selected TCP trace](figures/prediction-prefix.pdf) | Recorded PREGRASP prefix with confirmed observations at their original 32 ms timestamps. Positions use robot-base metres. |
| 4.3 · execution | [Recorded phase frames](figures/webots-execution.png) | Onboard RGB through PREGRASP, CLOSE, LIFT, TRANSIT, RELEASE and HOME, with original simulator times. |
| 4.4 · initialization | [Reset frame](figures/webots-reset.png) | Onboard reset view of the task workspace with its documented camera perspective. |
| 4.5 · environment state | [Camera frame](figures/webots-reset.png) and [oracle map](figures/oracle-state.pdf) | Same episode. The map shows recorded world positions of the base, CAN, HAM and APP. Learned perception is outside the evaluated path. |
| 4.6 · candidate ensemble | [Recorded pool](figures/candidate-pool.pdf) | Recorded draw-indexed component sums of a mixed 32-candidate PREGRASP pool. Unselected state trajectories were not retained. |
| 4.7 · assessment and selection | [Pool plot](figures/candidate-pool.pdf) and [check and margin ledger](figures/candidate-pool.json) | The marker shows the selection among admitted draws. Diagnostic sums for vetoed rows are shown for context and never enter production ranking. |
| 6.1 · schedule | [Conformance schedule](REPORT_CONFORMANCE.md#chapter-6-milestones-and-limitations) | Use the executed freeze and evaluation milestones in place of the original twelve-week placeholder. |
| C.1–C.3 · scene and tasks | [Recorded execution](figures/webots-execution.png) and [family comparison](figures/family-success.pdf) | Factory CAN pick-and-place and the three scenario families. A separate obstacle-avoidance task was not evaluated. |

## Results figures

- [Paired outcomes](figures/paired-outcomes.pdf) shows all 18 seed identities.
- [Scene families](figures/family-success.pdf) shows Wilson intervals for six trials per family.
- [One-tick object error](figures/object-prediction-error.pdf) shows RMSE and MAE across matched objects.
- [Latency and veto rates](figures/latency-and-veto.pdf) shows p50/p95 and the applicable
  hard-veto rates. Arm 3 has a measured zero at its prefilter. Arm 4 has no predictive veto.
- [Successful-episode costs](figures/successful-episode-costs.pdf) shows TCP path, simulated
  time and final CAN error. [Results](RESULTS.md) also reports all-trial costs.

## Reproduction

From the repository root, with the original development archives present, use
Gnuplot with Cairo terminals and the project Python environment.

```text
uv run --offline python tools/render_worldmind_report.py runs/worldmind/nov25-validation-report/analysis.json runs/worldmind/nov24-matched-arm5-equal/validation-clear-i00-s0/attempt-2/episode docs/figures
```

PDFs are vector exports. PNGs are previews and presentation assets. The
camera composites retain the recorded RGB pixels with timestamp captions.
