# Statistics and measurement rules (ADR-0029)

This document is the contract for how campaign numbers are computed and labelled. The code
implementing it lives in `src/vlarobot/eval/metrics.py` (unit tests:
`tests/test_eval_metrics.py`). It is part of the ADR-0029 Week-1 deliverable.

## Pairing unit: (scene, seed)

- The **pairing unit is `(scene, seed)`** — one scene name (the scenario name with its
  `-s<seed>` suffix stripped, `scene_id()`) plus one seed. Arms are run on **identical**
  `(scene, seed)` pairs (frozen in `configs/eval.yaml`: VLA arms 3 seeds/scene, scripted
  baseline the same 3 seeds), so every comparison is paired, never two independent samples.
- `paired_comparison(a, b)` **requires identical keys** in both arms and raises otherwise —
  an unpaired comparison cannot silently proceed.

## Episode-level vs scene-level inference (the independence trap)

Repeated seeds within one scene are **correlated** (same geometry, same obstacles, same
family). 30 episode pairs (10 scenes × 3 seeds) are therefore **not 30 independent
observations**.

- **Episode-level paired results are preserved** (`episode_pairs`, `episode_b`,
  `episode_c`, `episode_mcnemar_p`) — reported as *descriptive*. The episode-level exact
  McNemar p-value **assumes independence and is anti-conservative** here; it must always be
  printed with that caveat.
- **The inferential analysis is scene-level**: each scene collapses to one discordant count
  (did A's within-scene rate beat B's, or vice versa), and the exact sign test is computed
  on those. `independent_units` = the **scene count**, and that is the n that may be cited
  as independent.
- Per-scene rows (`ScenePairRow`) keep the within-scene variation visible: per-arm success
  counts, rates, and sample standard deviations per scene. Per-scene variation is always
  reported alongside any aggregate.
- **McNemar pairing unit + this limitation** is recorded here as required by ADR-0029; any
  report text quoting a p-value must state which level it came from.

## Intervals and summaries

- Success rates are reported with the **Wilson 95% interval** (`wilson_interval`), not a
  Wald interval — Wilson has better coverage at small n and at 0/n and n/n boundaries
  (e.g. 0/10 → (0.000, 0.278), 8/10 → (0.490, 0.943)).
- **No single-run claims.** Every rate is `summarize_success` → count + rate + CI.
- Continuous spread uses mean ± **sample standard deviation (ddof=1)**; n = 1 reports NaN
  for the standard deviation rather than 0.

## Continuous vs probabilistic error — never conflated

| Target type | Metric | Function |
|---|---|---|
| Continuous (distances, angles, errors in m/deg) | **RMSE** and **MAE** | `rmse`, `mae` |
| Probabilistic binary outcomes (predicted probability of success) | **Brier score** | `brier_score` |

Brier = mean squared error of probabilities against 0/1 outcomes. It is mathematically
`rmse²` for probability inputs, but the two are reported as different quantities with
different names and never swapped (test:
`test_brier_is_for_probabilities_and_stays_separate_from_rmse`).

## Collision: three separate measurements that must never merge

1. **Collision PROXY (displacement/tilt)** — `collision_proxy()`. Object displacement from
   its **reset pose** > **5 mm** or tilt > **5°** (frozen thresholds in
   `configs/eval.yaml`, `collision_proxy`). This is a **proxy**, labelled as such in every
   output column (`proxy_*`). The max excursion over the episode drives the hit (a
   transient knock is not missed); the final reset→last displacement and final tilt are
   also reported. Comparison is strict `>` (at exactly the threshold = no hit). Rows with
   no reported pose (NaN) are skipped; an episode with no tracked objects reports
   `proxy_hit=False` with `None` magnitudes (not a zero measurement).
2. **Actual collision detection** — simulator contact nodes, recorded by the EpisodeRecorder
   into `episode-XX.contacts.jsonl` (one row per step whose diagnostics reported non-robot
   contact nodes; an empty file means no contact detected). Reported as `contact_steps` /
   `contact_nodes`, **never** folded into the proxy and never derived from it. The proxy
   can fire with zero contacts (near-miss displacement) and contacts can occur with a
   non-firing proxy — both cases are asserted in tests.
3. **FK clearance — its own standalone measurement** — `min_fk_clearance_m(fk_points,
   objects)`: minimum surface-to-surface clearance (m) between supplied FK points and
   object spheres; negative = penetration; `None` when either side is empty. Pure geometry
   (eval must not import `backends/`), so the caller supplies the points. It is **not** a
   collision proxy substitute and is reported in its own column
   (`fk_min_clearance_m`).

## MEASURED / INFERRED / UNVERIFIED labels

Every published number carries one of:

- **MEASURED** — computed from recorded data in this repository (tests, recordings,
  profiling runs).
- **INFERRED** — derived from MEASURED values by stated arithmetic (e.g. projected
  wall-clock = measured per-episode time × episode count).
- **UNVERIFIED** — not yet run in this environment (e.g. Windows paths, Webots-only
  behaviour, Colab training). Never presented as a result.

## Blocked: the eight Table 5.1 metrics

The submitted **WorldMind Phase I report (Oct 2026)** enumerates the eight Table 5.1
metrics; the PDF is not in this repository (HUMAN STEP — owner supplies it). Until then,
`metrics.py` provides the primitives and episode plumbing only; **no metric is claimed to
be "the Table 5.1 metric"**, and the conformance mapping (`REPORT_CONFORMANCE.md`) stays
blocked. All eight will be preserved verbatim, supplemented only (ADR-0029
clarifications 2/8).
