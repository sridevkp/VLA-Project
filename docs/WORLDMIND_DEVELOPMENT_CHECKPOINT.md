# WorldMind independent development checkpoint — 2026-11-15

Branch `work/analytic-risk-followup`, isolated worktree `VLA-Project-analytic`. Main-worktree
training data/source archives, Colab configuration, registered scripted controller and accepted
23/24 baseline were not edited. No TEST geometry was generated for experiments. December 12,
November 27 and the literal 40% learned-policy gate are unchanged.

## Implemented

- Deterministic action-chunk robot states through injected FK/IK and velocity-limited joint
  tracking, base-left rotation deltas and explicit gripper tracking. Snapshots do not alias.
- Conditional rigid attachment for an explicitly observed held object, aperture release,
  ballistic translation and support-plane settling. A close command cannot manufacture a
  successful grasp. Contact/impact/angular dynamics remain declared approximations.
- Conservative swept arm capsule bounds with required link/object coverage, explicit scene completeness, point-speed certificates,
  self-collision adjacency pairs and object checks. Missing certificates/geometry fail closed.
  Actual TIAGo collision geometry coverage is not yet certified or validated; unit geometry
  examples do not establish full-robot collision acceptance.
- Separate continuous errors for robot TCP, joints, object position and orientation.
  Probabilistic Outcome rollout requires an explicit validation-derived calibrator/provenance;
  no guessed success probabilities or Brier results are emitted.
- RiskGate rejects failed/unknown/missing/duplicated hard checks and nonfinite scores before
  J ranking. Ties are stable; rationales contain actual components and weights. Empty-safe
  recovery re-observes, tries an alternate candidate family and aborts. Verification retries
  at most once. Scripted and VLA execution both cross the live SafetyMonitor.
- All five frozen arm switches and one TRAIN/VALIDATION plan/recorder/metric path. Prediction
  and predictive veto are independently ablated; removing predictive veto never removes
  motor safety. Comparison needs five explicit factories, not a mislabeled scripted expert.
  Fake-backend 5×18 validation integration is explicitly simulated and not gate evidence.
- Explicit seeded K policy draws log all draw seeds without perturbing physical commands.
- Approximate planning-only Jacobian IK resets per chunk and checks exact FK residuals,
  bounded joints and trust region; invalid results are rejected. Execution IK is unchanged.

## Measured read-only TRAIN replay

80 admitted episodes, recorded accepted joint commands; future observations are evaluation
only. Horizons 1, 10 and 50 control ticks; window starts stride 50. This is conditioned action
replay on the successful collection, not an autonomous task success/generalization claim.

| Family | TCP RMSE at 1 tick | TCP RMSE at 50 ticks | Conditional carried-CAN RMSE at 50 ticks |
|---|---:|---:|---:|
| clear | 4.62 mm | 6.69 mm | 14.25 mm (28 windows) |
| cluttered | 4.48 mm | 6.44 mm | 15.88 mm (26 windows) |
| reach-limited | 4.49 mm | 6.30 mm | 20.84 mm (11 windows) |

The object diagnostic initializes attachment and full base transform from an oracle TRANSIT
entry of an audited successful carry, then replays only commanded joints. It measures
conditional carry dynamics, not grasp classification, release/placement prediction or VLA
performance. Constant initial base pose ignores subsequent drift, which contributes to error.
The stationary-CAN 50-tick reference has 204/205/251 mm RMSE respectively.
Raw source hashes, quantities, MAE, orientation RMSE and counts are in `docs/evidence/`.

## Planning latency

Earlier local probes showed 32×50 FK/tracking steps at p50 0.57–0.83 s, a serial full-pose
IK projection of 21–38 s for 1,600 solves, and a bounded Jacobian path with full state
copying at p50 3.895 s / p95 4.162 s. Those figures were cost projections or preliminary
probes, not a valid full-system latency result.

**Week 3 evidence correction (2026-11-17).** The authoritative recorded profile is now
`docs/evidence/planning-optimization-final.json` (hash-pinned, identical inputs):
K=32, H=50 (1,600 control steps), optimized nominal analytic-kernel p50 `0.4677 s`,
p95 `0.4803 s`, cold batch `0.4746 s`, and pre-optimization p50 `2.0996 s`. The evidence
records `max_absolute_state_difference = 0.0`, identical vetoes, prepared-FK difference
`0.0`, unchanged safety/trust thresholds, `includes_model_inference = false`,
`test_accessed = false`, and `uncertified_full_arm_sweep_remains_vetoed = true`.

The profiled inclusive contributors were, before optimization, IK `2.011 s`, state deep copy
`1.271 s`, FK `0.908 s`, and Jacobian `0.154 s`; after optimization, IK `0.646 s`, FK
`0.195 s`, and world/robot copy `0.016 s`/`0.013 s`. These inclusive profiler entries
overlap and are not additive.

This is **nominal-kernel-only** evidence. Full-system latency, VLA-proposer latency,
full-arm sweep latency, and scored per-decision latency remain **UNVERIFIED**; the
below-1-second milestone is not claimed as met. No K, horizon, safety threshold, veto,
or experiment design was reduced to mask latency. The phase-separated measurement
protocol is in `docs/WEEK3_RISKGATE_VALIDATION_PLAN.md`.

## Verification and remaining evidence

Full isolated CPU suite: 272 passed, one TEST-geometry-generation test intentionally deselected.
Latest targeted model/risk/integration suite after candidate/verifier additions: 25 passed.
Source/test/tools ruff and new-module type checks pass. The inherited old Colab notebook has
a preexisting E402 lint issue and was deliberately not edited; this is not an all-repo lint
pass. The inherited reset test was aligned with the existing final 32 ms reset step, using
the already accepted main-worktree expectation; controller behaviour was not changed.

A fresh serial 18-episode VALIDATION expert campaign is running at
`runs/validation-evidence-v1/campaign`; every episode has a fresh Webots session, rendering
on and explicit accepted registration. This new validation evidence never overwrites the
TRAIN baseline or contributes training demonstrations. Its outcomes remain pending until
complete. They will support geometry/error validation, not validation-based rescue of the VLA.

The report PDF is absent from the repo; exact Eq. 3.8 J-component definitions and Table 5.1
metrics remain external inputs. Weights/calibration are not guessed or claimed approved.
VALIDATION progress at checkpoint: 3/18 completed, 2 physical successes with 140.44/133.45 mm
CAN rise. Reach-limited i02 seed 0 failed PREGRASP (7.946 mm IK residual vs the unchanged
5 mm limit), with negligible lift. The other 15 episodes are running; this is not a complete
validation success-rate claim, and the failure is not being used to rescue or tune the VLA.

No trained SmolVLA checkpoint has been supplied here. Existing main-worktree import preflight
and frozen TRAIN VLA-only harness remain ready; 10/24 is the literal gate with baseline 23/24.

## Reproduction

Run in the isolated worktree with Python 3.11, development dependencies and optional sim
(IKPy) dependencies. Pure-model tests require no Webots or GPU:

```text
uv run pytest tests/test_analytic_risk.py tests/test_kinematic_surrogate.py tests/test_rollout_comparison.py -q
uv run pytest -m "not webots" -k "not generate_scenes_counts_families_and_test_split_guard"
uv run ruff check src tests tools
uv run ruff format --check src tests tools
```

For read-only replay, use the accepted main-worktree source/data as the explicit FK adapter:

```text
python tools/profile_registered_replay.py --project ../VLA-Project --out runs/replay-new-version
python tools/profile_registered_replay.py --project ../VLA-Project --out runs/attachment-new-version --attachment-only
```

The script refuses existing output directories and verifies static registration provenance.
