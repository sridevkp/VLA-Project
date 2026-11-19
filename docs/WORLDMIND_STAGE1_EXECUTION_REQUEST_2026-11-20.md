# Stage 1 Webots evidence request — PREPARED, NOT AUTHORIZED, NOT EXECUTED

Owner authorization is required before any command in this document is run. This request covers
only uncalibrated Stage 1 VALIDATION diagnostic evidence collection. It does not authorize
scoring, calibration selection, production activation, Packet B, Packet D, TEST/TRAIN access,
a geometry diagnostic, or motion authorization for `full_system`.

## 1. Requested action and exact output path

Reserve one new, never-overwritten evidence directory:

```text
/home/tj330/programs/VLA-Project-week3/runs/worldmind/stage1-validation-v1-2026-11-20
```

The path must not exist before the run. No prior evidence directory, v4/v5 artifact, frozen
campaign or TEST/TRAIN artifact may be modified.

## 2. Frozen VALIDATION scope

- Split: `validation` only. TEST and TRAIN remain sealed.
- State source: `oracle` only.
- Backend: `webots`, connecting to an already-running `sim-server`; do not launch a new geometry
  diagnostic or reuse the accepted v5 inspection as a new measurement.
- Base seed: `0`; seeds: `[0, 1, 2]`; 18 unique `(scene, seed)` episodes.
- Exact frozen episode list:

```text
validation-clear-i00-s0 0
validation-cluttered-i01-s0 0
validation-reach-limited-i02-s0 0
validation-clear-i03-s0 0
validation-cluttered-i04-s0 0
validation-reach-limited-i05-s0 0
validation-clear-i00-s1 1
validation-cluttered-i01-s1 1
validation-reach-limited-i02-s1 1
validation-clear-i03-s1 1
validation-cluttered-i04-s1 1
validation-reach-limited-i05-s1 1
validation-clear-i00-s2 2
validation-cluttered-i01-s2 2
validation-reach-limited-i02-s2 2
validation-clear-i03-s2 2
validation-cluttered-i04-s2 2
validation-reach-limited-i05-s2 2
```

- Frozen pipeline identity: `scripted_chunk_adapter`, `full_system`, `candidate_count=32`,
  `horizon_steps=50`, `control_dt_s=0.032`, `decision_seed_base=0`.
- Config: `configs/worldmind_diagnostic.yaml`, status `diagnostic_uncalibrated`,
  `diagnostic.mode=evidence_collection_only`, `calibration_stage=1`,
  `scored_evaluation_allowed=false`.
- No Eq. 3.8 weights, ranking direction, component instances, calibration candidate, calibration
  evidence, calibration evaluator or selected parameters.

## 3. Command status

### 3.1 Inert preflight only — the only currently executable command

After the owner decision set in ADR-0069 is resolved and the approved definitions file exists,
run only this read-only preflight:

```text
cd /home/tj330/programs/VLA-Project-week3
uv run python -m vlarobot.cli worldmind-diagnostic \
  --config configs/worldmind_diagnostic.yaml \
  --out runs/worldmind/stage1-validation-v1-2026-11-20 \
  --backend webots --state-source oracle --split validation
```

This command only validates configuration, bindings, frozen plan and scope. It creates no output
directory, launches no Webots process, records no episode and scores nothing. As of this request it
still fails closed at `task.destination is still a blocked template placeholder`.

### 3.2 Evidence-collection command — exact command not available

There is no execution-capable Stage 1 CLI command in the current tree:

- `worldmind-diagnostic` is preflight-only.
- `eval-worldmind` requires a `ready` production config, components, calibration evidence/candidate
  and selected weights; it must not be used for Stage 1.
- `eval --pipeline scripted` is the frozen scripted evaluator with bookkeeping/success summary; it
  does not consume the Stage 1 diagnostic config and must not be substituted.

Therefore the exact first-episode command cannot be supplied legitimately until the owner either
authorizes a minimal Stage 1 evidence runner with the flags above or designates an already-approved
command. Do not infer one from the production command in
`docs/WORLDMIND_OWNER_DECISION_READINESS_2026-11-18.md §9.9`; that command is for scored VALIDATION,
not Stage 1.

## 4. Expected evidence artifacts

For the reserved output directory, a Stage 1 runner must preserve the frozen recorder layout and
add non-scored WorldMind decision evidence:

```text
runs/worldmind/stage1-validation-v1-2026-11-20/
  plan.json
  episodes.csv
  summary.json
  timings.json
  episode-XX.steps.npz
  episode-XX.actions.jsonl
  episode-XX.predicted.jsonl
  episode-XX.contacts.jsonl
  episode-XX.meta.json
  frames/episode-XX/NNNNNN.jpg            # only if frame recording is enabled
  worldmind-sidecars/
    provenance.json
    decisions.jsonl
    prediction_pairing.jsonl
    errors.jsonl
    risk_rejection.json
    phase_latency.json
    verification_recovery.json
    instrumentation.json
```

`summary.json` and `episodes.csv` are frozen recorder outcome bookkeeping. In Stage 1 they are
uncalibrated diagnostic evidence only; their success/gate fields must not be used for calibration
selection or presented as post-calibration performance. `worldmind-sidecars/provenance.json` must
record effective bounds, timing flag, backend declaration, state source, frozen plan and source
hashes, and must not carry calibration identity or selected parameters.

## 5. No-scoring and no-TEST restrictions

- Keep `scored_evaluation_allowed=false`, `calibration_stage=1` and
  `mode=evidence_collection_only`.
- No `components`, `calibration` block, weights, normalization scales, ranking direction,
  candidate, evaluator or approval flag may be added to the Stage 1 config.
- Do not create, rank or approve calibration evidence or select parameters from this collection.
- Do not activate production WorldMind components or claim a calibrated result.
- `split` must be `validation`; TEST remains sealed until a separate authorization window.
- TRAIN demos or campaigns must not be read as part of this request.
- No new geometry diagnostic, v4/v5 retry, broad readiness audit or new calibration framework.

## 6. Preflight and authorization gates

Before execution may be authorized:

1. Owner supplies the six decisions in ADR-0069, including an actually approved definitions file
   with `review_status=approved`.
2. The inert `worldmind-diagnostic` command above passes with zero blockers and creates no output.
3. The exact evidence command, its implementation status, output path, backend, state source,
   scene list and stop conditions are explicitly approved by the owner.
4. A running `sim-server` is confirmed; Stage 1 does not launch Webots itself. Each episode's
   observation validity/freshness must be measured in its own session. The accepted v5 freshness
   result does not transfer to a new session.
5. Every action reaching a backend continues to pass `safety_monitor.validate()`. No safety
   check, arm-sweep blocker or full-system veto is weakened.
6. `full_system` remains blocked. `unresolved_forwarded_geometry` (`PipeSection`,
   `StraightStairsRail`) and `unverified-arm-sweep` remain distinct motion blockers. The accepted
   v5 table-grounding PASS is not a full-arm safety certificate.
7. Packet B stays blocked with `speed_bound_m_s = 28.49316700274912`; Packet D is unauthorized.
8. No push or merge.

Until all gates are satisfied and the owner explicitly authorizes evidence collection, nothing in
this request should be run.
