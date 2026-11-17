# REPORT_FIGURES.md — submitted Phase I report figures → implementation & evidence (DRAFT)

Controlling source: the confirmed submitted Phase I report (SHA-256
`23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`). This document maps
every figure of that report to (a) what the figure currently contains, (b) the real
implementation/artifact it must correspond to, and (c) the evidence that can back it.

Status labels:

- **AS-REPORTED** — the figure exists in the report as a drawn diagram; verify content
  against implementation before reuse.
- **PLACEHOLDER** — the report itself labels the figure a placeholder/screenshot
  placeholder; it must be replaced with real content or dropped.
- **REPLACE** — the figure's content is factually wrong for this project (wrong robot
  or simulator) and must be replaced.
- **PENDING-EVIDENCE** — no recorded artifact yet exists to populate the figure; it may
  only be filled from a hash-pinned artifact under `runs/` (or its `docs/evidence/`
  mirror) once the corresponding authorized run has happened.

Conformance overlay (from `docs/REPORT_CONFORMANCE.md`): use one primary status per
figure — **Implemented and verified**, **Implemented but not physically validated**,
**Missing**, **Blocked**, or **Explicitly waived by approved decision** — plus an evidence
qualifier. In particular, inert unit tests and the K=32/H=50 nominal-kernel latency profile
are not scored per-decision traces. Figures 4.6 and 4.7 therefore remain
`PENDING-EVIDENCE` for campaign-backed candidate/risk visualization.

No figure in this tree may be regenerated with invented data. Every replacement image or
plot must be derived from a recorded artifact whose hash is pinned, and the derivation
noted next to the figure.

## Figures 3.x — architecture and data flow

| Figure | Report content | Status | Implementation correspondence / action |
|---|---|---|---|
| 3.1 | Overall architecture of the proposed world-model-based VLA system | AS-REPORTED | Three-process layout (sim / brain / model server, ZeroMQ+msgpack, deterministic lockstep) per `docs/ARCHITECTURE.md`; the report diagram must be checked against the committed architecture before reuse in the final report |
| 3.2 | Task instruction, visual observation, policy/planner, robot-action workflow | AS-REPORTED | Matches the propose-and-evaluate loop (`vla/action_adapter.py` → `risk/gate.py` → `planning/execution.py`); ADR-0029 deviation (Eq 3.2) must be visible in any reused version — the VLA does not condition on predicted state internally |
| 3.3 | Placeholder: module-interaction diagram showing data exchanged | PLACEHOLDER | Real interactions are enumerated in `core/wire.py` message contracts; redraw from the wire schema when the final report is prepared |
| 3.4 | Data-flow diagram of the proposed system | AS-REPORTED | Same correspondence check as 3.1; include the safety-monitor veto path (`execution/safety_monitor.py`) which every applied action passes |

## Figures 4.x — experimental results (report §4)

| Figure | Report content | Status | Real evidence available / action |
|---|---|---|---|
| 4.1 | "Webots simulation environment showing the robot, task objects and obstacles" | **REPLACE** | The submitted image shows a UR5 in V-REP/CoppeliaSim — not TIAGo in Webots. Replacement must be a real TIAGo/Webots scene (world `worlds/factory.wbt`). Candidate source: recorded camera frames from the frozen baseline campaign (`runs/eval/baseline-train-s0-fresh-v2-registered/frames/`, 620 jpgs) or a new screenshot of the actual scene, once authorized |
| 4.2 | Placeholder: world-model prediction — current state, candidate action, predicted outcome | PLACEHOLDER | Real traces exist: per-episode `episode-XX.predicted.jsonl` + `episode-XX.steps.npz` in the frozen VLA campaign (`runs/eval/vla-train-s0-step20000-recovered-v1/`, primary worktree) and read-only replay evidence (`docs/WORLDMIND_DEVELOPMENT_CHECKPOINT.md`: TCP RMSE 4.3–6.7 mm by family). A plot derived from these files is legitimate; label the source artifact |
| 4.3 | Placeholder: robot executing a selected trajectory in Webots | PLACEHOLDER | Recorded camera frames exist for the scripted baseline (`runs/eval/baseline-train-s0-fresh-v2-registered/frames/`) and grasp-repeat runs (part of the 1,083 jpgs under primary `runs/eval/`) and for the VLA campaign (`.../frames/`, 24 jpgs) |
| 4.4 | Screenshot placeholder: Webots scene initialization and robot configuration | PLACEHOLDER | Requires a scene screenshot of `worlds/factory.wbt` at reset; not yet captured. PENDING-EVIDENCE (capture needs an authorized sim session, or reuse of a recorded frame if acceptable to the owner) |
| 4.5 | Screenshot placeholder: perception output and estimated environment state | PLACEHOLDER | Recorded RGB frames exist (see 4.3); the "estimated environment state" side is the oracle-state provenance record (`eval/recorder.py` `state_source` per episode). A composite from these two recorded sources is legitimate |
| 4.6 | Screenshot placeholder: candidate trajectories and their predicted outcomes | PLACEHOLDER | Candidate-generation and prediction mechanics are implemented and unit-tested (`planning/candidates.py`, `risk/comparison.py`, `world_model/rollout.py`). No scored per-decision campaign trace exists yet for arms 3–5 (week 3 pending authorization). PENDING-EVIDENCE for a campaign-backed figure; a replay-derived illustration is possible but must be labelled as such |
| 4.7 | Screenshot placeholder: risk assessment and selected trajectory | PLACEHOLDER | Same status as 4.6: `risk/gate.py` mechanics complete and unit-tested; hard-veto precedes cost ranking; **weights not yet calibrated on VALIDATION (ADR-0029 week 3)**. PENDING-EVIDENCE |
| 6.1 | Placeholder: final Gantt chart across twelve weeks | PLACEHOLDER | Operationally superseded by ADR-0029's four-week campaign; the final report should either redraw the Gantt to the executed schedule or annotate the deviation. No measurements involved |

## Appendix figures C.1–C.3

| Figure | Report content | Status | Real evidence / action |
|---|---|---|---|
| C.1 | Placeholder: complete Webots simulation scene | PLACEHOLDER | Scene = `worlds/factory.wbt` (frozen in baseline provenance). Same capture requirement as 4.4 |
| C.2 | Placeholder: obstacle-avoidance scenario | PLACEHOLDER | Closest real counterpart is the **cluttered** scenario family (`configs/scenarios.yaml`: families `clear / cluttered / reach-limited`). There is no dedicated obstacle-avoidance task; the final report should relabel this figure to the cluttered family or drop it |
| C.3 | Placeholder: goal-reaching / object-manipulation scenario | PLACEHOLDER | The CAN pick-and-place task (`execution/scripted_can.py`, `clear`/`reach-limited` families); recorded frames and step recordings exist for the completed campaigns |

## Evidence inventory (recorded, hash-pinned or mirrorable)

| Asset | Location | Notes |
|---|---|---|
| Scripted-baseline TRAIN campaign frames | `runs/eval/baseline-train-s0-fresh-v2-registered/frames/` (620 jpgs) — primary worktree | Frozen 23/24 baseline; provenance-pinned (12 files) |
| Grasp-repeat diagnostics frames | `runs/eval/grasp-train-clear-i03-s0-registered-v1/repeat-{1..4}/frames/` — primary worktree | Part of 1,083 jpgs under `runs/eval/` |
| VLA-only campaign evidence | `runs/eval/vla-train-s0-step20000-recovered-v1/` (24 jpgs; per-episode `steps.npz`, `predicted.jsonl`, `actions.jsonl`, `latency.jsonl`, `meta.json`; `summary.json`, `provenance.json`, `timings.json`) — primary worktree | Frozen 0/24 gate-FAILED result; 225 files hash-pinned, manifest mirrored in `docs/evidence/` |
| Scripted VALIDATION campaign frames | `runs/validation-evidence-v1/campaign/frames/` (425 jpgs) — analytic worktree | 15/18 result |
| Independent-validation candidate frames | `runs/independent-validation-candidates-v3/*/frames/` (43 jpgs) — analytic worktree | Diagnostic candidates, not gate evidence |
| AD1 preserved-pair analysis | `runs/ad1-train-clear-i00-s100-frame0-v1/` — analytic worktree | `docs/VLA_TRAIN_POSTMORTEM.md` pairs (episodes 08/11/14) reference per-episode meta in the primary VLA campaign dir |

## Replacement procedure (for the final report, post-approval)

1. Pick the figure; identify the exact source artifact (path + SHA-256) from the table
   above; never draw from hand-entered numbers.
2. Generate plots/scripts under `tools/` (deterministic, seeded) or capture screenshots
   during an authorized sim session; record the derivation command in the figure caption
   or in `docs/evidence/`.
3. Update `docs/DECISIONS.md` with the figure-finalization decision; never overwrite the
   submitted PDF's figures silently.
