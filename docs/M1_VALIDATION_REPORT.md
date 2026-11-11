# M1 Validation Report (campaign 2026-11-11)

## Verdict table
| Gate/Phase | Verdict | Numbers / interval | Evidence | Notes |
|---|---|---|---|---|
| Phase 0 preflight | IN PROGRESS | ruff/ty/pytest green (MEASURED), baseline rerun 0/1 (MEASURED) | `runs/campaign-2026-11-11/phase0/` | Bench/equivalence pending |
| Phase 1 determinism/reset | NOT RUN | - | - | - |
| Phase 2 gt_pad grasp | NOT RUN | - | - | - |
| Phase 3 M1-07 (>=16/20) | NOT RUN | - | - | - |
| Phase 4 M1-06a equivalence | NOT RUN | - | - | - |
| Phase 5 kinematic model | NOT RUN | - | - | - |

## MEASURED / INFERRED / UNVERIFIED
- **MEASURED:** repo at `95f4e8d` then campaign commits; `ruff check .` pass, `ty check src tests` pass (0), `pytest -m "not webots"` 180 passed; Webots `R2025a` (`~/.local/opt/webots/webots`); `cli doctor` clean at preflight/after aborts. Baseline attempt 1 hung (ping timeouts to `tcp://127.0.0.1:50581`, trace 46 records, last `sim_time_s=13.024`, tail around DESCEND with `ik_rejection`, no summary) and was killed. Baseline rerun completed 0/1 (`final=FAILED`, `fail_state=CLOSE`, `dist_m=0.245`, `home/place/height/tilt=False`, `startup_s=18.34`, `episode_s=245.70`, `close_misaligned` xy_error `0.057236 m` > `0.008 m`).
- **INFERRED:** attempt 1 stall likely involved server busy/unresponsive REP handling or blocked backend/controller interaction; cause not fully isolated until instrumented runs produce per-request logs.
- **UNVERIFIED:** live 12-pose sweep, determinism reproducibility, M1-07/M1-06a outcomes, sim-vs-urdf grasp comparison, Windows live validation.

## Defects found / fixed / remaining
- Found (MEASURED): baseline run-episodes can hang with client ping timeouts while processes remain (attempt 1). Baseline rerun failed at CLOSE alignment (misaligned, not grasp/place success).
- Fixed: campaign hygiene (stray server removed), AGENTS/manual policy scoped; instrumentation/watchdog work in progress.
- Remaining: hang root cause confirmation, post-reset settle/fingerprints, tool-frame live verification, debug markers, `grasp_servo=gt_pad`, full evidence/screenshots, phase gates.

## Flags / config
- Campaign defaults unless noted: `kinematic_model=urdf`, `grasp_servo=none`, `pregrasp_realign/closed_loop_waypoints` per `configs/robot.yaml`, `export_sim_kinematics=false`, `allow_unverified_tool=false`, `debug_markers=false`. Any defect-fix flags added will be recorded with ADR/default/why.

## Reproduce commands
(see per-phase summaries and `docs/MANUAL_VALIDATION.md`)

## HUMAN CHECKLIST (<= 10 minutes)
- Pending (will list 6-10 screenshots and Windows/CI decisions once evidence exists).

## Phase summaries

### Phase 0 (preflight/baseline) - interim
- Verdict: IN PROGRESS/PARTIAL. Baseline rerun (rendering on, `--batch --mode=fast`, seed 0) 0/1 FAILED at CLOSE (MEASURED: `dist_m=0.245`, home/place/height/tilt False, `startup_s=18.34`, `episode_s=245.70`, `close_misaligned xy_error_m=0.057236>0.008`); attempt 1 aborted hung (MEASURED: ping timeouts, 46 records, last sim 13.024, no summary).
- MEASURED correction (0a): field-by-field first-10 comparison shows attempt1 (render-on, hung) == headless (no-render) for first 10 and same end (46 records, DESCEND exit/`ik_rejection`, sim 13.024), while both differ from rerun (render-on, completed) at record 0 and rerun ended 51 records CLOSE misaligned. Identical configs differed and different-rendering config matched attempt1 => **run-to-run nondeterminism**, not a supported rendering effect. Rendering-equivalence remains UNPROVEN.
- Failed/blocked: full `bench-steps` ms/step matrix (images/telemetry/rendering variants) and independent-ping health socket not completed; no screenshots yet. Hang diagnosis: server now logs request start/end/duration (>2s warning) and watchdog busy status (MEASURED implementation), but independent ping during busy REP not implemented.
- Correction 0(b) (MEASURED): runner hardened to record exception/failure-path outcomes and always terminate process tree; tests added (`tests/test_episode_runner_failures.py`). True server-stall attribution remains UNVERIFIED without a reproducing instrumented stall (busy/age logging/status now available). Independent health socket not added (REP constraint; 0b/diagnostics prioritized).
- Next (PRIORITY CHANGE): stop Phase 1 determinism x5/soft-vs-fresh; run Phase 2 `grasp_servo=gt_pad` (oracle, bounded corrections) with live pad verification, then 10 gt_pad vs 10 baseline fresh-world episodes. M1 acceptance revised per instructions.
- Human steps: none required to continue autonomous runs; Windows live validation and visual confirmation remain HUMAN (see `docs/MANUAL_VALIDATION.md`).

### Phase 2 (priority, in progress) - interim after setup
- Verdict: IN PROGRESS (a-d pending). 5-episode fresh-world nondeterminism/settle check started with `reset_post_settle_enabled: true` (MEASURED launch: `runs/campaign-2026-11-11/phase1_settle_check`, determinism-check 5, seed 0, batch fast rendering on); outcome UNVERIFIED at time of update.
- MEASURED: post-reset settle gate implemented (flagged, default off in repo) with step counting/fingerprint fields; `grasp_servo` config/plumbing and servo helper scaffolding added; runner hardened (0b) and 182 webots-free tests passing.
- Failed/blocked: pad-position verification probe (a), fake-backend servo convergence tests (b), 3 watched episodes with screenshots (c), and 10-vs-10 batches (d) not yet executed; full `gt_pad` servo control logic incomplete.
- Next: complete nondeterminism check record-0 comparison (tol 1e-3), run (a) safe pad probe and set servo to MEASURED pad value if |diff|>5mm (ADR), implement/verify servo + (b)/(c)/(d) per rules.
- Human steps: none required to continue autonomous runs; Windows/visual review remain HUMAN (see MANUAL_VALIDATION.md).

### Nondeterminism/settle check (5 fresh-world episodes, settle gate enabled) - completed with failures
- Verdict: NOT COMPARABLE across 5 (MEASURED). Episode 00 completed with trace; episodes 01-04 hung during `reset` and produced no `episode-0{1..4}.jsonl`, so the runner crashed on missing traces during comparison (runner now reports missing traces and exits non-zero without crashing).
- MEASURED (episode 00 only): `post_reset_settle` settled in **53 steps** (20 consecutive stable, thresholds lin/ang/tilt-delta 1e-4/1e-4/0.01, final speeds ~2.33e-5 m/s, 9.38e-5 rad/s, tilt 2.449°), `sim_time_s` at record 0 **1.696s**, `physics_step_count_at_reset=53`, final state **CLOSE** (failed). Record-0 key fields: base_world_xyz_m `[-0.5380553511747798,-3.7103890689800054,0.08823997795100583]`, can_world_xyz_m `[-0.7998876879166725,-4.389982223212384,0.8110620487771911]`, achieved_ee_base_xyz_m `[0.20348613587079004,-0.9854741773286547,0.9630241019993075]`.
- MEASURED record-0 diffs across 5: **UNVERIFIED/unavailable** (only episode 00 has a trace). Finding recorded: **fresh-world traces differ from record 0 (base pose/tilt at rest ~2.4 deg; settle gate not promoted)**. `reset_post_settle_enabled` restored to repo default `false`; **no more determinism checks** (M1 evidence rule: batch statistics, >=10 fresh-world episodes).
- Reset timeout check (episodes 1-4, from `runs/campaign-2026-11-11/phase1_settle_check/episode-0{1..4}.log` and `.out`):
  - MEASURED: each episode logged `request_start op=reset seq=2 sim_time=None` and server watchdog `Simulator busy: request=reset` ages (ep01 max 11.825s, ep02 max 13.883s, ep03 max 13.298s, ep04 max 11.829s) with controller `Terminating` lines after; no `request_end op=reset`. Runner summary `episode_s` ~12.86/13.62/15.12/13.31s and outcomes `RUNNER` with no trace files.
  - Configured (MEASURED): client reset timeout `zmq_reset_timeout_ms=120000` (`REQUEST_TIMEOUTS_MS["reset"]=120000`), ping 5000ms, runner startup timeout 180s, shutdown timeout 30s; no separate per-episode wall-clock execution timeout in runner.
  - Classification (MEASURED/INFERRED): observed busy ages/end (~12–14s) are far below the 120s client reset timeout, so **not a client-timeout cutoff** of legitimate settle (~53 steps ≈ ~21s observed in ep00). **INFERRED**: real server/controller hang or stall during reset leading to controller/Webots termination/crash before reset completed (client did not time out). Reset timeout not increased (no timeout-limited cutoff evidenced); settle thresholds unchanged.
- Failed/blocked: episode 01-04 outcomes/traces UNVERIFIED (no trace). Reset hang under settle-enabled run remains UNVERIFIED in cause and was not reproduced after restoring default.
- Human steps: none.
