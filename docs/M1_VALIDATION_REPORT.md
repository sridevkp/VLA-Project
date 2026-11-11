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
- Found+fixed (MEASURED, ADR-0022): settle-check "crashes" were the runner killing the tree after a settle-gate reset failure (angular jitter 1.0-1.6e-4 rad/s vs 1e-4 threshold, max_steps cap); fix records non-settle as non-fatal, thresholds/default-OFF unchanged.
- Fixed: campaign hygiene (stray server removed), AGENTS/manual policy scoped; instrumentation/watchdog (per-request logs, busy status, exit codes, controller diagnostics file, faulthandler).
- Remaining: probe crash (sim 8.768 s) exact cause UNVERIFIED (pre-instrumentation); post-reset base tilt ~2.4 deg at rest recorded; tool-frame live contact verification, `grasp_servo=gt_pad` validation, 10-vs-10 batches, revised M1 acceptance, Windows live validation.

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

### Phase 2 (priority, in progress) - checkpoint after settle-check, (a) probe attempts, runner/report fixes
- Verdict: IN PROGRESS (a partially attempted/INCOMPLETE, b partial, c/d pending). No M1 acceptance/batch evidence yet.
- MEASURED: settle check completed with failures (see section below); runner determinism comparison/reporting fixed to handle missing traces and exit non-zero, with test (`tests/test_episode_runner.py`); servo/descent helpers and Webots-free servo tests present (`tests/test_gt_pad_servo.py` passing). (a) Dynamic descent pad probe (`scripts/verify_pad_probe.py`) failed: controllers/Webots terminated/crashed during expert `stop_after=PREGRASP` (~sim_time 8.768s, `runs/campaign-2026-11-11/phase2/pad_probe.log`), no `pad_probe.json`. Static reset-pose probe succeeded (`runs/campaign-2026-11-11/phase2/pad_static.json`, seed 0): finger midpoint and arm_7 GT diagnostics measured, `kinematics_telemetry` absent.
- Pad-bottom-vs-PROTO (>5 mm criterion): **UNVERIFIED** (no contact-at-grasp measurement). Nominal/proto pad-center offset retained in servo; no measured offset applied and no ADR.
- Failed/blocked: (a) live contact-based verification incomplete; (b) spec-complete fake-backend cases (explicit 50 mm/wrong tool-axis FK open-loop vs servo) to be tightened; (c) 3 watched gt_pad episodes+screenshots and (d) 10-vs-10 batches blocked on stable live runs; Webots/controller crash/instability observed.
- Next: finalize (b) tests, attempt (c)/(d) with physical-state guard, process hygiene, screenshots (first 3/condition + every failed episode) and per-image descriptions; record CONTAMINATED/FAILURE evidence if instability recurs.
- Human steps: review screenshots from (c)/(d) when produced (reset, PREGRASP exit, DESCEND mid, CLOSE, LIFT top, RELEASE, end) with one-line descriptions.

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

### Phase 3 crash forensics (reduced repro, bisect, settle-gate fix) - cause found and fixed
- Verdict: CAUSE FOUND (MEASURED), fix applied and tested Webots-free; 10-episode plain fresh-world confirmation run still pending (resume gate for (a)-(d)).
- MEASURED delta analysis vs known-good (`runs/` hold-none-*, m1-07-realign-*, ee30aca-era baseline rerun): only differences are the settle-gate code+config (`8127201`), inert servo helpers (`2c704b1`), and probe scripts; launch flags identical (`--batch --mode=fast`, rendering on, no `--extern-urls`).
- MEASURED reduced repro (all optional features OFF, `runs/campaign-2026-11-11/phase3_reduced_repro/`, `--n 5 --fresh-world`): 5/5 completed reset and ran to scripted end (DESCEND×4, home×1), `webots_exit_code=0`, clean shutdowns. **Baseline is stable.** Episode-level failures (3× "move timed out arm_6_joint", 2× IK 0.006394 m vs 0.005 m tol; 0/5 success) are pre-existing known behavior, not crashes.
- MEASURED bisect (settle gate ON, `phase3_bisect_settle/`): 5/5 completed reset; ep02 reproduced the crash with the full error captured (`request_error op=reset duration_s=13.6458 error=Post-reset settle failed: reason=max_steps steps_used=500 consecutive=0/20 final={... angular 1.615e-4}`); client raised `Simulator rejected 'reset'`; runner killed the tree; `webots_exit_code=0` everywhere. **The `Terminating.` lines were runner cleanup, not a Webots crash.**
- Fix (commit `6bd41c1`, ADR-0022): non-settle recorded (`fatal: false`) + WARN instead of raise; thresholds and default-OFF unchanged; 3 new Webots-free tests (jitter max_steps no-raise, settled path, end-to-end fingerprint recording). Full gates: `ruff`/`format`/`ty` pass, `pytest -m "not webots"` 188 passed.
- MEASURED finger GT 3-pose probe (`phase3_finger_gt/finger_gt_poses.json`; lookup = Pose-derived endPoint Solids `gripper_left_finger_link`/`gripper_right_finger_link` of each finger SliderJoint, pad centre at `finger_tip_offset_m=0.1741` along Solid local -z, averaged; arm_7 GT from its own endPoint Solid): arm_7->finger-midpoint **distance 0.202825 m constant**; vector in the sim arm_7 frame **(0, 0, +0.202825) m**, per-axis spread <= 1.7e-7 m across reset + arm_1 ±0.08 rad + arm_2/arm_4 ±0.05 rad (torso fixed, tool > 0.80 m, no can approach). vs PROTO: pad-centre 0.1901 m (+12.7 mm), pad-bottom 0.235 m (−32.2 mm) — measured reference sits between pad centre and pad bottom; no measured offset applied to servo (no contact-at-grasp measurement yet). urdf-FK frame differs from the sim Solid frame by a constant ~0.0144 m x-component (magnitudes agree to 1e-6). New telemetry: `arm_7_joint_ground_truth_orientation_rows` (endPoint Solid orientation).
- Instrumentation (commit `4700bdf`): per-episode controller diagnostics file + `faulthandler` via `VLAROBOT_CONTROLLER_STDERR_PATH`, `episode-NN.exit.json` (exit codes + controller tail), always-written `episode-summary.json`. Note: `cli doctor` may list its own wrapper shell when the command text mentions a `backends/webots` path (benign self-match).
- Pending: 10 consecutive plain fresh-world episodes (resume gate) -> then (a) pad probe re-attempt, (c) 3 watched gt_pad episodes + screenshots, (d) 10-vs-10 batches, revised M1 acceptance. Probe crash (sim 8.768 s) exact cause UNVERIFIED (pre-instrumentation); superseded by the instrumented finger-GT probe path. Windows live validation remains UNVERIFIED/HUMAN.
- Human steps: none yet for Phase 3; screenshots requested at (c)/(d).
