# M1 Validation Report (campaign 2026-11-11)

## Verdict table
| Gate/Phase | Verdict | Numbers / interval | Evidence | Notes |
|---|---|---|---|---|
| Phase 0 preflight | IN PROGRESS | ruff/ty/pytest green (MEASURED), baseline rerun 0/1 (MEASURED) | `runs/campaign-2026-11-11/phase0/` | Bench/equivalence pending |
| Phase 1 determinism/reset | NOT RUN | - | - | - |
| Phase 2 gt_pad grasp | NOT RUN | - | - | - |
| Phase 3 M1-07 (revised ADR-0025: (a) 5-consecutive + (b) 10-scene ≥8/10) | **GATE PASSED (MEASURED)** | (a) 10/10 full successes, Wilson [72%, 100%]; (b) 9/10 (90%), Wilson [60%, 98%] ≥ 8/10 | `runs/campaign-2026-11-11/phase4d_gtpad10/`, `runs/campaign-2026-11-11/phase6_scene10/` | 20-seed design replaced by ADR-0025; (b) ran 2026-11-14, only far corner `grid_dx-0.05_dy-0.05` failed (IK realign) |
| Phase 4 gt_pad CLOSE pacing (ADR-0024) | GATE PASSED (MEASURED) | gate 1/1 success; batch 10/10 full successes, Wilson [72%, 100%]; CLOSE stall 30.3–31.5 mm over 31–40 steps; can lift 78.6–83.8 mm; dist 0.0425–0.0585 m | `runs/campaign-2026-11-11/phase4_closepacing/`, `phase4d_gtpad10/` | live VERIFIED 2026-11-13; 4c+4d merged batch complete; captions for ep00–02 |
| Phase 5 kinematic model | NOT RUN | - | - | - |
| **M1 verdict (2026-11-14)** | **PASSED** | M1-07 (a)+(b) both passed; M1-06a live 5+5 **WAIVED, UNVERIFIED** (ADR-0026) | above rows + `docs/DECISIONS.md` ADR-0025/0026 | M1-06a waiver user-directed; architecture acceptance line amended |

## MEASURED / INFERRED / UNVERIFIED
- **MEASURED:** repo at `95f4e8d` then campaign commits; `ruff check .` pass, `ty check src tests` pass (0), `pytest -m "not webots"` 180 passed; Webots `R2025a` (`~/.local/opt/webots/webots`); `cli doctor` clean at preflight/after aborts. Baseline attempt 1 hung (ping timeouts to `tcp://127.0.0.1:50581`, trace 46 records, last `sim_time_s=13.024`, tail around DESCEND with `ik_rejection`, no summary) and was killed. Baseline rerun completed 0/1 (`final=FAILED`, `fail_state=CLOSE`, `dist_m=0.245`, `home/place/height/tilt=False`, `startup_s=18.34`, `episode_s=245.70`, `close_misaligned` xy_error `0.057236 m` > `0.008 m`).
- **INFERRED:** attempt 1 stall likely involved server busy/unresponsive REP handling or blocked backend/controller interaction; cause not fully isolated until instrumented runs produce per-request logs.
- **UNVERIFIED:** live 12-pose sweep, determinism reproducibility, M1-06a live 5+5 equivalence (waived for the M1 verdict, ADR-0026), sim-vs-urdf grasp comparison, Windows live validation.

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
- Review screenshot caption sets (no new shots for phase6 — headless run):
  `runs/campaign-2026-11-11/screenshots/phase4d_gtpad10/CAPTIONS.md` (43 shots, ep00–02) and
  `runs/campaign-2026-11-11/screenshots/phase4_closepacing/CAPTIONS.md` (gate episode).
- Optional: confirm the far-corner IK failure visually (the only failing position).

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
- MEASURED finger GT 3-pose probe (`phase3_finger_gt/finger_gt_poses.json`; lookup = Pose-derived endPoint Solids `gripper_left_finger_link`/`gripper_right_finger_link` of each finger SliderJoint, pad centre at `finger_tip_offset_m=0.1741` along Solid local -z, averaged; arm_7 GT from its own endPoint Solid): arm_7->finger-midpoint **distance 0.202825 m constant**; vector in the sim arm_7 frame **(0, 0, +0.202825) m**, per-axis spread <= 1.7e-7 m across reset + arm_1 ±0.08 rad + arm_2/arm_4 ±0.05 rad (torso fixed, tool > 0.80 m, no can approach). vs PROTO: pad-centre 0.1901 m (+12.7 mm), pad-bottom 0.235 m (−32.2 mm) — measured reference sits between pad centre and pad bottom; **no measured offset applied to servo (no contact-at-grasp measurement yet)**. urdf-FK frame differs from the sim Solid frame by a constant ~0.0144 m x-component (magnitudes agree to 1e-6). New telemetry: `arm_7_joint_ground_truth_orientation_rows` (endPoint Solid orientation).
- Instrumentation (commit `4700bdf`): per-episode controller diagnostics file + `faulthandler` via `VLAROBOT_CONTROLLER_STDERR_PATH`, `episode-NN.exit.json` (exit codes + controller tail), always-written `episode-summary.json`. Note: `cli doctor` may list its own wrapper shell when the command text mentions a `backends/webots` path (benign self-match).
- ADR-0023: gt_pad servo uses MEASURED pad geometry + contact-height probe stop height. URDF chain modification was refuted live (2026-11-12, `rep01.jsonl` FK put chain tip 0.377m above real pad due to sim-vs-URDF arm_7 frame origin offset and 180° orientation difference, causing elbow-down joint-limit lock and base tip); the URDF chain KEEPS legacy `[0, 0, -0.1741]` unconditionally. The contact-height probe succeeded 3/3 (`pad_contact_probe.json`: trigger c, mean offset −2.340 mm, std 0.243 mm < 2 mm), and `probe_pad_reference_to_pad_bottom_m` is configured.
- Pending: ~~revised-gate (b) 10-scene mini-generalization run~~ **DONE 2026-11-14, see Phase 4(e)
  below** (ADR-0025). Windows live validation remains UNVERIFIED/HUMAN.
- **Corrected-close baseline re-run (2026-11-13, MEASURED):** `phase5_baseline5/` **0/5
  [0%, 43%]** — 2× CLOSE misaligned (guard rejected at e_xy 51.4 mm before the gripper
  command) and 3× arm_6 DESCEND timeout; the paced CLOSE never executed on the baseline
  pipeline in this run, so its live evidence remains the gt_pad batch (10/10 with stalls at
  30.3–31.5 mm). Seed outcomes are not reproducible across batches (seed 1: home → misaligned;
  seed 4: IK-fail → timeout), confirming chaos near the failure boundary.
- Human steps: review screenshots/captions for Phase 4(c)+(d) —
  `runs/campaign-2026-11-11/screenshots/phase4d_gtpad10/CAPTIONS.md` (ep00–02, 43 shots) and
  `.../screenshots/phase4_closepacing/CAPTIONS.md` (gate episode).

### Phase 4 — gt_pad stall-based CLOSE/RELEASE pacing validation (ADR-0024)

- **Status:** Code committed (commit `6eaafd5`); **gate PASSED live 2026-11-13** (1/1 success,
  `phase4_closepacing/`); **10-episode batch (4c+4d) COMPLETE — 10/10 full successes
  (MEASURED, 2026-11-13, see below).**
- **Code change:** `scripted_can.py` CLOSE now runs up to 300 steps with stall detection
  (`still_count >= 15` after step 10, or `opening <= 0.001` for 3 steps). RELEASE waits up to
  `GRIPPER_OPEN_TIMEOUT_S / TIME_STEP` steps until `opening >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE_M`.
  Root cause of 0/3 success in `phase4_gtpad3/`: single `backend.step(1)` after CLOSE was
  insufficient for fingers to stall against the can; gripper remained open and LIFT left the can
  on the table. ADR-0024 details the fix.
- **Webots-free gates:** `ruff/format` clean, `pytest -m "not webots"` **198 passed**.
- **Phase 4 gate (done 2026-11-13, MEASURED):** 1 fresh-world `gt_pad` episode
  (`phase4_closepacing/`, seed 0) verified all three signals: CLOSE exit `gripper_opening_m =
  0.0314 m` > 0.004 (stall at can radius, ~31 sim steps); `can_world_xyz_m[2]` +86.7 mm during
  LIFT; `dist_m = 0.0493` < 0.10, `success=true` (height error 12 µm, tilt 0.068°, home reached).
  **First successful gt_pad episode of the campaign.** Also refutes the "finger–can collision
  absent" hypothesis (see ADR-0024): pre-pacing runs lifted the arm before fingers reached the
  can; `contact_nodes` stays `[]` even on successful stall (telemetry gap).
  Screenshots + captions: `runs/campaign-2026-11-11/screenshots/phase4_closepacing/CAPTIONS.md`.

- **Phase 4(c)+(d) — 10 gt_pad episodes with first 3 watched (COMPLETE 2026-11-13, MEASURED):**
  ```
  uv run python -m vlarobot.cli run-episodes --n 10 --fresh-world --webots ~/.local/opt/webots/webots --grasp-servo gt_pad --trace-dir runs/campaign-2026-11-11/phase4d_gtpad10/
  ```
  **Result: 10/10 full successes** — every episode finished `home` with all four criteria
  (`place_distance_ok`, `can_height_ok`, `can_tilt_ok`, `home_reached`) true; task success
  **10/10 (100%), Wilson 95% CI [72%, 100%]**; final distance 0.0425–0.0585 m (gate 0.10);
  can lift during LIFT 78.6–83.8 mm; CLOSE stalled at 30.3–31.5 mm opening over 31–40 steps
  (baseline bug version: 1 step, 44 mm, lift 0.0); arm_6 margin at PREGRASP exit −4.4…−7.5 mrad
  (within ±1.39 rad limit). Mean wall 1358 s/episode (917–1522 s), total 3.78 h.
  Trajectory groups (state-boundary/servo md5): {00,01,02,07,09}, {03}, {04,05}, {06}, {08} —
  **5 distinct trajectories, all passing** (seeds vary only the IK-restart RNG, ADR-0025).
  Flags: `pregrasp_realign=true`, `closed_loop_waypoints=true`, `grasp_servo=gt_pad`,
  `kinematic_model=urdf`, snapshot `b5bdfe6e…`. Baseline beside it: `phase3_gate10b/`
  **0/10 [0%, 28%]** (home 3/10 [11%, 60%]) — with the caveat that it ran the single-step CLOSE
  bug (ADR-0024); its failure kinds (4× arm_6 DESCEND timeout, 2× DESCEND IK-solve-failed,
  1× CLOSE misaligned, 3× home-never-lifted) **do not appear in this batch at all**.
  **Scope label:** one fixed scene — this is *repeatability at a single fixed scenario*, not a
  scene-generalization rate (ADR-0025). One ~8.4 h inter-episode stall (session-idle shell
  freeze, INFERRED) fell between ep02 and ep03; excluded from wall figures above.
  Screenshots + captions (ep00–02, 43 shots):
  `runs/campaign-2026-11-11/screenshots/phase4d_gtpad10/CAPTIONS.md`.

- **Phase 4(e) — M1-07 gate (REVISED by ADR-0025, user-directed 2026-11-13; replaces the
  20-seed design):** (a) 5 consecutive fresh-world gt_pad successes with identical criteria —
  **SATISFIED** (10/10 above); (b) 10-scene mini-generalization run — CAN xy offset on a ±5 cm
  3×3 grid + 1 rotated-yaw scene, Supervisor-applied at reset, fresh world per scene, unchanged
  criterion, **≥ 8/10 passes**. Implementation **landed 2026-11-13** (first piece of the M2
  ScenarioGenerator): `ScenarioSpec.can_xy_yaw_offset` (world-frame dx, dy, yaw),
  `eval/scenarios.py::mini_grid10()` (10 named scenes), `--scenario` CLI →
  `RunnerSettings.scenario` → policy → `backend.reset`, Supervisor field-write composition in
  the Webots backend, pure `compose_can_reset_pose()` helper (import-safe for tests),
  `EpisodeOutcome.scenario` per-position column; tests in `tests/test_scenarios.py` +
  `tests/test_webots_backend_reset.py`. If (b) fails:
  M1 closes as PASSED WITH LIMITATION with failing positions carried to M2.
  *(Typo fix 2026-11-13: precursor was written "≥ 12/10" — impossible; corrected to ≥ 8/10 = 80%,
  consistent with the M1-07 acceptance in `docs/ARCHITECTURE.md` §M1-07. No criterion changed.)*

- **Phase 4(e) LIVE RESULT (MEASURED, 2026-11-14, EXIT=0, process hygiene clean):**
  ```text
  uv run python -m vlarobot.cli run-episodes --n 10 --fresh-world \
    --webots ~/.local/opt/webots/webots --grasp-servo gt_pad \
    --scenario mini-grid10 \
    --trace-dir runs/campaign-2026-11-11/phase6_scene10/
  ```
  **9/10 full successes (90%), Wilson 95% CI [60%, 98%] ≥ 8/10 → gate (b) PASSES → M1-07
  PASSES → M1 PASSED** (M1-06a waived per ADR-0026).

  | # | scene | success | final | fail_state | dist (m) | wall (s) |
  |---|---|---|---|---|---|---|
  | 00 | grid_dx-0.05_dy-0.05 | **N** | FAILED | PREGRASP | 0.2055 | 111 |
  | 01 | grid_dx+0.00_dy-0.05 | Y | home | — | 0.0505 | 1666 |
  | 02 | grid_dx+0.05_dy-0.05 | Y | home | — | 0.0670 | 1403 |
  | 03 | grid_dx-0.05_dy+0.00 | Y | home | — | 0.0667 | 979 |
  | 04 | grid_dx+0.00_dy+0.00 | Y | home | — | 0.0507 | 1385 |
  | 05 | grid_dx+0.05_dy+0.00 | Y | home | — | 0.0548 | 1695 |
  | 06 | grid_dx-0.05_dy+0.05 | Y | home | — | 0.0426 | 1699 |
  | 07 | grid_dx+0.00_dy+0.05 | Y | home | — | 0.0412 | 2157 |
  | 08 | grid_dx+0.05_dy+0.05 | Y | home | — | 0.0651 | 1155 |
  | 09 | yaw90_center | Y | home | — | 0.0621 | 1514 |

  (Authoritative per-criterion table:
  `python3 /tmp/opencode/report_batch.py runs/campaign-2026-11-11/phase6_scene10`.)
  - **Failure (only position):** `grid_dx-0.05_dy-0.05` (farthest corner) — deterministic IK
    rejection at PREGRASP realign, `error=0.014551 m` (limit 0.005 m), `tilt=0.357°`
    (limit 15°), abort at sim 5.28 s, no DONE record. **Reproduced identically on relaunch**
    (pre-reboot ep00 aborted the same way), so it is NOT the server-restart artifact first
    suspected — the pre-reboot ep00/ep01 files are preserved in
    `phase6_scene10/pre-relaunch/` (there ep01 = full success on `dx+0.00_dy-0.05`,
    supplementary repeatability evidence).
  - **All 9 successes** met every unchanged criterion (DONE at HOME, dist 0.0412–0.0670 m
    ≤ 0.10, can z within 0.02 m of rest, tilt < 15°); CLOSE stall 30.4–31.4 mm over 31–46
    steps; arm_6 PREGRASP-exit margins −0.0042…−0.1053 rad (limit ±1.39); `yaw90_center`
    succeeded — yaw is physically inert for a cylinder but the scene plumbing
    (CLI → policy → wire → Supervisor field write) is exercised end-to-end
    (scene-0 can at `[-0.84985,-4.43999]` vs default `[-0.79988,-4.38998]`,
    Δ=(−0.0500,−0.0500) m, z unchanged — MEASURED).
  - **Base never commanded to move** — drift ≤ 3.9 mm across all 10 episodes (hold jitter
    only). The far corner needs base repositioning: **reported, carried to M2**, no
    base-positioning work in M1 (ADR-0025 scope limit).
  - Mean 1376 s/episode (111–2157 s), **3.82 h total episode wall** — below the 4 h
    threshold, so the speed/`--observe-images`/headless equivalence work is skipped.
  - Flags: `grasp_servo=gt_pad`, `pregrasp_realign=true`, `closed_loop_waypoints=true`,
    `kinematic_model=urdf`; snapshot `b5bdfe6e…`. No screenshots captured for this batch
    (headless run); visual evidence = phase4d/phase4 caption sets below.

- **Human steps required:** review the screenshot captions from 4(c) (now automated via
  `/tmp/opencode/watch_shots.py` + `gen_captions.py`; gate captions at
  `runs/campaign-2026-11-11/screenshots/phase4_closepacing/CAPTIONS.md`).
