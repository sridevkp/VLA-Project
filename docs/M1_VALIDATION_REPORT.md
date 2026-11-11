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
- Next: complete feasible bench measurements, then Phase 1 determinism/reset (post-reset settle + missing fingerprint fields) and fresh-world determinism checks.
- Human steps: none required to continue autonomous runs; Windows live validation and visual confirmation remain HUMAN (see `docs/MANUAL_VALIDATION.md`).
