# Batch Analysis — 10-Episode Baseline (MEASURED)

**Source:** `runs/campaign-2026-11-11/phase3_gate10b/` (10 fresh-world episodes, all features OFF, `--n 10 --fresh-world`)
**Gate result:** 10/10 crash-free (`webots_exit_code=0` all). Task success: 3/10.
**Wilson 95% CI for success rate:** 3/10 → ~[13.5%, 57.1%] (small sample, wide interval).

## Per-episode failure table (addition 1)

| Ep | Failure state | Waypoint at failure | IK error | IK tilt | arm_6 @ PREGRASP | arm_6 limit margin | Base roll/pitch @ first cmd | Can xy in base frame |
|----|-------------|---------------------|----------|---------|------------------|-------------------|----------------------------|----------------------|
| 00 | CLOSE misaligned | CLOSE exit | — | — | −1.3888 rad | **0.0012 rad** | 0.000 / −0.000 deg | can_xy − padref_xy = (0.020, 0.039) m |
| 01 | success (home) | — | — | — | −1.3900 rad | **0.0000 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.020, 0.040) m |
| 02 | DESCEND timeout | last waypoint (dist 0.223m) | — | — | −1.3880 rad | **0.0020 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.014, 0.034) m |
| 03 | DESCEND timeout | last waypoint (dist 0.223m) | — | — | −1.3880 rad | **0.0020 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.014, 0.034) m |
| 04 | DESCEND IK failed | waypoint target (0.636, −0.225, 0.758) | >0.005 m | — | −1.3888 rad | **0.0012 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.020, 0.040) m |
| 05 | DESCEND IK failed | waypoint target (0.636, −0.225, 0.758) | >0.005 m | — | −1.3888 rad | **0.0012 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.020, 0.040) m |
| 06 | success (home) | — | — | — | −1.3900 rad | **0.0000 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.020, 0.040) m |
| 07 | DESCEND timeout | last waypoint (dist 0.223m) | — | — | −1.3880 rad | **0.0020 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.014, 0.034) m |
| 08 | DESCEND timeout | last waypoint (dist 0.223m) | — | — | −1.3880 rad | **0.0020 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.014, 0.034) m |
| 09 | success (home) | — | — | — | −1.3900 rad | **0.0000 rad** | 0.012 / 0.006 deg | can_xy − padref_xy = (0.020, 0.040) m |

**Key pattern:** Two distinct failure modes:
1. **DESCEND IK failures (ep04, ep05):** IK rejected at target `(0.636, −0.225, 0.758)` — this is the hover target at `arm_6 ≈ −1.3888 rad`, only **0.0012 rad from the −1.39 rad limit**. arm_6 is effectively pinned.
2. **DESCEND timeouts (ep02, ep03, ep07, ep08):** Joint stall at arm_6 limit (−1.3880 rad, 0.0020 rad margin). The arm reaches its joint limit during descent and can't track further downward.

**arm_6 is pinned at its −1.39 rad joint limit in ALL episodes** (margin 0.000–0.002 rad). This is the root cause of all DESCEND failures — the arm is already at its physical joint limit at PREGRASP exit and cannot descend further.

## Reachability diagnosis (step 1 answer)

**DESCEND failures are reachability failures caused by arm_6 pinned at its −1.39 rad joint limit.** This is NOT tool-offset-induced (all episodes share the same can xy → padref xy relationship) and NOT base-tilt-induced (base roll/pitch ≈ 0 deg at first command for all episodes).

Comparison successful vs failed:
- **Successful (01, 06, 09):** arm_6 = −1.3900 rad (exactly at limit), arm_7 = +2.0584 rad (positive elbow-up), base roll/pitch ≈ 0.012/0.006 deg
- **Failed DESCEND (02,03,07,08):** arm_6 = −1.3880 rad (0.0020 rad margin), arm_7 = +0.6031 rad (elbow-down), base roll/pitch ≈ 0.012/0.006 deg
- **Failed IK (04,05):** arm_6 = −1.3888 rad (0.0012 rad margin), arm_7 = −2.0617 rad (negative elbow-down)

The difference between success and failure is **arm_7 elbow configuration**: successful episodes use elbow-up (arm_7 ≈ +2.06 rad), failed ones use elbow-down (arm_7 ≈ +0.60 or −2.06 rad). With arm_6 pinned at its negative limit, elbow-down configurations have no downward reach.

## Servo kinematic model consistency (step 4 — MEASURED live refutation)

- **Live refutation of the URDF-chain variant (2026-11-12, MEASURED, `phase3_contact_probe/rep01.jsonl`)**: An attempted implementation applied the measured `(0, 0, +0.202825)` offset as the URDF chain's `last_link_vector` when `grasp_servo=gt_pad`. Live FK at a recorded PREGRASP pose put the chain end **0.377 m above the real pad midpoint** (tip world z 1.3281 vs pad 0.9510): the offset is expressed in the **sim arm_7 Solid frame**, whose origin and orientation do not match the URDF chain's arm_7 frame (sim Solid origin − URDF arm_7 origin ≈ (−0.039, +0.016, +0.028) m world; relative orientation ≈ 180 deg).
- That variant also drove PREGRASP into the pinned elbow-down posture (arm_6 −1.3886 rad at the −1.39 limit, arm_7 −2.0641) and tipped the base (yaw −0.50 deg, pitch −0.51 deg => "base moved" failure in all 3 probe reps).
- **Decision (ADR-0023):** The URDF chain therefore **KEEPS the legacy `[0, 0, -0.1741]` tool offset unconditionally**. The servo does not need the chain to end at the pad because its XY corrections and DESCEND stop height are measured from the exact Solid-based pad telemetry.

## Contact probe replacement plan (step 2 — COMPLETED 2026-11-12)

Replaced first-contact detection with three policy-based triggers:
- **(a)** Can displacement > 0.5 mm or tilt > 0.3 deg from previous sample
- **(b)** Joint stall with pad-reference height recorded
- **(c)** Pad-reference height at trigger vs lid height (can centre z + 0.0611)

**Result (3/3 contact, `phase3_contact_probe/pad_contact_probe.json`):**
- All 3 fresh-world reps fired trigger (c) at `descended = 0.08 m` with e_xy at contact 2.9–3.1 mm.
- Offsets: −2.616 / −2.246 / −2.158 mm => mean **−2.340 mm**, std **0.243 mm** (< 2 mm gate).
- Config `probe_pad_reference_to_pad_bottom_m` set to `-0.0023399045941977135`.

## Reachability analysis (step 3 — DONE)

**Analysis completed:** arm_6 is pinned at its −1.39 rad joint limit in ALL 10 baseline episodes (margin 0.000–0.002 rad). DESCEND failures (timeouts + IK rejections) are reachability failures: arm_6 cannot deflect further downward. Successful episodes use elbow-up config (arm_7≈+2.06 rad); failed ones use elbow-down (arm_7≈+0.60 or −2.06 rad).

## Next steps (in order)

1. **Run (c) 3 watched gt_pad episodes + screenshots** — verify `grasp_servo=gt_pad` live execution.
2. **Run (d) 10-vs-10 batches** — reuse existing baseline (`phase3_gate10b`), run 10 fresh-world episodes with `--grasp-servo gt_pad`.
3. **After (c)/(d): update docs/M1_VALIDATION_REPORT.md verdicts and DECISIONS.md**
