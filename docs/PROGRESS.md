# Project progress

Current milestone: **M1 — Port controller behind interfaces**

Last updated: **2026-11-08**. M0 is committed. M1-01 through M1-05 are implemented. The approved
world-frame placement target and corrected safety profiles are in place. The latest seed-0
Webots-free IK audit reports a place-above endpoint error of `0.010475481 m` and tilt of
`0.256656122°`; the transit-only 0.015 m tolerance remains configured, with the 0.005 m grasp
tolerance and existing tilt limit unchanged. M1-06a and M1-07 are pending live runs. Reset replay
passed with the revised `1e-3` pose/joint/CAN tolerance; RGB is reported diagnostically, not as a
gate.

## M0 — Foundations (P0)

### M0-01: Establish repository and tooling

- [x] Add the `src/vlarobot/` package layout and lockfile.
- [x] Declare Python 3.11 and the `core`, `sim`, `perception`, `vla`, `dashboard`, and `dev` extras.
- [x] Configure Ruff, pytest (`webots` and `gpu` markers), pre-commit, `.gitattributes`,
  `.editorconfig`, and `.gitignore` (`.venv/`, `runs/`).
- [x] Add `configs/` and `scripts/` landing documentation.
- [x] Add Ubuntu/Windows GitHub Actions matrix for locked dependency sync, lint, type-check, and
  non-Webots tests.
- [x] Leave controller migration in M1 as decided; no controller logic or files were changed.
- **Local evidence:** M0 checks passed on Fedora with Python 3.11.18; M0 commits:
  `ae3e874 Build M0 project foundation` and `1b2e4da Add project architecture and agent guidance`.
- **Windows evidence:** CI is configured but has not yet run.

### M0-02: Document setup and version decisions

- [x] Document Windows native, Linux native, and WSL2 Webots setup.
- [x] Document CUDA PyTorch installation steps for Windows and Linux with the official wheel
  selector.
- [x] Record Webots/Python targets and unpinned torch, LeRobot, and MuJoCo statuses.
- [x] Record the §9 items that still need model, GPU, MuJoCo, collision-metric, or Windows
  verification.
- [x] Add ADR-formatted decisions, including the dependency license audit.

### M0-03: Implement the Webots smoke command and environment helper

- [x] Configure `WEBOTS_HOME` and prepend Webots' controller Python API to `PYTHONPATH` and
  `sys.path` using the host OS path separator.
- [x] Connect to the external controller, enable `Astra rgb`, step ten times, validate one RGBA
  image, and print simulation time and image shape.
- [x] Unit-test environment setup and fake-robot smoke behavior.
- [x] Verify on Fedora/Linux with Webots R2025a. Output:
  `sim_time=0.320s image_shape=(480, 640, 4)`.
- **Windows acceptance:** Human verification remains pending.

### M0-04: Verify locally-checkable open items

- [x] Audit the locked runtime and development dependency licences; no AGPL, GPL, or
  non-commercial dependency is selected. `ty` is MIT-licensed; see ADR-0001.
- [x] Verify Linux package synchronization, lint, type-check, and non-Webots tests.
- [ ] Verify package installation and checks in the Windows CI job.
- [x] List unverified items in `docs/VERSIONS.md`.

### M0 acceptance gate

- [x] Linux package/tool checks and Webots smoke command.
- [ ] Windows CI green.
- [ ] Webots smoke command verified by a human on Windows.
- **Gate status:** Windows checks remain pending at the user's request; the user explicitly
  approved proceeding to M1 before Windows validation.

## M1 — Port controller behind interfaces (P0)

- [x] Move controller implementation into `src/vlarobot/backends/webots/`; retain the original
  scripted controller as a comparison baseline.
- [x] Keep `controllers/tiago_controller/tiago_controller.py` as a thin shim. It selects the
  project `.venv` Python (or `WEBOTS_PYTHON`) and starts the server by default; set
  `VLAROBOT_WEBOTS_MODE=legacy` to run the original script.
- [x] Implement the core message types, protocols, and versioned wire serialization from §3.3–3.4.
- [x] Add Webots `SimServer`, lockstep `ZmqBackendClient`, and `cli sim-server` / `cli brain`.
- [x] Add deterministic `FakeBackend` and simulator-independent unit tests.
- [x] Move controller defaults to typed YAML-backed settings with compatibility aliases.
- [x] Implement safety validation for action types, finite values, workspace, joint position and
  velocity, gripper limits, and malformed commands. Fake and Webots backends validate before apply.
- [x] Derive `configs/safety.yaml` workspace profiles from corrected nominal PickPlace waypoints
  with a 0.10 m `original_envelope` margin and a tighter 0.02 m `strict` profile. The original
  controller had no workspace limit; the envelope is a behavioral baseline, not a validated safety
  limit.
- [x] Log validation rejection reasons and test waypoint acceptance under `original_envelope`,
  strict-profile containment, and out-of-box rejection under both profiles.
- [x] Replace the active `place_base` setting with the approved `place_world` CAN-center target
  `[-1.0, -4.3, 0.821]` m and convert it through `Frames.world_to_base()` in the Webots controller
  and through the full `base_pose_world` transform in the brain regression.
- [x] Reject the historical base-frame target `[-1.07, -3.9, 0.8]` under both safety profiles;
  keep joint, velocity, IK-tolerance, gripper, and invalid-command checks enabled.
- [x] Add optional controller and brain JSONL diagnostics; tracing is disabled unless requested.
- [x] Bound each sequential linear IK step to `max_joint_step_rad: 0.15`; interpolate PREGRASP and
  HOME joint commands by the same step. Each commanded point now waits for measured joint settling
  using original tolerances/timeouts; linear states use 3 cm Cartesian waypoints with the trust
  region, and a configurable no-progress watchdog fails with `joint stall`.
- [x] Add Webots-free full-flow continuity, swept-transit, and placement-grid analysis at
  `scripts/audit_can_path.py`; no target or safety parameter was changed.
- [x] Add original-worktree comparison fixture and `compare-traces` CLI for M1-06a; live comparison
  remains pending.
- [ ] **M1-06a refactor equivalence (UNVERIFIED):** compare the original stop-after-LIFT trace's
  exact `INIT → PREGRASP → DESCEND → CLOSE → LIFT` sequence to the matching brain prefix. Require
  commanded/achieved EE boundary positions and named DESCEND/LIFT Cartesian targets within `1e-3 m`, and all
  prefix IK tilts within the existing limit. Report per-run joint-step and limit-margin diagnostics;
  do not gate on joint equality (withdrawn in ADR-0009).
- [ ] **M1-07 scripted-expert hardening (UNVERIFIED; required before M2):** complete
  `TRANSIT → PLACE_DESCEND → RELEASE → RETREAT → HOME` and reach `DONE`; at least 16 of 20 live
  episodes must place the CAN within `0.10 m` of the target in the robot base frame. Preserve and
  inspect state, commanded/achieved pose, IK, joint, gripper/stall/contact, and CAN world/base traces.
- **Prior actual result (Fedora/Webots R2025a, before frame correction):**
  `uv run python -m vlarobot.cli can-regression --seeds 10` produced `success=0/10` and
  `rejections=9`; the old seed-0 distance of 0.560 m mixed world- and base-frame values and is
  invalid as a baseline-compatible score.
- **Local checks:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src
  tests`, `uv run pytest -m "not webots"`, and `uv build` passed; pytest reported **36 passed**.
- **Post-metric-correction checks:** Ruff check and format check passed; `uv run ty check src tests`
  passed; `uv run pytest -m "not webots"` reported **37 passed**. The subsequent corrected-target
  implementation is validated separately below.

- **Webots integration:** Fedora / Webots R2025a factory world, brain process:
  `uv run python -m vlarobot.cli brain --steps 10`

  ```text
  sim_time=0.416s image_shape=(480, 640, 3)
  ```

- **Webots RPC evidence:** ping, reset, apply, step, get_gt, and shutdown completed successfully;
  the returned ground-truth object key was `CAN`. This validates transport/backend operation, not
  the M1-06 pick-and-place regression.

- **Historical M1-06 waypoint evidence:** an instrumented run of the original controller in
  `worlds/factory.wbt` collected home `(0.203672298, -0.985455309, 0.963048983)`, descend/grasp
  `(0.672736661, -0.213295065, 0.764821419)`, pregrasp/lift
  `(0.672736661, -0.213295065, 0.884821419)`, transit/retreat
  `(-1.077327982, -3.842045869, 0.936284755)` / `(-1.086885823, -3.858086446, 0.935119120)`,
  and place-descend/release `(-1.077327982, -3.842045869, 0.816284755)` /
  `(-1.086885823, -3.858086446, 0.815119120)`, all base-frame meters. These extrema include two
  runs; the latest original run terminated `FAILED`. See ADR-0002 for profile bounds and rationale.
- **Historical target diagnosis:** With torso fixed at 0.35 m, a 65,536-sample IKPy reach
  sample occupied 19,092 5 cm voxels, with sampled bounds `[-0.45, -1.00, 0.05]` to
  `[1.15, 1.00, 1.80]` m. The coded `PLACE_BASE=[-1.07,-3.9,0.8]` has 3.206796 m IK position
  error; interpreting those coordinates as world and converting with the original frame transform
  gives base `[0.202930,-0.508161,0.726285]` and approximately zero IK error. CAN and
  pregrasp/descend/lift checks also had approximately zero IK error. The coded CAN-to-target probe
  first exceeded the 0.005 m IK threshold at waypoint 15/135 (0.452730 m from CAN; error
  0.012630 m). Sampled bounds are approximate, not a proof of reachability.
- **M1-06 original baseline:** Three instrumented original-controller runs in the current factory
  geometry, each limited to 120 seconds, all terminated `FAILED` in 94.51, 97.37, and 110.70
  seconds. Runs 1 and 2 failed on a retry `PREGRASP` IK rejection after the CAN was displaced;
  run 3 failed in `TRANSIT` at waypoint 10/134 (IK error 0.005562 m > 0.005 m). Thus the original
  controller has no demonstrated success in these trials; M1-06 cannot be judged against a
  successful baseline. The original implementation sets its terminal state after HOME without
  verifying final placement.
- **M1-06 scoring correction:** Brain-side final distance converts the final CAN world position
  into the robot base frame using the full 3D base pose, matching `Frames.world_to_base()` and the
  baseline formula. `RobotState` carries optional `base_pose_world`, requiring wire protocol
  version 2. The corrected target has not yet been run through live Webots episodes.
- **Approved target and frame evidence:** At the factory pose, `PLACE_WORLD` converts to base
  `(0.600512925, -0.425544073, 0.747345359)` m; the initial CAN center converts to
  `(0.684471701, -0.223124176, 0.732413946)` m. With torso at 0.35 m, IKPy position errors are
  below 0.000001 m for pregrasp, grasp/descend, lift, and place-descend. The place-above transit
  target has 0.010475 m error and 0.257° tilt. Before ADR-0006, the 0.005 m grasp tolerance rejected
  transit at waypoint 7/7. ADR-0006 records the measured, state-specific transit tolerance decision;
  the target, grasp tolerance, safety bounds, and tilt cap remain unchanged.
- **Historical placement geometry check (clearance results superseded; UNVERIFIED (assumed radii)):** `Table.proto` defines a 0.75 m top; the CAN bottom at the approved
  center height is 0.7599 m. The target is 0.209707 m from the nearest tabletop edge, leaving
  0.177957 m after subtracting the CAN radius. The initial CAN-to-target center distance is
  0.219419 m horizontally (0.219649 m in 3D); parallel-cylinder surface gap is 0.155919 m. Based
  on Webots R2025a collision shapes, conservative horizontal-projection surface-clearance lower
  bounds are 0.2826 m to the wrench, 0.3290 m to the hammer, 0.5246 m to the apple, 0.7974 m to
  the paint bucket, and 1.1517 m to the screwdriver. Darwin-op is also nearby; its articulated
  geometry is not included in those exact prop-shape calculations. These are geometric checks, not
  a live contact or placement-stability test.
- **Safety profiles:** `original_envelope` now bounds the corrected nominal commanded points plus
  0.10 m per axis: min `(0.103672, -1.085455, 0.652414)` m, max
  `(0.784472, -0.123124, 1.092414)` m. `strict` uses the same points plus 0.02 m. The measured
  held offset remains constrained by the existing `<0.10 m` lift check; its per-axis components fit
  inside the 0.10 m margin. Neither profile is a validated safety limit.
- **M1-07 trace implementation:** controller mode appends state, sim time, base and tool pose,
  CAN world pose, joint/head positions, commanded target, IK error/tilt/waypoint, gripper opening,
  stall evidence, and finger/CAN contact to `VLAROBOT_WAYPOINT_LOG`. `can-regression --trace-log`
  records corresponding brain-side state and waypoint data, including CAN world/base pose and
  `RobotState.gripper_holding`.
- **Reset restoration:** `WebotsBackend.reset()` reacquires the CAN by DEF, restores arm/torso/head
  and finger joint position fields plus the robot base and CAN transforms, resets physics, clears
  gripper close-stall telemetry, and reports episode-relative observation time. An initial torso
  field read was `-0.0001722422` against a legal minimum of `0.0`; reset baselines now project
  out-of-range startup readings to the nearest legal motor limit and log a warning. Simulator-free
  reset tests cover field restoration and this boundary case.
- **Reset equivalence threshold:** Pose, arm-joint, and CAN comparisons use `1e-3 m` and `1e-3 rad`.
  The prior `1e-6` requirement was unnecessarily strict relative to measured Webots render/physics
  repeatability; RGB remains a diagnostic only.
- **Live reset behavior (Fedora, Webots R2025a):** Three consecutive episodes used seed 0. Episode
  0 planned INIT/PREGRASP once; episodes 1 and 2 replayed the identical 18 joint-target commands.
  Both comparisons passed at `1e-3`: state sequence `INIT, PREGRASP`; max reset joint delta
  `4.72e-10 rad`; base pose delta `3.12e-8 m` / `2.96e-7 rad`; CAN pose delta `0 m` /
  `0 rad`; max commanded-joint delta `0`; max achieved waypoint position delta `2.22e-6 m`;
  max achieved waypoint orientation delta `0`; gripper-opening delta `1.80e-8 m`.
- **RGB reset diagnosis:** Camera `noise=0.0`; each reset observation is captured after one
  simulation step (episode-relative `t=0`). In the identical-command run, reset-image mean
  absolute differences were `0.00171` and `0.00206` with 8x8-window SSIM `0.999992` and
  `0.999971`. After warm-up steps 1/3/5, MAD remained below `0.0061` and SSIM above `0.99993`;
  differences did not strictly vanish but remained sparse render variation (peak pixel delta 54–92).
  The earlier 112-level maximum came from a different-seed run with divergent IK solutions, not
  camera noise or pre-render capture.
- **Reset timing and reload option:** Soft reset took `0.704`, `0.666`, and `0.838 s` (mean
  `0.736 s`) across the three episodes. `reset_mode` defaults to `soft`; `reload` is an explicit
  but unsupported option because it restarts the controller/server and requires ZMQ reconnect and
  retry behavior. Reload time is therefore not measured.
- **M1 gate redefinition:** Applied in `docs/ARCHITECTURE.md`: M1-06a compares the completed
  original-controller prefix over five runs per implementation; M1-07 requires at least 16/20
  complete-placement episodes and is a prerequisite to M2. Both gates remain pending live evidence.
- **Corrected-target live regression:** `timeout 300s uv run python -m vlarobot.cli can-regression
  --seeds 10 --trace-log runs/brain-can-reset-validation.jsonl` exceeded its 300-second timeout after
  terminal results for seeds 0–7; seed 8 started but did not finish, and seed 9 did not run.
  Seed 0 failed at TRANSIT with IK error `0.010245 m` against the unchanged `0.005 m` limit and
  zero safety rejections. Seeds 1–7 failed with an arm-7 velocity-limit rejection (one per seed):
  TRANSIT for 1, 3, and 5; DESCEND for 2 and 4; LIFT for 6 and 7. Each terminal seed reported
  `final_distance=0.220 m`; aggregate for those eight completed episodes was `success=0/8`,
  `rejections=7`. A subsequent two-seed run again failed seed 0 at TRANSIT (`0.010083 m` IK error)
  and seed 1 at DESCEND due to the arm-7 velocity limit. No complete 10-seed comparison has passed.
- **Transit IK profile (Fedora, Webots-free IKPy):** Lift endpoint error was `0.000000030 m`.
  Seven straight-line transit waypoints had errors `0.000000158`, `0.000000227`, `0.000000005`,
  `0`, `0`, `0.002703109`, and `0.010473834 m`; the final point is the first rejection at the
  unchanged grasp tolerance `0.005 m`. Its tilt was `0.256616°`, and the nearest arm-joint-limit
  margin was zero (arm 6 at its `-1.39 rad` limit). The point-level endpoint result is acceptable
  under the existing `0.015 m` general IK tolerance.
- **Transit remedy order:** Existing linear IK already warm-starts each waypoint from the previous
  solution; deterministic restarts now occur only when that warm start fails, and the nearest
  feasible candidate is selected. This still found no solution within `0.005 m` at the endpoint.
  An elevated via-route was tested without applying it: its vertical/horizontal segments reached
  errors up to `0.091506345 m`, so that path did not satisfy the unchanged tolerance either. The
  resulting state-specific `ik_tol_transit: 0.015` is used only for TRANSIT and RETREAT (HOME is
  joint-space); all remain subject to the existing `15°` tilt cap. PLACE_DESCEND and grasp-related
  moves retain `IK_TOL_GRASP=0.005 m`. Target, safety bounds, and the final `0.10 m` placement
  criterion are unchanged.
- **Latest reconciled transit measurement:** `uv run python scripts/audit_can_path.py` with seed 0,
  torso `0.35 m`, and the legal initial arm pose reports endpoint error `0.010475481 m` and tilt
  `0.256656122°`. This deterministic run supersedes earlier values `0.010473834/0.256616` and
  `0.010475343/0.256653`, which came from prior solver/RNG executions.
- **Full-flow continuity profile:** The same audit prints all 48 waypoint/state records from INIT
  through HOME and per-joint `|dq|`. No joint delta exceeds `0.15 rad`. Sixteen records have a
  joint-limit margin below `0.05 rad`; seven are the TRANSIT waypoints, where arm 6 is at its
  `-1.39 rad` lower bound from waypoint 5 onward. The current target therefore does not have the
  requested `0.05 rad` margin.
- **Branch-flip reproduction:** Without a trust region, arm 7 changed by `3.15896 rad` at DESCEND
  waypoint 2 and `3.90845 rad` at TRANSIT waypoint 5 in the seed-0 diagnostic. The bounded path
  kept all measured adjacent commands at or below `0.15 rad`. A reachable pregrasp target requiring
  an unrestricted `0.468583 rad` change from a deliberately displaced seed now fails explicitly
  inside the `0.15 rad` trust region (best position error `0.144119 m`).
- **Historical swept transit comparison (object-clearance values superseded; UNVERIFIED (assumed radii)):** Cartesian line: max joint step `0.150000 rad`, minimum joint-limit
  margin approximately zero, peak tilt `0.263349°`, estimated minimum payload/table clearance
  `0.120158 m`, estimated minimum payload/object clearance `0.116168 m` (hammer). Joint-space
  interpolation: max step `0.137162 rad`, margin approximately zero, peak tilt `2.192890°`, table
  clearance `0.112915 m`, object clearance `0.118443 m` (hammer). These are FK checks of the tool
  and held-CAN proxy, not arm-link collision or dynamic Webots evidence. Cartesian transit is
  recommended for its roughly `7.2 mm` greater tabletop clearance and `1.93°` lower peak tilt;
  neither route removes the arm-6 limit-margin issue.
- **Historical placement grid search (geometry screen superseded; UNVERIFIED (assumed radii)):** At 0.10 m grid spacing, 534 tabletop points passed the documented
  geometric filters and 16 passed sampled IK descent. The best route margin excluding the fixed
  LIFT start state was `0.043619 rad` at world `(-0.188828,-4.190293,0.821100)` m / base
  `(0.466393,0.381651,0.725361)` m. The next two margins were `0.031087` and `0.028296 rad`;
  see ADR-0007 for their coordinates. No placement candidate was applied.
- **M1-06a original-only noise-floor audit (2026-11-08):** Ran all six old-rule `compare-traces`
  pairings over clean original traces 01, 02, 03, and 05. Five pairs first failed the joint check
  at PREGRASP; pair 02-05 passed. Maximum pairwise joint deltas by state were `0.319572 rad`
  (PREGRASP), `4.078076 rad` (DESCEND), `4.095043 rad` (CLOSE), and `4.139445 rad` (LIFT), all
  largest on arm_7. Boundary EE targets and both Cartesian waypoints differed by at most
  `8.1e-15 m`; achieved boundary EE differences were at most `6.78e-6 m`. Fixture v1 seeds its IK restart RNG to 0 before the first PREGRASP solve. The
  original trace does not record torso state, so the physical cause of the solver branch variation
  is not known. See ADR-0009.
- **M1-06a revised gate and fixture v2:** Joint equality is withdrawn. The gate checks the expected
  state prefix, commanded/achieved EE boundary positions and DESCEND/LIFT Cartesian targets within `1e-3 m`,
  and the existing tilt limit. It reports per-run max joint step per joint, minimum joint-limit
  margin, and arm_7 steps over `1 rad` as diagnostics. Fixture v2 adds opt-in CAN, gripper, grasp,
  stall, IK, tilt, and joint-limit measurements. Existing run 04 remains untouched and labelled
  **anomalous: CLOSE->RETRY, CAN displaced**; v1 traces are not reused for the revised five-run
  batch. No v2 live runs or brain M1-06a traces have been collected.
- **M1 gate status:** M1-06a and M1-07 remain **UNVERIFIED** pending their required live evidence.
  M2 has not started. Windows CI remains unverified.

### 2026-11-08 pacing repair

- Implemented original per-command settling semantics in the brain: PREGRASP/HOME joint waypoints
  use 0.01/0.02 rad tolerances and 800-step timeouts; DESCEND/LIFT/TRANSIT/PLACE_DESCEND/RETREAT
  use 3 cm Cartesian points with 0.03 rad intermediate and 0.01 rad final tolerances, each with a
  600-step timeout. A 10-step configurable no-progress watchdog fails the active state as `joint
  stall`.
- Added explicit measured and commanded-target trace fields; the legacy
  `commanded_joint_positions_rad` is retained as a deprecated measured-feedback alias. Fixture v2
  and diagnostics use the explicit names.
- Webots-free tests cover the previously rejected unpaced rate-limited path, the paced
  PREGRASP→DESCEND→CLOSE→LIFT prefix, stall handling, and DESCEND seeding from the settled
  PREGRASP command. Live Webots timing and M1 gate evidence remain **UNVERIFIED**; collect fresh
  traces before making equivalence claims.
## M2 — Evaluation and data harness (P0)

- [ ] Define `ScenarioSpec` and a seeded scenario generator for objects and obstacles.
- [ ] Add Webots Supervisor scenario spawning/movement.
- [ ] Implement `EpisodeRecorder` and a LeRobot-convertible episode format.
- [ ] Implement success, grasp, collision, timing, placement-error, and report metrics.
- [ ] Verify `eval --pipeline scripted --n 20 --seed 0` is reproducible.

## M3 — Perception (P0)

- [ ] Implement RGB-D detection, segmentation, 3D localization, tracking, and color fallback.
- [ ] Calibrate camera extrinsics and depth scale against Supervisor ground truth.
- [ ] Verify median localization error ≤ 1.5 cm over 50 randomized scenes.
- [ ] Verify real-perception scripted success is at least 80% of oracle rate.

## M4 — Language and task specification (P0)

- [ ] Implement schema-validated `TaskSpec` parsing with LLM and rule-based demo fallback.
- [ ] Return `Clarification` for ambiguous target descriptions.
- [ ] Verify at least 95% correctness on the 30-sentence set and clarification for the two-bottle
  case.

## M5 — World state, scene graph, and surrogate simulator (P0)

- [ ] Implement `WorldState` and scene-graph relations.
- [ ] Import TIAGo and objects into MuJoCo; implement rollout, collision, reachability, and stability
  checks.
- [ ] Log prediction error against Webots outcomes.
- [ ] Verify at least 90% surrogate/Webots agreement over 200 candidate actions.
- [ ] Verify scene-graph unit tests and 32-candidate rollouts under one second on laptop CPU.

## M6 — Candidates, risk engine, and predictive planner (P0)

- [ ] Implement scripted grasp, pregrasp, base standoff, and placement candidates.
- [ ] Implement hard-constraint veto, calibrated utility, risk report, and numeric planner rationale.
- [ ] Implement execute-first-segment, re-observe, and replan behavior.
- [ ] Verify risk-engine collision reduction on at least 50 paired scenarios, with task success no
  worse by over 10 percentage points.

## M7 — Closed-loop verification and recovery (P0)

- [ ] Implement grasp and placement verification, failure taxonomy, and recovery ladder.
- [ ] Inject nudged-object and grasp-slip failures.
- [ ] Verify recovery success ≥ 70% and no unsafe action passes the safety layer.

## M8 — VLA integration (P1)

- [ ] Inspect and document `lerobot/smolvla_base` inputs, outputs, dimensions, and camera keys.
- [ ] Collect scripted-expert demonstrations in the canonical end-effector-delta action space.
- [ ] Fine-tune SmolVLA and serve the checkpoint.
- [ ] Implement action normalization, IK/safety adaptation, chunking, and invalid-action rejection.
- [ ] Integrate VLA-only baseline and risk-aware candidate-proposer mode.
- [ ] Verify the baseline target after initial evaluation, reject all injected out-of-limit actions,
  and run the proposer end-to-end.

## M9 — Learned residual dynamics, memory, and dynamic environments (P1)

- [ ] Train and calibrate a learned residual ensemble on Webots logs.
- [ ] Implement scene and episodic memory for candidate generation.
- [ ] Add moving-obstacle, nudged-object, occlusion, and sensor-dropout scenarios.
- [ ] Compare held-out prediction error with/without residual and report memory ablation honestly.

## M10 — Dashboard, ablations, and report (P0/P1)

- [ ] Implement camera/mask, scene-graph, candidate, risk, timeline, and metrics dashboard.
- [ ] Run six ablations with at least 30 seeds per condition and report confidence intervals and
  failure categories.
- [ ] Verify `eval --suite ablation` regenerates tables and figures from raw logs.

## P2 backlog

Not started and out of scope until M10: ROS 2 adapter, MoveIt 2/Nav2, IKFast, latent/video world
models, RL fine-tuning, sim-to-real preparation, multi-robot support, CBF/chance-constrained
planning, and active perception.

## 2026-11-08 analysis addendum

- **M1-06a semantics:** ADR-0008's joint-equality criterion is superseded by ADR-0009. The gate
  compares the state prefix, commanded/achieved EE boundary positions and named DESCEND/LIFT Cartesian targets
  within `1e-3 m`, and the existing tilt limit; joint values remain per-run diagnostics. The
  original-only noise floor is `4.139445 rad` on arm_7 at LIFT despite `8.1e-15 m` commanded and
  `6.78e-6 m` achieved EE-position variation. Live v2 comparisons remain pending.
- **Geometry:** The Webots-free audit now uses radii derived from the R2025a collision geometry
  where the PROTO was available. PaintBucket.proto could not be retrieved, so its exclusion radius
  is conservatively assumed as 0.20 m. Clearance and placement geometry screens are marked
  `UNVERIFIED (assumed radii)`. The refreshed baseline grid has 505 geometrically
  eligible points; no candidate was applied.
- **Posture study (analysis only, 100 deterministic pregrasp restarts):** Current behavior at torso
  0.35 m: full-flow minimum margin `0 rad`, maximum joint step `0.150000 rad`, maximum IK error
  `0.010727 m`, maximum tilt `0.263349°`. Selected pregrasp at torso 0.35 m: `0 rad`, `0.150000 rad`,
  `0.010474 m`, and `0.256619°`. Torso sweep 0.00–0.35 m: only 0.35 m completed the full flow;
  its selected-posture result was `0 rad`, `0.150000 rad`, `0.010506 m`, and `0.257334°`.
- **Margin target limitation:** The fixed initial/home arm posture places `arm_1_joint` at its lower
  bound (`0.070 rad`; margin `0 rad`). Torso height and world-frame base offsets do not change that
  joint value or limit. Therefore a full-flow minimum margin of `0.05 rad` cannot be achieved by
  the requested PREGRASP/torso/base sweeps alone. The full 1,960-setting base/torso sweep and the
  dependent candidate search have not been run; no base/torso/posture change or place candidate was
  applied. M1-06a and M1-07 remain unverified.
- **Candidate-search sequencing:** The existing combined audit emitted one baseline-setting search
  before the posture sweep. Its results are not accepted as the requested best-setting search. The
  default audit no longer runs candidates; `posture_study.py` defers that search until a best base
  setting is established. No candidate was applied.

## 2026-11-08 PREGRASP diagnostics addendum

- Fixed failed state-exit tracing to use the latest stepped observation and added a synthetic
  regression test. The untouched old `m1-watched-paced-02.jsonl` confirms its PREGRASP exit at row
  16 incorrectly used simulation time 0.0 and reset joints/base; its final waypoint was at 26.08 s.
- Added protocol-v3 read-only backend diagnostics for torso sensor, base drift, wheel state,
  contact point/node identity, and arm_7 FK/Supervisor position. Added 5 mm / 0.5 degree base
  invariant and default-preserving `pregrasp_route` flag. The new physical telemetry has not been
  verified in Webots; M1 gates remain **UNVERIFIED**.
- The saved run 02 does not show the cited 13 cm / about 42 degree base drift by 1.4 s. At 1.376 s
  its planar drift is about 10.4 mm; the larger 12.9 cm / about 42 degree change occurs by 26.08 s.
  This confirms base motion but not the user's inferred onset time or contact cause.
- Webots-free FK sweep against the verified table box predicts both motion semantics clear the
  configured clearance envelope at torso 0.35 m; with torso 0.0 m, the 0.08 m hand envelope
  intersects near the table edge in both paths. `safe_via` metrics are kinematic estimates only. No
  live route, posture, torso, base, IK, safety, or placement candidate experiment has been run.

## 2026-11-08 handoff commit and diagnostic corrections

- Recorded the M1 handoff in three commits on `m1-checkpoint-ik-transit`: `a2a2d68` (read-only
  diagnostics, wire protocol v3), `227b7a5` (pacing and guarded PREGRASP route), and `5df8cbc`
  (comparison tooling, scripts and docs). `runs/` was not modified.
- Fixed the arm_7 FK-versus-ground-truth diagnostic. It now compares the IKPy `arm_7_joint` link
  origin with the Supervisor world position of the same HingeJoint anchor through
  `RobotIO.joint_anchor_world_position`, and the dead `jointParameters.position` + `getSFVec3f()`
  branch is removed. A configurable 0.01 m tolerance sets
  `arm_7_joint_fk_ground_truth_mismatch`. Webots-free tests cover a 0.05 m offset (flagged) and
  identical inputs (zero error); unavailability is `None`, not zero. Live alignment is still
  unverified.
- Unified table geometry and clearance in `configs/robot.yaml`: `table_top_height_m` 0.75,
  `table_footprint_centre_xy_m` `[-0.23882837, -4.8402928]`, `table_footprint_size_xy_m`
  `[4.5, 1.5]`, and `table_clearance_required_m` 0.05. The runtime guard and
  `scripts/analyze_pregrasp_table.py` read the same values; the hardcoded `0.80 m` plane and the
  analysis script's separate `0.03 m` threshold are gone. Re-running the analysis on
  `m1-watched-paced-02.jsonl` at torso 0.35 m gives 0.041478 m (synchronized) and 0.054233 m
  (independent-rate) above the 0.05 m requirement; torso 0.0 m intrudes at sample 3
  (-0.246566 m) and sample 10 (-0.171349 m). These are 0.02 m below the previously reported values
  because the threshold changed, not because the geometry or paths changed.
- Corrected ADR-0010 (PREGRASP/HOME settle at 0.01/0.02 rad; protocol v3 exposes torso readback)
  and deduplicated the identical `enable_can_contact_tracking`/`enable_contact_tracking` methods.
- Trace additions for the watched runs: per boundary and waypoint,
  `supervisor_arm_link_world_xyz_m` (elbow `arm_4_joint`, wrist `arm_6_joint`, tool
  `arm_7_joint` endPoint), `supervisor_min_table_clearance_m`, the FK mismatch flag, and the
  configured table thresholds. Every field is `null` when the Supervisor value is unavailable.
- Webots-free checks: Ruff check and format pass, `ty check src` passes, and
  `pytest -m "not webots"` reports **80 passed** (73 baseline plus 3 joint-anchor tests, 3 FK
  mismatch tests and 1 Supervisor-link trace test). No live Webots behaviour has been validated;
  M1-06a and M1-07 remain **UNVERIFIED** and no posture, torso, base, or placement experiment has
  been run.

## 2026-11-08 PREGRASP determinism and telemetry pass

This pass is Webots-free instrumentation and test work only. No live Webots run was performed and
no `runs/` file was modified.

### Invalid Supervisor calls and arm_7 ground truth

- `RobotIO.joint_anchor_world_position` was calling `getPosition()` on `HingeJoint`/`SliderJoint`
  nodes. Webots only accepts `getPosition()` on Pose-derived nodes, so every call emitted
  `wb_supervisor_node_get_position() can exclusively be used with Pose`. The method is removed.
- `RobotIO.joint_endpoint_world_position` now reads the Pose-derived child `Solid` from the joint's
  `endPoint` field. It catches `AttributeError`/`TypeError`/`ValueError`/`RuntimeError`, returns
  `None` when the node, field, or value is unavailable, and caches each distinct failure so the
  warning is logged once (`_warn_lookup_once`, `_lookup_failed`).
- `WebotsBackend` records the world offset `endPoint - FK(arm_7)` at reset
  (`_arm7_endpoint_offset`) and reports the offset-change residual at each sample
  (`arm_7_joint_endpoint_offset_change_world_xyz_m`,
  `arm_7_joint_fk_ground_truth_error_m`, `arm_7_joint_endpoint_fk_validation_available`). A
  constant tool offset therefore cancels; only a change in that offset is an FK mismatch.
  Unavailable points are `None`, never zero.
- New tests use a fake supervisor whose `HingeJoint.getPosition()` raises like the real
  Supervisor; they assert the endpoint path never calls it and that failures warn once.

### Soft-reset inventory and reset fingerprint

Restored by `WebotsBackend.reset` (`src/vlarobot/backends/webots/backend.py`):

- all named motor targets from `_initial_joint_pos` (arm joints);
- torso to `torso_height` (0.35 m) and head pan/tilt (clamped), then the original-order settle;
- finger motor positions from `_initial_finger_positions`;
- CAN node `translation`/`rotation`;
- robot base `translation`/`rotation`;
- `simulationResetPhysics()` followed by one step; the CAN node is re-fetched by DEF;
- controller caches: gripper stall state, last joint/time, reset base pose/yaw, arm_7 endpoint
  offset, reset fingerprint, torso before/after.

Not restored by the soft reset: the Webots simulation clock (monotonic; the episode start time is
captured), Webots-internal RNG/noise and camera buffers, contact-tracking history, and any world
node other than the CAN, base, and named motors/sensors. `simulationResetPhysics()` requests a
physics-state reset but does not reload the world.

`RobotIO.reset_fingerprint()` records every motor target/velocity/control mode (arm, torso, head,
fingers, wheels), every joint sensor, the base world pose plus `getVelocity()`, and the CAN world
pose. It is emitted only on the reset observation (`RobotState.diagnostics["reset_fingerprint"]`)
and written by the INIT-enter trace boundary. Two same-seed consecutive episodes must produce an
identical fingerprint; this is covered by tests with an injected deterministic fingerprint backend.

### Torso semantics (why it is not a brain INIT command)

- The original controller raises the torso at startup:
  `controllers/tiago_arm_api/tiago_arm_api.py:357-364` commands `torso_lift_joint` and steps up to
  1000 times until the sensor is within 0.002 m, then `:366-374` clamps head pan/tilt, sets
  velocity 0.5, and waits `int(2000 / ts)` steps. The packaged copy is
  `src/vlarobot/backends/webots/pick_place.py:121-145`.
- The brain reproduces this in `WebotsBackend.reset` (torso-before read, target write,
  `simulationResetPhysics`, step) and `_settle_reset_posture`
  (`src/vlarobot/backends/webots/backend.py:371-411`) using the new config keys
  `reset_torso_settle_tolerance_m: 0.002`, `reset_torso_timeout_steps: 1000`, and
  `reset_head_settle_ms: 2000`. Brain INIT then seeds IK from the settled torso sensor and traces
  `reset_torso_before_rad`/`reset_torso_after_rad` separately (the "before" value legitimately
  differs between resets, so it is not part of the fingerprint).
- A completed soft reset with default config always re-commands 0.35 m. The torso can be ~0 only if
  reset is never called (fresh world), reset is aborted before/during `_settle_reset_posture`, the
  torso joint is absent from `io.sensors` (early return), or `torso_height` is configured to 0/None.
  An aborted reset raises; it does not silently continue at ~0.

### Base-drift evidence and threshold proposal

`scripts/analyze_base_drift.py` (new, Webots-free) and
`trace_compare.base_drift_by_state()` compute planar distance and yaw relative to the first
`state_boundary` base pose, including fixture traces that only log
`base_world_orientation_matrix`. Results on available traces (read-only):

| trace | state | max drift | max yaw | end | rate |
|---|---|---|---|---|---|
| `m1-pregrasp-interpolated-01` | PREGRASP | 3.37 mm | 0.08° | 3.37 mm | 0.97 mm/s |
| `can-reset-check` | PREGRASP | 0.98 mm | 0.02° | 0.87 mm | 1.39 mm/s |
| `can-reset-check-replay` | PREGRASP | 1.25 mm | 0.01° | 0.96 mm | 1.69 mm/s |
| `can-reset-check-seed0` | PREGRASP | 1.28 mm | 0.01° | 0.99 mm | 1.73 mm/s |
| `m1-06a-brain-01` | PREGRASP | 1.13 mm | 0.01° | 1.13 mm | 3.52 mm/s |
| `m1-06a-brain-01` | DESCEND | 1.69 mm | 0.05° | 1.13 mm | 5.27 mm/s |
| `m1-watched-paced-01` | PREGRASP | 120.71 mm | 43.27° | 120.71 mm | 2.19 mm/s |
| `m1-watched-paced-02` | PREGRASP | 129.55 mm | 42.51° | — | — |

Proposal (not applied; user approval required to change thresholds):

- Keep the hard reset invariant at `base_drift_position_tolerance_m: 0.005` and
  `base_drift_yaw_tolerance_deg: 0.5`. Healthy runs are ≤3.37 mm / 0.08°, so 5 mm gives ~1.6 mm
  headroom while remaining ~24x below the 120 mm failure mode.
- Optionally add a soft warning at 4 mm / 0.1° to catch creep just before the veto.
- Drift rate is diagnostic only. It is not discriminative: the pathological watched runs drift at
  2.19 mm/s, lower than the short-window 5.27 mm/s seen in a healthy DESCEND sample. Do not gate on
  rate.

The new trace field `base_drift_rate_m_per_s` is computed as `max_drift / observation.t` and is
reported per boundary/waypoint.

> **Superseded on 2026-11-08 by ADR-0014.** The single 5 mm limit and the
> `base_drift_position_tolerance_m` / `base_drift_yaw_tolerance_deg` keys no longer exist; they
> were replaced by the two-tier warn/fail keys plus a sustained-rate guard. The warning above still
> holds in spirit: the per-sample rate is not discriminative (a healthy DESCEND spike hit
> 5.27 mm/s while a pathological run averaged 2.19 mm/s), which is exactly why ADR-0014 gates on
> the **mean rate over a 0.5 s window at 10 mm/s**, not on instantaneous rate. Thresholds remain
> provisional pending original-controller fixture evidence.

### Opt-in timing counters

- Disabled by default. Enable with `ScriptedCanPolicy(timing_enabled=True)`, the
  `can-regression`/`repeat-check` `--timings` flag, or `VLAROBOT_BACKEND_TIMINGS=1` for the backend
  diagnostics counter. With timings off, no `perf_counter()` calls are made on the hot path and
  behaviour is unchanged.
- Buckets: `backend_reset_s`, `ik_solve_s`, `backend_apply_s`, `backend_step_s`, `trace_write_s`,
  and `supervisor_s` (backend diagnostics computation only). ZMQ round-trip time is included in
  `backend_apply_s`/`backend_step_s` but is not broken out separately. Counters are single-run
  wall-clock samples, not averages; expect jitter.
- No live share can be reported without a Webots run. Cheapest candidate fixes, if the counters
  show a dominant bucket: buffer trace records and flush once per state (trace writes), skip
  Supervisor diagnostics when no trace is requested, and reuse one observation per step instead of
  recomputing base/endpoint telemetry per boundary.

### Fixture instrumentation

`scripts/prepare_original_comparison.py` still writes the fixture under
`VLAROBOT_COMPARISON_FIXTURE_V2` and now also logs `torso_lift_joint_sensor_rad`,
`head_joint_sensor_rad`, `base_velocity_world_xyz_m_s`, and
`base_angular_velocity_world_xyz_rad_s`. Existing fixture v2 traces already include
`base_world_xyz_m` and `base_world_orientation_matrix`, so no v3 base logging is needed.
`compare_traces()` now checks INIT torso (default 2e-3 m) and head (default 0.02 rad) posture
against the configured values so INIT boundaries match.

### Repeat-check commands (not run)

(a) Three fresh-world interpolated runs — restart Webots and `sim-server` before each:

```text
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP \
  --pregrasp-route interpolated --trace-log runs/repeat-fresh-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP \
  --pregrasp-route interpolated --trace-log runs/repeat-fresh-02.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP \
  --pregrasp-route interpolated --trace-log runs/repeat-fresh-03.jsonl
```

(b) Two consecutive episodes in one server session (soft reset between, no restart):

```text
uv run python -m vlarobot.cli repeat-check --episodes 2 --seed 0 \
  --stop-after PREGRASP --pregrasp-route interpolated
```

(c) One guarded `safe_via` run (fresh world):

```text
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP \
  --pregrasp-route safe_via --trace-log runs/repeat-safe-via-01.jsonl
```

Add `--timings` to any command for the opt-in counters.

### Verification status

- `ruff check .` and `ruff format --check .` pass.
- `ty check` reports no errors in the files touched this pass; the remaining diagnostics are
  pre-existing `controller`/`torch` import and `model/`/`controllers/` issues.
- `pytest -m "not webots"` reports **90 passed** (80 before this pass).
- All Webots-dependent behaviour — warning-free Supervisor calls, torso/reset posture, reset
  fingerprint identity, base drift, repeatability, and timing shares — remains **UNVERIFIED**.
  M1-06a, M1-07, and all M1 gates remain **UNVERIFIED**.

## 2026-11-08 live repeatability run and FK-offset correction

The operator ran the three repeat commands. This section records the observed evidence and the
one code correction the data required.

### Fresh-world interpolated repeatability (a)

Three fresh-world, same-seed interpolated PREGRASP runs:

| trace | max drift | yaw | torso start | torso end | sim | wall |
|---|---|---|---|---|---|---|
| `repeat-fresh-01` | 3.36 mm | 0.076° | 0.349745 rad | 0.349747 rad | 3.488 s | 46.6 s |
| `repeat-fresh-02` | 3.41 mm | 0.076° | 0.349745 rad | 0.349747 rad | 3.488 s | 46.0 s |
| `repeat-fresh-03` | 3.41 mm | 0.076° | 0.349745 rad | 0.349747 rad | 3.488 s | 46.4 s |

The spread is ≤0.05 mm and 0.000°, well inside the 5 mm / 0.5° invariant. The torso settles at
0.349745–0.349747 rad (0.000253–0.000255 m below the 0.35 m target), inside the 0.002 m settle
tolerance. Wall time is about 13–14x simulation time; `--timings` was not passed, so the
per-component split is not available.

### Consecutive-episode fingerprint (b)

`repeat-check --episodes 2` reported `fingerprint=present` for both episodes and 28 raw float
differences at the old 1e-6 tolerance:

- `motors.*`: zero differences. All commanded targets, motor velocities, and control modes are
  bit-identical.
- `joint_sensors_rad`: only `arm_1_joint` (1.5e-4 rad) and `arm_5/6/7` (~1e-5 rad); torso, head,
  fingers identical.
- `base`: pose ~1.3e-4 m and orientation ~1.4e-5; linear/angular velocity differs up to
  3.4e-3 m/s and 7.2e-3 rad/s.
- `can`: pose ~2.7e-6 m and orientation ~4e-5.

Conclusion: the soft reset restores the *commanded* state exactly; the *measured* state differs
only by sub-millimetre physics settling. The base/CAN velocities are a world-load transient
(episode 0 is the first reset after load; episode 1 follows a settled episode), not a reset
contract. The raw 1e-6 comparison was therefore the wrong pass/fail rule.

`repeat-check` now classifies each leaf:

- `commanded` (`motors.*`): must match exactly; any difference fails.
- `measured` (sensors/poses): fails only above `--measured-tolerance` (default `1e-3`).
- `transient` (`base.velocity_*`, `base.angular_velocity_*`): reported with
  `max_transient_delta` but never fails.

With the observed data this reports `commanded=0 measured=0 transient=6
max_measured_delta≈1.5e-4 max_transient_delta≈7.2e-3` and PASS. This is a comparison-semantics
fix, not a change to reset behaviour.

### Guarded `safe_via` run (c)

`repeat-safe-via-01` failed at the PREGRASP exit with `base moved` at 2.816 s:

- drift `0.005001564877593894 m` (5.0016 mm), yaw `0.074°`, no contact nodes;
- the 5 mm threshold was exceeded by **1.56 µm**;
- the drift is monotonic and roughly linear at ~1.78 mm/s (interpolated: ~0.98 mm/s, 3.41 mm at
  3.488 s).

This is a threshold-edge failure, not a collision. The route itself is therefore not accepted at
the current threshold. Options (thresholds unchanged pending approval):

1. Keep 5 mm / 0.5° and treat `safe_via` as failing; the interpolated route passes at ≤3.41 mm.
2. Raise the position threshold to 8 mm (still ~15x below the 120 mm watched-paced failure and
   ~2.3x above the observed `safe_via` drift); keep yaw at 0.5°.
3. Use a route-specific threshold, which complicates the contract and is not recommended.

Recommendation: option 2 if `safe_via` is to remain a candidate; option 1 if the 5 mm invariant
is considered non-negotiable.

### FK-offset correction (false positive found)

The same traces showed `arm_7_joint_fk_ground_truth_mismatch: true` with an error that grew
monotonically from 0 to **0.065779936 m** — essentially the 0.067725 m tool offset. Cause: the
reset-time offset `endPoint − FK(arm_7)` was stored in world coordinates and compared with a later
world offset without rotating it with the wrist.

Fix in `src/vlarobot/backends/webots/backend.py`:

- `_arm7_fk_world_pose()` now returns both the world position and the arm_7 link world rotation;
- reset stores `_arm7_reset_link_rotation` alongside `_arm7_endpoint_offset_world`;
- each sample predicts the current offset as
  `(R_now @ R_reset.T) @ reset_offset` and reports the residual.

New tests assert that a 90° wrist rotation with a correctly rotating endpoint gives zero error,
while an endpoint that does not rotate is flagged. Live confirmation that the error returns to
~0 still requires a Webots run.

The trace now also writes `arm_7_joint_endpoint_offset_world_xyz_m`,
`..._reset_offset_world_xyz_m`, `..._offset_change_world_xyz_m`,
`arm_7_joint_ground_truth_source`, `arm_7_joint_endpoint_fk_validation_available`, and the INIT
`reset_torso_before_rad`/`reset_torso_after_rad` fields that the earlier trace omitted.

### Warnings

The Webots console showed no `wb_supervisor_node_get_position() can exclusively be used with
Pose` warning, so the Pose-only endpoint lookup is working in the live run. Remaining messages are
expected or unrelated: the `torso_lift_joint` / `gripper_left_finger_joint` reset-baseline clamps
at startup, and Webots' own `Unsupported file type '/api/v1/models'`.

### Verification

- `ruff check .` and `ruff format --check .` pass.
- `ty check` has no errors in the touched files.
- `pytest -m "not webots"` reports **97 passed** (90 before this round).
- M1-06a, M1-07, and all M1 gates remain **UNVERIFIED**.

## 2026-11-08 two-tier base-drift invariant, wheel hold, and per-state targets

This round implements the operator's custom decision: warn early, fail late, distinguish a shove
from creep, diagnose the wheel hold, and refresh base-frame targets from the current base pose.
All thresholds remain **provisional** until original-controller fixture evidence is available.

### Two-tier invariant

`configs/robot.yaml` now exposes:

```yaml
base_drift_warn_position_m: 0.005
base_drift_warn_yaw_deg: 0.5
base_drift_fail_position_m: 0.015
base_drift_fail_yaw_deg: 0.5
base_drift_rate_limit_m_per_s: 0.010
base_drift_rate_window_s: 0.5
hold_base: none
```

`ScriptedCanPolicy._base_moved()` now:

- records a per-state sliding window of `(sim_time, drift)` samples;
- emits `base_drift_warning` **once per state** when drift reaches 5 mm or 0.5°, and never fails
  on the warning alone;
- fails with reason `base moved` when drift exceeds 15 mm or 0.5°, or when the mean drift rate
  over at least 0.5 s exceeds 10 mm/s;
- writes `base_drift_rate_m_per_s` on every `state_boundary` as the window rate (0.0 before a
  full window exists), and `rate_exceeded` / `position_exceeded` on `base_moved`.

Tests cover four cases: below-warning creep is silent, the 5–15 mm band warns once per state
without failing, above-15 mm fails by position, and a 25 mm/s shove that stays under 15 mm fails
by rate (`rate_exceeded=True`, `position_exceeded=False`).

### Creep cause and wheel state

The live trace already records `wheel_motors`. At INIT both wheels are:

| wheel | control mode | target position | target velocity | sensor |
|---|---|---|---|---|
| `wheel_left_joint` | position | 0.0 rad | 10.1523 rad/s | ≈ −1.2e-7 rad |
| `wheel_right_joint` | position | 0.0 rad | 10.1523 rad/s | ≈ +6.6e-7 rad |

So the base is already under a Webots position hold at 0 with a 10.15 rad/s velocity limit. The
creep is therefore compliance of that hold under arm/torso reaction forces, not a free-rolling
wheel. The original controller never acquires or commands a wheel device: its device setup only
covers the arm + torso at `controllers/tiago_arm_api/tiago_arm_api.py:57-64` and the fingers at
`:66-76`. The wheel hold is the TIAGo PROTO default.

`hold_base: wheels_zero_velocity` adds `RobotIO.apply_hold_base()`, which switches both wheels to
velocity control with target 0 (`setPosition(inf)`, `setVelocity(0)`). The backend applies it
immediately after `simulationResetPhysics()` and re-asserts it after the reset settle, so it is
active at INIT. Default remains `none`; the live drift difference with the flag on over the same
route is still **UNVERIFIED** and requires an operator Webots run.

### Per-state drift and rate (brain traces)

`scripts/analyze_base_drift.py` now reports both the episode-average rate and the maximum
boundary-to-boundary rate per state. Existing `--stop-after PREGRASP` traces:

| trace | state | max drift | yaw | avg rate | max rate |
|---|---|---|---|---|---|
| repeat-fresh-01 | PREGRASP | 3.36 mm | 0.08° | 0.96 mm/s | 1.24 mm/s |
| repeat-fresh-02 | PREGRASP | 3.41 mm | 0.08° | 0.98 mm/s | 1.11 mm/s |
| repeat-fresh-03 | PREGRASP | 3.41 mm | 0.08° | 0.98 mm/s | 1.11 mm/s |
| repeat-safe-via-01 | PREGRASP | 5.00 mm | 0.07° | 1.78 mm/s | 2.50 mm/s |

At these rates the 10 mm/s sustained limit is ~4–9x above the observed creep, and 15 mm is ~3x
above the largest observed PREGRASP drift. Full-flow and original-controller fixture traces are
still needed before the thresholds can be called final.

### Base-frame target recomputation

The original recomputes the CAN position in base coordinates via
`can_base_pos()` at `controllers/tiago_arm_api/tiago_arm_api.py:354-355`, used at INIT `:406`,
CLOSE `:436`/`:441`, and LIFT `:459`/`:464`. The brain previously converted the CAN and place
world poses to the base frame **once at INIT** and reused those targets for every state.

`ScriptedCanPolicy._base_frame_targets()` now re-derives `pregrasp`, `grasp`, `transit`,
`place_descend`, `release`, and `retreat` from the current `base_pose_world` before each state,
using the same ground-truth CAN pose the original reads. LIFT deliberately keeps the INIT
pregrasp, matching the original's fixed `pregrasp` target. The safe-via planner also uses the
current base pose instead of the reset pose. With zero base motion the targets are unchanged;
under drift they track the world-anchored CAN/place positions. IK tolerances and safety limits
are untouched.

### Proposed thresholds for approval

Pending original-controller fixture traces, the evidence supports keeping the operator's
provisional values:

- warn at **5 mm / 0.5°** (interpolated passes silently at 3.41 mm; `safe_via` now warns);
- fail at **15 mm / 0.5°**;
- fail at **10 mm/s sustained over 0.5 s**.

Observed `safe_via` PREGRASP is 5.00 mm at 2.50 mm/s, so it warns but no longer fails. The
original-controller fixture should be run on the same route to confirm that its per-state drift
and rate stay below these limits; until then the thresholds are **provisional**.

### Verification

- `ruff check .` and `ruff format --check .` pass.
- `ty check` has no errors in the touched files.
- `pytest -m "not webots"` reports **102 passed**.
- Live `hold_base` drift comparison, full-flow per-state drift, original-controller fixture
  evidence, and FK-error-returns-to-zero confirmation remain **UNVERIFIED**.
- M1-06a, M1-07, and all M1 gates remain **UNVERIFIED**.

## 2026-11-08 full-flow failure: closed gripper, not base drift

The operator ran the full flow twice on the same route, once per `hold_base` mode. Both runs
failed the can regression (`success=0/1`) even though the base-drift invariant was silent.

### Wheel-hold A/B (same route)

| run | `hold_base` | max drift | max yaw | max window rate | warnings | fails | final distance |
|---|---|---|---|---|---|---|---|
| `hold-none-01` | `none` | 3.874 mm | 0.118° | 1.655 mm/s | 0 | 0 | 0.246 m |
| `hold-velocity-01` | `wheels_zero_velocity` | 3.855 mm | 0.115° | 1.655 mm/s | 0 | 0 | 0.258 m |

The wheel hold is a **no-op** for drift on this route: the PROTO position hold already keeps the
base within ~3.9 mm, and forcing velocity-zero control does not reduce it. Keep `hold_base: none`.

Per-state drift is also benign: every state stays at or below 3.74 mm / 0.118°, LIFT is the
largest, and HOME returns to ~0.5 mm. The 15 mm fail and 10 mm/s rate limits are not implicated.

### Root cause: the policy never opened the gripper

At INIT the finger sensors read ~1.4e-6 m (closed). `ScriptedCanPolicy` only sent
`GripperCmd(0.0)` at CLOSE and `GripperCmd(GRIPPER_OPEN)` at RELEASE, so DESCEND ran with closed
fingers. The closed fingers contacted the can on the way down and pushed it ~20 mm
(`can_displacement_world_xyz_m ≈ [0.0199, -0.0198, 0]`). CLOSE then reported a ~6.3e-6 m opening
(nothing between the fingers), LIFT height stayed ~0, and the can was left 0.246 m from the
placement target.

The original opens the gripper at INIT
(`controllers/tiago_arm_api/tiago_arm_api.py:414`, `open_gripper(cycle=not gripper_primed)`), and
aborts the run if it cannot open (`:303-326`). The brain was missing that step.

### Fix

`ScriptedCanPolicy._open_gripper()` now runs at INIT, before PREGRASP:

- sends `GripperCmd(GRIPPER_OPEN)` (`GRIPPER_OPEN = gripper_opening_limits[1] = 0.045 m`);
- steps up to `gripper_open_timeout_s` (new config key, default 4.0 s, matching the original);
- accepts a finger as open within 2 mm of the target, mirroring the original tolerance;
- fails INIT with reason `gripper failed to open` and a `gripper_timeout` trace event if the
  fingers do not open, rather than descending into the can.

The open happens before `_base_frame_targets()` is evaluated, so the world-anchored targets are
derived from the post-open base pose. The drift veto still starts at PREGRASP; the INIT open steps
contribute to `max_base_drift_m` but do not change the invariant's scope.

### Secondary finding: the base tilts, and the drift metric does not see it

The base orientation changes by up to ~2° in roll/pitch during the route (for example roll
−1.31°, pitch +1.48° between INIT and PREGRASP), while yaw changes by only 0.08°. The drift
invariant tracks x/y translation and yaw only, so this tilt is invisible to it. Because
`_base_frame_targets()` uses the live base pose, the per-state targets track the tilt correctly;
the original instead reuses the INIT `grasp_tip`, so this is a place where the brain is stricter
than the original. Whether the tilt should also be vetoed is **open** and needs fixture evidence
before any threshold is added.

The ~78–85 mm `arm_7_joint_fk_ground_truth_error_m` reported at grasp postures is a frame
mismatch in the diagnostic, not a control error: it compares the FK tool point (fingertip,
`TOOL_OFFSET = [0, 0, -0.1741]`) with the `arm_7_joint` endPoint Solid (the wrist link). The
elbow↔wrist distance matches ground truth within 0.2 mm at every posture, and the FK tip reaches
the commanded base-frame target within 2.7 mm, so the arm is where the IK says it is. Renaming or
re-referencing this diagnostic remains **UNVERIFIED** and is deferred.

### Verification

- `ruff check .` and `ruff format --check .` pass.
- `ty check` has no errors in the touched files (22 diagnostics, all in untouched controllers,
  `model/`, and scripts).
- `pytest -m "not webots"` reports **106 passed**, including:
  - `test_policy_opens_gripper_at_init_before_descend` (a fake backend that starts closed and
    ramps at `FINGER_SPEED`; the gripper is open at INIT exit and DESCEND enter);
  - `test_policy_fails_init_when_gripper_cannot_open` (stuck fingers abort INIT with
    `gripper failed to open`).
- The operator must re-run `can-regression` on both `hold_base` modes to confirm `success=1/1`
  and record the post-fix per-state drift. That live result is **UNVERIFIED**.
- Original-controller fixture evidence, final drift thresholds, the base-tilt question, and the
  FK diagnostic frame remain **UNVERIFIED**.
- M1-06a, M1-07, and all M1 gates remain **UNVERIFIED**.

## 2026-11-08 full-flow re-run: grasp fixed, base tilt exposed

The operator re-ran the full flow on both `hold_base` modes after the INIT gripper-open fix.

| run | `hold_base` | success | final distance | can displacement (world 3D) | max drift | max tilt |
|---|---|---|---|---|---|---|
| `hold-none-02` | `none` | **1/1** | 0.042 m | 0.2484 m | 3.64 mm | 1.50° |
| `hold-velocity-02` | `wheels_zero_velocity` | 0/1 | 1.740 m | 1.318 m | 3.55 mm | 1.52° |

The gripper-open fix is confirmed: in `none` the CAN now lifts 40.7 mm, is carried, and is
released 42 mm from the placement target. The wheel hold is still a no-op for drift (3.55 vs
3.64 mm) and is now two independent full-flow failures (`hold-velocity-01` 0.258 m,
`hold-velocity-02` 1.740 m). Keep `hold_base: none`.

> Correction (2026-11-08): an earlier version of this table reported "can displacement" as
> 0.2127 m for `hold-none-02`. That was only the **x component** of the total world displacement.
> The full world displacement is `[-0.2127, +0.1283, 0.0]` m, i.e. **0.2484 m**.

### Root cause of the remaining 42 mm error: base tilt during PREGRASP

The planar drift invariant is silent, but the base rocks while the arm extends:

| state | roll | pitch | Δroll vs INIT | Δpitch vs INIT | planar drift | yaw |
|---|---|---|---|---|---|---|
| INIT | 2.10° | 1.10° | — | — | 0 mm | 0° |
| PREGRASP exit | 0.80° | 2.58° | −1.30° | +1.48° | 3.35 mm | −0.08° |
| DESCEND exit | 0.96° | 2.59° | −1.14° | +1.50° | 3.43 mm | −0.09° |

Because the arm is commanded in the base frame, that 1.5° rock moves the CAN **27.9 mm** in base
coordinates while the PREGRASP target stays fixed. The fingertip therefore finishes PREGRASP
**32.6 mm** from the CAN centre, i.e. well outside the 13.3 mm per-side clearance of the 90 mm
open gripper around the 63.5 mm CAN.

`hold-none-02` and `hold-velocity-02` are bit-identical through PREGRASP (same tilt, same
offsets), so the wheel-hold mode is not the cause of the divergence:

| state | tip-to-CAN xy (`none`) | tip-to-CAN xy (`velocity`) | CAN displacement (`none`) | CAN displacement (`velocity`) |
|---|---|---|---|---|
| PREGRASP | 32.57 mm | 32.57 mm | 0.02 mm | 0.02 mm |
| DESCEND | 41.24 mm | 99.74 mm | 42.31 mm | 103.17 mm |
| LIFT | 41.86 mm | 151.30 mm | 42.77 mm | 132.03 mm |

DESCEND re-derives its target from the live base pose, so the tip travels diagonally from 32.6 mm
off-centre at approach height to the CAN centre at grasp height. The diagonal clips the CAN's
upper rim: in `none` it is pushed 42 mm but still caught, in `wheels_zero_velocity` it is tipped
over (LIFT height −29.6 mm) and later rolls off the table to z = 0.0315 m, giving 1.740 m.

### What the 42 mm actually is (evidence reconciliation)

An earlier draft claimed "the final distance (0.042 m) equals the CAN displacement (0.042 m)".
That is **not** what the trace shows. The two numbers are different quantities that happen to be
similar in magnitude because both are produced by the same tilt-induced eccentric grasp:

- `final_target_distance` is the **3D base-frame distance from the final CAN pose to
  `place_world = [-1.0, -4.3, 0.821]`**: `0.04153 m`.
- The CAN's total world displacement from its INIT pose is **0.24843 m**
  (`[-0.2127, +0.1283, 0.0]`). The old "0.2127 m" was only the x component.
- The CAN's **start-to-place** distance is `0.21964 m`; the robot never moves the CAN that far,
  so the residual is not "the CAN was placed where it was pushed instead of where it started".
- The final residual is approximately the **eccentric grasp offset at CLOSE**: the tip is
  ~43 mm off the CAN centre when the fingers close (`tip→CAN xy = 43.11 mm` at CLOSE). The
  DESCEND push is also ~42 mm (`can displacement = 42.31 mm`), because both come from the same
  base-tilt offset. Their equality is a coincidence of mechanism, not an identity.

Per-state evidence for `hold-none-02` (CAN world position, CAN displacement from INIT, lift
height, and fingertip→CAN xy error computed from the live base pose):

| state | CAN world xyz (m) | CAN disp. (mm) | lift (mm) | tip→CAN xy (mm) |
|---|---|---|---|---|
| INIT | `[-0.7999, -4.3900, 0.8111]` | 0.00 | 0.00 | 908.06 |
| PREGRASP | `[-0.7999, -4.3900, 0.8111]` | 0.02 | 0.01 | 32.57 |
| DESCEND | `[-0.8271, -4.3580, 0.8164]` | 42.31 | 5.32 | 41.24 |
| CLOSE | `[-0.8279, -4.3565, 0.8169]` | 44.07 | 5.88 | 43.11 |
| LIFT | `[-0.8123, -4.3858, 0.8518]` | 42.77 | 40.73 | 41.86 |
| TRANSIT | `[-1.0150, -4.2666, 0.8762]` | 256.36 | 65.17 | 36.49 |
| PLACE_DESCEND | `[-1.0079, -4.2546, 0.8127]` | 248.23 | 1.68 | 40.11 |
| RELEASE | `[-1.0088, -4.2572, 0.8132]` | 247.57 | 2.16 | 41.11 |
| RETREAT | `[-1.0127, -4.2619, 0.8111]` | 248.37 | 0.01 | 51.90 |
| HOME/DONE | `[-1.0126, -4.2617, 0.8111]` | 248.43 | 0.00 | 661.27 |

The tip→CAN error is already 32.6 mm at PREGRASP (outside the ~13 mm per-side gripper
clearance), jumps to ~41 mm during DESCEND (the tilt-induced diagonal), stays ~43 mm through
CLOSE/LIFT/TRANSIT, and only collapses to zero at HOME because the arm retracts and the
fingertip is no longer near the CAN. The CAN is carried with that offset and released with it,
which is why the final placement residual is ~42 mm.

Two secondary errors follow from the same offset:

- LIFT used the INIT pregrasp, so it dragged the CAN sideways by the same offset;
- the residual after placement is the eccentric grasp offset, not a failure to reach the place
  target: the CAN is released ~42 mm from `place_world` because it was held ~43 mm off-centre.

### Fix

- `ScriptedCanPolicy._realign_pregrasp()` runs at the end of PREGRASP, after the arm extension has
  rocked the base: it re-derives the pregrasp point from the live base pose, records
  `pregrasp_realign` with `xy_error_m`, and, when the error exceeds `align_tol` (5 mm), solves IK
  for the refreshed point and moves there (`pregrasp_realigned` with before/after error). The
  descent then starts centred on the CAN.
- LIFT now rises from the refreshed pregrasp (`lift_target_refreshed` trace event) instead of the
  INIT pregrasp, so it no longer drags the CAN sideways.
- Base roll/pitch change since reset is now recorded on every `state_boundary`
  (`base_tilt_roll_deg`, `base_tilt_pitch_deg`), on `base_drift_warning` / `base_moved`, in
  `ScriptedCanResult.max_base_tilt_deg` (also printed by `can-regression`), and by
  `scripts/analyze_base_drift.py` (`tilt=` column). It is **telemetry only**; the veto still uses
  planar drift and yaw, because a 1.5° tilt is normal for this route and a 0.5° tilt limit would
  fail every episode.
- The whole realign path is gated by `pregrasp_realign` (`configs/robot.yaml`, default `true`;
  CLI `--pregrasp-realign/--no-pregrasp-realign`). With `false` the policy keeps the original
  semantics: no PREGRASP re-align, no refreshed LIFT, no CLOSE guard. M1-06a equivalence runs
  must use `--no-pregrasp-realign`; the comparer rejects a mixed-flag pair and reports the flags.
- CLOSE guard (`_guard_close_alignment`, `close_align_tol: 0.008 m`): before closing, the policy
  measures the live base-frame fingertip→CAN xy error and traces `close_alignment_check`. If the
  error exceeds the tolerance it performs **one** refined descent from the refreshed grasp target
  (`close_realign` with before/after error); if the residual is still over tolerance the episode
  fails with `failure_reason=misaligned at close` (`close_misaligned`). The guard is also gated by
  `pregrasp_realign`, so realign-off keeps the original semantics.
- Success is no longer a single distance test. `CanSuccessCriteria` reports each criterion
  separately on the `can-regression` line and in the `DONE` trace:
  `home_reached` (DONE at HOME), `place_distance_ok` (base-frame distance ≤ `place_tolerance_m`,
  0.10 m), `can_height_ok` (CAN z within `can_place_z_tolerance_m`, 0.02 m of its INIT
  table-rest height), and `can_tilt_ok` (CAN tilt < `can_tilt_limit_deg`, 15°). All four are
  required; no threshold was relaxed.

No IK tolerance, safety limit, workspace bound, or drift threshold changed.

### Verification

- `ruff check .` and `ruff format --check .` pass.
- `uv run ty check src tests` passes (0 diagnostics), down from the `ff041ff` baseline of 4
  test-file diagnostics (3 in `tests/test_scripted_can_policy.py`, 1 in
  `tests/test_webots_backend_reset.py`). The whole-repo `uv run ty check` still reports 22
  diagnostics in untouched files: 15 in `controllers/`, 3 in `model/`, and 4 in
  `scripts/analyze_pregrasp_table.py`. This scope difference is recorded in `docs/DECISIONS.md`
  (ADR-0017).
- `pytest -m "not webots"` reports **119 passed**, including:
  - `test_pregrasp_realign_corrects_tilt_induced_offset` (fake base rocks 1.3°/1.5° mid-PREGRASP;
    the realign fires, the residual is < 5 mm, and DESCEND starts < 5 mm from the CAN in world);
  - `test_pregrasp_realign_is_skipped_without_base_tilt` (nominal error ~1.8e-8 m, no extra move);
  - `test_pregrasp_realign_flag_is_recorded_and_can_be_disabled` (ON records
    `pregrasp_realign`/`pregrasp_realigned`/`lift_target_refreshed`; OFF records none);
  - `test_close_guard_redescends_once_when_can_is_nudged` (rocking base + one 12 mm nudge:
    check > 8 mm, one re-descent, residual ≤ 8 mm, episode reaches HOME);
  - `test_close_guard_fails_when_can_stays_dragged` (rocking base + CAN dragged at a fixed
    12 mm offset: `failure_state=CLOSE`, `failure_reason=misaligned at close`);
  - `test_success_criteria_are_reported_separately` and
    `test_can_success_criteria_require_every_criterion`;
  - `test_compare_rejects_mixed_pregrasp_realign_flags` and the matching/missing-flag cases;
  - `test_base_drift_by_state_reports_base_tilt`.
- Live confirmation is **UNVERIFIED**: the operator must run the realign-ON and realign-OFF
  batches in `docs/MANUAL.md` and record the post-fix `pregrasp_realign` / `pregrasp_realigned`
  errors, the CAN displacement at DESCEND, the CLOSE-guard error, and the four success criteria.
  The prediction from the trace is that the CAN is no longer pushed and the final distance drops
  from 42 mm to a few mm.
- Original-controller fixture evidence, final drift thresholds, and the FK diagnostic frame
  remain **UNVERIFIED**. M1-06a, M1-07, and all M1 gates remain **UNVERIFIED**.

## 2026-11-09 M1-07 diagnosis: closed-loop waypoints, GT tip telemetry, fresh-world runner

This section records (a) the per-run diagnosis of the six M1-07 full-flow runs, (b) the GT
fingertip telemetry instrument, (c) the closed-loop waypoint change behind ADR-0018, (d) the
housekeeping fixes and opt-in profiling counters, and (e) the fresh-world `run-episodes` runner
with its determinism-check mode. Nothing that gates behaviour changed: thresholds, the two-tier
drift invariant, `hold_base: none`, and the four success criteria are untouched.

### Six-run diagnosis (first failure cause per run)

Evidence: `runs/m1-07-realign-{on,off}-0{1,2,3}.jsonl`, rendered by
`scripts/analyze_can_run.py` (per-state rows below show CAN displacement and the
live-base fingertip→CAN xy error at state exit; "dz" is the CAN's vertical lift Δ, so a
strongly negative dz means the CAN was pushed/tipped down).

| run | realign | outcome | first failure cause | DESCEND/lift knock signature |
|---|---|---|---|---|
| `on-01` | ON | failed at CLOSE (`misaligned at close`; check 33.1 mm → one re-descent → residual 50.7 mm) | **contact/knock during DESCEND**: CAN pushed dz −29.60 mm (displacement 40.5 mm) | tip→CAN 35.59 mm at DESCEND exit |
| `on-02` | ON | failed at DESCEND (`joint_stall`) | **joint stall under the strongest tilt** (roll −2.14°, pitch −2.05°); CAN moved only 10.2 mm | descent dz +1.35 mm (no knock) |
| `on-03` | ON | failed at CLOSE (`misaligned at close`, check 277.6 mm; re-descent IK rejection at 20.7 mm error) | **CAN knocked during DESCEND** (dz −29.64 mm) then again before CLOSE (CLOSE z 0.031 m, off the table) | displacement 282 mm at DESCEND exit |
| `off-01` | OFF | ran to DONE; failed criteria | **CAN knocked during DESCEND and tipped 90°** at LIFT rise (dz −29.64 mm); height error −29.6 mm, distance 266 mm | tip→CAN 59–67 mm through DESCEND/CLOSE |
| `off-02` | OFF | ran to DONE; failed criteria | **CAN rolled off the table** (final distance 1.740 m, height error −780 mm, tilt 90°); pushed dz −28.35 mm during DESCEND | tip→CAN 99.7 mm at DESCEND exit; 151.3 mm at LIFT |
| `off-03` | OFF | **success** (distance 41.5 mm ≤ 100 mm, height +4.4 µm, tilt 0.06°) | CAN pushed ~42 mm (dz +5.32 mm, caught) but grasp and placement landed inside tolerance | eccentric carry ~43 mm through CLOSE/LIFT |

**Ordering:** in 5/6 runs the first decisive deviation is a contact/knock while the arm descends
onto the CAN (`can-regression`'s approach column clips the rim); base tilt is secondary — it
 correlates with the push but the stall in `on-02` is the only run where it is the proximate cause.
This matches the earlier `hold-none`/`velocity` study: pushed 42 mm during DESCEND in `none`,
tipped in `velocity`. The realign gate changes where the knock shows up (ON pushes at DESCEND
even after a clean 0.0–4.0 mm realigned approach; OFF pushes a larger
32.6 mm-eccentric descent), but neither setting fixes the contact itself.

### First-deviation evidence per run (trace decomposition, this session)

Per-run numbers recomputed from the raw JSONL boundaries (`can_displacement_world_xyz_m`,
`achieved_ee_base_xyz_m`, `can_base_xyz_m`; CAN half-height 0.0611 m from the standard Webots
`Can.proto`). The "real fingertip" column applies the arm_6→arm_7 placement error found by the
FK-vs-GT decomposition below (GT wrist sits 55.0 mm below the FK-predicted arm_7 origin in base
z on these poses), i.e. real fingertip height ≈ FK tip − 55 mm in base z:

| run | FK hover (tip − CAN top) | real fingertip at hover | real fingertip at first DESCEND waypoint | first CAN displacement ℓ1 | DESCEND exit displacement |
|---|---|---|---|---|---|
| `on-01` | +79.6 mm | +24.6 mm | **−14.8 mm** | 1.6 mm (at first DESCEND waypoint boundary) | 84.6 mm |
| `on-02` | +79.7 mm | +24.7 mm | −13.3 mm | 3.1 mm | 10.2 mm (stall, see below) |
| `on-03` | +77.2 mm | +22.2 mm | **−20.0 mm** | 188.4 mm | 953.9 mm |
| `off-01` | +67.7 mm | +12.7 mm | **−11.4 mm** | 7.4 mm | 61.8 mm |
| `off-02` | +67.7 mm | +12.7 mm | **−14.0 mm** | 7.1 mm | 103.2 mm |
| `off-03` | +67.7 mm | +12.7 mm | **−12.7 mm** | 1.9 mm (grazing, caught) | 42.3 mm |

All six runs agree: FK believes the tip hovers 68–80 mm above the CAN lid, but the
55 mm wrist error leaves the real fingertip only +12.7…+24.7 mm clear at hover, and by the first
DESCEND waypoint it is already **11.4–20.0 mm below the lid plane** — the commanded descent path
starts geometrically inside the CAN body. This is the rim contact that knocks `(dz −28…−30 mm)`
in 5/6 runs and stalls `on-02`.

**`on-02` stall detail (UNVERIFIED inference).** The `joint_stall` event carries no object
identification — only `target_base_xyz_m`, `joint_target`, and `no_progress_steps: 10` — so the
obstacle it stalled against is not directly recoverable from the trace. The CAN-rim inference is
arithmetic, not observation: the stalled waypoint's target z is **0.8055 m, i.e. 11.2 mm below
the CAN top (0.8167 m)** at that boundary, and the corrected real fingertip would reach ~55 mm
lower still; a stall against the rigid cylinder rim is consistent, but no contact-force or
contact ground-truth channel exists in the trace to confirm it live.

**FK-vs-GT frame mismatch (not the proximate cause).** Earlier supervisor-side measurement put
the FK tool tip ~70–83 mm away from the real finger-pad midpoint on live runs. That is a
`TOOL_OFFSET`/frame calibration discrepancy against the configured 0.1741 m `finger_tip_offset_m`;
the descent targets stay world-anchored to the GT CAN, so the offset degrades clearance
measurement fidelity, not the target itself. Because the six M1-07 traces predate the GT fields,
these numbers are **UNVERIFIED** until the telemetry below has run live.

### GT fingertip telemetry (instrumented this session)

Every state boundary now merges `_gt_finger_payload` into `_trace_measurements`, and
`gt_tip_telemetry` events fire exactly at PREGRASP exit, DESCEND enter, and CLOSE enter, carrying:

- `gt_midpoint_world_xyz_m`, `finger_tip_world_xyz_m` (per-finger tip points from
  Supervisor `endPoint` Solids), `ee_fk_world_xyz_m`, `finger_midpoint_fk_offset_world_xyz_m`/
  `finger_midpoint_fk_offset_m`, `finger_midpoint_source`;
- `fk_gt_offset_m` (|GT midpoint − FK tip|) and `fk_to_can_world_m`/`fk_to_can_xy_m`;
- `gt_to_can_world_m`/`gt_to_can_xy_m`, `finger_midpoint_base_xyz_m`,
  `finger_midpoint_to_can_base_xy_m`, `can_radius_m`;
- `clearance_left_m`/`clearance_right_m`: each finger-pad's xy distance to the CAN cylinder minus
  `can_radius_m` (negative = pad centre inside the CAN radius).

If `fk_gt_offset_m > TOOL_OFFSET_CALIBRATION_TOLERANCE_M` (5 mm), the event adds
`tool_offset_error`, a `calibration_warning`, and a report-only
`tool_offset_calibration_proposal` (`configured_finger_tip_offset_m = 0.1741` and a
`proposed_finger_tip_offset_m` from the GT `arm_7_joint` ground-truth point, with a note that the
direction must be confirmed against the tool axis). **The proposal is never auto-applied**;
calibration requires a live operator run and a decision entry. The m1-07 traces show
`gt_tip_telemetry: none in this trace` in `analyze_can_run.py`, as expected for pre-telemetry
records.

### Closed-loop waypoints (ADR-0018)

`closed_loop_waypoints: true|false` in `configs/robot.yaml` (default `true`;
`--closed-loop-waypoints/--no-closed-loop-waypoints` on `can-regression` and `run-episodes`):

- **DESCEND**: the grasp target is re-expressed in the live base frame at every Cartesian
  waypoint (`closed_loop_waypoint: true`, `live_target_base_xyz_m` on waypoint boundaries), so a
  base that shifts between waypoints drags the target with it instead of clipping the rim.
- **LIFT**: anchored to the CAN pose captured at LIFT entry, then re-expressed in the live base
  frame only — a grasped, rising CAN cannot pull its own target upward (no feedback loop).
- **CLOSE guard re-descent**: also live-tracked now, and attributed to CLOSE in
  `state_sim_durations`/`state_wall_durations` so DESCEND's settle accounting is not inflated by
  a state that already exited.
- **OFF = original semantics**: `_move_linear` keeps a fixed target and the original `alpha`
  schedule verbatim; fake-backend regression tests assert the OFF run's waypoint boundaries carry
  no `live_target_base_xyz_m` key at all.

The base-frame unit test rocks the base 6 mm sideways once the fingertip is inside the DESCEND
column: with closed-loop on, the live targets follow the planar drift (≥ 2 mm first-to-last);
with it off, they do not move.

### Reset fingerprint / repeat-check status

Carried over from the 2026-11-08 repeatability run (still current, no change this session):
`repeat-check` reports `fingerprint=present` with `commanded=0 measured=0 transient=6
max_measured_delta≈1.5e-4 max_transient_delta≈7.2e-3` and **PASS** under the
commanded/measured/transient classification. Episode-0 base/CAN velocity differences remain a
world-load transient, not a reset contract violation. The fresh-world runner additionally
normalizes `reset_fingerprint` and `controller_reset_state` out of trace digests so a
determinism comparison does not re-flag them.

### Housekeeping and opt-in profiling

- **Gripper force clamp** (`backend._set_gripper`, `gripper.close`): commanded force is clamped
  to `motor.getMaxForce()` before `setAvailableForce`, matching the simulator's internal clamp.
  No behaviour change; the AvailableForce warning disappears.
- **Settle/sim accounting**: the CLOSE guard's re-descent no longer counts into DESCEND; CLOSE's
  own sim/wall durations are recorded (`state_sim_durations["CLOSE"]`) and set before the guard
  runs.
- **Per-step counters (opt-in):** `--timings` on `can-regression` / the policy-level
  `TimingSummary` reports `ik_solve_s`, `backend_apply_s`, `backend_step_s`, `trace_write_s`,
  `backend_reset_s` totals with per-state splits, plus new `steps` and `wall_per_step_s`
  (accounted wall per control step). With `VLAROBOT_BACKEND_TIMINGS=1` the ZeroMQ client also
  accumulates `encode_s`/`send_s`/`recv_s`/`decode_s`, `requests`, and `image_transfer_bytes`
  (RGB rasters), printed as `timings zmq:`. Supervisor diagnostics time is traced as
  `backend_diagnostics_compute_s`. **Cheapest-fix proposal (not applied):** the largest
  policy-side slice is expected to be `trace_write_s` (one JSON line per boundary/waypoint), so
  buffering trace rows and flushing per state would cut file syscalls; the sim step itself
  (`backend_step_s`) dominates overall wall time and is not addressable from the brain side.
  Run one live episode with `--timings` + `VLAROBOT_BACKEND_TIMINGS=1` to confirm before acting.

### Fresh-world episode runner (`run-episodes`)

New CLI command (see `docs/MANUAL.md` §3b) built on `execution/episode_runner.py`:

- one Webots subprocess per episode (strict arg lists, never `shell=True`), `WEBOTS_HOME` /
  `PATH` resolution, ping-wait (`--startup-timeout-s`, 180 s default), polite terminate-then-kill;
- per-episode traces `runs/episode-runner/episode-NN.jsonl` + `episode-NN.log` (stale traces are
  deleted), one fixed-width summary table across all runs covering the four success criteria;
- `--determinism-check K`: same seed K times in fresh worlds; compares outcome dictionaries
  (per-episode index/paths/wall-times excluded), SHA-256 digests of normalized traces (volatile
  keys `reset_fingerprint`, `controller_reset_state`, `trace_path`,
  `backend_diagnostics_compute_s` dropped), and numeric deltas (`sim_time_max_delta_s`,
  `achieved_ee_max_delta_m`, `can_world_max_delta_m`, `commanded_ee_max_delta_m`); **exits 1 on a
  mismatch**;
- `--no-fresh-world` reuses one Webots process with soft resets (same semantics as
  `can-regression`);
- rendering stays ON (`--batch --mode=fast --stdout --stderr`) because the scripted policy reads
  RGB every step; `--headless` adds `--no-rendering` and prints an explicit UNVERIFIED caveat.

Operator commands (not run by the agent):

```bash
uv run python -m vlarobot.cli run-episodes --n 3 --fresh-world
uv run python -m vlarobot.cli run-episodes --n 3 --fresh-world --no-closed-loop-waypoints
uv run python -m vlarobot.cli run-episodes --determinism-check 3
```

### Verification

- `ruff check .` and `ruff format --check .` pass; `uv run ty check src tests` is clean
  (0 diagnostics).
- `pytest -m "not webots"` reports **144 passed** (119 before this round), including:
  - `test_closed_loop_descend_tracks_rocking_base` (descend-column base shift; live targets
    track ≥ 2 mm, OFF run has no live-target keys, identical waypoint counts);
  - `test_gt_telemetry_records_fingers_and_flags_tool_offset_error` (three-point events, 5 mm
    proposal with `report_only=True` and configured 0.1741);
  - `test_gt_telemetry_below_tolerance_has_no_calibration_proposal`;
  - `test_gt_telemetry_fires_at_all_three_full_flow_boundaries`;
  - the ZMQ client timing tests (opt-in counters, off by default);
  - `tests/test_episode_runner.py` (16 tests: exact Webots arg lists, `WEBOTS_HOME` resolution,
    outcome normalization, digest/delta comparisons, determinism match/mismatch) — no subprocess
    is launched in any new test.
- `runs/*.jsonl` operator evidence was not modified; the six M1-07 traces stay as recorded.
- **UNVERIFIED items:** live `run-episodes`/`--determinism-check` runs; headless flags; the real
  `TOOL_OFFSET` calibration correction; the original-controller fixture comparison; M1-06a,
  M1-07, and all M1 gates.

### Sim-derived kinematics export and validation sweep (ADR-0020, machinery; live UNVERIFIED)

- **Supervisor export:** `supervisor_kinematics.py` captures a read-only snapshot at the settled
  reset posture (Robot node frame, torso/arm parent+child `Solid` poses, both finger `endPoint`
  poses, Pose-derived nodes only). Hinge axes are derived from a <= 0.05 rad posture probe
  (restored immediately) with the `jointParameters.axis` used only as a cross-check; the torso
  slider is baked and never probed. Gated by `export_sim_kinematics` (default `false`): the
  backend attaches `sim_kinematics_snapshot` on reset and per-step `kinematics_telemetry`.
- **Sim chain:** `sim_kinematics.py` loads the JSON and builds an IKPy chain rooted in the Robot
  node frame (fixing the URDF rigid upstream XY offset by construction). Per hinge the measured
  family `T(θ)=Rot(â,θ)@K` is encoded as `URDFLink(axis=â)` then `FixedLink(K)` with absolute
  motor-limit bounds; the torso is a baked, asserted `FixedLink`; the tool frame is the finger-pad
  midpoint (or the `fingerTipOffsetFallback(UNVERIFIED)`). Load validation re-derives axes from the
  probe, rejects fit residuals > 1 mm and arm parent-chain gaps > 0.1 mm.
  `Kinematics(model_override="urdf"|"sim")` selects the chain; default remains `urdf`.
- **Validation sweep:** `kinematics_validation.py` + CLI `validate-kinematics` scores both chains
  in the Robot-node base frame over >= 40 deterministic stratified arm poses (torso fixed). A pose
  is aborted, not scored, on table-plane clearance failure or reject/unsettled. Accept (sim only):
  finger-midpoint mean < 2 mm and max < 5 mm. Failure analysis reports the first affected
  link/frame and fits named rigid-offset families with a strict even/odd **pose** holdout, so a
  constant offset is distinguished from pose-dependent axis/scale/structure error (identifiability
  stated). `export-sim-kinematics` writes/verifies the snapshot; `scripts/reevaluate_traces_sim_model.py`
  re-scores telemetry-bearing reports offline and reports "blocked on live export" (no numeric
  claim) when the snapshot/telemetry is absent.
- **Evidence (Webots-free):** `tests/test_supervisor_kinematics.py`,
  `tests/test_sim_kinematics_loader.py`, `tests/test_kinematics_validation.py` — 29 tests
  covering access discipline, probe derivation/restore, export-posture round trip, perturbed-angle
  FK, base-tilt invariance, loader rejections, exact acceptance, constant base/tool bias
  attribution, pose-dependent rejection, table pre-check/pose sampling, and the CLI sweep driver
  on a fake lockstep client. Gates: ruff check/format clean, `ty check` only the 2 pre-existing
  HEAD diagnostics, `pytest -m "not webots"` **180 passed**. Commits `4710fd7`, `587b176`,
  `05d18d7`, `801d62e`.
- **UNVERIFIED (live, required next):** no snapshot exported, no `validate-kinematics` sweep run,
  sim model not accepted/rejected on live data, and existing traces cannot be numerically
  re-evaluated (they lack `kinematics_telemetry`). Run the §3c commands against a watched factory
  world: `export-sim-kinematics` (≈60–120 s startup + ~10 s probes) then `validate-kinematics -n 40`
  (several minutes), persist the JSON report, and feed it to `scripts/reevaluate_traces_sim_model.py`.
  M1-06a, M1-07, and all M1 gates remain UNVERIFIED.

### ADR-0020 addendum — fast reset / chunked export / probe stationarity / session hygiene (UNVERIFIED)

- **Reset/export separation:** `WebotsBackend.reset` is probe-free and fast; export moved to a dedicated chunked `export_kinematics` request (`start`/`poll`/`status`/`cancel`) advanced one joint per poll with per-joint progress/heartbeat (steps used, measured angle moved, settled/reason) and a final snapshot. `cli export-sim-kinematics` now does `reset` + chunked polling with progress; `validate-kinematics` requires per-step `kinematics_telemetry` (enabled by `export_sim_kinematics`) and no longer expects a reset-attached snapshot.
- **Probe:** stationarity-based settle (`|delta sensor| < probe_stationary_delta_rad` for `probe_stationarity_steps` consecutive steps, config) with axis/angle from **measured** sensor/pose delta (not commanded convergence); per-joint and total step caps and explicit reasons. The earlier `PROBE_SENSOR_TOL` near-target workaround is removed. Axis derivation keeps the posture-0 parent frame consistent so loader recomputation matches stored axes.
- **Session/bind/client robustness:** `ping` carries `session_id` (pid/start/nonce) and clients fail loudly on mismatch; server resolves/binds `VLAROBOT_ZMQ_PORT`/config `zmq_port` and fails loudly on bind failure (no fallback). `ZmqBackendClient` uses per-request-type timeouts, lazy-pirate socket reset on timeout, timeout/elapsed logging, and chunked `export_kinematics` polling.
- **Process hygiene / ops tooling:** `cli doctor` lists stale `webots`/`tiago_controller`/sim-server processes and the configured port owner (`--kill` after printing). `run-episodes` allocates a free port per episode/session, passes `VLAROBOT_ZMQ_PORT` to the child, writes pid files, and terminates the full Webots/controller process tree cross-platform on exit/timeout/exception. `cli bench-steps --n 200` measures step+observation latency against a running server; config `observe_images` (default `true`, unchanged) gates image capture. Defaults unchanged.
- **Evidence (Webots-free):** fake chunked export separated from reset, ZMQ chunked export + session identity + lazy-pirate socket recovery + session-mismatch rejection tests, updated supervisor/loader/validation/backend-reset tests. Targeted affected suite and full `pytest -m "not webots"` / `ruff` / `ty` results recorded in the commit/run notes for this addendum.
- **UNVERIFIED (live, agent did not run):** valid `export-sim-kinematics` snapshot/report, trial (`--n 8`) and acceptance (`--n 40`, sim finger-midpoint mean < 2 mm & max < 5 mm) sweeps, telemetry-bearing re-evaluation, and `run-episodes --determinism-check 3` (fresh world, watched GUI). Prior live attempts failed (reset/export timeouts, stale-server suspicion, ZMQ REQ state errors) and are addressed by this addendum; `configs/generated/sim_kinematics.json` from an earlier attempt is **invalid/untracked** and must not be committed. M1-06a/M1-07/M1 gates remain **UNVERIFIED**.


### ADR-0020 follow-up — validity/atomic export, timeout config, ty 0, live observations (UNVERIFIED)

- **Commits:** `454c575` (instantaneous probe frames + verified atomic snapshot validity/hash/session; loader refuses invalid/hash-mismatch; failed export removes temp/target; loader accepts `parent_pose_probe`), `ba54f28` (configurable ZMQ timeouts, client not-up vs busy/stalled distinction, doctor port-free+GUI/batch, bench verdict, episode-runner startup ping config, validate report `samples` persistence, forward_kinematics ty fixes), `9b6568a` + `8e8c26b` (ty/type/import cleanups; `.gitignore` for `configs/generated/`).
- **Timeouts:** config keys added (`zmq_ping/reset/step/apply/get_gt/shutdown/export_poll_timeout_ms`) with defaults justified vs measured ~447 ms/step (RGB+telemetry). Reset/startup tolerate slow Webots start via startup wait + repeated short pings; client reports not-up vs busy using session responsiveness.
- **Type checking:** ty `0.0.99` pinned; baseline 2 at `1783274` → 27 at committed `5a86f75` → **0** now (`ty check src tests`).
- **Re-eval readiness:** future validate reports persist telemetry `samples` (re-evaluable offline with valid snapshot). Six earlier scripted traces lack `kinematics_telemetry` and cannot be backfilled; short re-collection (e.g. 3 episodes) with `export_sim_kinematics: true` required. No results guessed.
- **Live observations (operator, UNVERIFIED):** bench ~446.8 ms/step. Determinism check `--determinism-check 3` (per-episode ports/env/pids/tree-kill) ran but **non-deterministic** 0/3 with DESCEND/home variance and startup ping timeouts. Chunked export completed all joint probes (ok, ~120 steps total) but snapshot rejected on verification: hinge-fit residual up to **11.495 mm** (>1 mm cap) — instantaneous parent-frame probe fix + atomic validity now implemented but **live-UNVERIFIED**. `validate-kinematics` blocked (no `kinematics_telemetry` on server). Gates **UNVERIFIED** for live acceptance; Webots-free gates: `ruff check` pass, `ruff format --check` pass, `ty check src tests` pass (0), `pytest -m "not webots"` **185 passed**.


### ADR-0020 live export diagnosis/fix attempt (axis-line hinge, base pinning) — UNVERIFIED live

- **Commits:** pending in this session (axis-line hinge fit/model, base pose pinning during export, probe pose0 persistence for recomputation, chain-consistency handling, failed-export debug artifact, defensive telemetry/solid pose access).
- **Root-cause fixes (code):** hinge derivation/chain now model rotation about a fitted axis line `(a,c)` with pitch in residual (not origin-only rotation), addressing the ~9.75 mm fit residuals seen on prior live exports. Export probing pins the robot base pose and applies hold/extra quiesce to reduce drift; probe stores `parent_pose0/child_pose0` so loader recomputation remains valid when chain parent poses are normalized. Failed finalize writes `sim_kinematics.json.debug.json` (`valid:false`, `verification_error`) without leaving a loadable canonical snapshot.
- **Live analysis (from saved debug, UNVERIFIED as promotable snapshot):** recomputing stored probe poses with the axis-line model yields residuals < 1 mm for 6/7 arm joints (arm_1 ~0.64, arm_2 ~0.06, arm_3 ~0.49, arm_4 ~0.03, arm_5 ~0.57, arm_7 ~0.59 mm) but **arm_6 shows ~5.42 mm pitch** (screw component) in that debug sample; no verified canonical snapshot produced.
- **Environment:** Webots server startup/availability was intermittent in this session (batch runs sometimes died or became unreachable), preventing completion of a fresh verified `export-sim-kinematics` and any `validate-kinematics` sweep on a telemetry-enabled server. `validate-kinematics` requires `export_sim_kinematics: true` on a restarted server.
- **Gates (Webots-free):** `ruff check .` pass, `ruff format --check .` pass, `ty check src tests` pass (0), `pytest -m "not webots"` **185 passed**.
- **Status:** live export acceptance, validate-kinematics (n=8/40), re-evaluation, determinism, and M1 gates remain **UNVERIFIED**. Next: run fresh `export-sim-kinematics` on a stable watched Webots session, inspect per-joint residuals (especially arm_6 pitch) from debug/canonical output, then run telemetry-enabled validation sweeps if export verifies.
