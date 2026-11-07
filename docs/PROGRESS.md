# Project progress

Current milestone: **M1 — Port controller behind interfaces**

Last updated: **2026-11-07**. M0 is committed. M1-01 through M1-05 are implemented. The approved
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
  HOME joint commands by the same step, and fail a linear move with the trust-region reason if no
  in-region solution meets the existing tolerance.
- [x] Add Webots-free full-flow continuity, swept-transit, and placement-grid analysis at
  `scripts/audit_can_path.py`; no target or safety parameter was changed.
- [x] Add original-worktree comparison fixture and `compare-traces` CLI for M1-06a; live comparison
  remains pending.
- [ ] **M1-06a refactor equivalence (PENDING LIVE RUNS):** use an original-controller temporary
  worktree and the brain policy with the same approved `PLACE_WORLD` target/world. Compare only
  `INIT → PREGRASP → DESCEND → CLOSE → LIFT`, which the original actually completes: require equal
  state sequences and per-state joint/EE waypoints within `1e-3 rad` / `1e-3 m` over at least five
  deterministic-seed runs of each implementation. Report the first divergence.
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
- **Placement geometry check:** `Table.proto` defines a 0.75 m top; the CAN bottom at the approved
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
- **Swept transit comparison:** Cartesian line: max joint step `0.150000 rad`, minimum joint-limit
  margin approximately zero, peak tilt `0.263349°`, estimated minimum payload/table clearance
  `0.120158 m`, estimated minimum payload/object clearance `0.116168 m` (hammer). Joint-space
  interpolation: max step `0.137162 rad`, margin approximately zero, peak tilt `2.192890°`, table
  clearance `0.112915 m`, object clearance `0.118443 m` (hammer). These are FK checks of the tool
  and held-CAN proxy, not arm-link collision or dynamic Webots evidence. Cartesian transit is
  recommended for its roughly `7.2 mm` greater tabletop clearance and `1.93°` lower peak tilt;
  neither route removes the arm-6 limit-margin issue.
- **Placement grid search:** At 0.10 m grid spacing, 534 tabletop points passed the documented
  geometric filters and 16 passed sampled IK descent. The best route margin excluding the fixed
  LIFT start state was `0.043619 rad` at world `(-0.188828,-4.190293,0.821100)` m / base
  `(0.466393,0.381651,0.725361)` m. The next two margins were `0.031087` and `0.028296 rad`;
  see ADR-0007 for their coordinates. No placement candidate was applied.
- **M1-06a tooling:** `scripts/prepare_original_comparison.py` prepares the original temporary
  worktree with world-frame target conversion, seed 0, stop-after-LIFT, extern-controller mode, and
  JSONL tracing. `uv run python -m vlarobot.cli compare-traces A.jsonl B.jsonl` compares state and
  waypoint prefixes and reports the first divergence. Synthetic comparer tests pass. Original
  fixture files compile, but no live original-vs-brain runs have been made.
- **M1 gate status:** M1-06a and M1-07 are pending/unverified until their required live Webots runs
  are completed. M2 has not started. Windows CI remains unverified.
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
