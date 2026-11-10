# Architecture Decision Records

## ADR-0001: Establish M0 Python project foundation

- **Date:** 2026-11-07
- **Status:** Accepted
- **Context:** The repository had no Python package layout, lockfile, test configuration, or
  cross-platform CI. The architecture calls for Python 3.11, uv, Ruff, pytest, pre-commit, and a
  Windows/Linux CI matrix. M0 must not change controller behavior.
- **Decision:** Create a `src/vlarobot/` package, keep the existing controller files in place until
  the M1 migration, add empty configuration/script landing documentation, and use Python 3.11.18
  for local/CI tooling. Keep optional runtime extras declared but empty until runtime packages are
  selected. Configure pytest's `webots` and `gpu` markers.
- **Decision:** Use Typer for the CLI and import Webots' controller API only within
  `vlarobot.backends.webots`. Add `vlarobot.env.configure_webots_environment` to set `WEBOTS_HOME`
  and prepend the API path to `PYTHONPATH` and `sys.path` using the host OS path separator.
- **Decision:** Use `ty` for CI type checking rather than mypy. `ty` is MIT-licensed and has no
  runtime dependencies; it is currently beta, so the lockfile pins the tested version. This avoids
  introducing mypy's MPL-2.0 `pathspec` dependency.
- **Consequences:** CI installs and checks the package on Ubuntu and Windows; local Fedora checks
  pass. Actual Windows CI and Webots verification remain outstanding. Model, training, GPU, and
  simulator-surrogate dependencies remain unselected.
- **License audit:** Locked Python runtime and development packages are MIT, BSD, ISC, PSF, or
  Apache-2.0/BSD dual-licensed. The pre-commit hook repositories are MIT-licensed. No AGPL, GPL,
  or non-commercial dependency is in the lockfile. Package-license metadata was inspected from
  the resolved distributions on 2026-11-07. Model/checkpoint licences are not audited because
  none has been selected.
- **Source of truth:** `docs/ARCHITECTURE.md`; the older references to
  `docs/ARCHITECTURE_AND_PLAN.md` do not match the plan file present in this repository.

## ADR-0002: M1 core contracts and lockstep backend

- **Date:** 2026-11-07
- **Status:** Accepted, with M1-06 regression pending
- **Context:** M1 requires simulator-independent core contracts, a versioned simulator/brain
  protocol, a deterministic fake backend, and a Webots backend without changing the existing
  scripted controller logic.
- **Decision:** Implement the planned core types and protocols as dataclasses and `Protocol`
  interfaces. Use msgpack protocol version 1 with NumPy array shape/dtype metadata, strict
  envelope validation, and per-request sequence correlation. Implement a synchronous ZMQ REQ/REP
  transport with a default endpoint of `tcp://127.0.0.1:5555`.
- **Decision:** Keep the existing scripted controller available in
  `vlarobot.backends.webots.controller`. The Webots shim starts the lockstep server by default,
  selecting the repository `.venv` Python or `WEBOTS_PYTHON`; `VLAROBOT_WEBOTS_MODE=legacy`
  selects the retained controller for comparison.
- **Decision:** Load Webots/controller settings from `configs/robot.yaml` and workspace profiles
  from `configs/safety.yaml`; retain compatibility aliases for the old controller constants.
  Package both YAML files as install data, include the TIAGo URDF as package data, and resolve
  settings from the checkout or active Python prefix.
- **Dependencies and licences:** Added msgpack 1.2.4 (Apache-2.0), NumPy 2.4.6 (BSD-3-Clause and
  permissive notices), PyYAML 6.0.3 (MIT), pyzmq 27.3.0 (BSD-3-Clause), and Typer 0.28.0 (MIT);
  added IKPy 4.4.4 (Apache-2.0) to the `sim` extra. Its locked SciPy 1.17.1 and SymPy 1.14.0
  dependencies use BSD-style permissive licences. Installed distribution licence files were
  inspected on 2026-11-07; no GPL, AGPL, or non-commercial licence was selected.
- **Consequences:** Non-Webots tests and a live Fedora/Webots R2025a lockstep run pass. Camera
  extrinsics remain identity placeholders pending M3 calibration. Webots reset supports the
  loaded default scene only; seeded scenario spawning is deferred to M2.
- **Decision:** Derive workspace profiles from the original PickPlace flow's measured base-frame
  waypoints instead of guessing bounds. The `original_envelope` margin is 0.10 m and the tighter
  `strict` profile margin is 0.02 m. The legacy `PLACE_BASE = [-1.07, -3.9, 0.8]` is confirmed
  at `controllers/tiago_controller/config.py:6` in pre-M1 commit `1b2e4da`; this was the active
  target when ADR-0002 was recorded. The original controller had no workspace limit. Those
  historical profile bounds were later found to reflect the suspect target and were replaced by
  ADR-0004. Neither the old nor corrected profile is a validated safety limit.
- **Diagnostic qualification:** The original and current pick-and-place flows had no demonstrated
  successful placement in the documented trials. The earlier waypoint set is evidence of commands
  issued, not successful placement or a reachability proof.
- **Measured original waypoints (base-frame meters):** home
  `(0.203672298, -0.985455309, 0.963048983)`, descend/grasp
  `(0.672736661, -0.213295065, 0.764821419)`, pregrasp/lift
  `(0.672736661, -0.213295065, 0.884821419)`, transit/retreat
  `(-1.077327982, -3.842045869, 0.936284755)` / `(-1.086885823, -3.858086446, 0.935119120)`,
  and place-descend/release `(-1.077327982, -3.842045869, 0.816284755)` /
  `(-1.086885823, -3.858086446, 0.815119120)`. Collected from two instrumented original-controller
  runs against `worlds/factory.wbt`; each run logged repeated names at initialization and
  terminal-state flush. The latest run terminated `FAILED`, so this establishes commanded
  waypoint evidence, not successful placement.
- **Open acceptance:** M1-06 remains blocked by the live regression. The current Webots-backed
  brain comparison was executed with `uv run python -m vlarobot.cli can-regression --seeds 10`; it
  produced `success=0/10` and `rejections=9`, with the first and only near-success still 0.560m from
  the target and repeated safety rejections in the `INIT`, `pregrasp`, and `place_descend` phases.
  That 0.560m value compared a world-frame CAN point directly to a base-frame target and is not a
  valid baseline-compatible placement metric; ADR-0003 corrects the metric.
  Webots resets currently replay the default scene, so seed variation does not yet imply scenario
  randomization (scenario generation is M2). As of this date, the M1-06 gate is not met and no
  success claim is made.

## ADR-0003: Preserve full base pose for frame-correct regression metrics

- **Date:** 2026-11-07
- **Status:** Accepted
- **Supersedes:** ADR-0002's protocol-version-1 choice.
- **Context:** The scripted brain regression compared the CAN's world-frame ground-truth position
  directly with `PLACE_BASE`, although the original controller compares both in the robot base
  frame. `RobotState` previously exposed only a planar base pose, which cannot reproduce the
  original 3D transform over the simulator/brain boundary.
- **Decision:** Add optional `RobotState.base_pose_world` as a full `Pose3D`, populate it from
  Webots' base position and rotation, and increment the msgpack protocol version to 2. Compute
  regression distance as
  `norm(R_world_from_base.T @ (CAN_world - base_position_world) - target_base)`, matching
  `Frames.world_to_base()` and the original baseline scoring formula.
- **Consequences:** Protocol version 1 peers now fail fast as incompatible. Fake-backend and
  round-trip tests cover the added pose and transform.

## ADR-0004: Use an approved world-frame CAN placement target

- **Date:** 2026-11-07
- **Status:** Accepted; live regression blocked by existing IK threshold
- **Context:** The historical `PLACE_BASE=[-1.07,-3.9,0.8]` was unreachable in the coded base frame.
  The existing workspace profiles had been derived from that path and could not remain the active
  profiles. The user approved the world target `(-1.0,-4.3,0.821)` m as a CAN-center target and
  authorized regenerating the workspace profiles, while explicitly retaining all IK and safety
  thresholds.
- **Decision:** Store `place_world` in `configs/robot.yaml`; convert it at runtime through
  `Frames.world_to_base()` in the Webots PickPlace flow and through `RobotState.base_pose_world` in
  the brain regression. Score placement by transforming the final CAN world pose to the final robot
  base frame and measuring its Euclidean distance to the same target transformed into base.
  Preserve the old target only in historical controller sources.
- **Historical geometry evidence (clearances superseded; UNVERIFIED (assumed radii)):** In `worlds/factory.wbt`, the table top is z=0.75 m; the target CAN bottom
  is z=0.7599 m. The target is 0.209707 m from the nearest tabletop edge, or 0.177957 m after
  subtracting the CAN radius. Initial-CAN center distance is 0.219419 m horizontally and 0.219649 m
  in 3D; for parallel cylinders the surface gap is 0.155919 m. Webots R2025a collision shapes give
  conservative horizontal-projection surface-clearance lower bounds of 0.2826 m (wrench),
  0.3290 m (hammer), 0.5246 m (apple), 0.7974 m (paint bucket), and 1.1517 m (screwdriver).
  Darwin-op is also near the target; its articulated collision geometry is not included in these
  prop-shape calculations. This is a geometric screen, not proof of dynamic collision clearance or
  stable placement.
- **Workspace profiles:** Replace both profiles using the corrected nominal commanded waypoints.
  `original_envelope` is their axis-aligned bounds plus 0.10 m per axis; `strict` is the same bounds
  plus 0.02 m. The original controller had no workspace limit. The envelope is a behavioral
  baseline, not a validated safety limit. Joint limits, velocity limits, IK checks, and invalid
  command rejection remain active, and rejection reasons remain logged.
- **Reachability result:** With the configured torso at 0.35 m, IKPy reaches the CAN pregrasp,
  grasp/descend, lift, and place-descend targets with <0.000001 m position error. The place-above
  transit target has 0.010475 m error and 0.257 degrees tilt. `Motion.move_linear()` used the
  then-current 0.005 m `IK_TOL_GRASP` and failed at waypoint 7/7. At the time of this ADR, no target,
  IK tolerance, or safety threshold had been changed. ADR-0006 records the later approved,
  state-specific transit tolerance decision.
- **Diagnostics:** Add opt-in JSONL traces for controller and brain regression state boundaries,
  commands/achieved poses, IK result and waypoint, gripper opening/stall/contact evidence, and CAN
  world/base poses. Logging does not change the default action path. A live corrected-target
  comparison remains unverified.

## ADR-0005: Restore the default Webots scene between episodes

- **Date:** 2026-11-07
- **Status:** Accepted for soft-reset behavioral equivalence
- **Context:** Repeated brain episodes exposed stale CAN-node references and state drift in the
  robot, object, gripper telemetry, and simulation-time observations. Seeded episodes must begin
  from the same default scene before their task results can be compared.
- **Decision:** On reset, reacquire the CAN node by DEF, restore captured arm/torso/head and finger
  joint positions through Supervisor joint fields, restore the robot base and CAN transforms,
  reset physics, clear gripper close-stall telemetry, and report episode-relative observation
  time. If an initial joint field is outside that motor's legal range, use the nearest legal limit
  and emit a warning. Do not change task targets, safety bounds, or IK thresholds as part of reset
  handling.
- **Equivalence tolerance:** Compare reset poses, arm joints, and CAN pose at `1e-3 m` and
  `1e-3 rad`. The former `1e-6` threshold was below meaningful repeatability for the live Webots
  scene; observed transform drift was on the order of `1e-5 m` / `1e-5 rad`. The revised tolerance
  remains tighter than the millimeter-scale waypoint comparison used by M1. RGB is reported
  separately and is not used as a reset pass/fail threshold.
- **Evidence:** Simulator-free reset tests pass. Webots R2025a reported the startup torso field as
  `-0.0001722422` with legal range `[0.0, 0.35]`, so the baseline is normalized to `0.0` with a
  warning. A live three-episode probe ran with one planned INIT/PREGRASP command sequence, replayed
  unchanged in episodes 2 and 3. Both comparisons passed: worst reset arm-joint delta
  `4.72e-10 rad`, base position `3.12e-8 m`, base orientation `2.96e-7 rad`, CAN position `0 m`,
  commanded joint waypoint delta `0`, achieved waypoint position `2.22e-6 m`, and gripper opening
  delta `1.80e-8 m`. The state sequence was `INIT, PREGRASP` for all three episodes.
- **RGB diagnosis:** The camera's Webots `noise` field was `0.0`. Reset images are read after a
  simulation step (`t=0` is episode-relative), not before the first step/render. Across the two
  replay comparisons, reset RGB mean absolute differences were `0.00171` and `0.00206` (8x8-window
  SSIM `0.999992` and `0.999971`). After 1, 3, and 5 warm-up steps, MAD remained below `0.0061`
  and SSIM above `0.99993`; the difference does not strictly vanish, but it is sparse render
  variation rather than camera noise or an early-frame read bug. Peak per-pixel differences ranged
  from 54 to 92 levels while the image-wide MAD remained below `0.0061`.
- **Reset mode:** `configs/robot.yaml` exposes `reset_mode: soft | reload`, defaulting to `soft`.
  `reload` is intentionally deferred: reloading restarts the Webots controller/server process and
  requires ZMQ client reconnection/retry semantics, so it is not a cheap in-process reset. Selecting
  it currently fails explicitly. Measured soft reset calls took `0.666`, `0.704`, and `0.838 s`
  (mean `0.736 s`) in the probe. Reload timing is unavailable because reload is not implemented.
- **Consequences:** Multi-seed regression results cannot yet be treated as equivalent-start
  comparisons solely on reset grounds; the live behavioral reset gate now passes at the revised
  tolerance. Transit position tolerance is addressed separately in ADR-0006; M1-06 remains pending
  live equivalence and completion runs.

## ADR-0006: Scope relaxed position tolerance to held-object transit

- **Date:** 2026-11-07
- **Status:** Accepted; live regression verification pending
- **Context:** The corrected place-above target has a repeatable IK position error near `0.010475 m`
  with tilt near `0.257°`. The current `0.005 m` grasp tolerance rejects the final point of the
  linear TRANSIT path even though point-level IK is within the existing `0.015 m` general tolerance.
  Grasping, descent, and final placement still require precise positioning.
- **Diagnostic evidence:** With torso at `0.35 m`, the lift endpoint error was `0.000000030 m`.
  Seven straight-line transit waypoint errors were `0.000000158`, `0.000000227`, `0.000000005`,
  `0`, `0`, `0.002703109`, and `0.010473834 m`; the final point is the first point beyond `0.005 m`.
  The endpoint tilt was `0.256616°`. The minimum arm-joint-limit margin reached zero at the last
  waypoints; arm 6 was at its `-1.39 rad` lower bound. The endpoint is within `0.015 m`.
- **Ordered alternatives tested:** IK was warm-started from the preceding waypoint and deterministically
  seeded. A seeded multi-start diagnostic selecting the nearest candidate within grasp tolerance
  found no candidate at or below `0.005 m` at the endpoint. An elevated via-route was also evaluated;
  its intermediate IK errors reached `0.091506345 m` and its final endpoint remained about
  `0.010475 m`, so it did not solve the path under the original threshold. No new transit path was
  adopted.
- **Decision:** Warm-start each path waypoint from the previous joint solution. On an infeasible
  warm start, use deterministic seeded restarts and select the nearest feasible joint solution.
  Return a feasible warm-start solution immediately to avoid unnecessary restart cost. Add
  `ik_tol_transit: 0.015` and use it only for TRANSIT and RETREAT IK. HOME is
  joint-space motion and does not perform IK. Continue enforcing the existing `MAX_TILT_DEG=15`
  limit during transit (including while the CAN is held). Keep `IK_TOL_GRASP=0.005 m` for
  DESCEND, LIFT, PLACE_DESCEND, and other grasp-related moves.
- **Rationale:** Held-object transit does not require the grasp/contact positioning precision needed
  for approach, closure, or final placement. The final placement criterion remains unchanged:
  CAN-to-target distance at most `0.10 m` in the base frame, and the complete episode must reach
  DONE at HOME. This tolerance is not a placement-success substitute.
- **Unchanged constraints:** No placement target, safety bound, joint/velocity limit, or tilt
  threshold changed. M1-06a and M1-07 remain unverified pending their live runs.

## ADR-0007: Bound sequential IK and compare swept transit paths

- **Date:** 2026-11-07
- **Status:** Implemented for Webots-free analysis; live verification pending
- **Context:** The unbounded linear IK solver could jump between IK branches. In the seeded
  full-flow profile, arm 7 jumped `3.159 rad` at DESCEND waypoint 2 and `3.908 rad` at TRANSIT
  waypoint 5. Arm 6 reached its `-1.39 rad` lower limit during the final transit waypoints.
- **Decision:** Add `max_joint_step_rad: 0.15` to `configs/robot.yaml`. Each linear IK waypoint is
  solved within the intersection of real joint limits and a per-joint `q_previous ± 0.15 rad`
  trust region. If no in-region IK solution satisfies the state's existing position and tilt
  tolerances, fail the move with the best-error diagnostic rather than accepting a branch jump.
  All joint-space motor targets, including PREGRASP and HOME, are interpolated with the same
  maximum step. `Kinematics.interpolate_joint_path()` is shared by the Webots and brain paths.
- **Regression evidence:** A reachable pregrasp target solved from a seed offset by `-0.5 rad` on
  arm 1 had a feasible unrestricted solution requiring a `0.468583 rad` largest joint delta. The
  bounded solve failed clearly with a best position error of `0.144119 m`. The real scripted
  Cartesian sequence passed its IK checks with every adjacent joint step at or below `0.15 rad`.
  The current target still has zero minimum joint-limit margin: arm 6 reaches its `-1.39 rad`
  limit. The bounded solver prevents branch jumps but does not create limit margin.
- **Continuity audit:** With the legal initial arm pose and seed 0, the full INIT-to-HOME IK profile
  contained 48 records, including stationary INIT, CLOSE, and RELEASE records. No waypoint exceeded
  `0.15 rad`; 16 were flagged because the joint-limit margin was below `0.05 rad` (including
  stationary states at a limit). TRANSIT had seven such flags, with arm 6 at its lower limit from
  waypoint 5 through 7. The full per-joint table is reproducible with
  `uv run python scripts/audit_can_path.py`. This is a Webots-free IK profile, not a physical motion
  or collision test.
- **Metric reconciliation:** The latest seed-0 output from
  `uv run python scripts/audit_can_path.py` is `0.010475481 m` endpoint error and `0.256656122°`
  endpoint tilt. It supersedes earlier profile readings `0.010473834 m / 0.256616°` and
  `0.010475343 m / 0.256653°`; those are historical results from prior solver/RNG executions, not
  values from the current deterministic path audit.
- **Historical transit comparison (object-clearance numbers superseded; UNVERIFIED (assumed radii)):** With the approved target and a 0.02-rad FK sweep resolution, the Cartesian path had
  maximum command step `0.150000 rad`, minimum joint-limit margin approximately `0 rad`, maximum
  tool tilt `0.263349°`, minimum payload/table clearance `0.120158 m`, and minimum projected
  payload/object clearance `0.116168 m` (hammer). Joint-space interpolation used five segments,
  maximum step `0.137162 rad`, minimum joint-limit margin approximately `0 rad`, maximum tilt
  `2.192890°`, minimum payload/table clearance `0.112915 m`, and minimum projected
  payload/object clearance `0.118443 m` (hammer). Both paths remain within the configured
  `0.15 rad` step and `15°` tilt limits, with positive estimated payload clearances. Object checks
  use horizontal projection proxies inferred from Webots R2025a scene geometry; the arm links,
  dynamic collisions, and exact swept volumes are not modeled.
- **Recommendation:** Keep the Cartesian line for now. It has about `7.2 mm` more estimated
  tabletop clearance and about `1.93°` less peak tilt, while meeting the same joint-step cap. The
  joint-space remedy is implemented in the shared interpolation helper and evaluated in the
  Webots-free swept-path analysis, but is not selected as the default transit strategy.
- **Historical placement search (geometry screen superseded; UNVERIFIED (assumed radii)):** A 0.10-m grid over the factory tabletop applied a 0.10-m edge inset,
  0.15-m spacing from the starting CAN, and 0.15-m projected surface clearance from other objects.
  Of 534 geometrically eligible points, 16 passed the sampled IK/trust-region descent checks.
  Excluding the fixed LIFT start pose when scoring target-specific route margin, the best three
  (world coordinates in meters) were:
  1. `(-0.188828, -4.190293, 0.821100)`: minimum margin `0.043619 rad`, maximum waypoint delta
     `0.149989 rad`; base `(0.466393, 0.381651, 0.725361)`.
  2. `(-0.688828, -4.190293, 0.821100)`: minimum margin `0.031087 rad`, maximum delta
     `0.149961 rad`; base `(0.481472, -0.117945, 0.738648)`.
  3. `(-0.588828, -4.190293, 0.821100)`: minimum margin `0.028296 rad`, maximum delta
     `0.149998 rad`; base `(0.478456, -0.018026, 0.735990)`.
  The current target does not have the requested `0.05 rad` margin; its target-route margin is
  approximately zero. No candidate or target was applied. All candidates remain research
  diagnostics, not validated safe placements.
- **Comparison tooling:** `scripts/prepare_original_comparison.py` prepares a temporary original
  worktree with the approved world target, frame conversion, IK seed 0, stop-after-LIFT behavior,
  external-controller world setting, and JSONL state/waypoint trace schema. The CLI
  `compare-traces` reports the first divergent state/waypoint. Synthetic trace tests cover match,
  joint divergence, and waypoint-count mismatch. These tools do not constitute live M1-06a
  evidence; M1-06a and M1-07 remain unverified.

## ADR-0008: Define M1-06a equivalence at state boundaries and Cartesian targets

- **Date:** 2026-11-07
- **Status:** Superseded by ADR-0009
- **Context:** The brain path splits PREGRASP and HOME joint-space moves into commands no larger than
  `0.15 rad`, while the original controller issued each joint-space target as one command. Comparing
  every command makes the regression fail on a known execution-shape difference rather than a
  difference in the commanded state outcome. DESCEND and LIFT remain Cartesian moves with shared
  targets.
- **Decision:** M1-06a compares the state sequence, final commanded joint positions and commanded
  end-effector target at each state boundary, plus the named Cartesian targets for DESCEND and LIFT.
  Intermediate PREGRASP and HOME joint interpolation count and intermediate commands are intentional
  deviations and are excluded from equivalence. The final PREGRASP/HOME boundary targets remain
  compared. The trust-region cap and Cartesian transit behavior are retained.
- **Geometry decision:** The audit uses collision geometry from Webots R2025a `Can.proto`,
  `Wrench.proto`, `Hammer.proto`, `Apple.proto`, and `Screwdriver.proto`. The remote
  `PaintBucket.proto` could not be retrieved during this task, so its candidate exclusion uses the
  conservative 0.20 m radius around the object center. Geometry-screen clearances and candidate
  exclusions that include this assumed radius are explicitly marked
  `UNVERIFIED (assumed radii)`. No placement target or candidate is changed or applied.
- **Housekeeping:** Ignore `worlds/.factory.wbproj`, which Webots writes as local user state.
- **Consequences:** M1-06a trace tooling now validates behavior at the documented contract points;
  this changes the acceptance comparison semantics, not controller behavior or safety thresholds.
  M1-06a and M1-07 remain unverified until their live evidence is reviewed.

## ADR-0009: Withdraw M1-06a joint-equality criterion

- **Date:** 2026-11-08
- **Status:** Accepted; live v2 batch pending
- **Context:** Six pairwise `compare-traces` runs over clean original traces 01, 02, 03, and 05
  showed the old `1e-3 rad` boundary-joint equality requirement was not self-consistent. Five of
  six pairs first diverged at PREGRASP, during the first IK solve. Across the clean originals, the
  maximum pairwise joint differences were `0.319572 rad` at PREGRASP, `4.078076 rad` at DESCEND,
  `4.095043 rad` at CLOSE, and `4.139445 rad` at LIFT. The largest differences were arm_7. In
  contrast, all commanded boundary EE target and DESCEND/LIFT Cartesian waypoint differences were
  at most `8.1e-15 m`; achieved boundary EE position differences were at most `6.78e-6 m`. The
  `02-05` pair was bit-for-bit equal in recorded joints.
- **Seed audit:** Fixture v1 sets `self.rng = np.random.default_rng(0)` in `Kinematics.__init__`.
  `tiago_controller.py` constructs `Kinematics` before the task run; no solve occurs before
  PREGRASP; and `Motion.go_to()` calls `solve(..., restarts=True)`. Thus random restart candidates
  for the first PREGRASP solve are drawn from the seeded generator. The first observed joint
  divergence is at that PREGRASP boundary, not a later Cartesian state. The trace does not record
  torso state, so the exact physical source of the solver branch variation is not established.
- **Decision:** Withdraw joint equality as an M1-06a acceptance criterion. Equivalence requires the
  original trace to have exactly the expected stop-after-LIFT prefix and the brain trace to start
  with that prefix; commanded and achieved EE boundary positions and named DESCEND/LIFT Cartesian
  targets must match within `1e-3 m`; all recorded prefix IK tilts must remain within the existing configured
  tilt limit. Report per-run maximum absolute joint step per joint, minimum joint-limit margin, and
  arm_7 consecutive-command excursions over `1 rad` as diagnostics, without a joint-equality gate.
- **Fixture v2:** Keep the original IK, controller behavior, posture, torso/base settings, target,
  limits, and tolerances unchanged. Extend opt-in traces with CAN world/base positions, displacement
  and lift height, gripper/grasp/stall diagnostics, joint limits, IK error, and tool tilt. The old
  run 04 trace remains untouched and labelled **anomalous: CLOSE->RETRY, CAN displaced**. Failed or
  anomalous traces are renamed and retained, never deleted. Recollect the five-run original batch
  under new `m1-06a-original-v2-01` through `-05` filenames because v1 traces lack these fields and
  run 04 is not an isolated clean episode.
- **Guarantees and status:** This gate still checks prefix state progression, task-space command
  targets, Cartesian approach/lift geometry, and the existing tilt cap. It does not assert joint
  posture equivalence, grasp success, lift success, placement success, or collision safety. M1-06a
  and M1-07 remain unverified; posture/torso/base studies remain unapplied and M2 has not started.

## ADR-0010: Pace joint waypoints using measured completion

- **Date:** 2026-11-08
- **Status:** Accepted; Webots verification pending
- **Context:** The brain sent a new interpolated joint target after one 32 ms simulator step even
  when arm motors were configured to move at 0.5 rad/s. Measured arm_7 motion was about 0.016 rad
  per step while commanded targets advanced about 0.149 rad. The safety layer compared each next
  target against measured position and rejected the accumulated tracking error.
- **Decision:** Match the original wait semantics. Joint-space PREGRASP/HOME waypoints wait until
  every arm joint is within 0.01 rad (PREGRASP) or 0.02 rad (HOME), with the original 800-step
  timeout. Cartesian linear paths use
  3 cm waypoints and wait to 0.03 rad at intermediate waypoints and 0.01 rad at the final waypoint,
  with the original 600-step timeout. Every wait checks measured `RobotState.joint_pos`. A
  configurable `joint_stall_steps` watchdog (default 10) fails the active state with the exact
  reason `joint stall` when no arm joint moves for that many steps. Safety limits, command horizon,
  IK tolerances, target poses, and motor velocity settings are unchanged.
- **Intentional motion deviations:** PREGRASP and HOME retain the brain's 0.15 rad joint
  interpolation, now settling every command; the original issued one joint target and waited for
  completion. DESCEND, LIFT, TRANSIT, PLACE_DESCEND, and RETREAT use 3 cm Cartesian waypoints,
  sequential warm starts, and the already accepted 0.15 rad IK trust region. The original linear
  solver did not impose that trust region or use random restarts. The brain also has a fixed scripted
  sequence rather than the original retry/alignment/grasp-verification state machine. These
  differences are documented; this repair does not claim full controller equivalence.
- **Seed parity:** The comparison fixture seeds the original IK RNG at 0; the brain M1 trace runs
  with seed 0 and resets the same `default_rng(0)`. Both first PREGRASP solves use the same reset
  arm posture and the configured 0.35 m torso target: the original settles the torso to within
  0.002 m before capturing `home_q`, and the brain constructs its IK vector from reset arm sensors
  plus configured torso height. Protocol v3 now exposes the torso position sensor as
  `RobotState.diagnostics["torso_lift_joint_sensor_rad"]`, so a live trace can check the physical
  torso against the configured height; this was not traceable when this decision was recorded.
  Random restart candidates
  are drawn in the same arm-joint order. IK candidate stopping and selection rules differ and remain
  an intentional implementation difference.
- **Trace semantics:** `measured_joint_positions_rad` is sensor feedback and
  `commanded_joint_target_rad` is the latest requested joint target. The old
  `commanded_joint_positions_rad` remains a deprecated measured-position alias for older trace
  readers. Original fixture v2 writes all three fields; the comparer prefers explicit values.
- **Timing:** Policy and CLI report per-state simulation duration, cumulative waypoint-settle
  simulation duration, and wall-clock duration. Wall time includes IK and transport overhead. No
  live Webots timing claim is made before recaptured traces are reviewed.
- **Consequences:** Webots-free rate-limited tests reproduce the old safety rejection and exercise
  paced PREGRASP→DESCEND→CLOSE→LIFT, a stalled-joint failure, and DESCEND seed continuity. Original
  and brain trace batches must be recollected before a new equivalence claim. All M1 gates remain
  unverified.

## ADR-0011: PREGRASP contact and base diagnostics; guarded clearance route

- **Date:** 2026-11-08
- **Status:** Accepted for instrumentation; guarded route and Webots behavior unverified
- **Context:** A paced PREGRASP trace showed substantial base displacement and an end-effector
  position near/below the factory table. Existing traces do not contain torso sensor values,
  contacts, or arm-link ground truth, so the proposed joint-sweep/table-contact cause was not yet
  established from telemetry.
- **Decision:** Preserve the default `pregrasp_route: interpolated`; add opt-in `safe_via`, which
  deterministically searches joint configurations using FK and requires both joint-interpolated
  segments to keep the sampled elbow, wrist, tool, and hand points at least
  `table_clearance_required_m` (default 0.05 m) above the table plane when inside the footprint. No
  safety thresholds, command horizon, IK tolerances, targets, or motor speed are changed. Add a
  config-driven reset-pose invariant (0.005 m planar, 0.5 deg yaw) that fails the active state with
  `base moved` and includes observed contact-node context.
- **Single table geometry source:** `configs/robot.yaml` now holds `table_top_height_m` (0.75),
  `table_footprint_centre_xy_m` `[-0.23882837, -4.8402928]`, `table_footprint_size_xy_m`
  `[4.5, 1.5]`, and `table_clearance_required_m` (0.05). The runtime guard in
  `scripted_can.py` and `scripts/analyze_pregrasp_table.py` both read these values, which removes
  the previous split where the analysis script used `top + 0.03 m` and the runtime guard used a
  hardcoded `0.80 m` plane. Centre, size, and top are read from the `Table` instance in
  `worlds/factory.wbt` and the R2025a `Table.proto` (top surface at `size.z`, footprint
  `size.x x size.y`). The 0.05 m requirement itself is a policy value and remains
  **UNVERIFIED (inferred)**.
- **Reset posture:** The checked-in Webots TIAGo PROTO leaves the torso joint-position field at its
  default 0.0 m, while `torso_height` is 0.35 m. Brain reset now commands torso/head and waits for
  the torso sensor to be within 0.002 m; the head receives the configured pan/tilt and a two-second
  settle window. Existing traces do not include the torso sensor, so physical reset behavior is
  still unverified until a watched run records it. Original `initialize_posture` captures home arm
  posture and enables contact tracking after optional tool calibration. Task `INIT` builds
  waypoints, records initial CAN height, and opens/primes the gripper. The brain still skips explicit
  gripper priming, optional calibration/gripper test, initial-height bookkeeping, and later retry,
  alignment, grasp-verification, and recovery flow.
- **Telemetry contract:** `RobotState.diagnostics` is optional read-only backend telemetry and does
  not affect safety validation. It carries torso sensor, base drift, wheel mode/target/velocity/
  encoder, non-robot contact node/type/world points, and arm_7 FK versus Supervisor joint position.
  The failure boundary now uses the last stepped observation. Added telemetry changes strict
  dataclass wire fields, so protocol version advances from 2 to 3. Protocol v1/v2 peers are
  incompatible until both processes are restarted together.
- **FK versus ground truth:** The arm_7 comparison now uses `RobotIO.joint_anchor_world_position`,
  i.e. the world position of the `arm_7_joint` HingeJoint node. IKPy's full forward kinematics
  returns the transform of that joint's link frame, whose origin is the same joint anchor, so the
  two values describe the same point and can disagree. The previous implementation read the
  `jointParameters.position` field with `getSFVec3f()`; that field is an `SFFloat`, so the read
  always raised and the value silently came from the same Supervisor node, making the comparison
  unable to fail. The `endPoint` node is a different point and is used only as the Supervisor
  tool reference (`arm_link_world_xyz_m["tool"]`), never as FK ground truth. A configurable
  `fk_ground_truth_tolerance_m` (default 0.01 m) drives an explicit
  `arm_7_joint_fk_ground_truth_mismatch` flag; unavailability is reported as `None`, not zero.
- **Geometry and Webots-free analysis:** The factory table is at
  `(-0.23882837,-4.8402928)` with footprint `4.5 x 1.5 m` and top `z=0.75 m` from
  `worlds/factory.wbt` plus the R2025a `Table.proto`. FK sweeps using the existing PREGRASP joint
  endpoints and reset base pose predict that at torso 0.35 m the synchronized path has minimum
  hand-adjusted clearance margin 0.041478 m relative to `top + 0.05 m`, and original-rate
  independent motion has 0.054233 m; neither crosses the table. At torso 0.0 m both paths penetrate
  the clearance threshold: synchronized first at sample 3 (hand-adjusted margin -0.246566 m; world
  `(-1.4895,-4.0504,0.6334)`), original-rate first at sample 10 (margin -0.171349 m; world
  `(-1.5226,-4.0213,0.7087)`). These margins were previously reported against `top + 0.03 m` and are
  0.02 m smaller under the unified requirement; no geometry or path changed. The nominal trace lacks
  torso feedback, so the torso-0.0 sweep is a plausible explanation of the observed low EE height,
  not proof of the actual contact cause.
- **Candidate route estimate:** On the recorded reset pose, FK selected via target
  `(1.958,0.860,-1.714,1.910,0.356,-0.913,1.236) rad`. Estimates at torso 0.35 m: interpolated
  route minimum clearance 0.041478 m above the `top+0.05 m` threshold, max joint step 0.1372 rad,
  minimum arm joint-limit margin 0.000953 rad, simulated motor-time estimate 2.304 s. `safe_via`
  minimum clearance 0.045238 m, max joint step 0.1469 rad, minimum limit margin 0.000953 rad,
  estimated motor time 6.336 s. The unchanged reset posture is arm_1's limiting point for both
  routes. Estimates ignore Webots collision dynamics and table/arm collision geometry beyond the
  sampled points; they are not evidence of safe execution.
- **Evidence and status:** `scripts/analyze_pregrasp_table.py` reproduces the sweep using the same
  configured table geometry and clearance value as the runtime guard. Existing
  `m1-watched-paced-02.jsonl` confirms the old failure boundary incorrectly used reset time 0.0
  and reset base/joint values; no existing run includes contact, wheel, torso, or joint-node
  telemetry. Brain traces now also record, per boundary and waypoint,
  `supervisor_arm_link_world_xyz_m` (elbow `arm_4_joint`, wrist `arm_6_joint`, tool
  `arm_7_joint` endPoint), `supervisor_min_table_clearance_m` for links inside the footprint,
  `arm_7_joint_fk_ground_truth_mismatch`, and the table threshold values. Each field is `None`
  when the Supervisor value is unavailable. Instrumentation is testable without Webots; live
  traces must validate contact tracking, FK/ground-truth alignment, torso reset, the invariant,
  and the table geometry. All M1 gates remain **UNVERIFIED**.

## ADR-0012: Pose-only Supervisor lookups, reset fingerprint, and INIT posture replication

- **Date:** 2026-11-08
- **Status:** Accepted for instrumentation; live Webots behavior unverified
- **Context:** The arm_7 FK ground-truth diagnostic called `getPosition()` on the joint node,
  which Webots rejects for non-Pose nodes. The soft reset restores many but not all Webots state
  values, and there was no single record proving two same-seed episodes start identically. The
  torso is raised by the reset path rather than by brain INIT, and the original ordering and
  settle thresholds were not yet reproduced exactly.
- **Decision — Pose-only lookups:** Remove `RobotIO.joint_anchor_world_position`. Arm-link
  telemetry uses the Pose-derived child `Solid` in the joint's `endPoint` field. Lookup failures
  (`AttributeError`, `TypeError`, `ValueError`, `RuntimeError`, missing node/field, non-finite
  value) return `None` and are cached so each distinct failure warns once. At reset the backend
  records the world offset `endPoint - FK(arm_7)`; each sample reports the change in that offset,
  so a constant tool offset is not an error and only offset change flags an FK mismatch.
- **Decision — Reset fingerprint:** `RobotIO.reset_fingerprint()` captures every motor target,
  velocity, and control mode (arm, torso, head, fingers, wheels), every joint sensor, the base
  world pose plus `getVelocity()`, and the CAN world pose. It is emitted only on the reset
  observation and written by the INIT-enter boundary. The torso "before" value is exposed
  separately as `reset_torso_before_rad` because it legitimately differs between resets; the
  fingerprint records the post-settle torso value and commanded target. Two same-seed episodes in
  one server session must produce identical fingerprints; `cli repeat-check` enforces this.
- **Decision — INIT posture:** Reproduce the original ordering exactly. Torso is commanded first
  and stepped up to 1000 times until within 0.002 m; the head is then clamped to its limits, set
  to velocity 0.5, and given a two-second settle. Sources:
  `controllers/tiago_arm_api/tiago_arm_api.py:357-364` (torso) and `:366-374` (head); packaged
  copy `src/vlarobot/backends/webots/pick_place.py:121-145`. Implementation:
  `src/vlarobot/backends/webots/backend.py` reset plus `_settle_reset_posture`. New config keys
  are `reset_torso_settle_tolerance_m: 0.002`, `reset_torso_timeout_steps: 1000`, and
  `reset_head_settle_ms: 2000`. Brain INIT seeds IK from the settled torso sensor. A completed
  soft reset with default config always re-commands 0.35 m; ~0 is possible only if reset is never
  called, aborted before/during the settle (which raises), the torso joint is absent, or
  `torso_height` is configured to 0/None.
- **Decision — Drift analysis and thresholds:** Add `trace_compare.base_drift_by_state()` and
  `scripts/analyze_base_drift.py` (Webots-free). Available-trace evidence: healthy runs drift
  ≤3.37 mm / 0.08°; the watched paced runs drift 120.71 mm / 43.27° and 129.55 mm / 42.51°.
  Keep the hard invariant at 0.005 m / 0.5° until the user approves a change; optionally add a
  4 mm / 0.1° soft warning. Drift rate is diagnostic only because the pathological runs drift at
  2.19 mm/s, lower than a short-window healthy sample.
- **Decision — Opt-in timings:** Wall-time counters are off by default and change no behavior.
  Enable with `ScriptedCanPolicy(timing_enabled=True)`, `can-regression`/`repeat-check --timings`,
  or `VLAROBOT_BACKEND_TIMINGS=1` for the backend diagnostics counter. Buckets are
  `backend_reset_s`, `ik_solve_s`, `backend_apply_s`, `backend_step_s`, `trace_write_s`, and
  `supervisor_s`; ZMQ round trips are included in apply/step, not broken out. No live share is
  claimed.
- **Decision — Fixture additions:** The original-comparison fixture stays under
  `VLAROBOT_COMPARISON_FIXTURE_V2` and additionally logs torso/head sensors and base linear/angular
  velocity. No v3 base logging is needed because fixture v2 already logs
  `base_world_xyz_m`/`base_world_orientation_matrix`. `compare_traces()` checks INIT torso
  (2e-3 m) and head (0.02 rad) posture.
- **Evidence and status:** `ruff`/format pass, `ty` has no new errors in touched files, and
  `pytest -m "not webots"` reports 90 passed. No live Webots run was performed; warning-free
  Supervisor calls, torso/reset posture, fingerprint identity, repeatability, drift, and timing
  shares remain **UNVERIFIED**. M1-06a, M1-07, and all M1 gates remain **UNVERIFIED**.

## ADR-0013: Rotating FK offsets and classified reset-fingerprint comparison

- **Date:** 2026-11-08
- **Status:** Accepted; live confirmation pending
- **Context:** The first live repeat run showed `arm_7_joint_fk_ground_truth_mismatch: true` with
  an error growing from 0 to 0.065779936 m, essentially the 0.067725 m tool offset. The reset-time
  offset `endPoint − FK(arm_7)` was stored in world coordinates and compared with a later world
  offset without rotating it with the wrist. The same run showed that two same-seed consecutive
  episodes are not bit-identical: commanded motor state matches exactly, while joint sensors and
  poses differ by ~1.5e-4 and base/CAN velocities differ by up to 7.2e-3 because the first reset
  follows world load.
- **Decision — rotate the FK offset:** `_arm7_fk_world_pose()` returns the arm_7 link world
  position and rotation. Reset stores both the world offset and `_arm7_reset_link_rotation`; each
  sample predicts the current offset as `(R_now @ R_reset.T) @ reset_offset` and reports the
  residual. A constant tool offset cancels under wrist rotation. A 90° wrist rotation with a
  correctly rotating endpoint yields zero error; an endpoint that does not rotate is flagged. The
  trace now also records the offset, reset offset, offset change, ground-truth source,
  validation-availability flag, and INIT `reset_torso_before_rad`/`reset_torso_after_rad`.
- **Decision — classified fingerprint comparison:** `repeat-check` classifies each fingerprint
  leaf. `motors.*` (targets, motor velocities, control modes) is the deterministic contract and
  must match exactly. Sensors and poses are measured and fail only above `--measured-tolerance`
  (default 1e-3). `base.velocity_*` and `base.angular_velocity_*` are world-load transients and are
  reported via `max_transient_delta` but never fail. Observed two-episode result:
  `commanded=0 measured=0 transient=6 max_measured_delta≈1.5e-4 max_transient_delta≈7.2e-3`
  (PASS). This changes the comparison semantics, not reset behaviour.
- **Decision — drift thresholds unchanged:** Three fresh-world interpolated runs drift
  3.36/3.41/3.41 mm at 0.076° with torso 0.349745→0.349747 rad. `safe_via` failed at 5.0016 mm
  (1.56 µm over 5 mm) at 2.816 s with no contacts; drift is monotonic at ~1.78 mm/s vs ~0.98 mm/s
  for interpolated. The 5 mm / 0.5° invariant is kept until the user approves a change. Options
  recorded: keep 5 mm and reject `safe_via`; raise to 8 mm (still ~15x below the 120 mm watched
  failure); or use route-specific thresholds (not recommended).
- **Evidence and status:** Fresh runs confirm the Pose-only lookup produced no
  `wb_supervisor_node_get_position` warning; remaining console messages are expected reset-baseline
  clamps and Webots' unrelated `Unsupported file type` warning. `ruff`/format pass, `ty` has no
  errors in touched files, and `pytest -m "not webots"` reports 97 passed. Live confirmation that
  the FK error returns to ~0 after the rotation fix is still required. M1-06a, M1-07, and all M1
  gates remain **UNVERIFIED**.

## ADR-0014: Two-tier base-drift invariant, wheel hold, and per-state target refresh

- **Date:** 2026-11-08
- **Status:** Accepted; thresholds provisional pending original-controller fixture evidence
- **Context:** A single 5 mm base-drift limit failed `safe_via` by 1.56 µm at PREGRASP even though
  the drift was monotonic creep (~1.78 mm/s), not a collision. The trace also showed both wheel
  motors in Webots position control at target 0 with a 10.1523 rad/s velocity limit, and the
  original controller never acquires or commands a wheel device (device setup at
  `controllers/tiago_arm_api/tiago_arm_api.py:57-76`). The brain converted world-anchored CAN and
  place targets to the base frame once at INIT, while the original re-reads `can_base_pos()` in
  INIT, CLOSE, and LIFT (`:354-355`, used at `:406`, `:436`, `:441`, `:459`, `:464`).
- **Decision — two tiers plus a rate guard:** warn at 5 mm / 0.5° once per state via
  `base_drift_warning` and never fail on the warning alone; fail at 15 mm / 0.5° with reason
  `base moved`; fail when the mean drift rate over at least 0.5 s exceeds 10 mm/s. All six values
  are config keys (`base_drift_warn_*`, `base_drift_fail_*`, `base_drift_rate_*`) and remain
  provisional until the original-controller fixture confirms them. `base_drift_rate_m_per_s` on
  `state_boundary` is now the window rate, not the episode-average rate.
- **Decision — optional wheel hold:** `hold_base: none | wheels_zero_velocity` (default `none`).
  `wheels_zero_velocity` switches both wheels to velocity control at target 0 and is applied after
  `simulationResetPhysics()` and re-asserted after the reset settle. The default preserves the
  PROTO position hold and changes no behaviour.
- **Decision — per-state target refresh:** `_base_frame_targets()` re-derives the world-anchored
  CAN/place targets from the current base pose before each state. LIFT keeps the INIT pregrasp to
  match the original's fixed lift target. The safe-via planner uses the current base pose. With
  zero drift the targets are identical; under drift they track the world anchors. IK tolerances
  and safety limits are unchanged.
- **Evidence:** Existing `--stop-after PREGRASP` traces report interpolated max drift
  3.36/3.41/3.41 mm at 0.08° with max boundary rate 1.11–1.24 mm/s, and `safe_via` 5.00 mm at
  0.07° with max rate 2.50 mm/s. Under the new invariant interpolated is silent and `safe_via`
  warns without failing. `ruff`/format pass, `ty` has no errors in touched files, and
  `pytest -m "not webots"` reports 102 passed.
- **Remaining evidence:** Live drift with `hold_base: wheels_zero_velocity`, full-flow per-state
  drift, original-controller fixture traces, and confirmation that the FK error returns to ~0
  after the rotation fix all require operator Webots runs. M1-06a, M1-07, and all M1 gates remain
  **UNVERIFIED**.
- **Supersedes:** The drift-threshold portion of ADR-0013 (5 mm hard limit). The FK-offset
  rotation and classified fingerprint decisions in ADR-0013 remain in force.

## ADR-0015: Open the gripper at INIT before PREGRASP

- **Date:** 2026-11-08
- **Status:** Accepted; full flow confirmed 1/1 in `hold_base: none` (`runs/hold-none-02.jsonl`)
- **Context:** The M1 full flow failed `can-regression` (`success=0/1`, final distance 0.246 m)
  with the base-drift invariant silent. `runs/hold-none-01.jsonl` shows finger sensors at ~1.4e-6 m
  at INIT, `can_displacement_world_xyz_m ≈ [0.0199, -0.0198, 0]` during DESCEND, a CLOSE opening
  of ~6.3e-6 m, and a LIFT height of ~0. `ScriptedCanPolicy` only commanded `GripperCmd(0.0)` at
  CLOSE and `GripperCmd(GRIPPER_OPEN)` at RELEASE, so DESCEND pushed the can instead of straddling
  it. The original opens the gripper at INIT (`controllers/tiago_arm_api/tiago_arm_api.py:414`,
  `open_gripper(cycle=not gripper_primed)`) and aborts the run if a finger does not reach
  `OPEN_POS` within 2 mm (`:303-326`).
- **Decision:** `ScriptedCanPolicy._open_gripper()` runs at INIT before `_base_frame_targets()`:
  sends `GripperCmd(GRIPPER_OPEN)` where `GRIPPER_OPEN = gripper_opening_limits[1]` (0.045 m), steps
  up to the new `gripper_open_timeout_s` config key (default 4.0 s), and accepts the opening within
  `GRIPPER_OPEN_TOLERANCE_M = 0.002` m. A rejected command fails INIT with the safety reason; a
  timeout fails INIT with `gripper failed to open` plus a `gripper_timeout` trace event. The
  hard-coded `0.045` in the RELEASE command is replaced by `GRIPPER_OPEN`.
- **Scope:** The drift veto still starts at PREGRASP. INIT open steps update `max_base_drift_m`
  and `max_base_drift_yaw_deg` but are not vetoed, matching the original's pre-motion open. No IK
  tolerance, safety limit, or workspace bound changes.
- **Evidence:** `ruff check .` and `ruff format --check .` pass; `ty check` adds no diagnostics in
  touched files; `pytest -m "not webots"` reports 106 passed, including
  `test_policy_opens_gripper_at_init_before_descend` (closed-finger fake backend ramping at
  `FINGER_SPEED`; opening is within 2 mm of `GRIPPER_OPEN` at INIT exit and DESCEND enter) and
  `test_policy_fails_init_when_gripper_cannot_open` (stuck fingers abort INIT).
- **Remaining evidence:** Confirmed in `hold_base: none` (`runs/hold-none-02.jsonl`: 1/1, CAN lifts
  40.7 mm). `hold_base: wheels_zero_velocity` still fails, but for the base-tilt reason recorded in
  ADR-0016, not the gripper. The original-controller fixture, final drift thresholds, and the FK
  diagnostic frame remain **UNVERIFIED**, as do M1-06a, M1-07, and all M1 gates.
- **Supersedes:** Nothing. Extends ADR-0014, which remains in force.

## ADR-0016: Re-align the approach pose after the base tilt settles

- **Date:** 2026-11-08
- **Status:** Accepted; live `can-regression` confirmation pending operator run
- **Context:** The post-fix full flow succeeds in `hold_base: none` (1/1, final distance 0.042 m)
  but still pushes the CAN 42 mm during DESCEND. `runs/hold-none-02.jsonl` and
  `runs/hold-velocity-02.jsonl` show why: the base rocks by −1.30° roll / +1.48° pitch while the
  arm extends during PREGRASP, while planar drift stays at 3.35 mm and yaw at 0.08°. Because the
  arm is commanded in the base frame, that rock moves the CAN 27.9 mm in base coordinates, so the
  fingertip finishes PREGRASP 32.6 mm off-centre. The open gripper has only 13.3 mm per-side
  clearance around the 63.5 mm CAN. The per-state target refresh (ADR-0014) makes DESCEND target
  the live CAN position, but the arm starts the descent from the stale PREGRASP pose, so it
  approaches diagonally and clips the CAN. The successful run's final distance is 0.0415 m and its
  total CAN displacement is 0.2484 m; the two are not equal (the earlier "0.042 = 0.042" claim
  compared the final placement residual with only the x component of the displacement). Both are
  nevertheless produced by the same eccentric grasp: the tip is 43.1 mm off-centre at CLOSE and
  the DESCEND push is 42.3 mm. LIFT also reused the INIT pregrasp, dragging the CAN sideways.
- **Decision — re-align at the end of PREGRASP:** `_realign_pregrasp()` re-derives the pregrasp
  point from the live base pose, records `pregrasp_realign` with the XY error, and when the error
  exceeds the existing `align_tol` (5 mm) solves IK for the refreshed point and moves there
  (`pregrasp_realigned` with before/after error). It runs after the arm extension, where the tilt
  has settled, so DESCEND starts centred. If IK fails the run fails with the existing IK reason.
- **Decision — the realign path is behind `pregrasp_realign`:** the YAML key (default `true` for
  M1-07) and CLI `--pregrasp-realign/--no-pregrasp-realign` gate the PREGRASP re-align, the
  refreshed LIFT, and the CLOSE guard together. `false` preserves the original controller
  semantics for M1-06a equivalence runs: no extra waypoints or events. Traces record the flag on
  every `state_boundary`; `compare-traces` reports both flags and rejects a mixed pair as a
  documented deviation rather than a mismatch.
- **Decision — CLOSE guard:** before the fingers close, `_guard_close_alignment()` measures the
  live base-frame fingertip→CAN xy error (`close_alignment_check`). If it exceeds
  `close_align_tol` (0.008 m) the policy performs exactly one refined descent from the refreshed
  grasp target (`close_realign`, before/after error); if the residual is still over tolerance the
  episode fails with `failure_reason=misaligned at close` (`close_misaligned`). This is a
  measurement-based veto derived from the same live pose as the realign, not a new tolerance on
  the arm.
- **Decision — LIFT rises from the refreshed pregrasp:** the LIFT target is re-derived from the
  current base pose after the realign and recorded as `lift_target_refreshed`. This supersedes the
  "LIFT keeps the INIT pregrasp" portion of ADR-0014, which dragged the CAN by the same offset.
- **Decision — success is four separate criteria:** `home_reached`, `place_distance_ok`
  (base-frame distance ≤ `place_tolerance_m` = 0.10 m), `can_height_ok` (CAN z within
  `can_place_z_tolerance_m` = 0.02 m of the INIT table-rest height), and `can_tilt_ok` (CAN tilt
  < `can_tilt_limit_deg` = 15°). All four are required and each is printed by `can-regression` and
  written on the `DONE` trace. No threshold was relaxed.
- **Decision — tilt is telemetry, not a veto:** every `state_boundary` records
  `base_tilt_roll_deg` / `base_tilt_pitch_deg` since reset, `base_drift_warning` and `base_moved`
  carry them, `ScriptedCanResult.max_base_tilt_deg` is printed by `can-regression`, and
  `scripts/analyze_base_drift.py` reports a `tilt=` column. The veto keeps using planar drift and
  yaw only: a 1.5° tilt is normal for this route, so reusing the 0.5° yaw limit would fail every
  episode. Whether a separate tilt threshold is needed remains open.
- **Evidence:** `ruff check .` and `ruff format --check .` pass; `uv run ty check src tests` is
  clean (0 diagnostics); `pytest -m "not webots"` reports 119 passed, including a fake backend
  that rocks the base 1.3°/1.5° mid-PREGRASP (realign fires, residual < 5 mm, DESCEND starts
  < 5 mm from the CAN in world), ON/OFF flag traces, a no-tilt case (no extra move), a nudged-CAN
  guard success, a dragged-CAN `misaligned at close` failure, and the four-criterion result and
  comparer flag tests.
- **Remaining evidence:** Operator re-run of `can-regression` with `hold_base: none` to confirm the
  CAN is no longer pushed during DESCEND and the final distance drops from 42 mm. The
  original-controller fixture, final drift thresholds, and the FK diagnostic frame remain
  **UNVERIFIED**, as do M1-06a, M1-07, and all M1 gates.
- **Supersedes:** The LIFT-target portion of ADR-0014. The two-tier drift invariant, `hold_base`
  default, and per-state refresh remain in force.

## ADR-0017: `ty` scope is `src tests`, not the whole repository

- **Date:** 2026-11-08
- **Status:** Accepted
- **Context:** The "22 diagnostics" baseline was produced by whole-repository
  `uv run ty check`. All 22 are in files this project does not own as library code: 15 in
  `controllers/`, 3 in `model/`, and 4 in `scripts/analyze_pregrasp_table.py` (Webots
  controller/proto and standalone scripts). At `ff041ff`, the scoped command
  `uv run ty check src tests` reported **4 diagnostics, all in tests**: 3 in
  `tests/test_scripted_can_policy.py` (two `diagnostics` assignment types and a `Pose3D | None`
  attribute) and 1 in `tests/test_webots_backend_reset.py` (`FakeNode.getContactPoints`). The
  current tree reports **0** for the same command; the pre-existing whole-repo 22 are unchanged
  unless a change touches those files.
- **Decision:** The per-task gate is `uv run ty check src tests`. A whole-repo run is reported as
  a separate baseline, and any new diagnostic in `src` or `tests` is a regression to fix. Adding
  diagnostics to `controllers/`, `model/`, or `scripts/` is not silently accepted: if a change
  touches one of those files, that file's diagnostics are reported by name.
- **Evidence:** `uv run ty check src tests` → 0 diagnostics after the ADR-0016 work, down from
  the `ff041ff` baseline of 4 test-file diagnostics. Whole-repo `uv run ty check` → 22
  diagnostics, unchanged in the untouched `controllers/`, `model/`, and
  `scripts/analyze_pregrasp_table.py` files.
- **Supersedes:** Nothing.

## ADR-0018: Closed-loop waypoints, GT fingertip telemetry, and the fresh-world episode runner

- **Date:** 2026-11-09
- **Status:** Accepted
- **Context:** The six M1-07 run diagnosis (`runs/m1-07-realign-{on,off}-0{1,2,3}.jsonl`) shows
  the first failure cause is contact/knock during DESCEND in 5/6 runs and a joint stall under the
  strongest tilt in the sixth. The FK tool tip and the Supervisor GT finger pads disagree by
  roughly 7–8 cm on live runs, which is a frame mismatch, not the proximate cause. The M1-07 flow
  also needs a repeatable fresh-world runner and cross-platform determinism checking that did not
  depend on the operator manually restarting Webots.
- **Decision — closed-loop waypoints:** new config `closed_loop_waypoints` (default `true`,
  CLI `--closed-loop-waypoints/--no-closed-loop-waypoints`). When on, DESCEND, LIFT, and the
  CLOSE guard's re-descent re-express their world-anchored target in the **live base frame once
  per Cartesian waypoint** through `_waypoint_target_provider()`. DESCEND tracks the live CAN;
  LIFT is anchored to the CAN pose captured at LIFT entry so a grasped, rising CAN cannot create
  a target feedback loop. `_move_linear(target_provider=...)` interpolates the remaining fraction
  of the path from the achieved position; with a fixed target this telescopes exactly to the
  original `alpha` schedule, so closed-loop-off arithmetic is byte-identical to the previous
  controller. Traces record `closed_loop_waypoint` and `live_target_base_xyz_m` on every waypoint
  boundary. This deepens, not replaces, ADR-0016: the PREGRASP realign and its refreshed LIFT are
  unchanged.
  *Annotation (2026-11-09, second phase):* the "byte-identical" wording above was originally
  asserted only by key absence. It is now backed by `tests/test_move_linear_original_equivalence.py`
  against a reference reimplementation of `1b2e4da:controllers/tiago_controller/motion.py`
  `move_linear` (waypoints/seeds/solutions at 1e-9 under the bounded continuation; the literal
  unbounded 1b2e4da solve is compared at the waypoint level and via the shared `IK_TOL_GRASP` tip
  criterion, because the unbounded ikpy solve lands on divergent joint configs with the same tip).
  The precise claim is: same Cartesian waypoint arithmetic, same seeds, same acceptance criteria;
  the *joint solutions* differ only through the bounded-vs-unbounded solver continuation.
- **Decision — GT fingertip telemetry, report-only:** one helper, `_gt_finger_payload()`, merged
  into `_trace_measurements()`, supplies `gt_midpoint_world_xyz_m`, `finger_tip_world_xyz_m`,
  `ee_fk_world_xyz_m`, `fk_gt_offset_m`, `gt_to_can_xy_m`, `fk_to_can_world_m`/`fk_to_can_xy_m`,
  `finger_midpoint_base_xyz_m`, `finger_midpoint_to_can_base_xy_m`, `can_radius_m`, and the
  per-finger `clearance_left_m`/`clearance_right_m` (finger-pad distance to the CAN cylinder).
  `gt_tip_telemetry` events fire exactly at PREGRASP exit, DESCEND enter, and CLOSE enter. When
  `fk_gt_offset_m` exceeds the 5 mm `TOOL_OFFSET_CALIBRATION_TOLERANCE_M`, the event carries
  `tool_offset_error`, a `calibration_warning`, and a report-only
  `tool_offset_calibration_proposal` computed from the GT `arm_7_joint` ground-truth point;
  nothing is ever applied automatically. Calibration needs a live operator run, so the
  `finger_tip_offset_m` value (0.1741 m) stays as configured.
- **Decision — fresh-world episode runner:** `run-episodes` in `vlarobot.cli` (module
  `execution/episode_runner.py`) launches `webots <world>` as one subprocess per episode (strict
  arg lists, never `shell=True`, `pathlib`+`sys.executable`-safe, `encoding="utf-8"` on all file
  I/O), waits for the ZeroMQ ping, runs exactly one brain-side episode, terminates the
  simulator, and prints one fixed-width summary table covering the four success criteria.
  `--determinism-check K` runs the same seed K times in fresh worlds, normalizes both traces
  (dropping `reset_fingerprint`, `controller_reset_state`, `trace_path`,
  `backend_diagnostics_compute_s`) and compares outcome dictionaries (dropping per-episode
  index, paths, and wall-clock times), SHA-256 digests, and numeric deltas; the command exits 1
  on a mismatch. Accumulating `run-episodes` deletes stale per-episode traces before writing.
  `--no-fresh-world` reuses one Webots process with soft resets (same semantics as
  `can-regression`). Rendering stays ON by default because the scripted policy reads RGB every
  step; `--headless` maps to `--batch --mode=fast --no-rendering --stdout --stderr` and is
  explicitly UNVERIFIED (`.doc` caveat: `--no-rendering` can blank camera reads).
- **Decision — housekeeping in the same pass:** gripper close force is clamped to the motor
  `maxForce` (`backend._set_gripper` and `gripper.close`), matching the simulator's internal
  clamp so behaviour is unchanged while the AvailableForce warning disappears. The CLOSE guard's
  re-descent is attributed to CLOSE, not DESCEND, in `state_sim_durations`/`state_wall_durations`,
  so DESCEND settle accounting is not inflated by a state that already exited. Transport timing
  counters are opt-in (`VLAROBOT_BACKEND_TIMINGS=1`): `encode_s`, `send_s`, `recv_s`, `decode_s`,
  `requests`, `image_transfer_bytes`; the policy timing summary adds `steps` and
  `wall_per_step_s`, and `can-regression --timings` prints a `timings zmq:` block.
- **Evidence:** `ruff check .`, `ruff format --check .`, and `uv run ty check src tests` are
  clean; `pytest -m "not webots"` reports **144 passed**, including a fake backend that shifts the
  base 6 mm sideways only once the tip is inside the DESCEND column (closed-loop live targets
  track it by ≥2 mm planar drift; the OFF run has no `live_target_base_xyz_m` keys), a
  GT-injection fake backend asserting the three event sites and the >5 mm proposal, and the
  episode-runner helper tests (exact arg lists, `WEBOTS_HOME` resolution, digest/norm/delta
  comparisons). No subprocess is launched by the tests.
- **Remaining evidence:** Operator `run-episodes` runs (fresh-world and `--determinism-check`)
  are **UNVERIFIED**, as are the headless flags and the real GT calibration correction. M1-06a,
  M1-07, and all M1 gates remain **UNVERIFIED**.
- **Supersedes:** Nothing. ADR-0016, the two-tier drift invariant, `hold_base: none`, and the
  four success criteria remain in force.

## ADR-0019 — FK-vs-GT gap decomposition, tool-offset calibration proposal, and the graded OFF-equivalence test

- **Date:** 2026-11-09 (second M1 phase, Webots-free, from recorded traces).
- **Context:** The live FK-vs-GT tool-tip gap (~70–83 mm) needed a source before any further live
  run or calibration action. Recorded traces (`runs/*.jsonl`, 749 usable `state_boundary` rows of
  763; 12 skipped without complete FK/GT/base fields) carry the GT `arm_7_joint` endPoint Solid,
  the base pose, and the measured joints, so the gap could be decomposed Webots-free with the
  project's own URDF kinematics (`scripts/decompose_tool_offset.py`).
- **Finding (trace-side, not calibrated live):** the gap is a *chain-of-origin* divergence, not a
  tool offset alone. Per-segment residuals localize it: the arm_6→arm_7 origin placement is
  **(+55.0, −1.1, +0.4) mm expressed in the arm_6_link frame with component stds
  (0.06, 1.53, 0.67) mm** — near-rigid, i.e. the simulator places the arm_7 origin ~55 mm from
  where every repo URDF (arm_7_joint at (0,0,0) of arm_6_link) and therefore FK expect it.
  Upstream residuals are smaller and looser (u4 ≈ 26 mm std ~5; u6 ≈ 44 mm std ~11). The tip-side
  residual (GT arm_7 vs FK tip incl. the 0.1741 m `finger_tip_offset_m`) is constant
  (+199 mm) only in the tool frame's z: pad geometry (174.1 mm) plus the ~55 mm placement error
  plus ~−28/+13 mm lateral spread. Consequence: `calibrate_tool_offset()` — which is **never run**
  (gated by `calibrate_tool: false`, `configs/robot.yaml:31`; call sites
  `pick_place.py:163-167` and `tiago_arm_api.py:391-393`; no brain caller) — could only correct a
  constant tip vector and cannot represent the arm_6→arm_7 origin offset. A #s67-only correction
  of the FK tip drops the DESCEND approach by 55 mm (correct directions for the observed rim
  contact) but is a *proxy*; the root fix must be converted through one live run (see below).
- **Decision — calibration stays report-only:** no code change to the configured
  `finger_tip_offset_m` (0.1741 m). The predicted effect of applying the s67-only correction at
  PREGRASP exit (per-run xy at the CAN: on-runs 1.6–6.6 mm uncorrected → 2.7–5.7 mm corrected,
  i.e. marginally worse; off-runs 28.4 → 27.7 mm, marginally better) shows **XY realignment is
  dominated by the PREGRASP realign gate, not by tool error**, so auto-correcting the offset would
  trade a large vertical error (which explains the DESCEND rim knocks) for worse XY. Validation
  path (live, operator): set `calibrate_tool: true`, run one episode, confirm the
  `tool_offset_calibration_proposal` event's vector matches the s67 prediction, and re-check the
  first-DESCEND-waypoint fingertip-to-lid clearance. All PROTO-side geometry (actual Webots
  arm-link endpoint placement vs URDF) remains **UNVERIFIED** without that live run.
- **Decision — graded OFF-equivalence test:** `tests/test_move_linear_original_equivalence.py`
  replaces the key-absence assertion with two reference drives of the original
  `1b2e4da:controllers/tiago_controller/motion.py` `move_linear` arithmetic (3 cm spacing, alpha
  schedule, `IK_TOL_GRASP`/`MAX_TILT_DEG` rejection, 0.03/0.01 rad settle, 600-step budget) on the
  identical OFF-policy inputs: *bounded* mode (shipped `restarts=True` +
  `max_joint_step_rad=MAX_JOINT_STEP_RAD`) requires waypoints, per-waypoint seeds, and accepted
  joint solutions equal within **1e-9**; *literal-1b2e4da* mode (`restarts=False`, unbounded
  chain) requires waypoints within 1e-9, the seam seed within 1e-9, and solutions within the
  common acceptance criterion (tip within `IK_TOL_GRASP`) plus the policy's trust-region bound —
  because the unbounded ikpy solve legitimately wanders ~0.35 rad from the warm seed at the same
  tip (< 1 µm). `ADR-0018`'s "byte-identical" wording is annotated above. The settle semantics of
  the fake backend measurement (measurement converges to commands) also make the commanded-seam
  seed equal to the measured seed within the 0.01 rad final settle tolerance; the seam test
  asserts the recorded bound.
- **Evidence:** `uv run pytest tests/test_move_linear_original_equivalence.py` — 4 passed
  (bounded parity, literal mode, seam bound, solver-argument forwarding). Decomposition script
  output for 6 `m1-07-realign-{on,off}-0{1,2,3}` runs + 10 `hold-*`/`velocity*`/earlier traces:
  n=749/763, per-run and global tables in the run transcript; `docs/PROGRESS.md` M1-07 section
  updated with the per-run first-deviation evidence table. Fake-backend-only; no live run.
- **Remaining evidence:** all live/unverified items above. M1-06a, M1-07, and all M1 gates remain
  **UNVERIFIED**.
- **Supersedes:** Nothing. The `calibrate_tool` gate and the report-only proposal behaviour are
  unchanged.

## ADR-0020 — Sim-derived kinematics snapshot and FK validation sweep (live-UNVERIFIED machinery)

- **Status:** Accepted (implementation + Webots-free tests landed; **live export, sweep, and acceptance UNVERIFIED** — no Webots session in this environment).
- **Date:** 2026-11-10
- **Scope:** Milestone M1 (second phase). Read-only live export of Webots kinematics to JSON, a sim-derived IKPy chain loader/selector, and a live FK-vs-GT validation sweep with an explicit accept criterion and a pose-cross-validated failure analysis. No safety/IK-tolerance/threshold/target changes, no `core/` wire or interface changes (`RobotState.diagnostics` remains the free-form channel), and no Webots/ROS imports outside `backends/webots/`.
- **Context:** ADR-0019 decomposed the URDF FK-vs-GT gap from traces (n=749): the arm_4/arm_6
  residuals are near-constant in world/base but the arm_6→arm_7 offset is the dominant term in the
  arm_7 frame, and the URDF chain root (`Base link`) is rigidly displaced from the Robot node
  origin — so a tip-only correction cannot represent the error and live-measured link frames are
  required to localize it.
- **Decision — live export (read-only, reset posture):** `supervisor_kinematics.capture_snapshot`
  records world poses of the Robot node (`io.me`, the `frames.world_to_base` base), every torso/arm
  joint's parent `Solid` and `endPoint` child `Solid`, and both finger `endPoint` `Solid`s. Poses are
  read only from `Pose`-derived nodes; joint/`jointParameters` nodes are never asked for pose. Hinge
  axes are derived empirically from a small posture probe (<= 0.05 rad per joint, restored
  immediately) via the rigid conjugation of two measured postures; the `jointParameters.axis` field
  is never assumed and is only cross-checked in the readability report. The torso slider is baked at
  its export posture and never probed. The backend attaches the snapshot to the reset observation
  (`diagnostics.sim_kinematics_snapshot`) and per-step GT Solid telemetry
  (`diagnostics.kinematics_telemetry`) only when `export_sim_kinematics: true` (default `false`).
- **Decision — sim chain (Robot-node base, measured hinge family):** the chain root is the **Robot
  node frame**, fixing the URDF chain's rigid upstream XY offset by construction (a base-tilt
  invariance test pins that a rigid base rotation is identical between FK and GT). For each hinge the
  measured family is `T(θ) = Rot(â, θ) @ K` with `K = Rot(â, θ_ref)⁻¹ @ T_pc(θ_ref)` and absolute
  joint angles as variables; because ikpy composes `origin ∘ Rot(axis, θ)`, the per-joint element
  order is `URDFLink(axis=â, absolute motor limits)` **then** `FixedLink(K)` (placing `K` first would
  encode a different, constant-anchor hinge family — the perturbed-angle FK test pins the correct
  order). The torso is a baked `FixedLink` at the export posture, asserted at load and on every FK
  call (`SIM_TORSO_TOLERANCE_M = 0.002 m`). The tool frame is the midpoint of the two finger pad
  centres in the `arm_7` child frame; if only one finger is readable the loader falls back to the
  configured `finger_tip_offset_m` along local −z and marks the source
  `fingerTipOffsetFallback(UNVERIFIED)`. Load-time validation re-derives every axis from the recorded
  probe, rejects a fit residual > 1 mm, and checks the arm parent-child chain (≤ 0.1 mm).
- **Decision — chain selection:** `kinematic_model: urdf|sim` (default `urdf`, M1-06a equivalence
  baseline unchanged). `sim` loads `configs/generated/sim_kinematics.json` and raises if it is
  missing, so a normal run never silently changes geometry.
- **Decision — validation sweep and criterion:** `validate-kinematics` sweeps >= 40 deterministic,
  stratified arm poses inside the recorded absolute motor limits with the torso fixed; a pose is
  aborted (not scored) when any tracked link is below the table plane inside the footprint, or when
  rejected/does not settle. FK and GT are compared in the Robot-node base frame per pose. The **sim**
  model is accepted only if the finger-midpoint error is mean < 2 mm and max < 5 mm over the sweep;
  the URDF model is the informational baseline and is never auto-accepted. On failure the report
  states the first affected link/frame and fits named parameter families (constant base-frame and
  tool-frame offsets) with a strict even/odd **pose** holdout (never by run), so a rigid offset is
  distinguished from pose-dependent axis/scale/structure error and identifiability is stated
  explicitly.
- **Consequences:** a live export is the single source for the sim chain; the export-posture FK
  residual and per-joint probe/cross-check are reported on write, so axis/parent-chain defects fail
  closed at load or are visible before a sweep. Existing PREGRASP/DESCEND traces lack the Solid-pose
  telemetry the sweep requires, so their numeric re-evaluation is blocked on a live export + a
  telemetry-bearing validation report (`scripts/reevaluate_traces_sim_model.py` enforces this and
  claims no numeric result).
- **Evidence (this environment):** Webots-free machinery and tests —
  `tests/test_supervisor_kinematics.py`, `tests/test_sim_kinematics_loader.py`,
  `tests/test_kinematics_validation.py` (synthetic 7-joint world, perturbed-angle FK, base-tilt
  invariance, loader rejection cases, constant/pose-dependent bias attribution, table pre-check, CLI
  sweep driver over a fake lockstep client). Gates: `ruff check .`, `ruff format --check .`,
  `ty check src tests` (only the 2 pre-existing HEAD diagnostics in `backend.py`/`scripted_can.py`),
  `pytest -m "not webots"` — 180 passed. Commits `4710fd7`, `587b176`, `05d18d7`, `801d62e`.
- **Remaining evidence (UNVERIFIED):** live `export-sim-kinematics` JSON; live `validate-kinematics`
  sweep (>= 40 poses) and sim-model accept/reject decision; numeric re-evaluation of prior traces
  against the accepted sim model; the PROTO-side geometry and the `tool_offset` calibration path
  (unchanged from ADR-0019, still report-only). M1-06a, M1-07, and all M1 gates remain UNVERIFIED.
- **Supersedes:** Nothing. `calibrate_tool` stays `false`/report-only; `kinematic_model` defaults to
  `urdf` and the URDF equivalence baseline is unchanged.

## ADR-0020 addendum — Fast reset, chunked export, probe stationarity, session hygiene (2026-11-10)

- **Status:** Accepted (implementation landed; live/GUI runs **UNVERIFIED** — agent must not launch Webots or run live commands; provide commands only).
- **Scope:** Addresses live export/reset timeout and stale-server failures observed in the first live attempt. No safety/IK-tolerance/threshold/target changes. No M2.

- **Separate export from reset:** `reset` performs no probes and must remain fast (< 5 s). Kinematics export is an on-demand `export_kinematics` request handled by chunked polling (`start`/`poll`/`status`/`cancel`): each poll advances by probing at most the next joint and returns per-joint progress/heartbeat (steps used, measured angle moved, settled/reason) plus the final snapshot when done. `export-sim-kinematics` now does `reset` (posture) then polls the export; per-poll timeouts are bounded and progress is reported.

- **Probe design (stationarity, measured motion):** a joint is settled when `|delta sensor| < probe_stationary_delta_rad` for `probe_stationarity_steps` consecutive lockstep steps (config: `0.0001` rad, 3). The axis is derived from the **measured** sensor angle change and measured poses, never from assumed convergence to the commanded angle (avoids stalling on steady-state tracking error). Steps are capped per joint (`probe_max_steps_per_joint`, default 60) and in total (`probe_total_step_cap`, default 2000); every joint reports steps used, measured angle moved, and the reason for success/failure. The earlier `PROBE_SENSOR_TOL` "near-target" workaround is removed; the remaining settle parameters are the explicit stationarity/cap config above.

- **Session identity and bind hygiene:** `ping` replies include a `session_id` (`pid`, `start_time`, `nonce`); clients record/expect it and fail loudly on mismatch (stale server detection). The server binds `VLAROBOT_ZMQ_PORT` (or config `zmq_port`) and fails loudly on bind failure (no silent fallback). `cli doctor` lists stale `webots`/`tiago_controller`/sim-server processes and the configured port owner, and with `--kill` terminates listed processes after printing them.

- **Client robustness:** per-request-type timeouts (reset short, export long/per-poll); on timeout the client closes and recreates the REQ socket (lazy-pirate) so shutdown no longer hits a poisoned ZMQ state, and logs the request, timeout, and elapsed time.

- **Step cost / observation toggles:** `cli bench-steps --n 200` measures ms/step against a running server (each call is step + observation: state always; RGB capture controlled by `observe_images`; per-step `kinematics_telemetry` attached only when `export_sim_kinematics` is enabled). Config `observe_images` (default `true`, unchanged behavior) allows disabling image reads later when the brain does not need them. Defaults unchanged.

- **Evidence:** Webots-free/unit coverage for chunked export via the fake backend, session identity in ping, lazy-pirate socket recovery, stationarity-based probe capture, and updated supervisor/loader/validation tests. Full live export/sweep/determinism runs remain **UNVERIFIED** (commands only).


## ADR-0020 addendum (follow-up) — Snapshot validity, timeouts, ty baseline, live observations (2026-11-10)

- **Status:** Accepted (machinery/tests/docs landed; live acceptance/GUI runs **UNVERIFIED** — agent did not launch Webots). No safety/IK-tolerance/threshold/target changes. No M2.
- **Snapshot validity (stale/invalid refusal):** exports are finalized only after load/build/export-posture verification passes. The saved JSON carries `valid: true`, `content_hash` (SHA-256 over canonical payload excluding the hash field), `session_id`, Webots version and timestamp; written to a temp file, verified, then atomically renamed. On failure the temp file and target are removed so a pre-existing invalid/stale snapshot cannot remain loadable. `kinematics_from_payload`/`load_snapshot` refuse snapshots not marked valid or with a hash mismatch. Probe derivation now samples parent/child poses in their instantaneous parent frames (reduces spurious hinge-fit residual from base/parent drift during probing); loader accepts `parent_pose_probe` when present.
- **Timeout inventory (config keys, defaults, justification; bench-steps measured ~447 ms/step with RGB + `kinematics_telemetry` enabled — step+observation, so a single steady-state step << 5 s but reset/export/cold start span many steps):**
  - `zmq_ping_timeout_ms` 5000 (liveness/session). `zmq_reset_timeout_ms` 120000 (tolerates slow Webots start; many settle steps). `zmq_step_timeout_ms`/`zmq_apply_timeout_ms`/`zmq_get_gt_timeout_ms` 30000. `zmq_shutdown_timeout_ms` 10000. `zmq_export_poll_timeout_ms` 300000 (per-joint chunk, bounded by probe caps).
  - `run-episodes` startup wait uses `startup_timeout_s` (default 180 s) with repeated short pings (server-not-up retry) and per-episode free ports/env/pid files/process-tree termination. Client timeout errors distinguish `not up/unreachable` (no successful response yet) from `busy/stalled` (previously responsive) using session-aware state.
  - Bench-steps has been run (447 ms/step reported by operator); numbers above are justified against that measurement and cold-start/export behavior.
- **Tooling:** `doctor` reports port-free status, owner/listener, stale processes, and Webots GUI vs `--batch` mode. `bench-steps` prints a verdict estimating batch time from the most recent run-episodes traces. Validate-sweep reports persist `samples` for offline re-evaluation.
- **Type checking (item 1):** ty pinned `>=0.0.99,<0.1.0` (lock 0.0.99, no upgrade). At `1783274` (pre-addendum HEAD) `ty check src tests` reported **2** diagnostics (pre-existing `Chain.forward_kinematics` arg-type in `src/vlarobot/backends/webots/backend.py` and `src/vlarobot/execution/scripted_can.py`). Committed addendum HEAD `5a86f75` reported **27** diagnostics (introduced by export-job dict typing and probe-status update typing in addendum code plus the 2 pre-existing). Fix commits (`454c575`, `ba54f28`, `9b6568a`, `8e8c26b` and working fixes) resolved all to **0** (`uv run ty check src tests`: All checks passed), including fixing the 2 pre-existing diagnostics by passing `list(q)`.
- **Re-evaluation readiness:** Offline re-scoring requires per-pose `kinematics_telemetry` samples and a valid sim snapshot. Future validate reports persist samples; the six earlier scripted traces (e.g. `runs/m1-07-*`, `runs/hold-*-*`) contain no `kinematics_telemetry` and cannot be backfilled — a short re-collection (e.g. 3 episodes) with `export_sim_kinematics: true` is required to produce re-evaluable traces. No numeric results guessed.
- **Live observations (operator runs, UNVERIFIED, no agent Webots):** `bench-steps` measured ~446.8 ms/step on session pid 152341 (RGB+telemetry enabled). `run-episodes --determinism-check 3` (per-episode free ports/process hygiene) completed 3 episodes but outcomes were non-deterministic (0/3 success, states/timings/digests differed: episodes failed in DESCEND or reached home differently; repeated ping timeouts observed during startup). `export-sim-kinematics` chunked probes completed (120 total steps, all joints `reason=ok`, measured moves ~0.0499 rad) but verification rejected the snapshot with `hinge axis fit residual 11.4954 mm exceeds 1.0 mm` (arm_1/4/2/6 high) — instantaneous parent-frame probing and stricter validity/atomic-write were implemented after this observation and remain **UNVERIFIED live**. `validate-kinematics` reported no `kinematics_telemetry` on that server session (`export_sim_kinematics` not enabled for telemetry on the running server). M1-06a/M1-07/M1 gates remain **UNVERIFIED**.


## ADR-0020 addendum — Axis-line hinge fit and export pinning (2026-11-10, UNVERIFIED live)

- **Status:** Implemented in code/tests (Webots-free gates pass); **live-UNVERIFIED** (no verified canonical snapshot or validation sweep completed in this session).
- **Hinge model:** replaced origin-axis assumption with fitted axis line `(a,c)` (point on axis, perpendicular component) and pitch-aware residual: `D = T1*T0^-1` modeled as `Trans(c)@Rot(a,delta)@Trans(-c)`; chain uses revolute origin at `c` with `K = Rot(a,-theta_ref)@Trans(-c)@T(theta_ref)`. Fixes systematic multi-mm residuals when hinge axes do not pass through parent origins.
- **Export robustness:** robot base pose pinned and held/quiesced during probing to limit drift; probe records `parent_pose0/child_pose0` for stable recomputation; chain parent consistency enforced for sequential arm joints; on finalize failure a non-loadable debug snapshot (`*.debug.json`, `valid:false`) is written while canonical target/temp are removed.
- **Open:** debug analysis of a failed run showed arm_6 pitch ~5.4 mm (>1 mm cap) under the new model; requires fresh live export to confirm and diagnose if persistent.


## ADR-0020 addendum — Feasible sweep, episode hooks, snapshot hygiene, forensics (2026-11-10)

- **Date:** 2026-11-10
- **Status:** Implemented (Webots-free); live evidence **UNVERIFIED**
- **Decision:** Redesign `validate-kinematics` to a feasible 12-pose sweep (6 spread + 6 near
  PREGRASP/DESCEND, tool-vertical joint-space bias) with sub-moves bounded to
  `0.4 * velocity_limit * horizon`, stationarity settle (cap 60 steps/sub-move), pre-run step budget,
  and fail-fast-continue per pose. Acceptance: mean < 3 mm, max < 6 mm on `arm_7` endpoint and wrist
  in base frame, whole/near-task reported separately; finger midpoint only when tool frame verified
  (otherwise FALLBACK/UNVERIFIED). No safety/IK/threshold/target changes.
  `run-episodes` accepts `--kinematic-model urdf|sim` and per-run `--closed-loop-waypoints/--pregrasp-realign`,
  records model/snapshot hash/path in every episode summary, and refuses `sim` when snapshot
  missing/invalid or identity differs/unverifiable against the running session (clear available fields).
  Verified snapshots are written as `sim_kinematics.<session-hash>.json` with stable symlink-free pointer
  `sim_kinematics.pointer` (+ compat `sim_kinematics.json`); failed finalize leaves no loadable/temp/debug
  artifact.
- **Evidence (Webots-free):** determinism forensics `docs/DETERMINISM_FORENSICS_2026-11-10.md`
  (episode-00..02: first substantive divergence record 1, reset fingerprint identical; missing
  session/world/step identity fields documented). Finger geometry derived from R2025a
  `TiagoGripper.proto` (remote): pad-center midpoint in `arm_7` frame ≈ `(0,0.1901,0) m`, pad-bottom
  ≈ `(0,0.2350,0) m`; vs scalar `finger_tip_offset_m=0.1741` as `[0,0,-0.1741]` difference norm ~0.258 m
  under that interpretation (**UNVERIFIED**). Gates: ruff/format/ty pass, pytest `-m "not webots"` 180 passed.
- **Consequences:** live 12-pose acceptance, determinism re-run, episode comparison runs, and offline
  re-eval of telemetry-bearing reports remain **UNVERIFIED** without Webots. `--kinematic-model sim`
  runs are refused unless snapshot/session identity is verifiable (conservative).
