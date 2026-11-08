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
