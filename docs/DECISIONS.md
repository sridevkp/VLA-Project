# Architecture Decision Records

## ADR-0043: Bind migrated account reads and relocate processor addresses in memory

- Date: 2026-11-16. Implemented/tested locally; new-account authentication, transport,
  T4 and CUDA restoration remain external stop gates. No training/evaluation runs.
- Resolve the owner-selected shared-project entrypoint once, without assuming the
  original shortcut ID. Verify the expected new API account, resolve API shortcut
  target/owners and bind mounted payload bytes to unique downloadable API children
  plus immutable approved SHA-256 pins. No programmatic compute/mount identity
  attestation is claimed; those account selections require explicit operator UI checks.
- Sharing does not move ownership/quota or prove the persistence issue fixed.
  All Drive access remains read-only. Installation logs and new checkpoint writes
  go to local storage; independent Linux receipts remain mandatory.
- Stage only the Linux-verified017000. Snapshot the11 verified processor metadata
  files and relocate only the policy/tokenizer filesystem addresses in memory.
  Original checkpoint JSON/normalization tensors, source/data/model settings and
  frozen seeds/steps/batch/cadence remain unchanged. Record account/asset provenance.
- A new standalone migration notebook preserves the approved notebook. Training
  command is comment-only; actual CUDA state restoration and a new session-bound
  full-size SSH canary precede any future owner launch. No credentials or listener
  are created automatically. No new Python training dependency is introduced.


## ADR-0042: Prepare private receiver without provisioning credentials or services

- Date: 2026-11-16. Status: CPU readiness work completed; network/GPU gates blocked.
- Existing Tailscale is online; no SSH listener or expected host key was found.
  Prepare a private-address-only SSH config and forced command dispatcher without
  creating credentials, starting a server or changing tailnet/firewall policy.
- Forced commands permit only fixed golden-input emit and fixed-root receive.
  Owner approval/provisioning and verified host-key distribution precede activation.
- Full-size local receiver test passed, including fsync and fresh-process SHA.
  Do not substitute local throughput for authenticated Colab network measurements.
- The current runtime remains17-only/single-invocation by design. Preserve18/19
  checkpoints after interruption; do not silently reset to17 or count logical
  committed steps as all physical optimizer calls. Partial resume remains blocked
  until its lineage and remaining-update plan are separately reviewed.
- Add an offline strict-final import gate; learned inference/evaluation commands
  are prepared, not executed. No TEST access or frozen experiment changes.
- No new dependencies installed. Tailscale/OpenSSH already exist on this Linux
  host; any Colab networking enrollment remains an owner-managed external action.


## ADR-0041: Require independently acknowledged recovery artifacts

- Date: 2026-11-16. Status: implemented/tested locally; external transfer and GPU
  execution await owner-reviewed endpoint/authentication and readiness checks.
- All12 files in the Linux 017000 backup match their COMPLETE sizes and hashes,
  including the owner-supplied tensor identities. Preserve that backup unchanged.
- Owner reports a completed recovery whose model/export/archive later disappeared
  from Drive. Root cause remains unproven; prepare read-only account quota,
  ownership and tensor/trash metadata diagnostics rather than blaming quota/cache.
- Future recovery writes locally in Colab; the existing frozen save cadence gains
  synchronous authenticated SSH handoffs to independent Linux storage. Final
  portable model export is handed off before the full resumable checkpoint.
- Linux receiver uses exclusive artifact directories, size/SHA checks, fsync and
  read-back receipts. Missing receipts fail closed. Retain partial attempts and
  provenance; never overwrite original runs. A full-size canary and capacity checks
  precede training, alongside the actual CUDA state-restoration gate.
- No Python dependencies added. Existing OpenSSH client/service and owner-managed
  credentials/host keys are required; no networking service is launched here.
- No dataset/model/training configuration, core interface, gate or TEST changes.
  Recovery remains separately labeled; dataloader cursor replay is not claimed.


## ADR-0040: Resolve the confirmed Drive project shortcut before constructing run paths

- Date: 2026-11-15. Root-cause evidence: owner-authenticated Colab lstat/resolve
  results identify the project ancestor as a Drive shortcut; COMPLETE.json is a
  regular file. The old all-ancestor rejection was overbroad for that entrypoint.
- Resolve only the explicitly trusted project root, requiring its exact confirmed
  canonical target. Construct original, new recovery, export and metadata paths
  from that result. No ancestor allowlist or broad symlink exception is installed.
- Reject symlinked checkpoint roots/descendants, traversal and canonical escapes;
  keep canonical original/recovery isolation and every size/SHA-256 gate.
- Preserve failed recovery directories. v3 helpers use a fresh versioned output
  and retain the actual CUDA restoration gate before optimizer updates.
- Real and malicious symlinks are created only in requested temporary regression
  fixtures. Windows symlink tests skip if OS privileges prohibit their creation;
  no repository or deployment symlinks are created.
- No dependency, approved source archive, production checkpoint or frozen criterion
  changes. No training or TEST evaluation executed.


## ADR-0039: Preserve original production evidence during state recovery

- Date: 2026-11-15. Status: recovery procedure prepared, training not executed.
- Owner reports missing tensors at 018000–020000 and verified model/optimizer at
  017000. Full 017000 manifest verification remains required in authenticated Colab.
- Approved selector already rejects missing required tensors; stale manifests do
  not prove missing files were accepted. Do not invent an upstream deletion cause.
- Apply stricter all-entry, read-back and final-export guards in a separate helper
  process. Preserve approved bundles and source files on disk.
- Copy verified 017000 to a new versioned recovery directory. Restore full state
  and override output path only; target 20000, batch8, seed0, save1000 remain frozen.
- Original logs remain separate; replacement lineage and update counts must be
  explicit. Dataloader cursor/worker state are not restored, so exact replay is
  not claimed. No automatic training, evaluation or TEST access is authorized here.
- No public core interface or dependency changes.


## ADR-0038: Qualify physical validation and geometry evidence by provenance

- Date: 2026-11-15. Implemented locally on the isolated analytic branch.
- Preserve mixed-runtime validation separately; the fresh canonical campaign is
  15/18. Audit physical lift/placement independently of HOME.
- Source-derived geometry corrections remain diagnostic until full coverage is
  certified. Unknown geometry retains veto power.
- Missing authoritative report definitions remain parameterized, not invented.
- Record owner-reported production progress separately from local verification;
  never restart active Colab training to apply local safeguards.
- No public core interface, dependency, training artifact or frozen criterion changes.


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

## ADR-0021 - Campaign 2026-11-11: AGENTS manual-steps scope and hang instrumentation (MEASURED/INFERRED)
- Status: Accepted (campaign operational), 2026-11-11
- Context: Commit `95f4e8d` added AGENTS workflow step 7 ("when the user needs to do something manually, always provide clear step-by-step commands...") at explicit user request to remember manual step-by-step guidance. Autonomous validation campaign requires agent-executed live runs without per-step approval.
- Decision (MEASURED): Keep step 7 as human-requested. Scope: agent runs autonomous live commands (`run-episodes`, `bench-steps`, determinism/servo/gate runs) without approval; when human action is needed (Windows/visual checks/failed automated attempts), provide self-contained procedures in `docs/MANUAL_VALIDATION.md` (prereqs, Webots launch GUI/headless, env vars Linux/Windows PowerShell, ports/doctor, exact ordered commands, expected output/timing, screenshots, what to send back). Windows steps recorded as UNVERIFIED. AGENTS.md not edited again without ADR.
- Hang diagnosis (MEASURED so far): Phase 0 attempt 1 stalled at sim 13.024s (DESCEND, ik_rejection context) with repeated client ping timeouts to `tcp://127.0.0.1:50581` while server/controller processes remained; no per-request timing logs existed then. Rerun completed 0/1 CLOSE. `SimServer` now logs `request_start/end` with `duration_s`, warns on slow requests >2s, and runs watchdog/status reporting (`runs/campaign-2026-11-11/sim-status-<port>.json`, busy request+age). Independent ping answerable during busy REP handling is **not implemented** (INFERRED limitation: REP single-threaded request/response blocks mid-operation); future hangs should be diagnosable from logs/status age. Client liveness policy: busy with known request under budget acceptable; silent/unresponsive beyond budget => kill process tree and record (campaign rule).
- Consequences: Improved MEASURED evidence for stalls; full out-of-band health socket remains future work if needed.

## ADR-0022 - "Terminating." crashes were runner cleanup of a settle-gate reset failure; non-settle is now recorded, not fatal (MEASURED)
- Status: Accepted, 2026-11-11
- Context: Campaign 2026-11-11. Four settle-check episodes (settle gate ON) and one pad-probe run terminated at ~12-15 s with `Terminating.` lines in the Webots console; live environment suspected unstable, feature work halted for forensics (2-working-day time box).
- Cause (MEASURED, reproduced with instrumentation in `runs/campaign-2026-11-11/phase3_bisect_settle/episode-02.controller.log`): the settle gate's angular-velocity threshold (1e-4 rad/s) sits below the base's persistent post-reset angular jitter (ep00 settled at 9.38e-5 rad/s in 53 steps/4.84 s; ep02 stuck at 1.615e-4 rad/s after 500 steps/~13.6 s wall, `consecutive=0/20`). `_settle_post_reset` exhausted its cap and raised `RuntimeError("Post-reset settle failed: reason=max_steps ...")` -> error reply -> client `RuntimeError: Simulator rejected 'reset'` -> runner records RUNNER and kills the process tree. Webots and controller exit code 0 in every episode (`episode-NN.exit.json`); the `Terminating.` lines are the runner's cleanup, NOT a Webots crash. Knife-edge nondeterminism: jitter straddles the threshold across fresh worlds.
- Delta from known-good (MEASURED): baseline rerun (ee30aca-era) and earlier hold-none/m1-07 runs differ only in settle-gate code+config (`8127201`), inert servo helpers (`2c704b1`), and probe scripts; launch flags identical. Reduced repro with all features OFF: 5/5 episodes completed reset and ran to scripted end, `webots_exit_code=0` (`phase3_reduced_repro/`). Bisect with settle gate ON: 5/5 completed reset, ep02 reproduced the failure; 4/5 ran to end (`phase3_bisect_settle/`). Baseline is stable; the gate was the sole crasher.
- Decision: non-settle outcome is WARN + recorded with `"fatal": false` in the fingerprint `post_reset_settle` info instead of raising; thresholds (1e-4/1e-4/0.01, 20 consecutive, 500 max) and default OFF are unchanged; no threshold or criteria changes. Genuinely contaminated physical state remains vetoed by the physical-state guard. Forensics instrumentation kept: per-episode controller diagnostics file + `faulthandler` (`VLAROBOT_CONTROLLER_STDERR_PATH`), `episode-NN.exit.json` (exit codes + controller log tail), always-written `episode-summary.json`.
- Evidence: `phase3_reduced_repro/` (5/5 stable, features OFF), `phase3_bisect_settle/` (repro + full error), `phase1_settle_check/` (original 4/4), instrumentation commits `4700bdf`, fix commit `6bd41c1`; Webots-free tests in `tests/test_webots_backend_reset.py` (jitter/max_steps no-raise, settled path, end-to-end fingerprint).
- Consequences: settle-gate resets may take up to ~13.6 s (cap) without failing the episode; the gate stays default OFF and is NOT promoted (a non-settling base is recorded, not accepted as clean). Probe-crash cause (sim 8.768 s, pre-instrumentation) remains UNVERIFIED and is classified separately. Finger GT verified separately: arm_7->pad-midpoint vector constant `(0,0,+0.202825) m` in the sim arm_7 frame across 4 safe poses (spread <= 1.7e-7 m; `phase3_finger_gt/finger_gt_poses.json`).

## ADR-0023 - gt_pad servo uses the MEASURED pad geometry; descent stop height comes from the live contact-height probe (user-directed, 2026-11-11)
- Status: Accepted (user-directed additions 1+2); contact-height probe run live 2026-11-12 (3/3 contact, see below); gt_pad episode runs pending.
- **Live refutation of the URDF-chain variant (2026-11-12, MEASURED, `phase3_contact_probe/rep01.jsonl`)**: an attempted implementation applied the measured `(0, 0, +0.202825)` offset as the URDF chain's `last_link_vector` when `grasp_servo=gt_pad`. Live FK at a recorded PREGRASP pose put the chain end **0.377 m above the real pad midpoint** (tip world z 1.3281 vs pad 0.9510): the offset is expressed in the **sim arm_7 Solid frame**, whose origin and orientation do not match the URDF chain's arm_7 frame (sim Solid origin − URDF arm_7 origin ≈ (−0.039, +0.016, +0.028) m world; relative orientation ≈ 180 deg). That variant also drove PREGRASP into the pinned elbow-down posture (arm_6 −1.3886 rad at the −1.39 limit, arm_7 −2.0641) and tipped the base (yaw −0.50 deg, pitch −0.51 deg => "base moved" failure in all 3 probe reps). The URDF chain therefore KEEPS the legacy `[0, 0, -0.1741]` tool offset unconditionally; the servo does not need the chain to end at the pad because its XY corrections and DESCEND stop height are measured from the exact Solid-based pad telemetry.
- **Contact-height probe RESULT (2026-11-12, MEASURED, `phase3_contact_probe/pad_contact_probe.json`)**: 3/3 repetitions reached contact from fresh worlds (PREGRASP reached in all 3, no failures; run with the legacy chain). All three fired trigger (c) (measured pad reference crossing the can-lid plane) at `descended = 0.08 m`; e_xy at contact 2.9/3.1/3.1 mm; can centre z 0.81107 m (reproducible to <1 um); reference-to-lid offsets **−2.616 / −2.246 / −2.158 mm => mean −2.340 mm, std 0.243 mm** (std < 2 mm gate: single value usable). Negative sign explained: with 20 mm commanded steps the trigger fires on the step where the pad reference crosses the lid plane, i.e. the pads straddle the can just below the rim without side contact (no can displacement => triggers (a)/(b) silent). Config `probe_pad_reference_to_pad_bottom_m` filled with the signed mean **−0.0023399045941977135**; the config validation was amended from "must be positive" to "must be finite (signed)" so the measured value loads; the gt_pad DESCEND stop equation `pad_bottom_z = pad_ref_z − offset` with the signed value replicates the probe's contact state exactly (stop when `pad_ref_z <= lid − 2.34 mm`, i.e. within one 5 mm servo step of the probe state, std 0.24 mm << step 5 mm).
- Decision (1): the gt_pad servo pad reference is the arm_7 endPoint Solid pose composed with the MEASURED constant offset `(0, 0, +0.202825) m` in the sim arm_7 Solid frame (new config `pad_offset_arm7_simframe_m`, note "measured at 4 poses, spread <= 1.7e-7"). The legacy `finger_tip_offset_m` (0.1741, PROTO slider geometry) is NOT used by `grasp_servo=gt_pad` (it remains for the FK tool-tip exporter and legacy telemetry). The servo fails loudly when neither the Solid pose+orientation nor the composed key is available; the UNVERIFIED PROTO-pad-centre fallback was removed.
- Sign convention (MEASURED, phase3_finger_gt): with the sim arm_7 Solid +z pointing wrist -> fingers (approach direction, toward the can), the pad reference (measured midpoint) sits at +0.202825 m from the Solid origin. Relative to that reference: the PROTO pad centre is at **-12.7 mm** (12.7 mm on the wrist side, i.e. FARTHER from the can) and the PROTO pad bottom is at **+32.2 mm** (32.2 mm on the can side, i.e. NEARER the can). The earlier "+12.7 / -32.2 mm" phrasing in ADR-0022 used unsigned norm differences; the correct signed reading is pad centre -12.7 mm, pad bottom +32.2 mm along +z.
- Decision (2): the DESCEND stopping height for `grasp_servo=gt_pad` is NOT computed from PROTO arithmetic. New config `probe_pad_reference_to_pad_bottom_m` (null until measured) is filled by the live contact-height probe (`scripts/pad_contact_probe.py`): 5 mm commanded descent steps from the servo-aligned hover at fixed XY, abort at first finger contact (`contact_nodes`), record Solid-based reference height at contact, can centre z (GT) and lid z (can centre z + `can_half_height_m` = 0.0611, CAN.proto R2025a), and the resulting reference->pad-bottom offset; 3 repetitions from fresh worlds, mean/std reported; std > 2 mm => explicitly flagged, no single value used. gt_pad descent refuses to run while the probe offset is null.
- Batch diagnostics (addition 3): every episode records at the first post-reset command (`first_command_state` event) base roll/pitch, base linear/angular speed norms (new backend diagnostics `base_velocity_linear_m_s`/`base_velocity_angular_rad_s`), tilt change and drift since reset; at PREGRASP exit, DESCEND enter/mid and CLOSE enter (`gt_tip_telemetry`), and every servo log, the 3D vector (can centre - pad reference) in world and base frames (`can_minus_pad*_world/base_xyz_m`).
- Webots-free verification: ruff/format/ty pass, `pytest -m "not webots"` 192 passed. Live probe and gt_pad episode runs UNVERIFIED until the 10-episode stability gate passes.

## ADR-0024 - Stall-based gripper CLOSE/RELEASE pacing in scripted_can (LIVE VERIFIED)

- **Date:** 2026-11-12
- **Status:** Implemented (Webots-free tests pass); **live VERIFIED 2026-11-13**
  (gate episode `runs/campaign-2026-11-11/phase4_closepacing/episode-00.jsonl`, 1/1 success).
- **Context:** Phase 4 `grasp_servo=gt_pad` 3-episode batch (`phase4_gtpad3/`) showed 0/3 placement
  success (`dist_m ≈ 0.219 m`) despite all episodes reaching HOME. Post-hoc analysis: the original
  `scripted_can.py` CLOSE code issued exactly **one** `backend.step(1)` after sending
  `GripperCmd(0.0, force=50.0)`. One sim step (32 ms) is insufficient for the finger servos to stall
  against the can. The original controller (`controllers/tiago_arm_api/tiago_arm_api.py`
  `close_gripper()`) runs up to **300 steps** checking for stall. With only 1 step, the gripper
  command was accepted but fingers barely moved; subsequent LIFT raised the arm without the can,
  so the can remained at its drop location and the final placement distance exceeded the 0.10 m gate.
- **Decision (CLOSE pacing):** Replace the single-step `backend.step(1)` after `GripperCmd(0.0, ...)`
  with a stall-detection loop of up to `max_close_steps = 300` steps:
  - Track `previous_opening` and `still_count` (count of steps where `|Δopening| < 2e-5`).
  - Fast-exit on step 0 when `current_opening <= 0.001` and `|Δopening| > 0.01` (finger already
    essentially closed from a prior position near zero — avoids a spurious 300-step wait on the
    fake backend where the gripper is instantaneous).
  - Stall declared when: `i > 10 and still_count >= 15` (finger position stable for 15+ consecutive
    steps after the first 10 steps), OR `current_opening <= 0.001 and still_count >= 3`
    (finger at hard limit for 3 steps). Both conditions break the loop.
  - `_base_moved` guard runs each iteration (abort if TIAGo base drifts during close).
- **Decision (RELEASE pacing):** Replace the single-step `backend.step(1)` after
  `GripperCmd(GRIPPER_OPEN, ...)` with an open-wait loop of up to
  `max_release_steps = int(round(GRIPPER_OPEN_TIMEOUT_S * 1000.0 / TIME_STEP))` steps:
  - Break when `observation.robot.gripper_opening >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE_M`.
  - `_base_moved` guard runs each iteration (abort on base drift during release).
- **Why not use `MIN_GRASP_OPEN` for the stall threshold?** The stall condition (`opening <= 0.001`)
  is intentionally tighter than `MIN_GRASP_OPEN = 0.004 m`. After the CLOSE loop exits, the
  post-state check (at LIFT enter) will verify `gripper_opening > MIN_GRASP_OPEN` to confirm
  the can is actually grasped (finger was forced open by the can's radius). The `0.001` threshold
  in the stall loop triggers when the finger is effectively at its hard mechanical limit (no can
  present or failed grasp); it does not substitute for the grasp success check.
- **Fake-backend behaviour:** The `FakeBackend` applies gripper commands instantaneously (opening
  jumps to commanded value in one step). The fast-exit on step 0 (`opening <= 0.001` jump from a
  higher value) handles this case; the stall loop exits after 1 step, so test run-time is unchanged.
  For the RELEASE loop, the fake backend opens immediately, so the loop exits after 1 step.
- **Webots-free verification:** `ruff check . && ruff format --check .` → clean; `pytest -m "not webots"` → **198 passed** (unchanged from pre-pacing count).
- **Live verification (MEASURED, 2026-11-13):** Gate episode (1 fresh-world `gt_pad`, seed 0,
  `phase4_closepacing/`) met all three acceptance signals:
  1. **CLOSE stall:** exit `gripper_opening_m = 0.0314 m` > `MIN_GRASP_OPEN = 0.004 m` —
     fingers stalled against the can (can radius 0.03175 m) instead of closing through it.
     CLOSE took ~31 sim steps (t=94.272 → 95.264) before the stall loop exited.
  2. **Can lifted:** `can_world_xyz_m[2]` 0.8116 → 0.8983 m during LIFT (**+86.7 mm**), carried
     through TRANSIT (to −1.037, −4.249) and placed at 0.8117 m.
  3. **DONE:** `dist_m = 0.0493 m` < 0.10 gate, `success=true`, `home_reached=true`,
     `can_height_ok=true` (error 12 µm), `can_tilt_ok=true` (0.068°). RELEASE pacing exited at
     `opening = 0.0433 m` (≥ GRIPPER_OPEN − tolerance).
  This is the **first successful gt_pad episode** in the campaign (pre-pacing: 0/3, baseline: 0/10).
- **Collision finding (MEASURED, 2026-11-13):** The earlier working hypothesis that finger–can
  collision geometry might be absent in `worlds/factory.wbt` is **refuted**. In the pre-pacing
  `phase4_gtpad3/` runs the single-step CLOSE left fingers at 0.044 m when LIFT began; the arm
  rose past the lid overhang before the fingers could reach the contact threshold, and the fingers
  then closed to 0.0005 m *above/after* the can during TRANSIT — i.e. the "close-through-can"
  observation was an artifact of premature LIFT, not missing collision geometry. With the 300-step
  stationary close, fingers reach the can surface while still alongside it and stall at 0.0314 m.
  Note: `contact_nodes` remained `[]` even at the successful stall — the field does not record
  finger–can contacts (telemetry gap; stall opening + can motion are the reliable signals).
- **Regression analysis (2026-11-13, Webots-free, MEASURED against git history):**
  The single-step CLOSE was a **brain-only regression introduced by the M1 port**, not a
  deliberate design change:
  - **Original (tree at `1b2e4da`) waited for the stall before LIFT** — two implementations,
    identical semantics:
    - `controllers/tiago_arm_api/tiago_arm_api.py:333-351` `close_gripper()`:
      `:340` `for i in range(300)`, `:345` `still = still + 1 if abs(cur - prev) < 2e-5 else 0`,
      `:347` `if i > 10 and still >= 15: break`, `:351` `return opening > MIN_GRASP_OPEN, opening`
      (`:19` `MIN_GRASP_OPEN = 0.004  # fingers stalled wider than this => something is held`).
    - `controllers/tiago_controller/gripper.py:46-64` `Gripper.close()`:
      `:54` `for i in range(300)`, `:60` `if i > 10 and still >= 15`, `:64`
      `return opening > MIN_GRASP_OPEN, opening`.
    - LIFT transition gated on the stall result: `tiago_arm_api.py:450-452`
      `grasped, _ = close_gripper()` → `wait(20)` → `state = "LIFT" if grasped else "RETRY"`;
      same at `controllers/tiago_controller/pick_place.py:121-123`.
  - **Brain at introduction (`a420649` "M1 port and safety envelope"):**
    `git show a420649:src/vlarobot/execution/scripted_can.py:73` `GripperCmd(0.0, force=50.0)` →
    `:83` `backend.step(1)` (one step) → straight to the next waypoint. No stall wait, no
    `grasped` gate; RELEASE likewise `:85`/`:95` single step. The port dropped the original's
    300-step stall loop.
  - **This is a defect for M1-06a equivalence** (the brain must reproduce original OFF
    semantics). It escaped the equivalence test because `dce35d5`
    (`tests/test_move_linear_original_equivalence.py`) covers only `move_linear`/IK arithmetic
    on the FakeBackend — it never exercises the gripper.
  - **Fix (this ADR, no flag):** the stall loops now live in the **common path** —
    `scripted_can.py:459` `if name == "grasp":` → `:510-556` (CLOSE, `max_close_steps = 300` at
    `:513`, stall at `:553-555`) and `:562` `elif name == "release":` → `:586-615` (RELEASE,
    exit at `:614`) — so **all pipelines including `grasp_servo=none`** get the original's
    semantics by default; nothing gates either loop on `grasp_servo`.
  - **Baseline `phase3_gate10b/` ran the buggy single-step version (MEASURED):** ep01/ep06
    CLOSE enter 9.536 → exit 9.568 = **1.0 step (0.032 s)**, exit `gripper_opening = 0.04405`
    (fingers barely moved from 0.045); RELEASE exit `0.00096` (one step, then fingers opened
    passively during RETREAT/HOME). Identical to pre-pacing `phase4_gtpad3/` (CLOSE 1 step,
    exit `0.04404`) vs the paced gate run (CLOSE **31 steps**, exit `0.03143`). **The baseline's
    close/lift timing was therefore the same single-step version, and its 3 "reached HOME, can
    never lifted" failures (ep01/06/09, `dist 0.2196`) may include this same bug.** The other 7
    baseline failures (4× DESCEND move-timeout with arm_6 pinned at its −1.39 rad limit, 2×
    DESCEND IK-solve-failed in the 0.15 rad trust region — arm_6 also at −0.0012 margin, 1× CLOSE
    misaligned) are a different kind and are unaffected by close pacing.
  - **Remaining divergence (documented, not fixed):** the original gated LIFT on `grasped`
    (`"LIFT" if grasped else "RETRY"`); the brain is a linear waypoint flow with no RETRY state
    and no `MIN_GRASP_OPEN` check before LIFT (`MIN_GRASP_OPEN` is configured in
    `configs/robot.yaml:23` but unused by the policy). With the stall loop in place a failed
    grasp still fails the episode via the placement-distance gate. Adding RETRY would change
    the policy structure and is out of scope for M1.
- **New dependency:** None. No new imports or packages.

## ADR-0025 - M1-07 gate revised: fixed-scenario repeatability + 10-scene mini-generalization (user-directed, 2026-11-13)

- **Context (MEASURED, phase4d gt_pad ×10 batch):** the episode seed only seeds the IK
  random-restart RNG (`scripted_can.py:257` → `kinematics.py:92 default_rng(seed)`); the world
  resets to fixed poses, so the CAN pose is identical across all episodes of both campaign
  batches (|Δ| ~1e-7 m). gt_pad seeds 0/1/2 produced bit-identical trajectories (21 DESCEND
  servo records with identical md5, CLOSE t=98.848, end t=111.904, all `success=true`); seeds
  3/4 diverged at trajectory level (different servo series, end t=97.024/90.464) yet passed every
  criterion. The original 20-seed gate therefore measured repeatability at ONE fixed scene, not
  scene generalization, and costs ~8 h wall time.
- **Decision (user-directed, 2026-11-13):** M1-07 is satisfied by BOTH:
  (a) **5 consecutive fresh-world gt_pad episodes** with the unchanged success criteria
      (`DONE` at `HOME`, base-frame distance ≤ 0.10 m, can z within 0.02 m of rest, tilt < 15°);
  (b) **a 10-scene mini-generalization run:** CAN xy offset on a ±5 cm grid (9 positions =
      3×3 at {-0.05, 0, +0.05}² m) plus 1 rotated-yaw scene (same world otherwise), applied
      via the Supervisor at reset, one fresh world per scene, unchanged success criterion.
  Per-position results are reported. **≥ 8/10 full successes in (b) → M1 PASSED; otherwise
  M1 PASSED WITH LIMITATION**, with the failing positions carried into M2.
  This **replaces the 20-seed gate** (`docs/ARCHITECTURE.md` M1-07 updated in the same change).
- **Implementation (first piece of M2's ScenarioGenerator):** a config-driven helper that
  spawns/moves the CAN to (x, y, yaw) at reset, exposed through `ScenarioSpec` — a public
  interface change, recorded here per AGENTS.md rule 3 — and honored by both the Webots backend
  (the Supervisor field writes already used in `backend.reset`) and `FakeBackend` (CI). Nothing
  is thrown away: M2's seeded `ScenarioGenerator` extends this helper to vary can pose and
  distractors by seed, and `cli eval --pipeline scripted` will consume it.
- **Scope limits:** if reaching an extreme grid position would require moving the base, that is
  REPORTED, not fixed — no base-positioning work in M1. No thresholds, tolerances, or success
  criteria change anywhere in this decision.
- **Status:** decision ACTIVE; **implementation landed 2026-11-13** —
  `core/types.py::ScenarioSpec.can_xy_yaw_offset`, `backends/webots/frames.py::
  compose_can_reset_pose` (pure, import-safe), `backends/webots/backend.py::reset` composing
  the offset into the Supervisor field writes, `eval/scenarios.py::mini_grid10()`,
  `cli run-episodes --scenario` → `RunnerSettings.scenario` → policy → `backend.reset`,
  `EpisodeOutcome.scenario` for per-position reporting; tests in `tests/test_scenarios.py`
  and `tests/test_webots_backend_reset.py`; gates `ruff check`/`ruff format --check`/
  `pytest -m "not webots"` green (208 passed). The grid/scene constants live in
  `eval/scenarios.py` (evaluation scene set, not robot tuning — the robot config schema is
  strict). **Live evidence MEASURED 2026-11-14:** gate (b) run
  (`runs/campaign-2026-11-11/phase6_scene10/`, command in `docs/M1_VALIDATION_REPORT.md`
  Phase 4(e)) completed EXIT=0 with **9/10 full successes (90%), Wilson 95% CI [60%, 98%] ≥
  8/10 → gate (b) PASSES → M1-07 PASSES**. Per-scene: only `grid_dx-0.05_dy-0.05`
  (farthest corner) failed — deterministic IK rejection at PREGRASP realign
  (`error=0.014551 m` vs `0.005 m` limit, abort at sim 5.28 s), reproduced identically on
  relaunch (so not the server restart); `yaw90_center` succeeded (dist 0.0621 m) with the
  scene plumbing exercised. Base never commanded to move — drift ≤ 3.9 mm across all 10
  episodes (hold jitter only); the far corner needs base repositioning, reported and carried
  to M2 per the scope limit above. Mean 1376 s/episode (111–2157 s), 3.82 h total episode
  wall. A separate
  **~8.4 h inter-episode stall** was observed between ep02 and ep03 of
  the phase4d batch (no machine suspend; per-request step durations normal at mean 0.344 s;
  stall brackets the opencode session-idle window) — INFERRED to be background-shell freezing
  during session idle; batch throughput therefore depends on session activity.

## ADR-0026 - Waive M1-06a live 5+5 equivalence batch for the M1 verdict (user-directed, 2026-11-14)

- **Context (MEASURED):** `docs/ARCHITECTURE.md` §M1 acceptance requires live evidence for
  **both** M1-06a and M1-07 ("do not start M2 until both gates are accepted"). M1-07 is now
  satisfied (ADR-0025: (a) 10/10, (b) 9/10 ≥ 8/10). M1-06a's live batch was never collected:
  **0/5** `m1-06a-original-v2-*` traces exist on disk and only **1/5** brain traces
  (`runs/m1-06a-brain-01.jsonl`); every prior ADR lists M1-06a as UNVERIFIED, and ADR-0009
  already records that the trace-comparison *tooling* "does not constitute live M1-06a
  evidence". The original-side runs are operator-heavy (fixture worktree at `1b2e4da`,
  Webots restart + URL refresh per run; steps in `docs/MANUAL.md`).
- **Decision (user-directed, 2026-11-14):** the M1 verdict is closed on the revised M1-07
  gate alone. The M1-06a live 5+5 equivalence batch is **WAIVED for the M1 verdict** and
  recorded as a known evidence gap. Equivalence evidence for M1 instead rests on: the
  Webots-free unit tests (`tests/test_move_linear_original_equivalence.py`), the
  `compare-traces` contract tooling (`src/vlarobot/execution/trace_compare.py`, ADR-0008/0009
  semantics), and the original-vs-brain defect analysis that produced ADR-0024 (the one
  real behavioural divergence found — one-step CLOSE — was fixed and is live-verified by
  phase4/phase4d/phase6).
- **Consequences:** M1 is declared **PASSED** with M1-06a's live prefix-equivalence
  explicitly **UNVERIFIED** (no 5+5 run); this waiver changes the evidence bar for the
  M1 verdict, not any controller behaviour, threshold, tolerance, or success criterion.
  M2 may start. The waiver can be revisited later by running the `docs/MANUAL.md` batch;
  nothing in M2 depends on it.

## ADR-0029 - OWNER DECISION: four-calendar-week campaign (WorldMind Phase I conformance); supersedes ADR-0028 timeline (2026-11-14)

- **Status:** ACTIVE — controlling owner decision. Do not redesign or renegotiate its
  milestones, deadlines, thresholds, scene counts, experiment arms or scope.
  (Numbering note: ADR-0027/0028 are not present in this file; the owner's numbering is kept.)
- **Contract:** the submitted **WorldMind Phase I report (Oct 2026)** is the contract; where it
  is silent, the Steering Directive applies. The report PDF is **NOT in this repo** — owner must
  supply it before conformance work (HUMAN STEP, week 1). **No threshold/criteria changes.**
  **VLA is a CORE deliverable.**
- **Deadline:** hard, **4 calendar weeks from 2026-11-14 → 2026-12-12**. Week boundaries:
  W1 ends 11-20 (report 11-21), W2 ends 11-27 (gate by 11-27), W3 ends 12-04, W4 days 1-3 =
  12-05..12-07 (frozen test), days 4-7 = 12-08..12-11 (docs + demo).
- **Compute:** GPU = Google Colab (T4-class, sessions disconnect): all training resumable with
  checkpoints on Google Drive; inference runs LOCALLY (export the checkpoint; no tunnel to Colab).
- **SIZES (frozen now in `configs/eval.yaml`; the test split stays untouched until week 4):**
  train 8, validation 6, test 10 scenes, **by seed**, three families (clear, cluttered,
  reach-limited). Arms: scripted baseline; VLA-only; VLA + heuristic risk veto (no prediction);
  VLA + world model (no veto); full system. VLA arms use **3 seeds per scene**.
- **DESIGN:** VLA (SmolVLA) = primary proposer in `CandidateGenerator`; surrogate predicts each
  sampled chunk; **RiskGate** — hard constraints fail closed, then ranking by
  `J = wg*Jg + wl*Jl + wr*Jr + wu*Ju` (report Eq 3.8); **empty safe set → defined outcome**
  (re-observe, next candidate family such as base standoff, else abort with a recorded reason);
  **verify-and-retry once**. Scripted expert = demonstrator, baseline and fallback proposer.
  **Document the deviation from report Eq 3.2** (SmolVLA cannot consume a predicted state, so the
  architecture is propose-and-evaluate). **Oracle state with provenance flags** (report 4.3 allows
  it). **Surrogate = analytic geometric-kinematic model, no MuJoCo** (deviation from the previous
  M5 plan, documented); learned residual only if week 3 finishes early. **Collision metric =
  displacement (>5 mm) / tilt (>5 deg) proxy plus FK clearance, labelled a proxy.**
- **GATE, end of week 2 (decided):** on the 8 train scenes x 3 seeds, VLA-only success
  `>= min(40%, 60% of the scripted baseline's rate on the same scenes)`. Pass → VLA is the
  primary proposer. Fail → scripted proposer becomes primary for the main comparison; the VLA
  stays as its own arm and is reported as a MEASURED partial result. **Never tune on
  validation/test to rescue it.**
  - **Interpretation resolved (owner answer, 2026-11-14):** the **literal `min`** as written —
    the threshold is the WEAKER of the two bounds (baseline 90% → threshold 40%, not 54%).
    `max` was considered and rejected: it would tighten a written criterion.
  - **Baseline repetitions resolved (owner answer, 2026-11-14):** the scripted baseline runs
    **3 seeds per scene, the same seeds as the VLA arms** (paired (scene, seed) structure for
    McNemar; 24 train episodes; frozen in `configs/eval.yaml`).
- **WEEK 1:** speed (profile per-step cost; rendering stays ON for demo collection; larger
  descent increments once aligned, final approach unchanged; per-episode ports for 2-3 parallel
  instances; prove outcome equivalence over >= 10 episodes each before using
  `observe_images=false` or no-rendering for non-demo runs). Scenario generator (can xy/yaw and
  obstacle positions at reset), recorder, metrics, `cli eval --pipeline baseline --split train`,
  baseline numbers. Demo collection: `collect-demos` → LeRobot format (RGB, state, instruction,
  7-D EE-delta + gripper actions through the existing IK/safety layer), ~80 episodes with
  perturbed can poses in the reachable envelope; check LeRobot docs for demo-count guidance
  first; verify the policy config (dims, camera keys, normalisation) and write
  `docs/VLA_NOTES.md`. Save real screenshots/traces from week 1 on for the report.
- **WEEK 2:** finish demos; Colab notebook/script for resumable fine-tuning with exact commands
  and runtime estimate (owner runs it); local model server, ActionAdapter (normalisation,
  EE->IK, limits, chunk execution, invalid-output rejection with tests); VLA-only evaluation;
  surrogate in parallel with prediction error vs Webots outcomes (RMSE/MAE for continuous
  targets, Brier score for probabilistic binary outcomes — not conflated) by quantity and scene
  family; apply the gate.
- **WEEK 3:** K sampled chunks per decision (state the sampling method), surrogate prediction,
  RiskGate, retry loop; weights and thresholds chosen on VALIDATION only; full pipeline on
  validation.
- **WEEK 4 (days 1-3):** frozen test campaign across all arms; Wilson intervals; McNemar on
  paired scenes (**pairing unit = (scene, seed); repeated seeds within scenes accounted for by
  scene-level clustering**, not naive episode-level independence); failure categories; veto and
  no-safe frequencies; planning latency; **Table 5.1 metrics exactly** (every metric of the
  submitted report is preserved; additional metrics may supplement, not replace). (Days 4-7):
  `docs/MATH.md` (Eqs 3.1-3.12, 4.1, 5.1 → code), `docs/REPORT_CONFORMANCE.md` (table: each
  report objective/table/equation → implementation, evidence path, any deviation),
  `docs/REPORT_FIGURES.md` (every figure of the submitted report with the real trace +
  screenshot; Fig 4.1 shows a UR5/V-REP window and must be replaced; Figs 4.2-4.7 are mock-ups;
  risk values in Fig 4.7 must come from logs), demo run. Results only from MEASURED data.
- **REPORTING:** 5-line summary and gate check at each week boundary plus a HUMAN STEPS list
  (what the owner must run on Colab, when). MEASURED/INFERRED/UNVERIFIED labels; no single-run
  claims; process hygiene; commit small. **Missed gate: say so and propose the smallest further
  cut; never silently expand scope or relax a criterion.**
- **Deviations to document (clarification 7):** (a) propose-and-evaluate VLA vs report Eq 3.2;
  (b) analytic surrogate vs the previous MuJoCo-based implementation plan (M5).
- **Conformance guard (clarifications 2/8):** Table 5.1 metrics are preserved verbatim; the
  final report contains actual measured results only.
- **Training guard (clarification 9):** Colab training resumable from Drive checkpoints; local
  inference only; never depend on a Colab tunnel.

## ADR-0030 - Seeded scenario generator design: families, obstacle entries, pre-registration (week 1, 2026-11-14)

- **Context (week-1 deliverable of ADR-0029):** `ScenarioSpec.can_xy_yaw_offset` covered CAN xy/yaw
  only; ADR-0029 requires three families (clear, cluttered, reach-limited), obstacle positions at
  reset, split-by-seed scene counts (train 8 / validation 6 / test 10, frozen in `configs/eval.yaml`),
  and a test split untouched until week 4.
- **Decision — generator (`eval/scenarios.py::generate_scenes(split, seed, allow_test=...)`):**
  scenes are drawn from `np.random.default_rng(SeedSequence([seed, split_salt]))`, so
  `(split, seed)` deterministically defines the whole scene tuple (byte-identical on repeat);
  families are assigned round-robin by index (train 3/3/2, validation 2/2/2, test 4/3/3).
  The `test` split raises unless `allow_test=True` (fail-closed until week 4).
  `mini_grid10` and `SCENARIOS` are untouched (M1 evidence must stay reproducible).
- **Decision — geometry lives in `configs/scenarios.yaml` (pre-registered, week 1):**
  world layout facts (CAN/HAM/APP poses, drift-guarded against `worlds/factory.wbt` by test),
  family CAN radius bands (clear/cluttered 0–0.05 m, reach-limited 0.06–0.10 m — beyond the
  measured M1-07 grid, r ≤ 0.0707), and obstacle modes:
  *parked* r 0.55–0.70 m at θ ∈ [−30°, +10°] (on-table, ≥ 0.5 m from the CAN, away from the
  −x place corridor) for clear/reach-limited; *near_can* r 0.13–0.20 m with
  θ ∈ [−170°, +120°] for cluttered.
- **Decision — place-corridor cone exclusion:** the near_can θ range excludes the cone
  (120°, 190°) around the CAN because the fixed `PLACE_WORLD` (−1.0, −4.3) sits at 155.8°,
  0.22 m from the default CAN; an obstacle ON the target would make placement physically
  impossible for every experiment arm (degenerate scene for all arms). MEASURED worst-case
  cluttered-obstacle clearance to `PLACE_WORLD` over seeds 0–59 × train+validation:
  **0.0815 m** (worst angular separation 24.9°); the test asserts a 0.06 m design floor.
- **Decision — pre-registration + anti-rescue:** geometry is committed before any baseline or
  VLA result exists. Later changes are allowed only as feasibility repairs (scene unreachable
  for the scripted expert by construction), each documented here. Such repairs RAISE the
  week-2 gate threshold (`min(0.40, 0.60 × baseline)`), so they can never lower the bar for
  the VLA; val/test remain untouched until their scheduled weeks.
- **Decision — obstacle wire contract (`ScenarioSpec.objects`, no `core/types.py` change):**
  each entry is a mapping `{"def": <existing DEF>, "xyz": [world x, y, z],
  "yaw_offset": rad}` — absolute world translation, yaw composed onto the world-file rotation
  (same world-Z convention as `can_xy_yaw_offset`). The Webots backend places them at reset,
  caches the world-file pose on first contact, and restores cached nodes when a later scenario
  omits them; the CAN DEF is rejected in objects (use `can_xy_yaw_offset`); missing DEF and
  malformed entries fail loud; spawning new nodes stays `NotImplementedError`. `FakeBackend`
  stores the scenario unchanged (no physics there).
- **Decision — collision-proxy groundwork:** `WebotsBackend.get_ground_truth()` now also reports
  the poses of obstacles tracked by a scenario (keyed by DEF). This is the read path for the
  ADR-0029 displacement (>5 mm)/tilt (>5°) collision PROXY — labelled a proxy, with FK clearance
  kept as a separate measurement in the metrics module (step 3).
- **Consequences:** scene identity is stable for report joins
  (`train-clear-i00-s0` style names + `metadata` = family/split/seed/index); the generator is
  pure Python (no Webots import) and unit-tested; `configs/scenarios.yaml` ships via
  package-data. Nothing in M1 behaviour changes (default scenarios have empty `objects`, and
  the no-objects code path performs zero extra DEF lookups — asserted by existing tests).

## ADR-0031 - EpisodeRecorder: file contract, RecordingBackend decorator, Pillow dependency (week 1, 2026-11-14)

- **Context (week-1 deliverable of ADR-0029):** eval and demo collection need per-episode
  evidence (numeric streams, applied actions with safety verdicts, RGB frames, provenance) that
  LeRobot conversion can consume later, without touching policy/controller behaviour
  (AGENTS rule 10).
- **Decision — file contract per episode** (`eval/recorder.py::EpisodeRecorder`):
  `episode-XX.steps.npz` (t, joint_names/pos/vel, gripper, ee_xyz, base_pose, per-DEF gt_xyz
  with **NaN rows where a step reported no pose**), `episode-XX.actions.jsonl` (one record per
  applied action with `accepted`/`reason` from `safety_monitor.validate()`, timestamped from the
  last recorded observation), `episode-XX.predicted.jsonl` (world-model predictions; empty until
  M5), `episode-XX.meta.json` (scenario + metadata, reset seed, pipeline, **state-source
  provenance flag** for report 4.3, step/frame counts, outcome), and `frames/episode-XX/*.jpg`.
  The layout (image directory + tabular arrays + per-episode metadata) is LeRobot-convertible;
  the converter ships with `collect-demos`.
- **Decision — `RecordingBackend` decorator, not policy changes:** wraps any backend; `reset`/
  `step` capture the returned observation plus one `get_ground_truth()` read, `apply` captures
  the action and its safety verdict, everything else delegates via `__getattr__`. The scripted
  policy run through the wrapper must produce an **identical `ScriptedCanResult`** to the
  unwrapped run (asserted: `test_recording_backend_is_policy_transparent`). The wrapper reads
  `get_ground_truth()` once per recorded step, so backends used with recording must keep that
  method side-effect free (true for Fake/Webots: pure field reads).
- **Decision — fail-loud misuse guards:** one open episode at a time; exactly one `reset` per
  episode (`on_reset` raises on a second reset — `probe_initial_pregrasp` is a separate CLI
  command with its own episode, unaffected); a mid-episode joint-key change raises; invalid
  `frame_hz`/`jpeg_quality` raise at construction. Frame capture is rate-limited by
  `frame_hz` (sim time, not wall clock) and can be disabled (`frame_hz=0` or
  `record_frames=False`) for non-demo runs.
- **Decision — Pillow for JPEG frames:** new dependency `pillow>=10.4` (licence **MIT-CMU**,
  permissive; pure-Wheel, no compiler needed) added to the `dev` dependency group (tests) and a
  new `project.optional-dependencies.eval` extra (production eval installs:
  `uv sync --extra eval`). Imported lazily inside `_write_frame`, so a missing Pillow fails with
  an actionable message only when frames are requested. **Windows install UNVERIFIED** in this
  environment (Linux MEASURED: Pillow 13.0.0 installs via uv); standard wheels exist for CPython
  3.11 on Windows, but that path has not been run here.
- **Decision — env sync caveat (MEASURED):** tests need `uv sync --extra sim --extra dev`; a
  plain `uv sync --extra dev` prunes `ikpy`/`scipy` and breaks the suite. Documented in
  `docs/SETUP.md`.
- **Consequences:** recordings are byte-deterministic for identical fake-backend runs (npz array
  contents, JSONL, and JPEG bytes compared across two runs by test); the npz container itself is
  not byte-stable (zip entry timestamps) so determinism assertions compare array contents.
  Wall-clock never enters the recording (all timestamps are sim time). No public interface in
  `core/` changed.

## ADR-0032 - Metrics module: statistics primitives, collision-proxy vs contacts vs FK clearance, scene-clustered pairing (week 1, 2026-11-14)

- **Context (week-1 deliverable of ADR-0029):** campaign reporting needs one authoritative
  implementation of the statistics and the three collision-adjacent measurements, with the
  (scene, seed) pairing rule and the independence limitation encoded in code rather than
  convention. Contract: `docs/STATISTICS.md`.
- **Decision — statistics primitives** (`eval/metrics.py`): Wilson 95% interval for all
  proportions (better small-n/boundary coverage than Wald; e.g. 0/10 → (0.000, 0.278));
  mean + sample std (ddof=1, NaN for n=1, never 0); **RMSE/MAE only for continuous
  targets**, **Brier only for probabilistic binary outcomes**, never conflated;
  `mcnemar_exact(b, c)` = two-sided exact p (Binomial(b+c, ½), doubled smaller tail,
  capped at 1). `summarize_success` enforces count+rate+CI (no single-run claims).
- **Decision — pairing unit is (scene, seed), inference unit is the scene.**
  `paired_comparison(a, b)` requires identical `(scene, seed)` keys (raises otherwise) and
  returns BOTH levels: episode-level McNemar (pairs preserved, p labelled **descriptive
  and anti-conservative** — repeated seeds within a scene are correlated, 30 pairs ≠ 30
  independent observations) and the **scene-level exact sign test** as the cluster-aware
  analysis, with `independent_units` = scene count. Per-scene rows keep within-scene
  variation (rates + sample std per arm) visible. `scene_id()` strips the `-s<seed>`
  suffix for pairing.
- **Decision — three collision-adjacent channels stay separate (labelled everywhere):**
  1. **PROXY** = object displacement from its **reset pose** > 5 mm OR tilt > 5° (frozen
     thresholds from `configs/eval.yaml`; strict `>`; max excursion drives the hit, final
     reset→last values also reported; NaN rows skipped; no tracked objects → `hit=False`
     with `None` magnitudes). Columns are prefixed `proxy_`.
  2. **Actual collision** = simulator contact nodes, recorded by the EpisodeRecorder into
     `episode-XX.contacts.jsonl` (new channel, added with this ADR) and read back by
     `load_episode`; reported as `contact_steps`/`contact_nodes`, never merged with or
     derived from the proxy (both cross-cases asserted by tests).
  3. **FK clearance** = `min_fk_clearance_m(points, objects)`, pure-geometry
     surface-to-surface clearance (negative = penetration), its own column
     `fk_min_clearance_m`. Eval may not import `backends/`, so callers supply FK points.
     It is not a proxy substitute.
- **Decision — recorder extension (same commit):** `steps.npz` now also carries `gt_quat`
  (N × D × 4, (x, y, z, w), NaN-padded like `gt_xyz`) for the tilt half of the proxy;
  `EpisodePaths` gains `contacts`; `meta.json` gains `contact_steps` and `gt_names`.
  The actual-collision channel exists because `GroundTruth.contacts` is never populated —
  real contacts surface in `robot.diagnostics["contact_nodes"]` (Webots
  `RobotIO.contact_nodes()`), which the RecordingBackend already captures per step.
- **Decision — output writers:** `write_csv` (first-seen column order) and `write_json`
  (dataclasses/NumPy/tuples/Path encoded) for per-episode rows and summaries.
- **Status of Table 5.1:** the submitted WorldMind Phase I report is still not in the repo
  (HUMAN STEP). This module provides primitives/plumbing only; **no function is claimed to
  implement a Table 5.1 metric**, and the conformance mapping stays blocked until the
  owner supplies the report (ADR-0029 clarification 2/8: all eight metrics preserved
  verbatim, supplement only).
- **Consequences:** 14 new tests (Wilson/McNemar/Brier known values, tilt geometry,
  proxy thresholds + strictness + NaN/empty handling, channel separation, FK clearance,
  episode-vs-scene pairing counts, writers); full suite **242 passed**, ruff/format green.
  No public interface in `core/` changed; no frozen config changed.

## ADR-0033 - Baseline eval command, seed-major episode plan, FakeBackend can-aware GT (week 1, 2026-11-14)

- **Context (week-1 deliverable of ADR-0029):** week 1 needs `eval --pipeline baseline
  --split train` to run the scripted baseline, record every episode, and emit a reproducible
  report (episodes CSV + summary JSON) before the week-2 VLA gate.

- **Decision — CLI and fail-closed guards** (`cli.py:eval`, ADR-0029): `--pipeline` accepts
  `baseline` and the alias `scripted` (canonical name `baseline`); other arms fail loud
  ("not implemented") in week 1. `--split` is `train|validation` only; the `test` split is
  not offered and `build_episode_plan` refuses it without an explicit week-4 override, so the
  frozen test set stays untouched (ADR-0029). `--n` defaults to 0 (every planned episode) and
  fails loud when `n` exceeds the plan. `--backend fake|webots` with provenance labelled
  `fake` vs `measured`; frames off by default. `--out` defaults to
  `runs/eval/<pipeline>-<split>-s<seed>`. A shared Webots session (`launch_webots_session`,
  one process, soft reset per episode) serves a run so N episodes do not pay N startups.

- **Decision — seed-major episode plan and pairing semantics** (`eval/runner.py`,
  `docs/STATISTICS.md`): `configs/eval.yaml` freezes 3 baseline seeds per scene, and the
  generator draws geometry from the seed. The plan is therefore **seed-major**: for each
  `base_seed + k` it generates that seed's full split, so a train baseline is
  `[seed, seed+1, seed+2] × 8 scenes = 24` episodes. The pairing key is `(scene_id, seed)`
  where `scene_id` strips the generator's `-s<seed>` suffix; because the seed also perturbs
  geometry, a paired unit is a scene slot/family within one repetition, not a fixed geometry.
  Consequences for McNemar (descriptive/anti-conservative at episode level) and the
  scene-level cluster-aware analysis are recorded in `docs/STATISTICS.md` (ADR-0032).

- **Decision — backend/policy seam (no `backends/` import in eval):** `eval/runner.py` depends
  only on `core` and `eval` modules. The CLI supplies the connected backend and
  `execution.scripted_baseline.scripted_policy_factory`, which adapts `ScriptedCanPolicy` to
  the plain outcome mapping `eval.metrics` reads. `fk_min_clearance_m` is reported as
  unmeasured (`null`) until an FK-point source is wired (standalone measurement, never
  invented). Each episode is wrapped by `RecordingBackend`; an exception becomes a recorded
  failure row so one bad episode does not lose the run.

- **Decision — FakeBackend can-aware ground truth:** `FakeBackend` now reports the CAN at
  world home (`CAN_WORLD_XYZ`, mirroring `configs/scenarios.yaml` `world.can_xyz`, drift-
  guarded by test) plus `can_xy_yaw_offset`, and obstacle DEFs from `scenario.objects` as
  absolute world translations with `yaw_offset` encoded as a world-Z (x, y, z, w) quaternion.
  This makes the scripted expert's CAN lookup and eval GT/proxy paths usable on the fake
  backend (default scenarios now report `{"CAN": ...}` rather than an empty pose map; affected
  tests updated). The fake backend remains pure Python with no simulator dependency.

- **Decision — report contents:** `episodes.csv` (one row per episode, flattened metrics
  including proxy/contact/FK channels), `summary.json` (Wilson 95% success interval, per-scene
  variation with rates/CIs and seed lists, week-2 gate threshold `min(0.40, 0.60×baseline)`,
  failure-state counts, proxy/contact tallies, continuous stats mean±std over finite values,
  backend/provenance/state-source labels), `plan.json`, and `timings.json` (per-episode wall
  time + run total; not a metric). A `--resume` run reuses a completed, plan-matching episode
  recording and re-runs incomplete/exception recordings.

- **Consequences:** a fast, deterministic fake-backend eval (`eval --backend fake --n 1`)
  produces byte-identical metric rows and summary statistics across runs (MEASURED by tests).
  **Webots-measured baseline numbers are not produced in this commit** (running the full 24-
  episode train baseline takes ~9.2 h single instance at the phase6 rate; throughput/
  equivalence proofs and the MEASURED baseline are step 4/5 follow-on work, reported
  UNVERIFIED here). No public interface in `core/` changed; no frozen config changed.


ADR-0037 — Isolated deterministic analytic rollout and five-arm mechanics (2026-11-15). Conditional attachment requires explicitly observed holding and never declares grasp from closing. Swept geometry requires coverage/speed certificates. Probabilities, report Eq. 3.8 components and validation-selected weights are explicit inputs, not guessed defaults. Approximate Jacobian planning IK is rejected on exact FK residual failure and does not replace execution IK. Registered replay and latency evidence are in WORLDMIND_DEVELOPMENT_CHECKPOINT.md; the 32-candidate latency target is not yet met. No contract, source training bundle, dataset, model, baseline or test split changed.
ADR-0045 — Content-verified historical canary version advance (2026-11-16).
User-supplied live observations show a matching canary SHA/size/ID/owner/parent
at receipt version1 and current version2. Drive version includes server metadata
changes and is not a content hash. The direct cause of the rejection is the old
historical-version equality predicate; the precise remote change is unproven.
Receipt upload uses a separate generated file ID and does not directly edit the
canary. Accept a positive numeric nondecreasing historical version only after a
fresh full API SHA-256 read, strict expected identity/ownership/parent/size and a
stable current version across that read. Reject rollback, malformed/missing version,
any content mismatch or mid-read metadata/version change. Preserve original
config/receipt byte-for-byte; append independently verified advances to a local
revalidation log. Checkpoint upload, COMPLETE validation and export checks remain
unchanged. Reconnect the failed setup's token broker using existing second-account
authentication; create no new remote folder/file. No new scopes/dependencies or
changes to ADR-0029, dataset, training configuration or evaluation criteria.

ADR-0044 — Second-account Drive API checkpoint handoff (2026-11-16).
At the owner's request, v6 supports a new folder directly under the authenticated
second account's My Drive root, replacing the unavailable private SSH route.
Checkpoint creation remains Colab-local. Each synchronous upload verifies current
principal, ownership, parent, size and stable version, plus server SHA-256 or full
independent API read-back SHA-256 if that field is absent. MD5 alone cannot pass.
Receipts are uploaded and read-back verified. Any failed persistence gate aborts
before another optimizer update. Final portable model is verified before full
resumable state. Full checkpoint archives include the pinned external processor
snapshot as well as original normalization, scheduler, optimizer and RNG files.
The existing Colab kernel refreshes a private local access-token file; tokens are
outside checkpoint/archive paths. No new worker dependency or infrastructure.
This deliberately replaces independent-provider storage with point-in-time Drive
verification, not a claim of indefinite retention. A small API canary measures no
full-size throughput; the final portable export must be independently downloaded
to Linux promptly. Original artifacts, approved configuration, dataset, splits,
evaluation criteria and ADR-0029 are unchanged. No training/evaluation authorized
or executed by preparation. Interrupted 18/19-to-20 lineage remains a separate gate.

### Owner-authorized A-D2 CPU infrastructure (implementation note)

The isolated `vlarobot.diagnostics.ad2` package verifies immutable A-D1 inputs and
compares explicitly supplied reset criteria. Its coordinator accepts inert CPU fixture
records only; there is no live adapter, launcher or physical diagnostic command. No
exploratory tolerance is enabled. Static collision-source enumeration is not swept-volume
certification. Live A-D2 remains NO-GO. See `AD2_CPU_INFRASTRUCTURE_REPORT.md` for fixture
and hash evidence and remaining approval boundaries. No change to ADR-0029, public core
interfaces, frozen0/24 result, FAILED gate or scripted-primary decision.

## Week 3 planning-latency evidence correction addendum (2026-11-17) — append-only

This addendum corrects the stale latency interpretation in ADR-0037 without rewriting that
ADR or changing ADR-0029. The authoritative recorded evidence is
`docs/evidence/planning-optimization-final.json`, an identical-input profile of the nominal
analytic kernel at K=32 candidates, H=50 prediction steps, and 1,600 control steps per
batch. It reports optimized p50 `0.4677 s`, optimized p95 `0.4803 s`, cold batch `0.4746 s`,
and pre-optimization p50 `2.0996 s`, with zero state difference, identical vetoes, unchanged
safety/trust thresholds, no model inference, no TEST access, and the uncertified full-arm
sweep still vetoed.

This evidence is explicitly **nominal-kernel-only**. Full-system latency, VLA-proposer
latency, full-arm-sweep latency, and scored per-decision latency remain **UNVERIFIED**.
The below-1-second milestone is not claimed as met. No candidate count, horizon, safety
threshold, veto, seed, split, frozen gate, or experiment design was changed. No physical,
inferred, training, IK, Webots, actuation, or scored evaluation was executed for this
addendum.

## ADR-0046 — Additive risk-gate decision observability and validation preflight (2026-11-17)

**Status:** Accepted. **Scope:** bounded, sidecar-only, inert.

- **Context:** the post-recovery Week 3 phase requires typed decision evidence and exact
  validation readiness without reopening frozen results. Execution-path inspection found no
  production end-to-end WorldMind composition (`cli.brain()` only steps a backend;
  `cli.eval()` only runs the scripted baseline); the full chain exists only in tests/tools.
  Archived scored evidence cannot retroactively supply executed-candidate↔prediction IDs,
  typed hard-check codes, per-candidate J components, pre-apply veto detail, decision IDs, or
  phase-separated timing without behavior-sensitive planner/recorder changes.
- **Decision:** add three pure modules — `eval/decision_evidence.py` (typed, fail-closed
  schemas), `eval/decision_metrics.py` (pure aggregation, pairing, latency summary), and
  `eval/decision_adapters.py` (optional delegate-only probes that call the original once and
  return the original object) — plus a synthetic fixture, 18 inert tests, a read-only frozen
  hash checker (`tools/verify_week3_frozen_evidence.py`), and documentation. All missing values
  stay `None` with an explicit reason; no value is invented. Risk rejection rate is
  `hard_vetoed / (hard_vetoed + safe_unselected + selected)`, invalid/unevaluable excluded and
  reported separately, zero denominator → `None`. Prediction pairing requires executed-candidate
  ID identity, explicit `base` frame, aligned timestamps and explicit recovery/abort boundaries.
  Latency is opt-in whole-phase `perf_counter_ns()` with independent, non-additive p50/p95.
- **Telemetry is sidecar-only.** Do not wrap `RiskGate.select()` (called N+1 times by
  `ComparisonPlanner.select`); wrap injected `predict`/`assess`, whole `execute_first`, verifier
  calls and delegate-only monitor/backend probes. Nothing is wired into a production path.
- **No guessed bindings.** The Eq. 3.8 component meanings and eight Table 5.1 names/units are a
  proposed mapping for owner review in `docs/WEEK3_DECISION_OBSERVABILITY.md` §6. No production
  `definitions.json` is created; scoring stays BLOCKED until a reviewed binding matching report
  SHA-256 `23d5ff15…1522b` is supplied.
- **Frozen facts unchanged:** scripted TRAIN 23/24, scripted VALIDATION 15/18, VLA-only TRAIN
  0/24 FAILED, scripted-primary, A-D2 CLOSED/NO-GO, K=32/H=50, seeds/splits/gates, archived
  evidence and scored summaries. No public `core/` interface changed. Deadline 2026-12-12;
  TEST window 2026-12-05..12-07 untouched. No Webots, real IK, inference, training, actuation,
  scored evaluation, TEST access, push, merge, PR #2 or `main` change in this phase.
- **Status consequences:** risk rejection rate and scored prediction error remain `Missing /
  BLOCKED` (definition and aggregation are INERT-UNIT-TESTED only). Nominal-kernel latency remains
  scope-limited; full-system/scored per-decision latency is UNVERIFIED.

## ADR-0047 — Production WorldMind composition, fail-closed config and inert evidence (2026-11-17)

**Status:** Accepted. **Scope:** implementation and inert FakeBackend verification only.

- **Context:** ADR-0046 added decision observability but recorded that no production end-to-end
  WorldMind call graph existed. The full chain was exercised only by tools/tests. This phase adds
  that single production composition without changing the frozen baseline, planner math, gate,
  safety monitor, recorder schema, or any public interface in `core/`.
- **Decision:** compose the existing `SeededCandidateGenerator` (K=32), owner-registered
  `AnalyticSurrogate` (H=50, dt=0.032 s), `ComparisonPlanner`/`RiskGate`, `execute_first` plus the
  live `SafetyMonitor`, `PlacementVerifier`, bounded `RecoveryState` and frozen `run_evaluation`
  in `execution/worldmind_pipeline.py`. Add a separate `execution/scripted_chunk_adapter.py` that
  emits reviewed, seeded, H-step `ActionChunk`s and is explicitly **not** the frozen
  `ScriptedCanPolicy`. It cannot discover geometry, perform a new grasp, simulate contact or
  recover from bad IK; unreviewed poses, residual breaches and out-of-bound jitter fail closed.
  Jitter uses rejection sampling and is never clipped or silently accepted.
- **Safety and latency:** `SafetyMonitor.validate()` still precedes every `backend.apply()`;
  execution is one control tick per decision. H=50 predictions are paired only with truthfully
  observed states through `allow_truncated=True`; the unexecuted horizon is never claimed as
  observed or scored. `ComparisonPlanner.select()` retains the exact N+1 gate contract: K=32
  produces 33 `RiskGate.select()` calls per decision. Sidecars record this count and the
  whole-selection latency window; hard checks and gate ranking remain fused inside the planner
  and are never reported as separately measured phases.
- **Fail-closed configuration:** `configs/worldmind_production.yaml` is a `status: blocked`
  template. `eval/worldmind_config.py` requires the authoritative report SHA-256, exactly four
  `{Jg,Jl,Jr,Ju}` definitions and exactly eight Table 5.1 `{definition, unit}` entries, approved
  VALIDATION calibration evidence, frozen seeds/splits/K/H/tick, a declared backend/state source,
  and explicit registered Python bindings for every component, check and factory. The CLI
  `eval-worldmind` uses `empty_worldmind_bindings()` and therefore cannot run until reviewed
  bindings are registered in code. It permits VALIDATION only; TEST and TRAIN stay sealed.
- **Evidence:** `eval/worldmind_evidence.py` writes additive non-overwriting sidecars
  (`decisions.jsonl`, `prediction_pairing.jsonl`, `errors.jsonl`, `risk_rejection.json`,
  `phase_latency.json`, `verification_recovery.json`, `provenance.json`) under a fresh output
  directory. Records include real draw/decision seeds, candidate IDs, gate-visible hard checks and
  J components, dispositions/vetoes, the selected index/score/rationale, pre-apply verdict,
  backend apply/step evidence, verification and recovery transitions, paired prediction identity
  and error metrics, and measured phase durations. Missing values stay absent with reasons;
  absent authoritative definitions, calibration or bindings are never invented.
- **Verification:** 14 new inert FakeBackend tests cover composition, seeded determinism,
  veto precedence, no-safe recovery, live pre-apply veto, verification recovery, selected-trajectory
  pairing, exclusive sidecars, config failure modes and CLI compatibility. No Webots, real IK,
  neural inference, training, actuation, scored evaluation or TEST access was executed.
- **Frozen facts unchanged:** scripted TRAIN 23/24, scripted VALIDATION 15/18, VLA-only TRAIN
  0/24 FAILED, scripted-primary, A-D2 CLOSED/NO-GO, K=32/H=50, all seeds/splits/gates/thresholds,
  archived evidence and scored summaries. Deadline 2026-12-12 and TEST window 2026-12-05..12-07
  are unchanged. No dependency added; `configs/worldmind_production.yaml` is added to packaged
  data files. No push, merge, PR #2 or `main` change.

## ADR-0048 — Concrete WorldMind registration, calibration gates and structured preflight (2026-11-17)

**Status:** Accepted. **Scope:** implementation and inert FakeBackend verification only; the pilot
remains blocked and was not run.

- **Context:** ADR-0047 left `eval-worldmind` bound to `empty_worldmind_bindings()`, so no concrete
  component could ever resolve and no calibration/report/preflight machinery existed. This phase
  binds the existing implementation pieces and adds review/approval gates without inventing any
  scientific value or running the pilot.
- **Concrete registration:** `backends/worldmind_bindings.py` registers the existing
  `AnalyticSurrogate`, live `SafetyMonitor`, `PlacementVerifier`, corrected `BoundedPoseSolver`
  (hash-pinned URDF registration `dfc542e1…5ea0`), corrected scripted waypoint factory and a
  finite-chunk hard check. All geometry is built lazily behind the backend boundary; constructing
  the registry does not launch Webots, solve IK, infer, train, actuate or access TEST. The surrogate
  intentionally registers `check_transition=None`, so with no certified full TIAGo swept-arm
  geometry `unverified-arm-sweep` remains a hard rejection. `eval/worldmind_bindings.py` holds only
  pure binding identities, so config/validation stay importable without any simulator. Eq. 3.8
  components, weights, ranking direction, risk thresholds and calibration evaluators remain
  **unregistered** and therefore fail closed.
- **Calibration:** `eval/worldmind_calibration.py` accepts only an owner-approved VALIDATION
  candidate whose evidence is 18 frozen VALIDATION episodes, hash-locked, on permitted parameters
  (`weights`, `maximize`) only. TEST/TRAIN paths are rejected by string and path check. Selection is
  the deterministic, declared `registered_evaluator_max_score_then_candidate_id`; the evaluator is
  an explicit registered binding, never inferred. No score, weight or objective is invented.
- **Report candidate:** `configs/report/definitions.candidate.json` records the exact J component
  symbols and the exact eight Table 5.1 names, with unresolved units and owner decisions listed
  explicitly. `report_contract.py` enforces `review_status` so the candidate is reviewable but not
  scorable; `ReportDefinitions.load()` still requires `review_status=approved`.
- **Preflight:** `eval/worldmind_preflight.py` collects every gate (split, state source, fresh
  output, config status, backend declaration, report candidate, bindings, calibration, frozen
  VALIDATION plan, backend boundary probe, calibration evidence) and returns a structured result;
  `ready` is true only when all pass. A missing probe or evaluator blocks. `eval-worldmind` now
  constructs `production_worldmind_bindings()` and uses this preflight; the shipped blocked config
  exits `eval-worldmind preflight failed` with code 1 and no output directory.
- **Oracle boundary:** provenance now labels that the privileged oracle supplies only the target
  pose while robot fields come from the declared observation, sets `perception_equivalent=false`,
  and separates collision proxy (displacement/tilt), backend-recorded contacts, and the absent
  full-arm sweep certificate. Oracle-assisted results are never equated with perception performance.
- **Verification:** 18 inert FakeBackend tests pass; Ruff, format check, `ty check src` and
  `compileall` pass; frozen evidence checker ends with `test_split_accessed: false`. The exact
  fail-closed CLI command above was run against the shipped config and captured. No Webots, real
  IK, inference, training, actuation, scored campaign or TEST/TRAIN access occurred. Frozen facts
  from ADR-0047 are unchanged. No push, merge, PR #2 or `main` change.
- **Remaining owner decisions (pilot blockers):** authoritative report PDF and approved
  definitions; approved VALIDATION calibration artifact/evaluator; four Eq. 3.8 weights, ranking
  direction and component formulas/units (and any non-safety `Jr` threshold only with an exact
  authoritative definition); a certified full-arm swept-volume check; reviewer sign-off on the
  scripted waypoint cadence/orientations; and authorization for the controlled scored pilot.

## ADR-0049 — Calibration bootstrap, Eq. 3.8 proposal, swept-arm gate and waypoint cadence (2026-11-17)

**Status:** Accepted. **Scope:** inert implementation, review artifacts and local verification only;
the controlled VALIDATION pilot remains **NO-GO** and was not run.

- **Calibration bootstrap (four separated stages):** add a `diagnostic_uncalibrated` config path
  (`configs/worldmind_diagnostic.yaml`) that keeps every hard check, frozen VALIDATION split,
  seeds, 18 episodes, K=32, H=50, 0.032 s tick, oracle target source, report SHA and surrogate/
  safety/verification/adapter gates, but contains no Eq. 3.8 weights, ranking direction,
  calibration candidate or calibration evaluator. `build_calibration_evidence()` writes only
  unapproved VALIDATION evidence in `evidence_collection_only` mode; `rank_calibration_candidates()`
  returns an inert `CalibrationProposal` with no parameters, selection or approval record;
  `worldmind-calibration-proposal` prints a deterministic selection statistic and never persists
  it. Only after owner approval may a separate `ready` config freeze those parameters;
  `evaluate_calibration()` labels its output `selection_statistic_not_independent_evaluation`,
  `independent_evaluation: False`, `test_split_accessed: False`,
  `frozen_evidence_modified: False`. No hard constraint was weakened, no calibration result was
  invented, post-selection statistics are never presented as unbiased evaluation, and frozen
  VALIDATION scenes/seeds are unchanged.
- **Eq. 3.8 proposal:** `eval/worldmind_objectives.py` and
  `configs/report/objectives.proposal.json` supply explicit engineering-choice formulas for `Jg`,
  `Jl`, `Jr` and `Ju`, with dimensionless normalisation, units, required inputs, missing-data
  failure, all-minimise ranking and hard-veto precedence. They are marked
  `engineering_choice_not_report_quotation`, `authoritative: false` and
  `scored_execution_enabled: false`. They are registered nowhere in production; the scientific
  component/weight/evaluator registries remain empty and fail closed.
- **Swept-arm gate:** `world_model/arm_sweep.py` adds a conservative interpolated full-arm check
  that calls `SweptArmCheck` on explicit sub-intervals using capsule bounds derived from current
  geometry. It fails closed on missing/incomplete geometry, invalid transforms, unsupported
  shapes/contacts, moving scene objects or unsupported motion. The opt-in
  `unverified_urdf_sweep_check` is deliberately outside `WORLD_MIND_BINDING_IDS`, the production
  registry has no `arm_sweep_checks`, and `analytic_surrogate_factory()` still uses
  `check_transition=None`; therefore `unverified-arm-sweep` remains a hard veto until an
  owner-reviewed full-arm certificate exists.
- **Waypoint cadence:** the production adapter now emits the frozen baseline station order
  `INIT(open) -> PREGRASP -> DESCEND -> CLOSE -> LIFT -> TRANSIT -> PLACE_DESCEND -> RELEASE ->
  RETREAT -> HOME`, with explicit gripper and joint stations, state-specific reviewed IK
  tolerances and per-decision re-derivation from the latest oracle state. The hold steps sum to
  H=50 but are only a scripted cadence proxy, not measured settle/contact equivalence.
  `execution/scripted_can.py` was not modified. Remaining IK, orientation, gripper-stall and
  recovery differences are documented in `docs/WORLDMIND_WAYPOINT_CADENCE_REVIEW.md` as physical
  validation requirements.
- **Verification:** Ruff, format check, `ty check src`, `compileall`, the frozen-evidence checker
  and the 38 focused WorldMind tests pass. The full non-Webots suite is
  `1 failed, 569 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  hash mismatch against frozen Drive evidence. Frozen evidence was not altered. The exact
  first diagnostic command is recorded in `docs/WORLDMIND_PILOT_READINESS_REPORT.md`; with the
  shipped blocked template it fails closed with code 1 and creates no output directory.
- **Unverified and blocked:** no Webots execution, real IK solve, inference, training, actuation,
  scored evaluation, TEST/TRAIN access, pilot execution, full-arm certificate or physical
  waypoint validation occurred. No dependency, public `core/` interface, frozen result, split,
  seed, threshold, threshold source or baseline controller behaviour changed. No push, merge or
  PR update was performed.

## ADR-0050 — Dedicated no-actuation Webots diagnostic (2026-11-17)

**Status:** Accepted. **Scope:** inert implementation and local verification only. The first
Webots diagnostic was **not run**; it remains blocked on separate owner authorization of the exact
command recorded in the final task specification.

- **Read-only boundary:** a new backend-internal `capture_read_only_state()` helper and an
  `inspect` ZMQ operation return simulator identity, robot identity, an observation and the oracle
  ground-truth payload without calling `step`, `apply`, `reset` or the kinematics-export probe.
  The frozen `backends/webots/backend.py` source was deliberately left byte-identical. `core/wire.py` and all
  public `core/` interfaces are unchanged. The brain-side adapter wraps the client so `apply`,
  `step`, `reset` and `export_kinematics` raise and increment attempt counters before reaching
  the transport; the diagnostic calls only `ping()` and `inspect()`.
- **Dedicated command and config:** `worldmind-webots-diagnostic` is separate from
  `eval-worldmind`, `worldmind-diagnostic`, `eval` and all scored commands. It has no `--split` or
  `--state-source` option; `configs/worldmind_webots_diagnostic.yaml` is VALIDATION-only and
  `oracle`-only, names the declared backend/robot/world/Webots version, pins the world SHA-256,
  and rejects scoring, TEST/TRAIN, calibration, component, weight and evaluator keys.
  `launch` starts one Webots process with `VLAROBOT_WORLD_SHA256` exported to its controller;
  `connect` requires the owner to start that server with the same value.
- **Evidence:** each run creates one exclusive output directory and writes `diagnostic.json`,
  `state.json` and `provenance.json` with `open("x")`; prior evidence is never modified. The
  report records session identity, endpoint, robot/Webots identity, geometry and registration
  hashes, oracle declaration, observation timestamp/age/validity, dependency availability, safety
  readiness, forbidden-call counters, missing swept certificate, explicit no-actuation, and exact
  PASS/FAILED outcome. Oracle state is explicitly labelled **not** perception-equivalent.
- **Safety precedence:** full-arm sweep remains uncertified and `unverified-arm-sweep` stays a
  hard motion veto. A missing certificate does not block read-only inspection, and a diagnostic
  `PASS` never sets `motion_allowed=true`. Missing geometry, non-finite/stale state, identity or
  world-hash mismatch, unavailable dependencies, any forbidden-call attempt and any observed
  actuator-command marker fail closed. No safety threshold was changed, no end-effector-only
  clearance was substituted and no FakeBackend evidence was treated as physical validation.
- **Verification:** Ruff, format check, `ty check src`, focused WorldMind/inert diagnostic tests,
  compileall and the frozen-evidence checker pass locally. No Webots process, real IK solve,
  inference, training, actuation, scored evaluation, TEST/TRAIN access or pilot execution was
  performed. Windows execution remains unverified. No dependency or version pin changed.

## ADR-0051 — Repair no-actuation diagnostic attestation (2026-11-18)

**Status:** Accepted. **Scope:** inert implementation, regression tests and local verification only.
The first authorized Webots diagnostic failed closed at `7b5b71b`; its four evidence files and
hashes remain byte-identical. Webots and the diagnostic were **not** rerun.

- **Envelope parsing:** `ZmqBackendClient.ping()` returns the full reply envelope, while the
  SimServer emits `session_id` and `world_sha256` under `payload`. The adapter now extracts only
  that documented payload; absent payload/non-mapping replies fail closed. No expected value is
  copied into an observed field and the wire protocol is unchanged.
- **Robot identity:** `TIAGO_ROBOT` is the world DEF identifier, not a node `name` field. The
  read-only helper reads `getDef()`, cross-checks it with Supervisor `getFromDef()` and node
  `getId()`, and keeps DEF, node name and robot type as distinct reported fields. A missing or
  uncross-checked DEF and a DEF/name mismatch both fail closed.
- **Runtime version:** the installed controller API exposes no runtime `getVersion()` source;
  `webots_version(io)` therefore yields `None`. The configured `R2025a` string and the local
  `resources/version.txt` are not substituted for runtime evidence. No additive server field is
  introduced because no independently observed simulator version exists to emit; a missing
  version remains a hard blocker until an owner-approved runtime source is defined.
- **Attestation distinctions:** the report separates missing/incomplete session identity,
  missing world hash, actual world-hash mismatch, missing/uncross-checked robot DEF, actual robot
  identity mismatch, missing runtime version and actual version mismatch. Observation validity is
  computed from observation/staleness checks alone so attestation failures are not mislabelled as
  bad state; the overall outcome still fails closed on either class.
- **Verification:** Ruff, format check, `ty check src`, `compileall`, 35 focused inert tests, the
  frozen-evidence checker and all four preserved SHA-256 values pass/verify. The full non-Webots
  suite is `1 failed, 583 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch. No Webots execution, real IK, inference, training, actuation,
  scored evaluation, TEST/TRAIN access, threshold/seed/scenario change or push/merge/PR update
  occurred. `unverified-arm-sweep` remains a hard motion veto.

## ADR-0052 — Launch-time Webots runtime version attestation (2026-11-18)

**Status:** Accepted. **Scope:** inert implementation, tests and local verification only. The first
authorized diagnostic remains preserved and FAILED; Webots and the diagnostic were **not** rerun.

- **Supported evidence sources:** the R2025a Webots documentation defines `--version` as “Display
  version information and exit”; the installed binary contains that flag. Webots also supplies
  `WEBOTS_VERSION` to controllers, as shown by the installed `lib/controller/matlab/launcher.m`.
  The installed Python controller API has no runtime version getter (`Supervisor.getVersion()` does
  not exist), so a missing controller-reported version is not substituted from a nonexistent API.
  Neither the configured `R2025a` string nor `resources/version.txt` is ever copied into observed
  evidence.
- **Launch-time probe:** `runtime_attestation.py` resolves the executable, hashes that exact path
  with SHA-256, and queries it with `[resolved_executable, "--version"]` using an argument list,
  no shell, a 20 s timeout, and captured stdout/stderr. A canonical `R20XXa` token is extracted only
  from the probe output. The command, return code, probe error, hash, path and UTC capture time are
  recorded as provenance. The configured expected value is compared only after observation; a
  mismatch remains a hard blocker.
- **Process binding:** after `Popen`, the attestation binds the spawned PID and OS-reported
  executable path (`/proc/<pid>/exe` on Linux; `QueryFullProcessImageNameW` on Windows where
  available, otherwise `Popen` PID plus command provenance). Linux launcher scripts are recognised
  by their shebang interpreter, so `webots` → `bash` → `bin/webots-bin` is recorded honestly rather
  than mislabelled as a direct binary match. An OS-level mismatch or missing PID fails closed.
- **Controller cross-check:** `WEBOTS_VERSION` is parsed in the controller and reported separately
  as `controller_webots_version` with its provenance. With verified executable evidence, a missing
  controller version is not a hard blocker; any controller version that mismatches configuration
  or disagrees with the executable evidence is a hard blocker. The report distinguishes launch
  evidence unavailable, executable-reported version unavailable, incomplete identity, executable
  path mismatch, process mismatch, executable version mismatch and controller mismatch.
- **Blast radius:** the opt-in flag is set only by the read-only Webots diagnostic. Existing launch
  paths keep their previous behaviour when it is absent. Session, world-hash and robot-DEF checks,
  read-only call blocking, `motion_allowed=false`, and the `unverified-arm-sweep` veto are
  unchanged. A verified version closes only the runtime-version evidence blocker; it does not
  authorize IK, actuation, calibration, scored evaluation or TEST/TRAIN access.
- **Trust limitation:** on this Linux host the resolved `webots` path is a shell wrapper that
  delegates to `bin/webots-bin`. The probe/hash are tied to the wrapper path and the spawned PID is
  tied to that command and its interpreter; they are **not** a cryptographic hash of the delegate
  binary. Windows process introspection and execution remain unverified in this environment. The
  first read-only rerun is ready for separate owner authorization but was not performed here.
- **Verification:** Ruff, format check, `ty check src`, `compileall`, 50 focused inert tests, the
  frozen-evidence checker and all four preserved diagnostic SHA-256 values pass/verify. The full
  non-Webots suite is `1 failed, 602 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch. No Webots execution, real IK, inference, training, actuation,
  scored evaluation, TEST/TRAIN access, push, merge or PR update occurred.

## ADR-0053 — Bare Webots command fallback to WEBOTS_HOME (2026-11-18)

**Status:** Accepted. **Scope:** resolver-only repair plus inert tests; no diagnostic rerun is
performed by this decision.

- **Root cause:** an explicit string such as the pinned diagnostic's `"webots"` was looked up only
  through `PATH` and the working directory. A valid `WEBOTS_HOME` launcher was never considered, so
  resolution failed before any process, probe or attestation could run.
- **Resolution:** path-like explicit values are validated directly. Bare commands resolve through
  `PATH` first, then `WEBOTS_HOME` and the working directory, with the platform executable name
  added on Windows. Missing, non-executable and directory candidates fail closed, and multiple
  executable filesystem candidates are rejected as ambiguous. The no-argument resolver keeps its
  original `WEBOTS_HOME`-before-`PATH` precedence for existing callers.
- **Attestation identity:** one resolved executable value is supplied to both the version probe and
  the launch command. The inert launch test asserts that the probe argument, the `Popen` command and
  the recorded attestation path are identical. No safety threshold, world, robot config, pinned
  diagnostic config or frozen evidence was changed.
- **Verification:** Ruff check and format check, `ty check src`, `compileall`, 39 focused inert
  tests and the frozen-evidence checker pass. The full non-Webots suite is
  `1 failed, 607 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch. No Webots process, actuation, motion, scored evaluation,
  TEST/TRAIN access, push, merge or PR update occurred.

## ADR-0054 — Concrete full-arm sweep adapter, local geometry diagnostic and motion-readiness criteria (2026-11-18)

**Status:** Accepted. **Scope:** review-gated integration, inert validation and local verification
only. No Webots execution, IK solve, trajectory, actuation, motion authorization, scored campaign,
TEST/TRAIN access, push, merge or PR #2 change is included.

- **Pinned geometry coverage:** the adapter and diagnostic use the unmodified TIAGo URDF
  (`1a6de992…`), corrected registration (`dfc542e1…`), `robot.yaml` (`1a9f0c2b…`) and factory world
  (`35d87f36…`). All nine relevant arm links (`TIAGo front arm` through `wrist_ft_tool_link`) have
  19 supported collision primitives (cylinder/box/sphere); unsupported URDF shapes are `[]`. Joint
  limits and link reaches yield the conservative URDF point-speed bound
  `28.49316700274912 m/s`; the frozen tick is `0.032 s`. The contract uses margin `0.02 m` and
  eight interpolation substeps. External Webots PROTO definitions are **not** resolved locally.
- **Concrete integration:** `backends/worldmind_bindings.py` now contains a strict
  `ArmSweepConfig`/`ArmSweepCertificate` loader, full-arm FK capsule transforms, and an
  `AnalyticSurrogate.check_transition` callback path built from `world_model/arm_sweep.py`. It
  retains every sub-interval, full-arm, static-scene, finite-transform and fail-closed check; a
  checker result of `True`/`False`/`None` still maps to clear / `arm-sweep-collision` /
  `unverified-arm-sweep`. `production_worldmind_bindings()` supplies the blocked contract, so the
  production factory still leaves `check_transition=None` and the hard veto remains. Supplying an
  invalid or unapproved explicit `arm_sweep_config` raises; it never weakens the veto.
- **Owner certificate is not self-issued:** a callback is returned only when the contract has
  `status: owner_approved` and a certificate with matching ID/hash, approved flag, approver,
  approval ID, immutable approval record (`role: independent_safety_owner`) and non-empty reviewed
  reference-case file, all hash-pinned and agreeing with the computed contract and geometry
  hashes. The loader rejects `blocked` contracts carrying a certificate and `owner_approved`
  contracts without one; the callback builder independently rejects any status other than
  `owner_approved`. Adjacent-link and permitted-contact exceptions must be explicit capsule/object
  names; the shipped template approves none. No such certificate exists, so the path is implemented
  but deliberately unused.
- **Local-only geometry diagnostic:** `worldmind-geometry-diagnostic` reads only pinned files. It
  launches no Webots process, opens no socket, solves no IK, applies/steps/resets no backend,
  writes no score, and reads no TEST/TRAIN evidence. An inert guard canary counts and blocks
  `apply`, `step`, `reset`, `export_kinematics`, `inspect` and `get_ground_truth`; the collector
  itself has no live backend attached, so the zero counts are a structural check rather than a live
  boundary enforcement claim. The authoritative inert run after the status-gate fix is:
  `uv run python -m vlarobot.cli worldmind-geometry-diagnostic --config configs/worldmind_geometry_diagnostic.yaml --out runs/worldmind/validation-s0-geometry-inert-v3`
  It wrote `diagnostic.json`, `state.json` and `provenance.json` with outcome `PASS`,
  `motion_allowed=false` and `motion_readiness="NOT_READY"`. SHA-256 digests are
  `9cdee7cf2dec62365218dcecbfdf753110a92296d30f38aee12abe5f213f2806`,
  `bcf0c94679a2a35ee7ad253942f64ac0f2f81fb6868b9ddc392d7baba5beaa29` and
  `65022dc0007b2d3e393574a9b8b913268154dba0e4bc67423668088b17049649`. The earlier
  `validation-s0-geometry-inert` and `validation-s0-geometry-inert-v2` artifacts are unchanged
  intermediates; their evidence bytes match v3.
- **Motion-readiness criteria (engineering, necessary but never sufficient):** (1) every relevant
  arm, robot self-collision pair, environment obstacle and carried/contact object has reviewed
  collision geometry, with supported shapes and explicit exceptions; (2) pinned registration and
  frame/unit transforms are finite, rigid and reviewed; (3) margin, interpolation substeps and
  point-speed bound are positive, finite and derived from pinned sources; (4) deterministic
  reference cases cover known collision, clearance, near-boundary, interpolation, self-collision,
  TABLE/CAN, missing geometry, invalid transforms and unsupported contacts; (5) the concrete
  callback is wired and a simulator-only read-only diagnostic has verified the same pinned files
  without actuation. These gates are recorded separately in `engineering_gates`; a diagnostic
  `PASS` only means the inert evidence was collected.
- **Motion-readiness criteria (owner safety approval, independent and sufficient only with all
  engineering gates):** an `independent_safety_owner` must record a certificate ID/hash, approval
  ID, approver identity, scope limited to the reviewed simulator context, reviewed reference
  cases, and the exact contract/geometry hashes. This project code may verify that record but may
  not create, sign or self-approve it. Only then may the callback leave `None`; even then a separate
  execution authorization, live `SafetyMonitor.validate()` veto and non-scored simulator stepping
  remain mandatory. `motion_allowed=true` is never set by a geometry diagnostic.
- **Current assessment:** **NOT_READY**. The nine required arm links and 19 capsules are covered
  and transforms/margin/interpolation gates pass, but `complete_relevant_collision_geometry=false`
  remains because 27 external PROTOs are unresolved, robot self-collision is unobserved, floor
  `Plane` is outside the capsule/AABB checker, table nested solids are absent from the live export,
  finger/gripper/carried-object envelopes are unapproved, and no reviewed reference cases or owner
  certificate exist. The prior read-only Webots PASS is connectivity-only and is not motion or
  collision-safety approval.
- **Verification:** 13 deterministic cases in `tests/test_worldmind_full_arm_validation.py` pass;
  Ruff check, format check, `ty check src`, `compileall`, frozen-evidence checker, all preserved
  Webots diagnostic SHA-256 values and the 34 focused WorldMind/diagnostic tests pass. The full
  non-Webots suite is `1 failed, 620 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch. Windows and live Webots execution remain unverified in this
  environment. No dependency, public `core/` interface, frozen baseline, seed, split or threshold
  changed.

## ADR-0055 — Local Webots geometry resolution, conservative scene/sweep checks and dedicated read-only geometry inspection (2026-11-18)

**Status:** Accepted for local implementation and inert verification only. This does not approve
motion, create an owner certificate, authorize Webots launch, or update PR #2. No push, merge,
scored evaluation, TEST/TRAIN access, actuation, physics step/reset, IK solve, trajectory execution
or frozen-evidence change is included. The prior read-only Webots PASS remains connectivity-only.

- **Phase A — source resolution:** all 27 external `EXTERNPROTO` references in the pinned factory
  world and 5 nested dependencies were copied byte-for-byte from the pinned Webots R2025a asset
  cache into `assets/webots/protos/r2025a/`. `configs/geometry/webots-proto-sources-v1.yaml`
  (SHA-256 `2d6f84e260b36ff4b754793813216a6ef0c38d8d76aa225ebc534ed4eb0a7362`) records source
  URLs, asset hashes and classification: 29 relevant, 3 unrelated. PROTO text is parsed for nested
  `Solid`/`boundingObject` declarations, shape kinds and `selfCollision`; mesh/visual geometry and
  unsupported shape tokens fail closed. Relevant forwarded geometry with `IS`/`USE` bounding fields
  is explicitly listed as `unresolved_forwarded_geometry: [PipeSection, StraightStairsRail]` and
  remains a motion blocker. This updates the PROTO-availability statement in ADR-0054: the files
  are now pinned locally, but their forwarded geometry is still not certified.
- **Phase B — transforms, enclosure and subdivision:**
  `configs/geometry/scene-collision-v1.yaml` (SHA-256
  `68397a05106ca39b9d233c692081583652ad57f76049b6d288ebc2b6c7b9f313`) verifies hashes, finite
  rigid world/base/URDF transforms, the transformed table box, conservative floor half-space,
  gripper tool capsule, CAN envelope and base-pose tolerances (`0.003 m`, `0.005 rad`). Transformed
  corners are proven enclosed by the conservative box, and floor tests use the padded swept AABB
  corners. Static checks cover table, floor, gripper, carried CAN and self-collision; missing or
  unsupported scene geometry returns `None` or fails closed rather than clearing a transition.
  The sweep contract (SHA-256 `1c587b2b78b214734eaf2e154bec13b7e3ec331003f2f9e5fa92fff96116f393`)
  keeps the blocked status and derives an adaptive count from the pinned speed bound without
  changing the fixed eight-substep certificate contract:
  `28.49316700274912 m/s * 0.032 s / 8 = 0.1139726680109965 m > 0.02 m`,
  so `ceil(0.1139726680109965 / 0.02) = 46`, giving `0.019821333567129826 m` per sub-interval.
  Unsupported hard blocks include mesh/visual geometry, dynamic carried-object motion, arbitrary
  planes, `PipeSection`/`StraightStairsRail` forwarding, the remaining 20 obstacle PROTO types
  (42 world instances), `selfCollision FALSE`, and non-reviewed diagnostic primitives.
- **Phase C — read-only geometry inspection:** the dedicated
  `worldmind-webots-geometry-inspection` command and config (SHA-256
  `442cb25eb307fec9d5608863df2debf6c319d3cdac34b072313a463d2e35f233`) add a backend-only
  `inspect_geometry` RPC; `core/wire.py` and public `core/` interfaces are unchanged. The collector
  reuses the existing session, identity and launch-attestation checks, and its client permits only
  `ping` and `inspect_geometry`. The config forbids any swept-arm certificate; the evidence rules
  fail closed on `apply`, `step`, `reset`, `export_kinematics`, the older `inspect`,
  `get_ground_truth`, motor commands, a commanded/stepped session, stale or nonfinite observations,
  identity mismatch and missing or mismatched catalog hashes. Output files are `diagnostic.json`,
  `state.json`, `provenance.json` and `geometry.json`, all with `motion_allowed=false` and
  `motion_readiness="NOT_READY"`. No Webots process was launched and the CLI command was not run;
  tests inject synthetic snapshots to exercise the exact inert path. ADR-0058 extends this contract
  to the five nested Table `SolidBox` boxes without changing its no-actuation boundary.
- **Unchanged motion gate:** production still has `check_transition=None`, sweep status `blocked`,
  `certificate=None`, `unverified-arm-sweep` veto and `unresolved_forwarded_geometry` blocker. No
  owner-reviewed full-arm swept certificate exists, and this project will not create one. The
  authoritative inert artifacts under `runs/worldmind/validation-s0-geometry-inert-v3` retain
  SHA-256 values `9cdee7cf2dec62365218dcecbfdf753110a92296d30f38aee12abe5f213f2806`,
  `bcf0c94679a2a35ee7ad253942f64ac0f2f81fb6868b9ddc392d7baba5beaa29` and
  `65022dc0007b2d3e393574a9b8b913268154dba0e4bc67423668088b17049649`.
- **Verification and residual risk:** the 53 focused inert tests pass; Ruff check, Ruff format
  check, `ty check src`, `compileall` and the frozen-evidence checker pass. The full non-Webots
  suite is `1 failed, 639 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch, which must not be repaired by editing frozen evidence. Windows
  execution and live Webots geometry inspection remain unverified in this environment. No new
  dependency, public `core/` interface, frozen pin, seed, split or threshold changed.

## ADR-0056 — Preserved live Webots geometry evidence reconciliation and conservative swept-bound repair (2026-11-18)

**Status:** Accepted for local diagnostic implementation and inert verification only. The single
preserved live Webots geometry inspection is accepted as valid evidence with outcome `FAILED`; no
retry, Webots launch, controller/brain process, motion, real IK, calibration, scoring, TEST/TRAIN
access, owner certificate activation, push, merge or PR #2 update occurred. Motion remains
`NO-GO` / `NOT_READY`. This decision repairs collision-coverage diagnostics and makes unsupported
claims fail closed; it does not approve the robot for motion.

- **Frozen evidence acceptance:** the artifacts under
  `runs/worldmind/validation-s0-webots-geometry-live-v1/` were re-hashed before use and remain
  unchanged: `geometry.json` `034061c296abcada98b32fb2f22d7b421354a3a90bebf58e5feb6cea8b1a69d9`,
  `diagnostic.json` `4f2e8cca63a3914c595fc13a6ae635625bf4e946084dbb958c91c4099e30156b`,
  `state.json` `2b970678a27a784a5860500e95ca9e31b00bc4a20466e05e519090dfe07bc7b1`,
  `provenance.json` `769fe70f2a03ff6c54ad23362113facf37afc9438fcacc2d1f4d9efbf2eb66d3`, and the
  inspection log `4e9c1a965f0bdb6544926f559d0344eef3b4bc4a0be2f8d5f4a458308e9c4ac6`. The export
  records 57 objects, 110 shapes (108 supported and two unsupported `Plane` shapes), zero
  `apply`/`step`/`reset`/`export_kinematics`/`get_ground_truth`/motor commands/physics steps, and
  `inspect_geometry=1`. It is an observation only, not a swept certificate.
- **Strict `SFBool` normalization:** `normalize_webots_sfbool()` accepts exactly `False`, `True`,
  `0` and `1`, rejecting other integers, strings, `None`, floats and containers. The report
  preserves `self_collision_enabled` raw, adds `self_collision_enabled_normalized` and
  `self_collision_enabled_observed_type`. The pinned `Tiago.proto` declares `selfCollision FALSE`;
  the observed raw integer `0` is therefore valid and normalizes to `False`. A real mismatch or an
  ambiguous value adds a blocker and fails the inspection closed. No raw observation is erased.
- **Component reconciliation:** `configs/geometry/live-geometry-evidence-v1.yaml` (SHA-256
  `91c8e1ed23900e348eb1088d0987d3a8ccd747fcd6f8a2df99185aa80895eb0d`) pins all six diagnostic
  artifacts, the catalogs/PROTO sources, expected counts and tolerances. Every hash is verified
  before any JSON is consumed. The pure module `worldmind_live_geometry_reconciliation.py`
  classifies nine component groups using exactly the five required categories:
  matched/conservative, observed/not conservative, missing from live export, unsupported, and
  irrelevant with justification. The result is `diagnostic_not_certified`, `motion_allowed=false`,
  `motion_readiness=NOT_READY`, with one matched, four observed/not conservative, two missing,
  one unsupported and one justified-irrelevant component. Seven evidence entries are pinned after
  ADR-0057; the current config SHA-256 is
  `d4577fec440dcb804c7cee171fc3daea045fbe742523b3e426990a20cee0b3e2`.
- **Concrete collision findings:** only the two observed pinned horizontal floor planes are
  matched; arbitrary plane orientations and unrecognized plane geometry are rejected. `Table.proto`
  has zero root `boundingObject` declarations and at least three nested `SolidBox` instances with
  `boundingObject Box`; the pinned world translation `[-0.23882837, -4.8402928, 0.0]` and size
  `[4.5, 1.5, 0.75]` are recorded, but the table is absent from the live export and its runtime
  collision transform is ungrounded, so table geometry is not approved. `PipeSection` forwarding
  (`boundingObject USE PIPE` → `DEF PIPE Cylinder`, with `height IS height` and `radius IS radius`)
  is traced; all seven live cylinders match pinned instances and produce seven review-only capsule
  envelopes, but the catalog remains blocked on unresolved forwarding. `StraightStairsRail` is
  absent from the live export and remains blocked by JavaScript-computed transforms and
  `TexturedParallelepiped` forwarding. The apparent `0.2365 m` gripper requirement was a
  wrong-frame interpretation; ADR-0057 supersedes this finding with frame-correct `front` evidence
  that fits the unchanged `0.1 m` pinned radius. Static CAN radius `0.03175 m` and height `0.1222 m`
  match the pinned envelope and the tool-frame
  distance is about `0.8087 m`, but attachment, release and carried-pose propagation are unknown;
  static geometry is not dynamic safety proof. Self-collision is disabled, self-collision pair
  coverage is unproven, and no grasp-contact exception is approved.
- **Swept-bound repair:** `SweptArmCheck` now pads the full uncertain motion interval
  (`speed*dt`) plus explicit support-error terms instead of the unprovable half-interval term.
  `required_substeps()` subtracts those support errors from the usable margin and fails closed when
  they consume it. ADR-0057 supersedes the original three-error expression with the exact
  five-term inequality and strict positive-residual gate; no unknown term is ever replaced by
  zero. The preserved numbers are fixed eight-substep travel `0.1139726680109965 m > 0.02 m`,
  adaptive 46-substep travel `0.019821333567129826 m`, catalog-derived registration bound
  `0.004952867331749651 m`, and reference one-pose AABB undercoverage
  `0.0010469683238133465 m`, which is explicitly not a global bound. Joint-displacement,
  global-geometry, numerical and transform bounds are absent, so the verdict is `FAIL_CLOSED` and
  the hard veto stays in force.
- **Unchanged gate:** production still has `check_transition=None`, blocked sweep status, no
  certificate, the `unverified-arm-sweep` veto and `unresolved_forwarded_geometry` blocker.
  Another live Webots inspection is not required to establish these local gaps: the preserved
  export already grounds them. The smallest useful future motion-readiness experiment is an
  owner-reviewed full-arm swept certificate with finite joint-displacement and global geometry
  approximation bounds, reviewed self-collision/gripper/CAN dynamic envelopes, and a grounded
  runtime table transform; it is outside this task.
- **Verification:** the 73 focused inert tests pass; `ruff check .`, `ruff format --check .`,
  `ty check src` and `compileall` pass. The accepted frozen-evidence preflight prints
  `report_sha256_ok=true`, 12 locked baseline sources, 225 VLA campaign manifest entries, 12
  pinned and 12 frozen planning sources, and `test_split_accessed=false`. The full non-Webots
  suite is `1 failed, 675 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch, which was not repaired by editing frozen evidence. Windows and
  live Webots execution remain unverified in this environment. No new dependency, public
  `core/` interface, frozen pin, seed, split or motion threshold changed.

## ADR-0057 — Fail-closed swept-collision residual, frame-aware envelopes and non-approving motion-readiness dossier (2026-11-18)

**Status:** Accepted for local diagnostic implementation and inert verification only. Motion stays
`NO-GO` / `NOT_READY`. The dossier is `approved: false`, `motion_allowed: false` and
`status: diagnostic_not_certified`; it is not and cannot become an owner safety certificate. No
Webots retry, launch, actuation, real IK solve, calibration, scored campaign, TEST/TRAIN access,
frozen-evidence edit, push, merge or PR #2 update occurred.

- **Exact swept inequality and residual:** the certified transition condition is
  `margin > max(point_speed*dt/substeps, joint_displacement*reach) + geometry_approximation_error +
  registration_uncertainty + numerical_error + transform_error`. A bound is accepted only when
  finite, nonnegative and traceable. `required_substeps()` uses `ceil` to keep discrete substeps at
  least as conservative as the continuous bound; `evaluate_sweep_safety_bound()` computes the
  strict residual and certifies only `residual_m > 0`. Missing joint-displacement, global-geometry,
  numerical or transform bounds leave `required_margin_m` and `residual_m` as `None`; the result is
  `FAIL_CLOSED`, never a zero-filled pass. With the pinned `0.02 m` margin, speed
  `28.49316700274912 m/s` and `0.032 s` tick, the eight-substep travel is
  `0.1139726680109965 m` and the adaptive 46-substep travel is `0.019821333567129826 m`; the
  registration bound is `0.004952867331749651 m`. The one-pose AABB undercoverage
  `0.0010469683238133465 m` is explicitly not a global geometry bound.
- **Frame-aware gripper resolution:** `ee_pose` is the finger-pad tool frame while the URDF
  collision frame is `front`, with `ee_to_front_translation_m = [0.0, 0.0, 0.1741]` and the pinned
  front-frame segment `[0,0,0] -> [0,0,-0.21875]`. In the correct frame the finger boxes need
  `0.03908115001161714 m`, the static gripper body needs `0.06587500891530741 m`, and the combined
  requirement `0.06587500891530741 m` fits the unchanged pinned `radius_m: 0.1`. The earlier
  `0.23651330651909497 m` came from reading the front-frame segment in wrist coordinates; the
  runtime-literal diagnostic gives `0.19258175941889452 m`. Both are documented frame-mixing
  artifacts, not a reason to silently enlarge the safety bound. Dynamic gripper motion remains
  blocked by the swept certificate and self-collision gates.
- **CAN lifecycle:** `world_model/carried_object.py` bounds the six states
  `independent_pre_grasp`, `attached`, `carried`, `released`, `independent_post_release` and
  `ambiguous`. Missing attachment evidence, unsupported or ambiguous states, missing release
  evidence and unverified pose propagation all fail closed. Contact safety is never asserted
  without an explicit owner-reviewed grasp-contact exception. The preserved evidence supports only
  static CAN geometry (`radius 0.03175 m`, `height 0.1222 m`); current state is `ambiguous`, so
  `carried_geometry_available=false` and `contact_claim_allowed=false`.
- **Self-collision coverage:** the pinned URDF has 59 links, 58 joints, 30 links with collision and
  61 collision bodies. The nine required arm links give 19 capsules and every one of the
  `C(19,2) = 171` pairs is checked: 39 adjacent-link, 117 non-adjacent and 15 same-link pairs,
  plus 19 gripper/arm and 20 carried/robot pairs. Adjacent and grasp contacts are never
  auto-exempt; `adjacent_pairs` and `permitted_contact_objects` remain empty. The 42 URDF
  collision bodies outside the pinned arm capsules stay uncovered, so full-body coverage is
  `UNVERIFIED`.
- **Table and obstacle trace:** `_trace_table_source()` expands five source-derived `Table` /
  `SolidBox` boxes from the pinned world, `Table.proto` (`d7f2a20f3f6c3f515ce96ae10665941813b226c16442abf92d17e62259379bdb`)
  and `SolidBox.proto` (`f9e18612ce325bc7abfd2d94cbd1c391a244f54ce5a2789a97fd1f901dc0dfba`).
  The coarse catalog proxy contains all five boxes, but the status remains
  `source_derived_unverified` with `runtime_collision_tree_grounded=false`. The exact missing
  observation is a read-only Webots expansion of the five nested nodes with resolved PROTO hashes,
  evaluated `translation`/`rotation`/`size`/`enableBoundingObject`, resulting `boundingObject
  Box.size`, and world/base poses. `PipeSection` remains catalog-blocked; `StraightStairsRail` is
  catalog-blocked and absent from the live export.
- **Independent cases and dossier:** `tests/fixtures/motion-readiness-reference-cases-v1.json`
  (SHA-256 `b5e3d1f6de9a1c3da896a2da961dff95a2713b25c8f31b42102b3ecfa236a188`) contains 12
  independent inert cases covering swept bounds, substep rounding, self-collision, CAN lifecycle,
  gripper/table reports and checker fail-closure; all 12 replay with their expected outcomes.
  `worldmind-motion-readiness-dossier` writes the artifact to the caller's `--output` path and
  intentionally exits 1 because authorization fails. No generated dossier is committed; each run is
  reproducible from the pinned evidence. Gate statuses are `PASS frozen_evidence`,
  `UNVERIFIED swept_residual`, `PASS gripper_envelope`, `UNVERIFIED can_lifecycle`,
  `UNVERIFIED self_collision`, `UNVERIFIED table_transform`, `PASS reference_cases`,
  `PASS production_binding` and `FAIL motion_authorization`, giving
  `PASS=4, FAIL=1, UNVERIFIED=4` and `overall_gate=FAIL`.
- **Pins:** `configs/geometry/live-geometry-evidence-v1.yaml` SHA-256
  `d4577fec440dcb804c7cee171fc3daea045fbe742523b3e426990a20cee0b3e2`;
  `configs/geometry/scene-collision-v1.yaml` SHA-256
  `c0ba359b25c9163ee3b6c95b86baa36152e2d852d00460d2b86203e3ca63d035`;
  `configs/worldmind_arm_sweep.yaml` SHA-256
  `d1f75347fc9b13d48e8ab400caf1466c0f33758d531e2850f64a114d721ade7f`;
  `configs/worldmind_webots_geometry_inspection.yaml` SHA-256
  `442cb25eb307fec9d5608863df2debf6c319d3cdac34b072313a463d2e35f233`. Frozen live artifacts are
  unchanged: `geometry.json` `034061c296abcada98b32fb2f22d7b421354a3a90bebf58e5feb6cea8b1a69d9`,
  `diagnostic.json` `4f2e8cca63a3914c595fc13a6ae635625bf4e946084dbb958c91c4099e30156b`,
  `state.json` `2b970678a27a784a5860500e95ca9e31b00bc4a20466e05e519090dfe07bc7b1`,
  `provenance.json` `769fe70f2a03ff6c54ad23362113facf37afc9438fcacc2d1f4d9efbf2eb66d3`, and the
  inspection log `4e9c1a965f0bdb6544926f559d0344eef3b4bc4a0be2f8d5f4a458308e9c4ac6`.
- **Smallest remaining experiment:** read-only Webots expansion of the five nested Table SolidBox
  nodes with resolved PROTO hashes, evaluated translation/rotation/size/enableBoundingObject,
  resulting `boundingObject Box.size`, and world/base poses; it closes the table-transform gap
  without motion or IK. The separate owner decisions still required are finite independent joint,
  global-geometry, numerical and transform bounds; gripper-frame approval; a grasp-contact
  exception; reviewed adjacent-pair exclusions; the runtime table observation; and an independent
  safety-owner certificate.
- **Verification:** 85 focused inert WorldMind tests pass; `ruff check .`, `ruff format --check .`,
  `ty check src` and `compileall` pass. Frozen-evidence verification reports
  `report_sha256_ok=true`, 12 locked baseline sources, 225 campaign manifest entries, 12 pinned and
  12 frozen planning sources, and `test_split_accessed=false`. The CLI dossier command writes the
  JSON and exits 1 with `overall=FAIL`, `motion_readiness=NOT_READY`, `approved=False`,
  `webots_launch=0 apply=0 step=0 scored_evaluation_allowed=False`. The full non-Webots suite is
  `1 failed, 687 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch, which was not repaired. Windows and live Webots execution remain
  unverified. No dependency, public `core/` interface, frozen live-artifact byte, seed, split or
  production safety threshold changed.

## ADR-0058 — Selector-only read-only inspection of five nested Table boxes (2026-11-18)

**Status:** Accepted for local implementation and inert verification only. The single read-only
Webots execution that would ground the five nested runtime boxes is ready for explicit owner
authorization but was **not** launched here. No actuation, physics step/reset, real IK solve,
candidate execution, calibration, scored evaluation, TEST/TRAIN access, owner-certificate
activation, push, merge or PR #2 update occurred. Motion remains `NO-GO` / `NOT_READY`.

- **Selector-only RPC:** `configs/worldmind_webots_geometry_inspection.yaml` moves to
  `geometry_inspection.schema_version: 2` and pins `table_targets` with parent `Table`, nested
  `SolidBox`, bounding shape `Box`, names `main frame`, `foot 1`, `foot 2`, `foot 3`, `foot 4`, and
  five finite nonnegative tolerances of `0.0001` (m or rad as applicable). The adapter forwards a
  selector-only descriptor containing those names and the pinned PROTO identities/hashes
  (`Table.proto` `d7f2a20f3f6c3f515ce96ae10665941813b226c16442abf92d17e62259379bdb`;
  `SolidBox.proto` `f9e18612ce325bc7abfd2d94cbd1c391a244f54ce5a2789a97fd1f901dc0dfba`) through
  the existing backend-only `inspect_geometry` RPC. No expected translation, rotation, size, pose or
  hierarchy value is sent to Webots. `core/wire.py` and public `core/` interfaces are unchanged;
  the optional `table_targets` argument is internal to backend capture and dispatch.
- **Raw runtime observation:** `geometry_probe.inspect_table_targets()` traverses exactly one
  reachable runtime `Table` through supported `getBaseNodeField("children")` expansion and emits
  runtime schema `1` rows. Each row preserves expanded node identity, parent/child hierarchy,
  evaluated `translation`/`rotation`/`size`, `enableBoundingObject`, actual `boundingObject Box.size`,
  world pose from `node.getPose()`, robot-base pose from `node.getPose(robot)`, and explicit
  unsupported reasons. Missing nodes remain `missing` with empty evaluated/transform fields;
  unreadable PROTO expansion or unsupported node types remain `unsupported`. No source value is ever
  copied into an observation.
- **Pure reconciliation:** `worldmind_webots_geometry_inspection.reconcile_table_targets()`
  derives expectations independently from the pinned world, PROTO sources and scene catalog via the
  new public `expand_table_source_geometry()` helper. It reports each of the five boxes as
  `Matched`, `Mismatched`, `Missing`, or `Unsupported`, records the exact tolerance and error for
  every checked quantity, and preserves source provenance. Duplicate rows are `Unsupported`; any
  non-`Matched` target has `runtime_verification=UNVERIFIED`. Missing, mismatched or uninspectable
  observations are never repaired with source arithmetic. Table blockers feed `geometry_blockers`,
  and a non-grounded result adds the `unverified-runtime-table-transform` motion blocker.
- **Evidence boundary:** the CLI contract remains exactly `--config` and `--out` (verified with
  `--help`), and expected evidence files remain `diagnostic.json`, `state.json`, `provenance.json`
  and `geometry.json`. Reports add `table_target_inspection`, `table_transform_verification`,
  `table_transform_grounded` and geometric `table_target_reconciliation`; all retain
  `motion_allowed=false`, `motion_readiness="NOT_READY"`, all pre-existing hard blockers and the
  forbidden calls `apply`, `step`, `reset`, `export_kinematics`, `inspect` and `get_ground_truth`.
  The config SHA-256 is `442cb25eb307fec9d5608863df2debf6c319d3cdac34b072313a463d2e35f233`.
- **Verification and residual risk:** 66 focused inert tests pass, including all five boxes,
  missing/duplicate/mismatched/disabled/unsupported cases, nested PROTO expansion failure, selector
  forwarding and forbidden-call guards. `ruff check .`, `ruff format --check .`, `ty check src`,
  `compileall`, and the frozen-evidence verifier pass. The full non-Webots suite is
  `1 failed, 701 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch, which was not repaired by editing frozen evidence. Windows and live
  Webots execution remain unverified. No dependency, public `core/` interface, frozen live-artifact
  byte, seed, split or production safety threshold changed.

## ADR-0059 — Preserved table-live repair: internal PROTO ID dedup and pinned-vs-live base pose (2026-11-18)

**Status:** Accepted for local diagnostic repair and inert verification only. The single authorized
live inspection remains valid evidence with outcome `FAILED`/FAIL-INCOMPLETE. No Webots launch or
retry, actuation, physics step, real IK solve, calibration, scoring, TEST/TRAIN access, owner
certificate activation, push, merge or PR #2 update occurred. Motion remains `NO-GO` /
`NOT_READY`.

- **Evidence acceptance:** all five preserved artifacts under
  `runs/worldmind/validation-s0-webots-table-live-v1/` were re-hashed before use and remain
  unchanged: `diagnostic.json` `24673d587adf6a77ccb3eacb053a3123b1f8404a634945d524a9745cc34a6921`,
  `geometry.json` `09adc279a4709ad8b79219d2b33748362c51e0fd32f84f44fce37e60d7d32607`, `state.json`
  `061a477bed8de7eab202ce49f603a66613da0079d52442bd4f08a5d0dcd16287`, `provenance.json`
  `67e028a83369de00e164a21c1299a4a78296b69f166ea3ed5e80f7c2e4af8e0a`, and the log
  `622c0d4559cd0ddc54bbc7f22ec6cc145b5bf099758791433c860611b9c1ec80`. The report records zero
  `apply`/`step`/`reset`/`export`/`inspect`/ground-truth calls, zero motor commands and zero physics
  steps. It is a read-only diagnostic observation, never a swept-volume or motion certificate.
- **Missing-feet root cause:** `Table.proto:54-95` instantiates `main frame` and `foot 1..4` as five
  nested `SolidBox` children; `SolidBox.proto:85-138` expands each to a real `Solid` with a
  `boundingObject Box`. No deeper Supervisor API is needed: the existing
  `getBaseNodeField("children")` traversal reached all five. The preserved probe reported no
  traversal errors and a valid enclosing `Table` (`id=33400`), but internal PROTO nodes reported
  `getId() == -1`. `walk()` keyed visited nodes by that sentinel, so `main frame` won and the four
  feet were discarded before name selection. This is a probe identity bug, not evidence that the
  feet are absent from the runtime collision tree. Existing evidence cannot close the four-feet
  runtime gap and source-derived geometry is not backfilled.
- **Traversal repair:** `geometry_probe.py` now deduplicates valid nonnegative runtime IDs by
  number and invalid IDs by retained Python wrapper identity in `walk()`, `ancestors()`,
  `snapshot().visit()` and device ancestry. The raw `id: -1` remains in every observation; no
  synthetic identity, private API, expected geometry or tolerance is introduced.
- **Base-pose root cause:** the original expectation uses the pinned initial robot pose at
  `factory.wbt:59-61` (translation `[-0.53683, -3.70906, 0.0885438]`), whereas the observed
  `node.getPose(robot)` uses the settled live base pose in `state.json` (translation
  `[-0.5370347697854961, -3.7093265007325016, 0.088267810476998]`). With the documented
  composition `T_BT = inverse(T_WB) @ T_WT`, the pinned expectation reproduces the preserved
  errors `0.002728548164635969 m` and `0.003505397960478432 rad`; the live-base expectation
  reproduces the observed transform to numerical precision. The live/pinned base rotation delta is
  exactly the observed residual, and the live pose matches the exported robot `T_world_owner`.
  Wrong composition order produces multi-metre translation error; applying the URDF registration
  (`configs/calibration/urdf-base-train-v1.yaml`, used only for `T_world_urdf` in
  `scene_geometry.py:401-402`) produces about `0.07 m`/`0.071 rad` error. The mismatch is therefore
  consistent base-pose registration drift, not table geometry, frame selection, composition order
  or URDF registration error. The original pinned comparison is retained for audit and the
  `0.0001` m/rad tolerance is unchanged.
- **Offline diagnostic:** the new pure `eval/worldmind_table_live_diagnosis.py` verifies all five
  artifact hashes and all provenance/diagnostic source hashes before parsing JSON, then audits the
  preserved reconciliation, reports both pinned-base and live-base comparisons, frame consistency,
  wrong-order/URDF alternatives and the shared sentinel-ID diagnosis. It changes no verdict, closes
  no runtime gap by arithmetic and never authorizes motion. One additional owner-authorized
  read-only Webots observation with the repaired probe is still required to ground all five feet;
  this task did not launch it.
- **Verification:** 23 focused inert tests pass, including five shared-`-1` nested boxes, genuinely
  absent feet, incorrect hierarchy, missing/ambiguous fields, snapshot dedup, hash tampering,
  frame consistency, composition order and URDF discrepancy. `ruff check .`, `ruff format --check .`,
  `ty check src`, `compileall` and the frozen-evidence checker pass. The full non-Webots suite is
  `1 failed, 711 passed, 3 skipped`; the only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch, which was not repaired by editing frozen evidence. No dependency,
  public `core/` interface, config, pinned evidence byte, tolerance, split, seed or production
  safety threshold changed.

## ADR-0060 — Point-of-use observation freshness for read-only Webots inspection (2026-11-18)

**Status:** Accepted for local diagnostic repair and inert verification only. No Webots launch or
retry, no actuation, physics step, real IK solve, calibration, scoring, TEST/TRAIN access, owner
certificate activation, push, merge or PR update is included. The preserved v3 evidence remains
byte-identical and `FAILED`; motion remains `NO-GO` / `NOT_READY`.

- **Root cause:** `capture_read_only_geometry()` called `capture_read_only_state()` before the
  supervisor `boundingObject` export. The observation's server-side capture timestamp was therefore
  stamped before the slow export and table traversal that followed it within the same RPC. The
  client then recomputed `observation_age_s` during finalization *after* session teardown, so the
  measured age mixed genuine pre-receipt work with post-consumption shutdown latency. The preserved
  v3 run recorded `observation_age_s≈12.8636 s` without a receipt-age field, so the exact split
  cannot be recovered from that evidence.
- **Repair — capture at the real point of use:** the read-only geometry export now runs first and
  the state observation is captured afterwards, in the same unstepped session. The expensive
  export can no longer age the observation. This is a genuine freshness repair, not a timestamp
  rewrite: no capture value is mutated, backdated or synthesized, and the `5.0 s` limit is
  unchanged.
- **Repair — consumption boundary:** `DiagnosticSnapshot` now records
  `consumption_wall_time_s`, stamped at client receipt/validation while the read-only session is
  still live and before teardown. `observation_age_at_consumption()` uses that recorded point of
  use; an explicit `now_wall_time_s` still overrides it for diagnostic recomputation.
  `observation_age_s_at_receipt` is retained, and the report/state still fail closed when the age
  at consumption exceeds `observation_freshness_max_s`. Post-consumption teardown and report
  finalization are not observation use and cannot be counted as staleness.
- **Fail-closed boundary unchanged:** capture-timestamp validation, future/inconsistent timestamp
  rejection, forbidden-call counters, commanded/stepped-session rejection, identity/world-hash
  checks, `motion_allowed=false` and the `unverified-arm-sweep` veto are unchanged. Stale
  observations still produce `observation_valid=false`, the `observation is stale` blocker and an
  overall `FAILED` outcome.
- **Verification:** focused WorldMind/diagnostic tests plus new point-of-use and export-order
  cases pass; `ruff check .`, `ruff format --check .`, `ty check src`, `compileall` and the
  frozen-evidence checker pass. The full non-Webots suite is `1 failed, 746 passed, 3 skipped`; the
  only failure is the pre-existing
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  frozen-helper hash mismatch. Live Webots verification of the repaired path still requires a new
  explicit read-only authorization and a single attempt with a new output path.

## ADR-0061 — Narrowed, measured Packet-B collision scope (diagnostic only) (2026-11-19)

**Status:** Diagnostic proposal. Adds an offline, explicitly diagnostic checker and a pinned
exclusion manifest; it certifies nothing, activates nothing, and does not change the shipped
`status: blocked` sweep contract, the `full_system` veto, the `0.0001 m/rad` tolerances, the seeds
or the TEST seal. No Webots launch, actuation, physics step, IK solve, calibration, scoring or
TEST/TRAIN access is included. Motion remains `NO-GO` / `NOT_READY`; `full_system` stays gated.

- **Frame defect found while measuring:** `UNVERIFIED_LINK_FRAMES["wrist_ft_tool_link"]` named
  `wrist_ft_tool_link_front_joint`, which places the *child* `front` link. The URDF places
  `wrist_ft_tool_link` with `arm_7_link_wrist_ft_tool_link_joint`, so the mapping double-counted the
  16 mm wrist→front offset in every wrist capsule. Corrected mapping: `wrist_ft_tool_link` →
  `arm_7_link_wrist_ft_tool_link_joint`. This is a measurement correction inside the diagnostic
  path; the shipped checker still uses the pinned contract and stays blocked.
- **Table geometry correction:** the shipped `StaticBox` for the table is a base-frame AABB of a
  world-axis-aligned table. Its top is `0.722153468 m`; the exact oriented box gives
  `0.658885820 m` in the base frame, i.e. the AABB is inflated by `63.27 mm` along the base `z`
  axis. The catalog now carries the exact oriented box (`axes_base`, `half_extents`) alongside the
  unchanged AABB, so diagnostics can use exact distances without changing shipped behaviour. With
  the exact box and a full sweep of all 18 episodes / `13,766` steps, the tightest gripper-box/table
  clearance is `0.053529841 m` at `episode-11:253` (`gripper_right_finger_link:0`), and every arm
  capsule keeps an AABB lower bound of at least `0.148097 m` from the table (tightest at
  `episode-06:306`; the exact distance there is `0.317494 m`), against a required separation of
  `0.039807 m`. The recorded task therefore clears the table by about `13.7 mm` at the fingertip and
  by more than `108 mm` for every capsule. The earlier "2.82 mm" figure is an AABB *lower bound*
  on the distance from a gripper box to the exact oriented box, not a measured clearance; the
  earlier `0.053881 m` / `0.066083 m` pair is withdrawn — both were gripper-box-scale values that
  this sweep does not reproduce (the same witness steps now measure `0.053986 m` and
  `0.054121 m`), and neither was a capsule/table distance.
- **Speed-bound derivation gap (kept, not silently changed):** the pinned
  `speed_bound_m_s = 28.49316700274912` cannot be reproduced from repository URDF values; the
  largest joint rate `2.35 rad/s` times the longest measured arm radius of `1.13913 m` gives
  `≈2.677 m/s`, about `10.6×` smaller. The pinned value is retained because it is the value the
  contract already declares and because a smaller bound would strengthen the check without an
  approved derivation. Diagnostics label this term as pinned-but-underived; fixing the derivation
  is an open owner decision.
- **Capsule-pair partition (measured, not assumed):** the 19 pinned arm capsules enumerate
  `171` pairs. Removed: `15` same-rigid-body, `39` directly-jointed (one URDF joint), `8` whose
  URDF collision volumes already intersect in the zero pose. Retained: `109`. Over all 18 recorded
  episodes / `13,766` expert steps, `101` retained pairs stay above the required separation at
  every recorded configuration and `8` do not. The `8` are genuine narrow-clearance pairs, not
  structural duplicates: the worst is `arm_6_link:0 × wrist_ft_tool_link:0` with a minimum slack of
  `-0.030749 m`.
- **No positive control exists in the recorded data.** With the full retained set checked, `0` of
  the `13,766` recorded steps pass, and `0` of them contain an actual capsule penetration. A search
  over `15,710` candidates (recorded steps, consecutive midpoints, and four-step convex mixes) found
  no passing configuration; the best worst-pair slack is `-0.030749 m` at
  `episode-00` midpoint `192`. The accept path of the narrowed checker therefore can only be
  exercised by an explicitly relaxed control clone, never by an expert transition. This is reported
  as a scope limitation: the recorded expert campaign is **not** a positive control for the pinned
  geometric allowance.
- **Gripper-mount exclusion rule (measured, per pair):** a `front` housing solid against an arm
  capsule is removed only when its *maximum* clearance over the whole recorded campaign is below
  the required separation, i.e. when no recorded expert configuration could ever satisfy the
  check. Measured over the `25` `front × wrist` pairs, `16` qualify and `9` do not. Removed pairs
  are `front:0 × arm_5_link:1` (`0.033725 m`), all five `front:{0..4} × arm_5_link:2`
  (`0.011507`–`0.026906 m`), all five `front:{0..4} × arm_6_link:0` (`0.000000`–`0.013605 m`) and
  all five `front:{0..4} × arm_7_link:0` (`0.000000`–`0.002558 m`). Retained (still vetoed) are
  `front:{0..4} × arm_5_link:0` (max `0.085725`–`0.095834 m`) and
  `front:{1..4} × arm_5_link:1` (max `0.040725`–`0.043886 m`); their recorded minima are
  `0.016501`–`0.032522 m`, so they continue to reject recorded configurations. Each pinned row
  stores the witness `episode-NN:step` of its maximum, and a witness recomputed from the recorded
  NPZ reproduces the pinned value to better than `1e-6 m` for all `16` rows.
- **Structural exclusion totals:** `62` capsule/capsule + `76` gripper-box/arm-capsule = `138` of
  the checked pairs, in five explicit reasons (`same_rigid_body` 16, `directly_jointed` 35,
  `urdf_volumes_intersect` 8, `parallel_linkage` 9, `gripper_mount` 16). The manifest loader
  rejects any unknown key, tampered count, mismatched URDF hash, re-derived reason, satisfiable
  mount row, duplicate row, or witness outside the pinned campaign; a dropped mount row is safe by
  construction (the pair returns to the vetoed set) and there is a test for that direction.
- **Intentional contact stays phase-scoped:** the diagnostic checker leaves `permitted_contact`
  as a caller-supplied predicate over `(primitive, object)` pairs. Only fingertip primitives can be
  exempted from the CAN envelope, and only for the phase the caller declares; `front` and every arm
  capsule remain vetoed against the CAN and every other object.
- **Explicit uncovered scope:** `42` of the `61` URDF collision bodies have no arm capsule and are
  not checked; wheel/ground contact, full-body dynamics, deformable contact and perception error
  are outside this scope. The manifest report prints the coverage statement verbatim so no run can
  cite this as full-body certification.
- **Verification:** `tests/test_worldmind_packet_b_scope.py` pins the manifest partition and
  tamper failure modes, the oriented-table values, the measured `101/8` campaign regression, the
  `16` measured gripper-mount witnesses, the pinned table-margin witnesses
  (`0.053530 m` gripper box, `0.148097 m` capsule lower bound), the strict rejection of a recorded
  expert configuration, the relaxed control clone as the only observed accept path, a raised-table
  fingertip-just-below-allowance rejection, a raised-table arm-capsule rejection, permitted and
  forbidden CAN contact, `SafetyMonitor` joint-limit and velocity rejection, `None` for malformed
  input, and the shipped blocked contract with empty production bindings.

## ADR-0062 — Geometry investigation hard stop; `full_system` blocked (2026-11-19)

- **Decision:** the geometric investigation is closed. Commit `77cefd4` (eight-pair
  classification, speed-bound verdict, check-verdict) is accepted as **diagnostic evidence only**,
  not as authorization to activate `ExactSweptArmCheck` / `packet_b_narrowed_sweep_check`.
- **Unchanged by decision:** `configs/worldmind_arm_sweep.yaml` remains `status: blocked`; the
  proposed `workspace_separated` exclusions are not activated; the pinned
  `speed_bound_m_s = 28.49316700274912` is not replaced by the derived full-chain (`≈5.37`) or
  repo-URDF (`≈2.677`) candidates. No exploratory search is started.
- **Rationale:** a sampled positive separation is not proof of separation throughout the joint
  workspace. §9.3 records `0 / 13,766` passing expert steps, no recorded positive control, `8`
  retained narrow-clearance pairs that still veto recorded steps, and `42 of 61` URDF collision
  bodies with no arm capsule. No limited operating-domain contract can be justified from this
  evidence.
- **Consequence:** `full_system` is `blocked` pending the safety decision in
  `docs/WORLDMIND_OWNER_DECISION_READINESS_2026-11-18.md` (Packet B / §9). It keeps vetoing every
  prediction with `unverified-arm-sweep` and recording `NO_SAFE_SET`. `production_worldmind_bindings()`
  keeps `components={}` and `calibration_evaluators={}`. Engineering on the other four arms is not
  blocked by this decision.

## ADR-0063 — Measured WorldMind instrumentation and implementation-ready scientific contract proposals (2026-11-19)

- **Decision:** surface measured decision counts and phase latencies in the canonical report, and
  make the Eq. 3.8 / Table 5.1 / calibration contract implementation-ready for approval, without
  selecting any numeric parameter.
- **Instrumentation:** `build_instrumentation()` in `eval/worldmind_evidence.py` aggregates real
  recorded decisions into `decisions_total`, `decisions_per_episode`, `candidate_evaluations_total`,
  `gate_select_calls` and per-phase p50/p95 latency. It is written to
  `worldmind-sidecars/instrumentation.json` and, additively, to
  `summary.json.worldmind_instrumentation`. `measurement_class` is `measured` only for a real
  Webots backend, otherwise `synthetic`. Effective `max_decisions` / `max_steps` /
  `timings_enabled` are recorded in sidecar provenance. No decision count is inferred from
  control-step counts and no latency is taken from a nominal kernel profile.
- **Scientific contract:** `configs/report/objectives.proposal.json` adds a `calibration_evaluator`
  block (inputs, outputs, evidence, fail-closed, `selection_statistic_not_independent_evaluation`);
  `configs/report/definitions.candidate.json` adds per-metric denominators, a derived failure
  taxonomy and a measurement-instrumentation pointer. All values stay
  `engineering_proposal_not_approved` / `[PROPOSED; OWNER DECISION]`; weights and ranking direction
  remain VALIDATION-selected by the owner. `Ju` stays a prediction-discrepancy proxy, not
  calibrated uncertainty; frozen thresholds are untouched.
- **Authorization:** ADR-0030 is scenario design only and grants no scored-run authorization. No
  scored run is authorized until a new Packet D signature exists; any partial campaign needs its
  own explicit approval and is reported as partial; TEST stays sealed to `2026-12-05..07` with a
  separate authorization.
- **Verification:** `tests/test_worldmind_production.py` asserts the new sidecar, the summary
  block, the Webots/fake `measurement_class` distinction and the bounds in provenance;
  `tests/test_worldmind_objectives.py` still asserts no proposal component or evaluator is
  registered in scored bindings.

## ADR-0064 — Decision-time amendment to Eq. 3.8 `Jr` and `Ju` (2026-11-19)

- **Status:** proposal only, owner approval required. Packet A is **not** approved unchanged. No
  component binding, weight, threshold, calibration candidate or scored run is activated.
- **Unchanged:** `Jg` and `Jl` formulas and implementations are unchanged. Their production
  availability was confirmed by the call-site audit. `MissingObjectiveInput` fail-closed, hard-veto
  precedence (`SafetyMonitor.validate` + `RiskGate` + `unverified-arm-sweep`), empty scientific
  registries, blocked `full_system`, sealed TEST window and frozen five-arm VALIDATION protocol all
  remain in force. No weight or scale is selected before approved VALIDATION calibration.
- **`Jr` contract change:** the old `d_min,k` required certified full-arm swept clearance that the
  shipped surrogate does not produce. The amended soft score uses `tilde_d,k`, the minimum exact
  segment/AABB distance from the predicted tool path to current observed object boxes minus a
  declared tool-envelope radius. It is explicitly **uncertified** and is not a full-arm, contact or
  self-collision certificate. Hard vetoes remain a separate, precedent gate; the proxy never vetoes
  or offsets one and fails closed on incomplete scenes, absent geometry or truncated predictions.
- **`Jr` CAN contact clarification:** intentional CAN contact during grasping is **not exempted**.
  The CAN remains in the current observed object set; tool-envelope overlap or negative clearance
  contributes the maximum soft deficit. This is only a ranking cost. Hard contact/grasp policy and
  the hard veto are unchanged; the soft proxy remains uncertified.
- **`Ju` contract change:** the old correction-residual formula required future observed states and
  would leak them into candidate ranking. The amended score is a deterministic command-following
  residual between candidate command fields and the predicted trajectory, aligned to the current
  observation by object identity. It is not calibrated uncertainty and claims no ensemble.
- **`Ju` alignment clarification:** scores require exactly `len(actions)+1` predicted states and
  current/predicted object-identity equality; `JointTarget` residuals use sorted joint names;
  gripper fields are accounted for `EEDelta`, `EEPoseTarget` and `GripperCmd`; empty or unsupported
  comparisons fail closed with `MissingObjectiveInput`. `Ju` measures predicted command-following
  consistency, not calibrated uncertainty.
- **Verification:** `tests/test_worldmind_objectives.py` checks decision-time availability of all
  four components, fail-closed missing inputs, hard-veto precedence, and that future observations
  cannot alter a candidate score. The geometric investigation is not reopened; no scored execution
  is authorized.

## ADR-0065 — Owner scientific approval and bounded Packet C authorization (2026-11-19)

- **Scientific approval:** commit `b8ee4a0` is accepted as the experimental specification for Eq. 3.8
  and the proposed Table 5.1 conventions, including the four-stage VALIDATION calibration
  procedure. `Jg`/`Jl` are accepted; `Jr` is an uncertified tool-envelope proxy with intentional CAN
  contact penalized; `Ju` is command-following consistency, not calibrated uncertainty; hard-veto
  precedence and fail-closed missing inputs remain in force.
- **Not selected:** no numeric weights, normalization scales, margin, or calibration candidate is
  approved by this decision. Scientific bindings stay empty, `scored_execution_enabled` stays false,
  and no scored VALIDATION or TEST access is authorized.
- **Packet B:** not approved. The narrowed sweep checker stays diagnostic-only, the pinned
  `speed_bound_m_s = 28.49316700274912` is unchanged, and `full_system` keeps its hard veto.
- **Packet C:** exactly one read-only v4 Webots inspection is authorized with the §8.3 command and
  output path. No stepping, reset, IK, actuation, scoring, or retry is authorized; evidence is kept
  on failure. No push or merge is authorized.
- **Packet C outcome (2026-11-19):** the single attempt FAILED — Webots did not reply to
  `inspect_geometry` within 30000 ms; runtime geometry remains unverified. No retry was made and
  `runs/worldmind/geometry-inspection-v4-2026-11-19/` is preserved byte-identical. The read-only
  attestation held (no apply/step/reset/inspect-geometry calls, no scoring, no TEST/TRAIN access).
- **Production preflight still blocked:** `configs/worldmind_production.yaml` is `blocked` with
  `<OWNER_DECISION>` placeholders; `components`, `calibration_evaluators` and `waypoint_sets` are
  empty (the reviewed state-derived `corrected_scripted_waypoints_v1` remains registered under
  `waypoint_factories`); no approved calibration candidate/evidence or selected numeric parameters;
  runtime geometry unverified; no Packet D authorization. Scored VALIDATION remains unauthorized.

## ADR-0066 — Offline Packet C communication diagnosis and launch-mode forensics (2026-11-19)

- **Scope:** read-only analysis of preserved evidence and code only. No Webots session, live request,
  retry, reset, actuation, scoring or geometry conclusion was produced. Packet C stays `FAILED`;
  its authorization is expired and not reusable.
- **Best-supported cause:** an isolated `inspect_geometry` RPC failure, not a startup/endpoint/
  controller-registration fault. The launch-mode client dialled the reserved free port
  `tcp://127.0.0.1:60455`, and the preserved sim-status file for that session records the same
  endpoint with `last_completed={"op":"shutdown","seq":2,...,"ok":true}` and `event=stopped`.
  Startup `ping` (`seq=0`) and the capture `ping` (`seq=1`) succeeded; `inspect_geometry` (`seq=2`)
  timed out at 30 s; the shutdown request reused `seq=2` because the client does not increment the
  sequence on timeout. This rules out a persistent REQ/REP desync and a stale controller answering
  on another socket. The identical config/world hashes completed the same export in ~13 s in the
  earlier v2 run, so a slow export near the 30 s ceiling and a lost request/reply remain
  distinguishable only with server-side request logs, which were not captured for this session.
- **Decision — smallest offline fix:** `launch_webots_session()` now sets
  `VLAROBOT_CONTROLLER_STDERR_PATH` per session (overridable via `extra_env`) so the server's existing
  `request_start`/`request_end` timing lines are preserved. This is launch-mode forensics only; it
  changes no simulation state and needs no live authorization. Focused tests cover the default path
  and the override.
- **Not authorized / not done:** no new Webots test, no change to the world file, controller
  registration, endpoint mode, pinned speed bound, `full_system` veto, or scientific registries.
  A larger `inspect_geometry` timeout would require a separately authorized bounded non-actuating
  live diagnostic and cannot be validated offline.
- **Stage 1:** `configs/worldmind_diagnostic.yaml` is correct inert scaffolding (frozen 18-episode /
  3-seed VALIDATION plan, `evidence_collection_only`, `scored_evaluation_allowed: false`,
  `calibration_stage: 1`, no components/weights/calibration). It remains fail-closed on owner-supplied
  values (backend declaration, task, bounds, authoritative PDF, approved definitions, safety config,
  verification/adapter numerics). No simulation was run and no parameter was selected.

## ADR-0067 — Session provenance, bounded diagnostic preparation and parameterized component factories (2026-11-19)

- **Session provenance (evidence only, no cross-session inference):** the preserved Packet C
  `diagnostic.json` blocker names endpoint `tcp://127.0.0.1:60455`; the status file
  `runs/campaign-2026-11-11/sim-status-60455.json` records the same endpoint. Its session start
  `1795092812.9244719` is `2026-11-19T12:53:32.924472+00:00` and its stop/`updated_at`
  `1795092856.3539882` is `2026-11-19T12:54:16.353988+00:00` (43.43 s span); the Packet C Webots
  log mtime is `12:54:16.338608+00:00` and `diagnostic.json` mtime is `12:54:17.169849+00:00`.
  This strongly associates the status file with the Packet C session, **but** the status path is a
  fixed legacy path keyed only by port (`Path.cwd()/runs/campaign-2026-11-11/sim-status-{port}.json`)
  and Packet C artifacts carry no independent copy of the session `pid`/`start_time`/`nonce`.
  Therefore the association is well-supported inference, not a cryptographic identity proof.
  Regardless of identity, `last_completed={"op":"shutdown","seq":2,...,"ok":true}` is written by
  `SimServer.handle()` **before** the reply is sent, and `event=stopped` is written in
  `serve_forever()`'s `finally`; neither proves the **client** received the shutdown reply. The
  Packet C report that the client-side shutdown timed out is not contradicted, and no successful
  Packet C shutdown is inferred.
- **Bounded diagnostic preparation (not executed):** `VLAROBOT_INSPECT_GEOMETRY_TIMEOUT_MS` is a new
  strict-validated, opt-in override for the `inspect_geometry` deadline only. Unset, the shipped
  default remains `REQUEST_TIMEOUTS_MS["inspect_geometry"] = 30_000` ms; invalid values raise.
  It exists so a separately authorized read-only diagnostic can use a 90 000 ms ceiling (≈7× the
  13 s measured export, 3× the Packet C ceiling) without changing any default or other operation.
  No Webots session, request or state change was made; Packet C evidence hashes are unchanged.
- **Parameterized component factories:** the existing `WorldMindBindings` registry supports factory
  groups (`surrogates`, `safety`, `verification`, `waypoint_factories`), but `components` holds direct
  callables. A new `component_factories` group in that same bindings object (not a separate registry)
  registers factories for the four owner-approved implementations in `worldmind_objectives.py`.
  Each factory requires a frozen parameter object with **all fields mandatory** (no defaults,
  strict validation in `__post_init__`), including explicit ee/goal/tool-path providers so no
  extraction semantics are invented; it returns the pipeline-signature callable and never swallows
  `MissingObjectiveInput`. No scale, margin, radius, weight, ranking direction or calibration
  candidate is chosen. `production_worldmind_bindings()` still returns `components={}`,
  `calibration_evaluators={}`, `waypoint_sets={}` and the config stays `blocked`; a ready config
  naming a component still fails closed.
- **Stage 1:** every owner placeholder in `configs/worldmind_diagnostic.yaml` is listed in the final
  report with its known authoritative source and required phase. No placeholder was filled from a
  guess; inert preflight still fails with `backend.declaration is still a blocked template
  placeholder` and creates no output directory.
- **Constraints held:** Packet A is an experimental specification only; Packet B stays blocked
  (pinned `speed_bound_m_s = 28.49316700274912`, `full_system` hard veto, no `workspace_separated`
  exclusions); Packet D is unauthorized; TEST is sealed; no scored run, parameter selection or
  geometry-checker activation. No push or merge.

## ADR-0068 — v5 read-only geometry PASS and Stage 1 source reconciliation (2026-11-19)

- **Authorization and scope:** exactly one separately authorized v5 read-only Webots geometry
  inspection ran with the strict `VLAROBOT_INSPECT_GEOMETRY_TIMEOUT_MS=90000` override. It used
  only `runs/worldmind/geometry-inspection-v5-2026-11-19`; no retry, reset, step, IK, actuation,
  scoring, Packet D run, TRAIN/TEST access or Packet C artifact modification occurred.
- **RPC classification:** the `inspect_geometry` request reached the controller and completed
  server-side; its response was received by the client and contained a valid result. Controller
  stderr records `request_start op=inspect_geometry seq=2` and `request_end op=inspect_geometry
  seq=2 duration_s=47.2952`, followed by `slow_request ... duration_s=47.2952`. This is the
  successful-RPC case, not a never-delivered request, a server-side stall, or a completed request
  whose reply was lost. The two 5 s `ping` warnings in CLI stdout were startup probes; stderr shows
  both pings completed in under 1 ms. The 90 s override applies only to `inspect_geometry`; the
  shipped default remains `30_000` ms.
- **Geometry and freshness:** `outcome=PASS`, `geometry_export_valid=true`, `geometry_blockers=[]`,
  `observation_valid=true`, `observation_age_s=1.0791707038879395`,
  `observation_freshness_limit_s=5.0`, `table_target_inspection.status=runtime_grounded`, and
  `summary` `Matched=5 Mismatched=0 Missing=0 Unsupported=0`. Endpoint was
  `tcp://127.0.0.1:33819`. Read-only counters were
  `backend_apply_calls=0`, `backend_step_calls=0`, `backend_reset_calls=0`,
  `backend_inspect_calls=0`, `backend_inspect_geometry_calls=1`, `backend_export_calls=0`,
  `backend_ground_truth_calls=0`, `motor_commands_sent=0`, `physics_step_count=0`,
  `scored_results_written=false`, `frozen_campaign_modified=false`, `test_split_accessed=false`
  and `train_split_accessed=false`.
- **Motion remains blocked:** `motion_readiness=NOT_READY` with blockers
  `diagnostic_no_actuation_mode`, `unverified-arm-sweep`, `unresolved_forwarded_geometry`.
  `safety_check_readiness=true` is an evidence gate only; it does not authorize motion. Packet B
  stays blocked with the pinned `speed_bound_m_s = 28.49316700274912` and the `full_system` hard
  veto.
- **Preserved evidence:** Packet C v4 remains `FAILED` and byte-identical; its diagnostic hash is
  `319e6c038d7a2d02e4b8a537c19412b98f0f24bbbb61fb18bd7eb36e13f45fc9` and its log hash is
  `1229887d267b8387ff9cc341022ac7d80ae17194edbd4e505c82378d6bb335e4`. v5 changed neither file.
- **Stage 1 reconciliation:** source-backed FROZEN values were filled from ADR-0004
  (`destination_world`), ADR-0016 (placement criteria) and the v5 declaration
  (`webots_factory_world_supervisor_geometry_export_v1`). Registered identities/inputs were filled
  from the explicit binding registries: `finite_chunk_v1`, `analytic_surrogate_corrected_v1`,
  `runtime_safety_monitor_v1`, `placement_verifier_v1`,
  `corrected_bounded_pose_solver_v1`, `corrected_scripted_waypoints_v1`, and surrogate/safety
  input names `forward`, `inverse`, `validate`, `tracking`, `monitor`. `task.target=CAN` is also
  frozen from the approved robot/scenario definition. All instruction/destination, bound,
  report-definition, safety-config, verification-numeric and adapter solve/jitter owner values
  remain blocked placeholders. This supersedes the earlier Stage 1 sentence in ADR-0067: inert
  preflight now fails at `task.instruction is still a blocked template placeholder` and creates
  no output directory.
- **Constraints held:** no weight, scale, ranking direction, threshold, calibration candidate,
  task geometry or component instance was invented; the config remains `diagnostic_uncalibrated`
  and non-scored. No push or merge.

## ADR-0069 — Stage 1 source completion, owner decision set and bounded execution request (2026-11-20)

- **Scope:** read-only reconciliation after the owner accepted v5. The v4/v5 communication
  investigation is closed; both evidence sets remain unchanged. No Webots process, geometry
  diagnostic, retry, reset, step, IK solve, actuation, scoring, TEST/TRAIN access, calibration
  selection or production component activation occurred. This ADR supersedes only the remaining
  placeholder sentence in ADR-0068; all other v5 findings stand.
- **Exact sources used for newly populated Stage 1 values:**
  - `task.instruction` = `Pick up the can and place it at the target location.` from
    `src/vlarobot/eval/demos.py::DEFAULT_INSTRUCTION` (also the literal used by
    `src/vlarobot/cli.py` and `src/vlarobot/execution/vla_campaign.py`).
  - `report.pdf` = `~/Downloads/Project_Report.pdf`; bytes SHA-256
    `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`.
  - `safety.config_path` = `safety.yaml`, resolved relative to `configs/` as
    `configs/safety.yaml`. This is the same file loaded by
    `src/vlarobot/backends/webots/config.py`; its effective profile remains `original_envelope`.
  - `verification.rest_center_world` = `[-0.799879, -4.38998, 0.810949]`, the frozen
    `world.can_xyz` in `configs/scenarios.yaml`. Only that source was used; no new scene geometry
    was inferred from the accepted v5 table inspection.
  - `verification.object_dimensions_m` = `[0.0635, 0.0635, 0.1222]`, derived exactly as
    `2 * configs/robot.yaml:can_radius_m (0.03175)` for X/Y and
    `2 * configs/robot.yaml:can_half_height_m (0.0611)` for Z.
  - `verification.home_tolerance_rad` = `0.02` from
    `configs/robot.yaml:home_settle_tolerance_rad`.
  - `adapter.ik_position_tolerance_m` = `0.005` and
    `adapter.ik_orientation_tolerance_rad` = `0.08726646259971647` from
    `configs/vla.yaml:ik_position_tolerance_m` and `ik_orientation_tolerance_rad`.
- **Still blocked, with reasons:** `report.definitions` has no approved artifact;
  `configs/report/definitions.candidate.json` is `review_status=candidate`, carries extra keys and
  marks units `[PROPOSED; OWNER DECISION]`, so `ReportDefinitions.load` cannot accept it. The
  destination ID has no approved source distinct from the free-text "target location" and
  `destination_world`. Bounds have no reviewed values. `up_axis_base` has no owner-approved
  base-frame vector. Fixed `home_joints` have no source: the existing controller records the
  current joint vector at reset rather than a frozen HOME map. Jitter values have no approved
  source; test-only literals are not evidence. The loader now fails closed at
  `task.destination is still a blocked template placeholder` and creates no output directory.
- **Smallest owner decision set:** (1) approved `definitions.json` with
  `review_status=approved`, bound report SHA and exactly the eight Table 5.1
  `{definition, unit}` entries; (2) `task.destination` string; (3) `bounds.max_decisions` and
  `bounds.max_steps` positive integers; (4) `verification.up_axis_base` as an explicit finite
  unit vector in the base frame; (5) `verification.home_joints` as an explicit joint-name to
  radian map with its source; (6) the three `adapter.jitter` values with their reviewed source.
  No weight, scale, ranking direction, threshold or calibration candidate is requested.
- **Blocker classification:** `unresolved_forwarded_geometry` is a distinct scene/PROTO
  geometry-coverage requirement, not another name for table grounding or `unverified-arm-sweep`.
  `configs/geometry/webots-proto-sources-v1.yaml` names `PipeSection` and `StraightStairsRail`
  as external PROTOs with forwarded `boundingObject` geometry. The v5 `table_target_inspection`
  PASS closes only live table grounding and observation freshness. All three motion blockers
  (`diagnostic_no_actuation_mode`, `unverified-arm-sweep`,
  `unresolved_forwarded_geometry`) remain; `safety_check_readiness=true` is not motion
  authorization. `full_system` stays blocked with `speed_bound_m_s = 28.49316700274912` unchanged.
  Packet B and Packet D remain unauthorized and TEST remains sealed.
- **Execution request:** prepared in
  `docs/WORLDMIND_STAGE1_EXECUTION_REQUEST_2026-11-20.md`; not executed. There is currently no
  execution-capable Stage 1 CLI command, so no measured Webots evidence collection was started.
  `worldmind-diagnostic` only preflights and `eval-worldmind` requires a ready scored production
  config with calibration and weights. No push or merge.

## ADR-0070 — Targeted Stage 1 source completion (2026-11-20)

- **Scope:** targeted lookup and population of only `verification.up_axis_base`,
  `verification.home_joints` and `task.destination` in
  `configs/worldmind_diagnostic.yaml`. No Webots process, geometry diagnostic, retry, reset,
  step, IK solve, actuation, scoring, TEST/TRAIN access, calibration selection, production
  activation, threshold/weight decision, push or merge occurred. This ADR supersedes only the
  three blocked-field conclusions in ADR-0069; all other findings stand.
- **`task.destination`:** `target location`, the exact object of the canonical instruction in
  `src/vlarobot/eval/demos.py::DEFAULT_INSTRUCTION` (file SHA-256
  `f25502d4fe5beafc23656a42a923f68333574c1ade0f6422fbed3229c2a2a9ef`). This is a wording
  source, not a new destination geometry or an independent metric definition.
- **`verification.up_axis_base`:** `[-0.008464554343786014, 0.028463807918849244,
  0.9995589842318061]`, the third row of `T_world_base_live_observed` in the accepted v5
  `runs/worldmind/geometry-inspection-v5-2026-11-19/geometry.json` (SHA-256
  `9258b376f80309ed074130c61e784b9e0c18c71d48b980102d364dd0e60d66e7`). The third row expresses
  world +Z in the robot base frame under that matrix convention; its norm is exactly 1.0.
- **`verification.home_joints`:** the seven accepted v5 reset joint observations in
  `runs/worldmind/geometry-inspection-v5-2026-11-19/state.json` (SHA-256
  `9c27698603c2d957b91e30d04c4fdb88024c1ee45b7d3a97132ad5b9d5ca00e9`). This is an observed
  reset vector, not a factory HOME constant; any later claim must retain that provenance.
- **Still blocked:** `report.definitions`, `bounds.max_decisions`, `bounds.max_steps` and the
  three `adapter.jitter` values. The diagnostic loader now fails closed at
  `bounds.max_decisions must be an integer >= 1`. The config remains
  `diagnostic_uncalibrated`; no scored evaluation, calibration candidate or production
  activation is enabled.

## ADR-0071 — Additive non-scored Stage 1 evidence runner (2026-11-20)

- **Scope:** made the prepared Stage 1 evidence collection executable offline without changing
  the diagnostic config's blocked status. This supersedes only the ADR-0069/ADR-0070 statement
  that no execution-capable Stage 1 command exists. No Webots process, live evidence collection,
  actuation authorization, scoring, calibration selection, metric/weight approval, production
  activation, TEST/TRAIN access, push or merge occurred.
- **Reuse boundary:** added `src/vlarobot/execution/worldmind_stage1_pipeline.py` and the
  `worldmind-stage1-evidence` CLI wrapper. They reuse the frozen `eval/runner.py` evaluator,
  recorder, diagnostic preflight, safety probes, surrogate, verifier and WorldMind sidecar
  writer. No new evaluation framework, no `RiskGate`, no Eq. 3.8 utility, no calibration
  evaluator/candidate, no approval artifact and no production preflight was added.
  `PIPELINE_ALIASES` gained only the existing `scripted_chunk_adapter` identity so the frozen
  recorder can label the scripted diagnostic pipeline correctly.
- **Explicit modes:** `--arm-mode` is required and has no default. `full_system_veto` records the
  configured hard checks plus `prediction_complete`, `prediction_constraints`, an explicit
  `unverified-arm-sweep` rejection when present, and `stage1_full_system_blocked`; it selects no
  candidate and sends no action. Its evidence class is `full-system blocked/veto diagnostic`.
  `scripted_proposer` changes the effective arm to `scripted_chunk_adapter`, selects the first
  hard-check-safe seeded candidate in deterministic order by live safety only, with no utility
  ranking, and is labelled `scripted diagnostic` in provenance, sidecars and CLI output. It must
  never be reported as `full_system` evidence.
- **Safety and oracle boundary:** every applied scripted action still passes the configured live
  `safety_monitor.validate()` chain inside `execute_first`; an abort records `SAFETY_ABORT` and
  applies nothing. `unverified-arm-sweep` is never bypassed or suppressed. `get_ground_truth()`
  supplies only the privileged target pose; robot state comes from the declared backend
  observation. No perception-equivalent or independent-evaluation claim is emitted.
- **Non-scored accounting:** Stage 1 provenance sets `gate_call_policy=stage1_no_risk_gate`,
  `gate_select_calls=0`, `expected_total=0`, `utility_ranking=not_performed`,
  `scored_evaluation_allowed=false`, `independent_evaluation=false` and
  `performance_claim=not_permitted_bookkeeping_only`. Latency sidecars state that RiskGate was
  not called and that hard-check/selection windows are measured directly without an Eq. 3.8
  ranking window. Calibration identities, component bindings, weights and selected parameters
  are excluded from Stage 1 provenance and summary.
- **Resume semantics:** the first run requires a new output directory. `--resume` requires the
  same frozen provenance byte-for-byte, loads existing decision/pairing/error sidecars, rejects a
  mismatch between completed recorder episodes and sidecar episode coverage, and rewrites the
  aggregate sidecars and recorder summary only for that explicitly resumed directory. It is not a
  new collection and cannot change mode, config, plan or scope.
- **Authorization remains unchanged:** the runner connects to an already-running `sim-server`
  and never launches Webots. Owner authorization is still required for live execution and for
  the exact `--arm-mode`. `full_system` remains blocked; `speed_bound_m_s = 28.49316700274912`
  is unchanged; Packet B remains blocked, Packet D unauthorized and TEST sealed.
- **Offline verification:** the initial `tests/test_worldmind_stage1_runner.py` run passed
  `9 passed`. Full gates then returned `All checks passed!`, `333 files already formatted`, and
  `1 failed, 803 passed, 3 skipped, 540 warnings in 571.41s (0:09:31)`. The only failure is the
  pre-existing, unrelated
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`; all
  Stage 1 tests and all 22 production tests passed. No Webots process or live evidence collection
  occurred.
- **Final focused verification:** three additional offline checks were added to the same test file;
  it now passes `12 passed in 4.73s`. They prove one permitted scripted decision yields exactly
  `reset -> apply -> step` with the final `SafetyMonitor.validate()` immediately before apply and
  `backend_stepped=true`; `full_system_veto` yields resets only, with no apply and no step; a live
  safety veto yields no apply and no step; resume reuses bytes without re-executing or duplicating
  episodes and rejects changed mode, config or plan with `resume provenance differs`; and patched
  `RiskGate`, `ComparisonPlanner`, Eq. 3.8 component evaluators and all calibration entry points
  are never invoked.
- **Live scope:** the existing diagnostic runner hard-requires the frozen 18-episode VALIDATION
  plan; there is no one-scene/one-seed CLI selector. A single-scene live proposal would require a
  scope change to the frozen campaign definition and is therefore not made. The first live command
  can only be the full frozen campaign after all owner approvals.

## ADR-0072 — Stage 1 evidence-integrity safeguards (2026-11-20)

- **Scope:** two additive safeguards in the accepted non-scored Stage 1 runner. No Webots process,
  live evidence, actuation, scoring, calibration selection, metric/definitions approval, config
  population, production activation, TEST/TRAIN access, push or merge occurred. The frozen 18-episode
  VALIDATION plan, exact `--arm-mode scripted_proposer` requirement, finite caps, seeded candidate
  generation and live safety chain are unchanged.
- **Bound exhaustion:** the Stage 1 episode policy now distinguishes loop exit by a finite bound
  from explicit abort/success/exception exits. If `max_decisions` or `max_steps` ends an episode,
  the outcome is `WORLDMIND_BOUND_EXHAUSTED` with a reason naming the reached bound and observed
  count. A prior `NO_SAFE_SET` or `VERIFICATION_RETRY` reason cannot masquerade as the termination
  cause. Explicit safety abort, verification abort, successful completion and exceptions retain
  their original states.
- **Interruption and resume:** resume provenance validation is unchanged. A coverage mismatch
  between completed recorder episodes and WorldMind sidecar episodes now writes
  `worldmind-sidecars/quarantine.json` and fails closed before rerunning any episode. The marker
  blocks all later resumes of that directory. Partial recorder files with no committed decision
  sidecars remain resumable at the episode boundary: `_completed_recording` treats them as
  incomplete and `open_episode` truncates them on rerun, so no completed-looking duplicate can be
  produced. Decision, pairing, error and provenance sidecars are not modified by a refused resume.
- **Verification:** `uv run pytest tests/test_worldmind_stage1_runner.py -m "not webots" -q`
  passed `16 passed in 9.02s`. `uv run ruff check .` returned `All checks passed!`;
  `uv run ruff format --check .` returned `333 files already formatted`. No Webots-dependent test
  was run.

## ADR-0073 — Bounded offline correction for Stage 1 endpoint failure and fail-stop (2026-11-20)

- **Scope:** exactly two behavioural corrections after the accepted diagnosis of the preserved
  18-episode Stage 1 campaign: solve full-horizon waypoint endpoints with the already-reviewed
  corrected kinematics and full registered bounds, and stop a campaign at the first recorded
  episode-level `EXCEPTION`. Necessary wiring in the two existing WorldMind pipeline generators and
  existing test stubs is included. No Webots process was launched; the failed campaign was not
  modified, resumed, rerun or superseded. No live campaign is requested. No config, approved
  definitions, report PDF, tolerance, seed, `K=32`, `H=50`, frozen VALIDATION plan, core
  interface, scoring rule, `full_system` behaviour, Packet B/D, TEST/TRAIN access, safety limit,
  push or merge was changed.
- **Root cause retained:** the preserved run failed at the first `PREGRASP` inside
  `ScriptedChunkAdapter.draw().propose()` because the one-tick
  `BoundedPoseSolver(max_joint_step_rad=0.15)` was applied to a full-horizon endpoint. All 18
  planned episodes were recorded as `EXCEPTION`, with zero decisions, candidates, predictions or
  backend actions. The campaign's CLI exit was `0`, which hid the early termination.
- **Endpoint correction:** `src/vlarobot/backends/worldmind_bindings.py` now registers a
  `_FullPoseEndpointSolver` through `corrected_pose_solver()`. It solves the full position and
  full orientation with the pinned corrected FK/IK, full registered joint bounds, the reviewed
  20-restart strategy and an explicit draw-seeded `numpy` RNG. It exposes only the execution
  limit `execution_max_joint_step_rad = SETTINGS.max_joint_step_rad = 0.15`; no `10 rad` trust
  region or other new constant was adopted. The endpoint selection uses the reviewed residual
  formulas and rejects non-finite solves.
- **Bounded execution retained:** `ScriptedChunkAdapter` now seeds each endpoint solve, requires
  a finite residual within the per-waypoint reviewed tolerance (`0.015 m` transit /
  `0.005 m` grasp; `0.08726646259971647 rad` orientation), and converts one endpoint into
  `hold_steps` explicit `JointTarget`s clipped to `0.15 rad` per tick from a simulated current
  joint state. It fails closed on a missing/non-finite current joint or a safety rejection.
  Every emitted joint and gripper action passes the injected `SafetyMonitor.validate()` before it
  enters a candidate chunk; the live `execute_first` veto remains unchanged. The endpoint may be
  farther than one tick from the current configuration, but each emitted tick is executable and
  bounded.
- **Rejected alternative:** constraining endpoint IK to `hold_steps x 0.15 rad` cannot solve
  preserved episodes 11 and 17 even with 100 restarts. Minimal path-length prototypes require
  9.95–11.20 rad across the scripted chain versus `50 x 0.15 = 7.5 rad`, so endpoint feasibility
  and per-tick executability must be separate decisions.
- **Fail-stop:** `EvalRunConfig.stop_on_exception` defaults to `False`; Stage 1 sets it to `True`.
  `run_evaluation` detects the first recorded `EXCEPTION` row, stops before the next episode, and
  writes `campaign_status="incomplete_fail_stop"`, `planned_episodes`, and a `fail_stop` object
  carrying the episode index, scene, state and actual recorded reason. `run_stage1_evidence`
  finalizes sidecars in the existing `finally` block, writes the updated summary, then raises
  `CampaignFailStop` through the CLI's existing nonzero-exception path. A partial campaign is
  never described as completed or scored.
- **Offline endpoint evidence:** all 18 preserved first observations were reconstructed read-only
  from the pinned v5 registration and the preserved `.npz` files using the exact first candidate
  draw seeds. All full-pose `PREGRASP` endpoints passed the reviewed residual gate; worst
  endpoint joint delta was `2.05520891543 rad` (episode 14), while the largest residuals were
  `0.00982120514033 m` and `0.00208532571287 rad` (episode 02). Episodes 11 and 17, the known
  hard cases, solved with residuals below `1e-13` and deltas `1.67690353702` and
  `1.96569235228 rad`. No file in the preserved campaign was written.
- **Documentation:** this ADR records the exact source hashes and evidence boundary;
  `docs/VERSIONS.md` records that no dependency, version pin, `core/` interface or frozen evidence
  byte changed.
- **Preservation hashes:** diagnostic config
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`, approved definitions
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`, report PDF
  `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`, v5 geometry
  `9258b376f80309ed074130c61e784b9e0c18c71d48b980102d364dd0e60d66e7`, and aggregate SHA-256 of
  all files in the preserved campaign
  `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712` were unchanged.
  The owner-approved uncommitted config population and untracked definitions remain outside this
  commit.
- **Source hashes (HEAD SHA-256 -> current SHA-256):**
  `src/vlarobot/backends/worldmind_bindings.py`
  `b35551e60e6c25836f6d71a51cb78913e53bc28a2e8f048044a67e6973447ece` ->
  `4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae`;
  `src/vlarobot/execution/scripted_chunk_adapter.py`
  `45ed185554723317747657697b1eed8f4e6b0b5c7b486c5607569cd0375a4470` ->
  `bac8b18276498d41ea84d069f0fbdd08d7e4619cd11ee59bcd070c749b215b52`;
  `src/vlarobot/execution/worldmind_pipeline.py`
  `3455d99d81140f77235be47e8ba9bfcf77226908363a481d3b473f12719347c5` ->
  `27f3d45165d5e4e34fa4b670e41a709ff25206f2e5400fa441afe1dda887fdf8`;
  `src/vlarobot/execution/worldmind_stage1_pipeline.py`
  `c674c6056f7a78eb0702570358ef9539a8fc5a531b1d8e979f094f806db1d812` ->
  `20ef594be17ed4b65744f878a06ee72f1f0ea1f156669944a5ef71c6fdd468b5`;
  `src/vlarobot/eval/runner.py`
  `b1e3f8d9a3f7c9e20ea00b0d2a64d42b172645a5568c5f61913843f782cb2b12` ->
  `7b64b744d48074c723a8a593b009507cf268aa79c6d02fea01d51d430c25e0c2`.
  New focused regression file
  `tests/test_worldmind_endpoint_failstop_regression.py`:
  `b7126bde13972929c325b232b936737f2bd0f1efd1f39b79d8035ea7ac3a84f2`.
  Existing test stubs changed only for the new solver/adapter signatures:
  `tests/test_worldmind_waypoints.py`
  `89028e771410e5b914b1134e60dd12ea04de9fe1eb802ea7acb6d827db5db560`;
  `tests/test_worldmind_production.py`
  `732087650b7f85fc14bf5f9431a87ec1bb7255046f7b1d5d18bec28d69ebcfdf`.
- **Focused verification:** the new offline regression file passed `6 passed` with no Webots.
  It covers preserved-episode endpoint generation and reviewed residuals, full bounds, per-tick
  bounded transitions, mandatory safety validation and fail-closed rejection, one-episode
  campaign fail-stop, truthful partial summary/CSV/plan evidence, and Stage 1 sidecar finalization
  before `CampaignFailStop`. The four existing WorldMind/eval test files passed
  `46 passed, 1 failed`; the only failure is the pre-existing template assertion that conflicts
  with the owner-approved uncommitted config population (`max_decisions: 1060`), which was
  intentionally not modified.
- **Full-suite gate:** `uv run pytest -m "not webots" -q` completed with
  `2 failed, 815 passed, 3 skipped, 540 warnings in 603.23s (0:10:03)`. The failures are
  pre-existing and outside this correction: `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  compares a committed helper manifest SHA `071cf660ee012a432fe51b1314ad7fd53a3cad072e1400c6a6e9867acc849063`
  against the current helper SHA `88fd4db97c1cb392c182da4379d11a5271b4576244521af783313403ae449137`,
  and `tests/test_worldmind_production.py::test_shipped_stage1_template_is_frozen_uncalibrated_and_fails_closed`
  rejects the owner-approved uncommitted config population (`max_decisions: 1060`). Neither was
  changed: the first is unrelated recovery-helper evidence, and the second is an authorized owner
  value explicitly excluded from this commit. The focused regression and Ruff/format gates pass.
- **Remaining unverified:** no Webots execution, physical settling, contact, swept-arm safety,
  calibration selection, scoring or live evidence collection was performed. The correction is an
  offline code fix with offline regression only; it is not a claim that a new live campaign would
  succeed.

## ADR-0074 — Offline readiness correction: config split and CLI fail-stop proof (2026-11-20)

- **Scope:** narrowly scoped, read-only-with-respect-to-evidence correction after acceptance
  review. The tracked `configs/worldmind_diagnostic.yaml` was restored byte-for-byte to its
  committed frozen placeholder template. A separate approved diagnostic instance
  `configs/worldmind_diagnostic.approved.yaml` now carries only the six owner-approved
  substitutions: `bounds.max_decisions=1060`, `bounds.max_steps=1059`,
  `report.definitions=report/definitions.approved.json`, `adapter.jitter.joint_std_rad=0.001`,
  `adapter.jitter.joint_limit_rad=0.005` and `adapter.jitter.max_attempts=10`. No other YAML
  field, candidate definition, authoritative report, seed, split, `K=32`, `H=50`, tolerance,
  safety limit or binding identity changed. No Webots process, campaign, scoring, calibration
  selection, `full_system` execution, TRAIN/TEST access, Packet B/D execution, geometry
  investigation, push or merge occurred.
- **Provenance clarification:** the byte set recorded in ADR-0073 as the owner-populated
  diagnostic config hash `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03` is
  now stored verbatim at `configs/worldmind_diagnostic.approved.yaml`. The restored shipped
  template has its own hash below. `tests/test_worldmind_endpoint_failstop_regression.py` was
  updated only to read the approved instance; its new hash supersedes the ADR-0073 value.
- **Config hashes:** shipped placeholder template `configs/worldmind_diagnostic.yaml`
  `9d096d4d3206f0a6287239257c57ed56e138843babaeea1c624bf8dea6426c0a`; approved instance
  `configs/worldmind_diagnostic.approved.yaml`
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`; approved definitions
  `configs/report/definitions.approved.json`
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`; authoritative report PDF
  `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`; aggregate SHA-256 over
  all preserved failed-campaign files
  `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712`. The preserved campaign,
  v5 geometry, approved definitions and report PDF were not written.
- **Corrected solver identity:** this decision record is the versioned provenance for the offline
  endpoint correction. The implementation identity is
  `vlarobot.backends.worldmind_bindings._FullPoseEndpointSolver`, exposed through the unchanged
  public binding identity `corrected_bounded_pose_solver_v1` and the unchanged factory
  `corrected_pose_solver()`. The corrected source
  `src/vlarobot/backends/worldmind_bindings.py` remains at SHA-256
  `4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae`; the other ADR-0073
  source hashes are unchanged (`scripted_chunk_adapter.py`
  `bac8b18276498d41ea84d069f0fbdd08d7e4619cd11ee59bcd070c749b215b52`, `worldmind_pipeline.py`
  `27f3d45165d5e4e34fa4b670e41a709ff25206f2e5400fa441afe1dda887fdf8`,
  `worldmind_stage1_pipeline.py`
  `20ef594be17ed4b65744f878a06ee72f1f0ea1f156669944a5ef71c6fdd468b5`, `eval/runner.py`
  `7b64b744d48074c723a8a593b009507cf268aa79c6d02fea01d51d430c25e0c2`).
- **Implementation choices, not calibrated parameters:** the 20 draw-seeded full-bound restarts
  and the deterministic selection key `(position_error + orientation_error, joint_delta)` are
  recorded as solver implementation choices. They are not Eq. 3.8 weights, component settings,
  ranking parameters, calibration candidates or approval evidence. No trust region, execution
  limit, safety threshold or scoring rule was introduced. Binding compatibility is preserved:
  no new binding identity or config field was added, and the registered factory still exposes the
  same `solve()` contract with a defaulted seed plus the execution-step property required by the
  adapter.
- **New regression hashes:** new readiness regression file
  `tests/test_worldmind_readiness_regression.py`
  `c22b498a842d19690bf7ea667f04c18be0629a9eba8b118ba49696cb86dda283`; updated endpoint/fail-stop
  regression file `tests/test_worldmind_endpoint_failstop_regression.py`
  `b43593615d6aebd62c432195e9ae34c15dec4bb25e687332eca11df37c5978b1`.
- **Focused verification:** endpoint/fail-stop regression `6 passed, 8 warnings in 6.02s`;
  readiness regression `3 passed in 2.88s`, covering the exact six-field YAML diff, independent
  loader acceptance of the approved instance, fail-closed loading of the restored template, and
  CLI `CampaignFailStop` exit code `1` with preserved partial evidence, finalized sidecars, two
  planned episodes, one recorded episode and no second-episode launch; shipped-template contract
  test `1 passed in 2.39s`; `uv run ruff check .` returned `All checks passed!`;
  `uv run ruff format --check .` returned `335 files already formatted`; `git diff --check` passed.
- **Future campaign path:** a fresh non-scored VALIDATION18 campaign should use the approved
  instance and a new output directory, proposed as
  `runs/worldmind/stage1-validation-v2-2026-11-20`. That directory was not created. Explicit
  owner authorization is still required before any live diagnostic campaign. Windows, Webots,
  physical settling, contact, swept-arm safety, calibration selection, scoring and live evidence
  collection remain unverified.

## ADR-0075 — Stage 1 V2 offline postmortem: absolute joint targets and partial-campaign finalization (2026-11-20)

- **Scope and preservation:** this ADR records an offline-only correction of the two authorized
  Stage 1 V2 defects. No Webots process, campaign, rerun, resume, scoring, calibration selection,
  `full_system`, Packet B/D, TRAIN/TEST access, geometry investigation, push or merge occurred.
  V1 and V2 evidence was not written. The preserved V1 aggregate remains
  `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712`; the preserved incomplete
  V2 aggregate remains `b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb`.
  The V2 campaign was executed exactly once at commit
  `8103a49ea1ef04702e6e9f07914ca9690986cc72` and failed stopped after episode 0; it is not
  retried or resumed. V2 authorization is exhausted; any V3 needs separate explicit owner
  authorization.
- **Runtime-recorded V2 facts:** `episode-00.steps.npz` records one state at `t=0.03200000000000003`.
  The recorded joint order is `arm_1_joint`, `arm_2_joint`, `arm_3_joint`, `arm_4_joint`,
  `arm_5_joint`, `arm_6_joint`, `arm_7_joint`, `torso_lift_joint`. The runtime-recorded initial
  `arm_6_joint` value is `1.8444461957053934e-06`. The primary recorded error is
  `ValueError: joint target for arm_6_joint violates position limits`; the sidecar error is
  `joint target for arm_6_joint violates position limits`. The V2 action log is empty, so the
  rejected target itself is reconstructed, not runtime-recorded.
- **Reconstructed rejection:** using the preserved episode-00 state, the pinned corrected
  registration/kinematics, the approved config's waypoints and jitter, decision seed
  `2968811710` and candidate draw seed `8787345`, candidate 1 reproduces the exact rejection at
  the first `TRANSIT` interpolation tick. The reconstructed current value at that tick is
  `arm_6_joint=-1.2534098586703584`; the raw endpoint IK value is `-1.3899999999987871`, which is
  inside the reviewed interval; the jitter draw is `-0.001213510088701924`; the jittered endpoint
  is `-1.3912135100874889`; the bounded per-tick delta is `-0.13780365141713058`; and the
  proposed absolute target is `-1.3912135100874889`, which is `0.0012135100874889 rad` below the
  lower limit. Endpoint residuals for that solve were `position_error_m=0.0032559632227919695`
  and `orientation_error_rad=0.0006959086486396269`, both inside reviewed tolerances. These
  values are explicitly offline reconstructions from preserved data, not new runtime evidence.
- **Root cause:** endpoint IK and full-horizon interpolation were not the defect. The producing
  operation was the pre-fix `ScriptedChunkAdapter._bounded_joint_actions` line
  `target[name] = current[name] + bounded_delta`: it bounded only the delta to
  `MAX_JOINT_STEP_RAD=0.15`, then wrote the resulting absolute target directly into
  `JointTarget`. It did not constrain that target to the reviewed absolute joint-position limits
  before line `self.monitor.validate(action, ...)`. `SafetyMonitor._validate_joint_target`
  correctly rejected the target through the unchanged comparison
  `if target < limits[0] or target > limits[1]`, yielding the exact V2 reason. The reviewed
  `arm_6_joint` limits are `[-1.39, 1.39]` in both runtime safety and corrected kinematics/URDF;
  velocity is `1.76 rad/s`, `command_horizon=1.0`, and the separate execution step bound is
  `0.15 rad`. The reconstructed delta was legal; the absolute target was not.
- **Source correction:** `ScriptedChunkAdapter` now takes absolute position limits from the
  injected `SafetyMonitor.limits`, fails closed on a non-finite or out-of-limit raw endpoint,
  redraws jitter values that would cross those limits, and constructs each one-tick target as
  either the in-limit endpoint or `current ± max_joint_step_rad`. Every generated action still
  passes through the unchanged `SafetyMonitor.validate()`. No safety threshold, velocity limit,
  command horizon, endpoint residual tolerance, seed, split, binding identity or waypoint cadence
  changed.
- **Secondary root cause and correction:** `eval.runner._build_summary` called the
  nonempty-only `summarize_success([])` for five planned scenes with no result rows after the
  first-episode EXCEPTION. `eval.metrics.summarize_success` raises `ValueError("flags must not be
  empty.")`, so `plan.json`, `episodes.csv`, `summary.json` and `timings.json` were never written
  and the explicit `CampaignFailStop` message was not reached. The runner now uses a zero-row
  summary helper that preserves all planned scene/seed records and does not call the
  nonempty-only aggregator. `run_stage1_evidence` still finalizes sidecars before raising
  `CampaignFailStop` with the original EXCEPTION reason, and the campaign loop still stops before
  episode 1.
- **Preserved provenance hashes:** approved config
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`; shipped blocked template
  `9d096d4d3206f0a6287239257c57ed56e138843babaeea1c624bf8dea6426c0a`; approved definitions
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`; report PDF
  `23d5ff15e39c49890e6e20dc1d09b861c072fe5ac8cbc0f0e5aeed8eafa1522b`; corrected registration
  `dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0`; frozen plan hash in V2
  provenance `013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2`.
- **Source hashes preserved from V2 / changed here:** `backends/worldmind_bindings.py`
  `4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae` unchanged;
  `execution/worldmind_pipeline.py`
  `27f3d45165d5e4e34fa4b670e41a709ff25206f2e5400fa441afe1dda887fdf8` unchanged;
  `execution/worldmind_stage1_pipeline.py`
  `20ef594be17ed4b65744f878a06ee72f1f0ea1f156669944a5ef71c6fdd468b5` unchanged;
  `execution/safety_monitor.py` `900c9218a9727a94cae0a1be9b071133c80925095cd7ca4b16241334adbc371a`
  unchanged; `execution/scripted_chunk_adapter.py` changed from
  `bac8b18276498d41ea84d069f0fbdd08d7e4619cd11ee59bcd070c749b215b52` to
  `890e2f5bd3621338db85cb60088f27c091b792ca437d2aeadd744dc7029432f4`; `eval/runner.py` changed
  from `7b64b744d48074c723a8a593b009507cf268aa79c6d02fea01d51d430c25e0c2` to
  `7475a13a294b33bdb7ec953882bf56852cabe3de83ac9cc74cd15f5f892e2955`.
- **Regression hashes:** new `tests/test_worldmind_v2_postmortem_regression.py`
  `041d383415123c018093b12e37175ed08206bf98d40952cc513bdcdb6fd97fc0`; updated
  `tests/test_worldmind_endpoint_failstop_regression.py`
  `b93d1f4400f415ead49b3c11ac43d46685066362b6fee5cb1fa38ef7bf037acd`;
  `tests/test_worldmind_waypoints.py`
  `9dc5b3e43c6c32e1a64289845290fec4daeebd01c974d892376ed94b756ee0e3`;
  `tests/test_worldmind_production.py`
  `f224ff5aa5e84275b786c4a6d33adeecb8dec307eca07186c42cf023b93ddee5`.
- **Focused verification:** the new V2 postmortem file has six passing tests: reconstructed
  delta-only rejection, corrected-path elimination, absolute/per-tick bounds without safety
  bypass, residual tolerance fail-closed, invalid-absolute-endpoint fail-closed, and
  frozen-shape 18-episode partial Stage 1 finalization with sidecars and original
  `CampaignFailStop` reason. The focused set
  (`v2_postmortem`, `endpoint_failstop`, `stage1_runner`, `readiness`, `waypoints`, `production`)
  returned `57 passed, 8 warnings in 13.40s`. Offline replay of all 32 reconstructed episode-0
  candidate draws with the corrected adapter returned `replayed_candidates=32 rejected=0
  actions_per_candidate=50`. `uv run ruff check .` returned `All checks passed!`;
  `uv run ruff format --check .` returned `336 files already formatted`; `git diff --check` passed.
  The full offline suite returned `1 failed, 825 passed, 3 skipped, 540 warnings in 575.60s`;
  the only failure is the pre-existing unrelated notebook helper SHA mismatch in
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  (`071cf660…` expected versus `88fd4db9…` present). It was not changed in this correction.
- **Remaining unverified and V3 recommendation:** no live Webots execution after the correction,
  no 18-episode physical campaign, no action application, settling, contact or swept-arm
  validation, no Windows run, and no scoring or `full_system` behavior was verified. **GO** only
  for a separately authorized, non-scored VALIDATION18 Stage 1 V3 diagnostic using a new output
  directory and the same frozen inputs; V2 must remain preserved and is not resumable. **NO-GO**
  for scoring, calibration selection, `full_system`, Packet B/D, TRAIN/TEST access or production
  activation until the live diagnostic and independent safety approvals are complete.

## ADR-0076 — Stage 1 V4 offline correction: explicit diagnostic open-loop chunk execution (2026-11-20)

- **Scope and preservation:** this is an offline source-and-test correction only. No Webots
  process, campaign, rerun, resume, scoring, calibration selection, `full_system`, Packet B/D,
  TRAIN/TEST access, geometry investigation, push or merge occurred. V1, V2 and V3 evidence was
  not written. Recomputed aggregates remain V1
  `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712`, V2
  `b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb`, and V3
  `d4204a6fae79c16988e68f0b7cb61b27c90ba614f55fbfa507d1f666cf3d4d29`. V2 and V3 each ran once;
  both authorizations are exhausted and neither campaign is retried or resumed. V4 is not
  authorized by this ADR.
- **V3 root cause:** the reviewed scripted proposer emitted the complete frozen 50-action chunk
  (`INIT 0-1`, `PREGRASP 2-11`, `DESCEND 12-17`, `CLOSE 18-20`, `LIFT 21-28`, `TRANSIT 29-36`,
  `PLACE_DESCEND 37-42`, `RELEASE 43-44`, `RETREAT 45-46`, `HOME 47-49`), but the legacy
  `execute_first()` path applied only `actions[0]`. Final-task verification therefore observed a
  one-tick state, failed on distance, and `retry_once` regenerated the same INIT-first chunk
  because `alternate_waypoints_binding: null`; the second failure aborted. No candidate,
  selection, safety, verifier or config threshold was proven wrong by that run.
- **Correction:** `Stage1ExecutionMode` adds two explicit values. The default
  `receding_horizon_tick` preserves the original one-tick path byte-for-behavior. The opt-in
  `diagnostic_open_loop_chunk` runs only under `arm_mode=scripted_proposer`; `full_system_veto`
  rejects it. `_execute_selected_chunk()` applies every selected index in order through
  `SafetyMonitorProbe` then `BackendApplyProbe`, advances exactly one reviewed `step(1)` per
  accepted apply, re-observes after every step, and records per-action payload, index, verdict,
  reasons, step and timestamp. It fail-stops on safety rejection, backend apply rejection or any
  execution exception, records the measured prefix and termination reason, and defers placement
  verification until after the complete chunk unless execution terminated early. Success pairs
  all 51 observed states with the 50-step prediction; early termination pairs the available
  prefix with an explicit abort boundary.
- **Invariants preserved:** candidate generation and first-safe selection, absolute joint limits,
  `0.15`-rad/tick bound, verifier tolerances, `retry_once → abort`, seeds, frozen plan, frozen
  approved config, all full-system blockers including `unverified-arm-sweep`, and the
  `scripted diagnostic` / `full-system blocked/veto diagnostic` evidence classes are unchanged.
  The path is labelled exactly `diagnostic open-loop chunk execution`; it never claims
  receding-horizon replanning. No new sidecar file was added; chunk evidence is nested in the
  decision sidecar, with execution mode also recorded in provenance, summary and CLI output.
- **Frozen inputs:** approved config
  `configs/worldmind_diagnostic.approved.yaml`
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`; approved definitions
  `configs/report/definitions.approved.json`
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`; frozen plan hash in V3
  provenance `013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2`; unchanged
  source hashes `backends/worldmind_bindings.py`
  `4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae`, `execution/runner.py`
  (at `src/vlarobot/eval/runner.py`)
  `7475a13a294b33bdb7ec953882bf56852cabe3de83ac9cc74cd15f5f892e2955`,
  `execution/worldmind_pipeline.py`
  `27f3d45165d5e4e34fa4b670e41a709ff25206f2e5400fa441afe1dda887fdf8`,
  `execution/scripted_chunk_adapter.py`
  `890e2f5bd3621338db85cb60088f27c091b792ca437d2aeadd744dc7029432f4`, and
  `execution/safety_monitor.py`
  `900c9218a9727a94cae0a1be9b071133c80925095cd7ca4b16241334adbc371a`.
- **Changed source hashes:** `src/vlarobot/cli.py` changed from
  `d4185b4d3b6019b9babd9260e4f2e35f805104aa2504293d2056ade44136bb3d` to
  `bc5ce3527f3a5a2d874c19a06e22dcf21d575515f8c842fd8d9429aee10f93f5`;
  `src/vlarobot/eval/decision_evidence.py` changed from
  `e2cb51bf27ab19a3ae17ee2bf2c9fc8ab55932edbb542fa4104bfeee73806813` to
  `28f74d98b02325fee43778dd3152d9e340bd55b9ea2bde5c04a09cf4ae608339`;
  `src/vlarobot/execution/worldmind_stage1_pipeline.py` changed from
  `20ef594be17ed4b65744f878a06ee72f1f0ea1f156669944a5ef71c6fdd468b5` to
  `b724da5736bde334022a77af97cbc2ff4092a19abb111b9e11b35e0c25ed544f`.
  Regression hashes: `tests/test_worldmind_stage1_runner.py` changed from
  `9e49c5d1e1e8eb817a6efac2118d5d9939e562a7ee62a67c8afb9b14c90d1743` to
  `d402bf5bf25b07cec023c05b77d0bf08f5bff80571e7cdbcb86d9a62ca4cb2c5`;
  `tests/test_worldmind_v2_postmortem_regression.py` changed from
  `041d383415123c018093b12e37175ed08206bf98d40952cc513bdcdb6fd97fc0` to
  `0751bee367b3fdaa926fc0133bec4a6e49d47b2867e97a30665f17096d2a3689`.
- **Verification:** focused set (`v2_postmortem`, `stage1_runner`, `eval_decision_evidence`,
  `eval_decision_adapters`) returned `39 passed in 6.24s`, including full 50-action execution
  with terminal-only verification, intermediate safety fail-stop, backend apply rejection,
  step-exception fail-stop, preserved cadence, preserved campaign hashes, and unchanged
  `retry_once → abort`. Relevant regressions (`worldmind_production`, `worldmind_waypoints`,
  `worldmind_readiness_regression`, `worldmind_endpoint_failstop_regression`) returned
  `35 passed, 8 warnings in 6.02s`. `uv run ruff check .` returned `All checks passed!`;
  `uv run ruff format --check .` returned `336 files already formatted`; `git diff --check`
  passed. The full offline suite returned
  `1 failed, 832 passed, 3 skipped, 540 warnings in 570.74s`; the only failure is the same
  pre-existing unrelated notebook helper SHA mismatch recorded in ADR-0075
  (`tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`,
  `restore_gate.py` expected `071cf660…`, present `88fd4db9…`), with no diff from `HEAD` in the
  helper, manifest or test.
- **Remaining unverified and V4 decision:** no Webots execution, no V4 campaign, no physical
  settling, contact or swept-arm validation, no Windows run, and no scoring, calibration
  selection, `full_system` activation or production use was verified. `unverified-arm-sweep`
  remains a hard full-system rejection. **GO** only for a separately authorized, non-scored
  VALIDATION18 Stage 1 V4 diagnostic that explicitly selects
  `--execution-mode diagnostic_open_loop_chunk`, uses the approved config and a new output
  directory, and preserves V1/V2/V3. **NO-GO** for retrying V2 or V3, scoring, calibration
  selection, `full_system`, Packet B/D, TRAIN/TEST access or production activation.

## ADR-0077 — Stage 1 V4 postmortem and C-only bookkeeping correction (2026-11-21)

- **Scope and preservation:** offline, read-only investigation of the preserved V4 evidence plus
  one minimal source-backed bookkeeping correction. No Webots process, campaign, rerun, resume,
  scoring, calibration selection, `full_system`, Packet B/D, TRAIN/TEST access, geometry
  investigation, push or merge occurred. V1-V4 were never written. Recomputed tree aggregates
  remain V1 `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712`, V2
  `b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb`, V3
  `d4204a6fae79c16988e68f0b7cb61b27c90ba614f55fbfa507d1f666cf3d4d29`, and V4
  `48ce9f020802dac08c3fa065f4b53ef7d0f547160a891f488b081966f4eeba47`. Frozen inputs are
  unchanged: approved config `configs/worldmind_diagnostic.approved.yaml`
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`, definitions
  `configs/report/definitions.approved.json`
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`, and frozen plan hash
  `013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2`.
- **V4 finding A (`arm_7_joint` veto):** runtime evidence records two vetoes, both at decision
  `d01`, action index `15`, reason `joint target for arm_7_joint violates velocity limits`. The
  runtime-recorded previous commanded targets were `-1.2686945979476707` and
  `1.6620230047232916`; the rejected targets were `-1.4186945979476706` and
  `1.8120230047232915`, giving exact command deltas of `-0.1499999999999999` and
  `0.1499999999999999` rad. The latest observed states paired to action 15 by preserved
  `observation_t_s` were `0.44138256595535447` and `-0.048049874615676044` rad, giving
  observed-state deltas of `-1.8600771639030251` and `1.8600728793389676` rad. Reviewed
  `SafetyLimits` are `1.76 rad/s` for `arm_7_joint`, `command_horizon=1.0 s`, and
  `max_joint_step_rad=0.15`; the exact check is `abs(target - current) / command_horizon`.
  Replaying `SafetyMonitor.validate()` gives `accepted=True` for both targets against the
  previous **commanded** target (`0.15 rad/s` implied) and `accepted=False` against the latest
  **observed** state (`1.8600771639030251` and `1.8600728793389676 rad/s` implied). Root cause:
  `ScriptedChunkAdapter._bounded_joint_actions()` updates its planner-local `current_joints` to
  each commanded target, while the live monitor validates against fresh observed
  `RobotState.joint_pos`; the one-tick open-loop execution does not achieve each target, so the
  two state estimates diverge and the veto is correct fail-closed behavior, not a bad limit.
- **V4 finding B (terminal placement/HOME):** offline reconstruction from the pinned source and
  recorded initial oracle state gives the reviewed cadence `INIT 0-1`, `PREGRASP 2-11`,
  `DESCEND 12-17`, `CLOSE 18-20`, `LIFT 21-28`, `TRANSIT 29-36`, `PLACE_DESCEND 37-42`,
  `RELEASE 43-44`, `RETREAT 45-46`, `HOME 47-49` with `GRASP_Z=0.02`, `APPROACH_H=0.12`,
  `PLACE_CLEAR=0.01`, destination `[-1.0, -4.3, 0.821]` m and the frozen HOME targets. For
  representative full episode `validation-clear-i00-s0`, runtime NPZ evidence shows maximum
  commanded-versus-observed joint error `1.4137613171101848 rad` (median `0.7820 rad`; 86/86
  joint actions exceed the `0.02 rad` HOME tolerance), end-effector path `1.3876388902408507 m`,
  CAN path `9.491110469569001e-06 m`, gripper range `1.4357139981294163e-06` to
  `0.024252222508518852` (commands close `0.0` and open `0.045`), final CAN distance
  `0.2531504291422367 m` versus `distance_tolerance_m=0.1`, HOME `False` with maximum error
  `1.1536445247351717 rad` versus `home_tolerance_rad=0.02`. At phase boundaries after the last
  applied action, the maximum joint error was `0.955859` rad at PREGRASP, `0.710020` at DESCEND,
  `0.640640` at LIFT, `0.289666` at TRANSIT, `0.927271` at PLACE_DESCEND, `0.563286` at RETREAT
  and `0.725871` at HOME. Runtime values come from decisions/sidecars/NPZ; waypoint targets and
  the one-tick assumption are offline reconstructions. Root cause: the reviewed 50-step chunk
  plans bounded target increments and assumes each target is reached in one simulator tick, but
  the diagnostic contract advances one tick per target without feedback or settling; terminal
  verification correctly consumes observed state and therefore fails closed.
- **V4 finding C (step accounting):** runtime evidence records `1730` backend steps and
  `1,748` NPZ states across 18 episodes, but meta `worldmind_control_steps` totals `1700`. The
  `30`-step gap equals the two `15`-step accepted prefixes in the safety-aborted chunks at
  `validation-clear-i03-s0-d01` and `validation-reach-limited-i05-s1-d01`. Source cause: the
  open-loop abort branch in `worldmind_stage1_pipeline.py` broke out before
  `steps += chunk.backend_step_count`; only complete chunks reached the accumulator.
- **Approved correction:** only finding C was changed. The accepted chunk's
  `backend_step_count` is now added immediately after chunk evidence is validated and before
  any safety/backend/exception early break, and the later unconditional add was removed so a
  complete chunk is still counted exactly once. This changes bookkeeping only; it does not
  change the 50-step cadence, one-tick simulator contract, candidate generation, selection,
  safety checks, motion limits, verifier thresholds, observed-state terminal checks, HOME
  criteria, scene/seed plan, `retry_once → abort`, fail-closed behavior, full-system swept-arm
  blocker or V1-V4 evidence. Findings A and B require changing frozen candidate/execution
  behavior or adding feedback absent from the approved open-loop chunk; no such change was made
  and separate owner approval is required before attempting one.
- **Changed hashes:** `src/vlarobot/execution/worldmind_stage1_pipeline.py` changed from
  `b724da5736bde334022a77af97cbc2ff4092a19abb111b9e11b35e0c25ed544f` to
  `01bbd7f73649b9d3096f977fddc3849d5b53adeb40d81f0f0ef5e47deac1c08a`. New
  `tests/test_worldmind_v4_postmortem_regression.py` has SHA-256
  `69ccbf99bf8970a2f806071bfc36523c720006e84060b15b267df1e7b4c3bc9c`. Unchanged pinned source
  includes `execution/scripted_chunk_adapter.py`
  `890e2f5bd3621338db85cb60088f27c091b792ca437d2aeadd744dc7029432f4`,
  `execution/safety_monitor.py`
  `900c9218a9727a94cae0a1be9b071133c80925095cd7ca4b16241334adbc371a`,
  `execution/worldmind_pipeline.py`
  `27f3d45165d5e4e34fa4b670e41a709ff25206f2e5400fa441afe1dda887fdf8`, and
  `backends/worldmind_bindings.py`
  `4e39a7b8272b71d6d6fec19503b396cce5e1c1d0b339b2da08717675049ffbae`.
- **Verification:** the new focused file returns `8 passed`; the combined relevant set
  (`v4_postmortem`, `v2_postmortem`, `stage1_runner`, `eval_decision_evidence`,
  `eval_decision_adapters`, `worldmind_production`, `worldmind_waypoints`,
  `worldmind_readiness_regression`, `worldmind_endpoint_failstop_regression`) returns
  `82 passed, 8 warnings in 10.91s`. These include veto reproduction, unchanged limit enforcement,
  observed-state HOME/terminal checks, fail-closed unachievable trajectories, complete and
  safety-aborted step accounting, pairing alignment, and V1-V4 aggregate hashes.
  `uv run ruff check .` returned `All checks passed!`; `uv run ruff format --check .` returned
  `337 files already formatted`; `git diff --check` passes. The full offline suite returned
  `1 failed, 840 passed, 3 skipped, 540 warnings in 599.83s`; the only failure is the pre-existing
  unrelated notebook helper SHA mismatch in
  `tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`
  (`071cf660…` expected, `88fd4db9…` present), with no diff from `HEAD` in the helper, manifest
  or test.
- **Remaining limitations and V5 decision:** no live Webots execution, no V5 campaign, no
  physical settling, contact, grasp, gripper or swept-arm validation, no Windows run, and no
  scoring, calibration selection, `full_system` activation or production use was verified.
  **NO-GO** for another live campaign with the current source: A and B remain unresolved and any
  correction needs separate owner approval for frozen cadence/simulator/candidate behavior
  changes. **GO** is reserved for a separately authorized, non-scored VALIDATION18 Stage 1
  diagnostic after that approval and after a fresh output directory is selected; V4 must remain
  preserved and is not resumable.

## ADR-0078 — Offline observed-state feedback execution contract (2026-11-21)

- **Scope and authorization:** owner-approved, offline Stage 1 A/B correction for V4 findings A
  and B. A new explicit `diagnostic_observed_state_feedback_chunk` execution mode is available
  only with `scripted_proposer`; `full_system` and the two prior execution modes are unchanged
  and there is no implicit default. No Webots process, live actuation, campaign, V4 rerun/resume,
  scoring, calibration selection, `full_system` execution, Packet B/D, TRAIN/TEST access,
  geometry investigation, push or merge occurred. This ADR defines a diagnostic execution
  contract, not a scientific result or authorization to run a campaign.
- **Frozen inputs preserved:** K=32, H=50, `control_dt_s=0.032`, `max_steps=1059`, the frozen
  scene/seed plan, original selected candidate chunks, scripted waypoint order and cadence,
  IK/candidate generation, `SafetyMonitor`, `PlacementVerifier`, terminal verification,
  `retry_once → abort`, `full_system` blocker, scoring/calibration inputs and V1-V4 evidence are
  unchanged. Candidate chunks are immutable: feedback may take extra simulator ticks and execute
  projected targets different from the selected targets, but never rewrites the selected action,
  the original H=50 prediction or its candidate identity. V1-V4 tree aggregates remain V1
  `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712`, V2
  `b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb`, V3
  `d4204a6fae79c16988e68f0b7cb61b27c90ba614f55fbfa507d1f666cf3d4d29`, and V4
  `48ce9f020802dac08c3fa065f4b53ef7d0f547160a891f488b081966f4eeba47`. Frozen config
  `configs/worldmind_diagnostic.approved.yaml`
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`, definitions
  `configs/report/definitions.approved.json`
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`, and pinned source files
  remain byte-identical to ADR-0077.
- **Execution contract:** each `JointTarget` is projected from the latest observed
  `RobotState.joint_pos` with the existing bound
  `solver.execution_max_joint_step_rad=0.15 rad`. The projected action, never the selected
  action, is sent to the unchanged `SafetyMonitor` and then to the backend only when accepted.
  One simulator tick is taken per accepted command, followed by re-observation; the loop repeats
  until the selected target converges, a reviewed budget stops it, or the safety/backend path
  fails closed. No assumption is made that a previous target was reached. Convergence tolerances
  are diagnostic-only and are read from reviewed `configs/robot.yaml`: `0.01 rad` for non-HOME
  actions and `0.02 rad` for HOME actions. Reviewed budgets are `joint_stall_steps=10`,
  `joint_progress_epsilon_rad=0.000001`, `joint_move_timeout_steps=800`,
  `linear_move_timeout_steps=600`, `gripper_open_timeout_s=4.0` and the global `1059` step
  ceiling. The 600-step linear timeout has no applicable linear action in the scripted chunk;
  the gripper timeout has no settling semantics, and `GripperCmd` remains one bounded,
  safety-validated command event with one observed step. No grasp, contact, release or settling
  claim is made. Precedence is safety/backend failure, then the global ceiling, then an
  applicable per-move timeout, then stall, then target convergence. The global ceiling spans all
  feedback attempts and is never reset between retries; no new numerical parameter or relaxed
  limit was introduced.
- **Evidence and non-comparability:** a version-1 `observed-state-feedback.jsonl` sidecar is
  created only in feedback mode, with versioned provenance in `provenance.json` and a
  `paths["observed_state_feedback"]` entry only for that mode. Each row records selected and
  executed actions separately, action indices, projection inputs/outputs and clamped joints,
  pre/post observations, each safety/apply decision, progress and stall counters, step numbers,
  per-action and total budget consumption, and termination reason. Feedback decisions emit no
  `prediction_pairing.jsonl` record and leave `decision.execution.chunk` and
  `decision.prediction_pairing` absent. Provenance states `comparability="non_comparable"`,
  `pairing_records_emitted=0`, and that extra ticks and projected targets prevent any one-to-one
  pairing with the immutable H=50 prediction. No prediction-error or performance claim is
  supported by this evidence.
- **Abort and accounting contract:** feedback terminations map to
  `global_step_ceiling → WORLDMIND_BOUND_EXHAUSTED`, `joint_move_timeout → JOINT_MOVE_TIMEOUT`,
  `joint_stall → JOINT_STALL`, `safety_rejected → SAFETY_ABORT`,
  `backend_apply_rejected → BACKEND_ABORT`, and `execution_exception → EXECUTION_EXCEPTION`.
  Only the global ceiling sets `loop_exited_by_bound=True`; measured feedback steps are added to
  the episode step count before abort, and sidecar invariants require exactly one observation
  before and after each applied command, one step per accepted apply, complete outcome counts and
  `episode_steps_after <= 1059`.
- **Changed and additive hashes:** `src/vlarobot/execution/worldmind_stage1_pipeline.py`
  changed from ADR-0077's
  `01bbd7f73649b9d3096f977fddc3849d5b53adeb40d81f0f0ef5e47deac1c08a` to
  `ad4401a263b33cc8e2dacb7f5db80c13fa859fdb85079594ca090791570be420`; new
  `src/vlarobot/eval/observed_state_feedback.py` is
  `aa7e06236563cbff726f73e6e6f46862c2859dcacf9433acad6334cec8e559a7`; changed
  `src/vlarobot/eval/worldmind_evidence.py` is
  `69748ec0615e19f16a1ef2dd987642f42a6df90465c14859efd95aa7cc0e7406`; changed
  `src/vlarobot/cli.py` is
  `723bdb1ca9c8ef4cf13a72c12731bdf8f00d7f4ddcff1e1a4295822697bfa5ef`; new
  `tests/test_worldmind_observed_state_feedback.py` is
  `3ed91143e68625c5bd0365c64d9a1620005c35353f3c601e6632dc01e095bc1b`. Unchanged regression
  files are `tests/test_worldmind_v4_postmortem_regression.py`
  `69ccbf99bf8970a2f806071bfc36523c720006e84060b15b267df1e7b4c3bc9c` and
  `tests/test_worldmind_stage1_runner.py`
  `d402bf5bf25b07cec023c05b77d0bf08f5bff80571e7cdbcb86d9a62ca4cb2c5`.
- **Verification:** the focused feedback file returns `9 passed`; the final combined relevant set
  (`v4_postmortem`, `stage1_runner`, `observed_state_feedback`) returns
  `33 passed, 8 warnings in 6.87s`. `uv run ruff check .` returned `All checks passed!`;
  `uv run ruff format --check .` returned `339 files already formatted`; `git diff --check`
  passed. The repository-wide offline suite returned
  `1 failed, 849 passed, 3 skipped, 540 warnings in 602.23s (0:10:02)`; the only failure is the
  same pre-existing unrelated notebook helper SHA mismatch recorded in ADR-0075/ADR-0077
  (`tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`,
  `071cf660…` expected, `88fd4db9…` present), with no diff from `HEAD` in the helper, manifest
  or test. All nine feedback regressions passed inside the full suite. An additional
  `uv run ty check src` probe is not clean (43 repository-wide diagnostics in existing
  type-model patterns, including the Stage 1 `_world_state` config call sites); it is not part of
  this ADR's offline acceptance workflow and no new sidecar-module diagnostics were reported.
- **Remaining limitations and recommendation:** the A/B correction was verified only offline with
  deterministic FakeBackend regressions. No Webots timing, physical joint response, grasp,
  contact, release, settling, swept-arm certificate, Windows run, V5 campaign, scoring,
  calibration selection, `full_system` activation or production use was verified. **CONDITIONAL
  GO** is reserved for a separately authorized, non-scored Stage 1 diagnostic that explicitly
  selects this mode with `scripted_proposer`, writes only to a fresh versioned output directory,
  and treats feedback rows as non-comparable diagnostics. **NO-GO** for a live/scored campaign,
  `full_system`, calibration selection, retrying or resuming V4, or treating these regressions as
  evidence of physical task success.

## ADR-0079 — Offline V5b postmortem corrections A-E (2026-11-21)

- **Scope and authorization:** owner-approved, offline implementation of the read-only V5b
  postmortem findings A-E. The implementation plan is
  `docs/WORLDMIND_V5B_POSTMORTEM_IMPLEMENTATION_PLAN_ADDENDUM_V1.md`. No Webots process, live
  campaign, V5c execution, scoring, calibration selection, `full_system`, Packet B/D, TRAIN/TEST
  access, geometry change, push or merge occurred. V1-V5b evidence is read-only and was not
  resumed or rewritten. This ADR adds diagnostic bookkeeping only; it is not a scientific result,
  performance claim or live authorization.
- **Frozen inputs preserved:** K=32, H=50, `control_dt_s=0.032`, `max_steps=1059`, the frozen
  scene/seed plan, candidate generation, selected candidates and waypoint order, IK/verifier
  thresholds, `SafetyMonitor`, `PlacementVerifier`, report definitions and `retry_once → abort`
  are unchanged. Recomputed V1-V5b tree aggregates match their approved values: V1
  `08258083104715443ebbbe0c77e095bd489b0ca2ffd1899c6173cea63ec1f712`, V2
  `b5b4075d8c6db133027ea900cecb5db995f286a52bdeb6d37e96231c07194ccb`, V3
  `d4204a6fae79c16988e68f0b7cb61b27c90ba614f55fbfa507d1f666cf3d4d29`, V4
  `48ce9f020802dac08c3fa065f4b53ef7d0f547160a891f488b081966f4eeba47`, V5
  `96d9540750257cd49bf0d3e01a9a4fd68d9e015d096e1187cc1979eebfe8df1e` and V5b
  `1946ff3ab5ce461ad92f3a354df4d292ecf469af2d6e7a77bd1322c530eae828`. Frozen config
  `configs/worldmind_diagnostic.approved.yaml`
  `9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03`, definitions
  `configs/report/definitions.approved.json`
  `457617762e8952796cd1f2fcfb0f3cf649735c727e91f7f1f179a103750dde14`, `configs/robot.yaml`
  `1a9f0c2b57d771e45b6908d05177eb64eb80634b7ab2d35c5dfbefb1f863ca1d`, `configs/eval.yaml`
  `79e4dfa708baea8bb85e839cbcbcc6e48bb2ca84eac830ebdbaffd19b9daaf4f`, and the frozen plan
  provenance hash `013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2` in all six
  run trees remain unchanged. `scripted_chunk_adapter.py`, `worldmind_bindings.py`,
  `safety_monitor.py`, `worldmind_pipeline.py` and `planning/verification.py` remain byte-identical.
- **Correction A:** feedback mode now runs a deterministic, fixed-seed-`0`, read-only solver
  residual precheck over re-anchored observed waypoints before candidate generation. It uses the
  reviewed per-waypoint IK tolerances exactly, never clamps, substitutes or repairs an endpoint,
  and emits a structured `unreachable_observed_target` diagnostic event with measured residual,
  exact tolerances, `candidate_generation_performed=false` and `backend_actions_applied=0`. A true
  solver exception or non-finite residual still fails closed with the original exception.
- **Correction B:** a separately labelled `diagnostic_home_completion` phase runs only after
  selected-chunk convergence and targets the approved `verification.home_joints` vector with
  `home_tolerance_rad=0.02`. Selected HOME payloads and their reviewed `robot.yaml` error tolerance
  are preserved; every diagnostic HOME intermediate is sourced from observation, safety validated,
  apply validated, step counted in the same 1059-step ledger and separately reported. HOME
  completion errors and terminal reasons are never mixed into selected-action errors and create no
  prediction pairing.
- **Correction C:** final selected-action errors are preserved. Per-action and episode termination
  distinguish `target_converged`, `global_step_ceiling`, timeout, stall, safety rejection, backend
  rejection and execution exception. The convergence summary excludes non-converged actions from
  converged tolerance statistics while retaining their indices, reason counts and final errors.
- **Correction D:** version-1 `collision-proxy-attribution.jsonl` attributes the existing
  displacement/tilt proxy per object, records threshold source and backend-observed contact steps,
  and never promotes a proxy to contact, grasp, release or a swept-arm certificate. Gripper
  completion uses existing reviewed `finger_speed=0.03`, opening limits `[0.0, 0.045]`,
  `control_dt_s=0.032` and `gripper_open_timeout_s=4.0`; the derived arrival tolerance is
  `0.00096 m` and timeout is `125` steps. Every apply is safety and backend validated, with
  explicit completion reason, target, error and unit. The FakeBackend rate-limited progression is
  opt-in, so default behaviour is unchanged.
- **Correction E:** each attempt snapshot records the complete observed `base_pose_world`
  position and quaternion (or explicit `null` when absent). `observed-state-feedback.jsonl` moves
  from schema 1 to schema 2; schema 1 rows remain readable verbatim. Schema 2 adds HOME completion,
  convergence summary and explicit gripper completion fields. No frozen scored metric or H=50
  prediction comparison changes; feedback remains non-comparable and unscored with zero pairing
  records.
- **Changed and added hashes:** `src/vlarobot/execution/worldmind_stage1_pipeline.py` changed from
  `ad4401a263b33cc8e2dacb7f5db80c13fa859fdb85079594ca090791570be420` to
  `a7135552750d43f0bef720c50eb9faa8c087c279588d4b6a326d2172811341c1`;
  `src/vlarobot/eval/observed_state_feedback.py` from
  `aa7e06236563cbff726f73e6e6f46862c2859dcacf9433acad6334cec8e559a7` to
  `961b423e9ef12877500564fb7efe3d98c922f593217813c0821055be72a79136`;
  `src/vlarobot/eval/worldmind_evidence.py` from
  `69748ec0615e19f16a1ef2dd987642f42a6df90465c14859efd95aa7cc0e7406` to
  `0cd500850324a000bbfb28b282d88d47fc453c364e82a6416b67d9a458acb4f6`;
  `src/vlarobot/backends/fake/backend.py` from
  `fd6c3e53f074b8454ac82150fb6dddc40ef90eded9323332ea62a089f2373cdb` to
  `5156026785bd642532019610c90549cbf6e8ed1c67c5829f4aa32c6caea6b9eb`;
  `tests/test_worldmind_observed_state_feedback.py` from
  `3ed91143e68625c5bd0365c64d9a1620005c35353f3c601e6632dc01e095bc1b` to
  `85c581bd19a5d6ae3eb86658f189b1b0e6bb1f6b201ef85bdddaea2bd6ad495f`. New
  `tests/test_worldmind_v5b_postmortem_regression.py` is
  `8de94f81bc642dcd22d78e69053b01327a73773396ae4b1ec3077246b70ac095`; new implementation plan
  addendum is `docs/WORLDMIND_V5B_POSTMORTEM_IMPLEMENTATION_PLAN_ADDENDUM_V1.md`. The pre-existing
  untracked `configs/report/definitions.approved.json` was left untracked and byte-identical.
- **Verification:** the 13 new A-E regressions plus the 9 existing feedback regressions return
  `22 passed, 8 warnings in 3.97s`; the combined relevant offline set (`v5b_postmortem`,
  `observed_state_feedback`, `v4_postmortem`, `stage1_runner`, `fake_backend`) returns
  `53 passed, 8 warnings in 8.24s`. `uv run ruff check .` returned `All checks passed!`;
  `uv run ruff format --check .` returned `341 files already formatted`; `git diff --check` passed.
  The repository-wide offline suite returned
  `1 failed, 862 passed, 3 skipped, 540 warnings in 647.30s (0:10:47)`; the only failure is the
  pre-existing unrelated notebook helper SHA mismatch recorded in ADR-0075/ADR-0077/ADR-0078
  (`tests/test_drive_handoff_notebook.py::test_v7_current_session_cells_and_helper_manifest`,
  `071cf660…` expected, `88fd4db9…` present), with no diff from `HEAD` in the helper, manifest or
  test. All 13 new regressions and all preserved-evidence tests passed inside the full suite. All
  six V1-V5b tree aggregates, all frozen config/eval/robot hashes and the frozen plan provenance
  hash were independently recomputed and matched after the code and documentation changes.
- **Remaining limitations and recommendation:** verification is offline and deterministic only.
  No Webots timing, physical joint or gripper response, grasp, contact, release, settling,
  swept-arm certificate, Windows run, scored campaign, calibration selection, `full_system`, V5c
  campaign or production use was verified. **NO-GO** for V5c live/scored execution, scoring,
  calibration selection, retry/resume of V5 or V5b, or production use. **CONDITIONAL GO** is
  reserved for a separately authorized, non-scored VALIDATION diagnostic that explicitly selects
  feedback mode with `scripted_proposer` and writes only to a fresh versioned output directory.

## ADR-0080 — Report candidate IK failure at the actual draw (offline only)

- **Supersedes ADR-0079 correction A** and section 2 of its implementation plan addendum. The
  independent seed-0 precheck could not establish equivalence with the approved draw seeds and
  simulated joint states. Remove its early veto. The other ADR-0079 corrections remain in force.
- **Generator contract:** keep K=32, H=50, decision seed derivation, draw seed order, eager
  sequential all-or-error generation, candidate IDs and payloads, waypoint construction,
  tolerances, and first-eligible scripted selection. A failed draw leaves later draws unattempted.
  The episode records the original exception and the runner fail-stops. A failed pool yields zero
  applies, completed generation observations, and prediction-pairing records for that decision.
- **Evidence:** feedback mode adds three version-1 event types to `diagnostic-events.jsonl`.
  `candidate_ik_residual_outside_tolerance` reports a finite failure on a specific draw.
  `candidate_ik_nonfinite_residual` stores non-finite components as null.
  `candidate_ik_solver_exception` stores the original exception type and message. Each row
  records the decision and episode, zero-based draw slot, draw seed, waypoint label, requested
  base-frame pose, effective tolerances and failed-decision apply count. These events report a
  failed solve under the approved policy. The feedback v1/v2 reader and historical rows are
  preserved.
- **Historical evidence:** V5b episode index 3 (`validation-clear-i03-s0`) records an
  `EXCEPTION` after 419 control steps with the message
  `scripted pose residual outside reviewed tolerance`. The failed target and numeric residuals
  were not independently reproduced from the archived run. The archive lacks the full observed
  world-frame base transform and has no failed-draw target or residual record. The values
  2.770961553780716 m and 0.3235694262270646 rad are injected regression inputs.
- **Scope:** offline implementation only. V5c readiness and any live diagnostic require separate
  owner authorization.
- **Offline verification:** focused tests returned 24 passed. The relevant offline set returned
  55 passed. Approved-source parity matched six generated pools and 9,600 action payloads.
  Ruff lint and format and `git diff --check` passed. The repository-wide non-Webots run
  excluded the Packet B scope module and returned 834 passed, 3 skipped and one failure.
  The failure was the pre-existing notebook-helper SHA mismatch (`071cf660…` expected,
  `88fd4db9…` present). The frozen config, definitions, recomputed VALIDATION plan and all six
  recomputed V1–V5b evidence tree hashes matched ADR-0079.

## ADR-0081 — Authoritative telemetry and execution outcomes for feedback diagnostics

- **Scope:** supersedes ADR-0079's diagnostic gripper telemetry and feedback accounting where
  they could omit a missing device or conflate an accepted command with a later failure.
  Applies to the explicit scripted observed-state feedback diagnostic. ADR-0080's generator
  and candidate failure rules continue to apply.
- **Gripper:** validate both configured names in the existing `finger_joint_sensor_rad`
  diagnostic mapping. Despite the historical key spelling, the TIAGo finger joints are
  prismatic. Their raw positions are metres in the reviewed stroke interval. The diagnostic
  measures the mean displacement from the configured closed end only after validating both
  readings. A missing, non-finite, nonnumeric, or out-of-range reading emits a version-1
  `invalid_gripper_telemetry` event identifying the affected devices and aborts. The reviewed
  tolerance and timeout still apply in either direction. Ordinary gripper command and telemetry
  code keeps its existing behavior.
- **Transport:** protocol v3 adds the opt-in `diagnostic_step` request for one tick. It returns
  `confirmed_ticks=1` only after the underlying Webots physics-step call reports success,
  including when visualization or observation construction subsequently fails. A stopped
  backend that declines to call physics returns 0. A termination return or exception from the
  physics call does not establish advancement and returns null. A lost, malformed or missing
  outcome is unknown. Ordinary `apply` returns its existing verdict, and transport failures
  yield an unknown apply in the diagnostic. The standard `apply` and `step` requests and
  responses retain their contracts. No unknown operation is automatically retried.
- **Recorder:** the diagnostic obtains the backend verdict or outcome before requesting the
  corresponding recorder call. A recorder exception preserves the confirmed backend outcome.
  `completed` means the recorder call returned successfully. Durability is determined by the
  recorder's later episode finalization and filesystem. Sidecar persistence itself can also
  fail. No success is inferred for writes that throw.
- **Feedback schema 3:** existing schema-1 and schema-2 rows remain readable. Each attempt
  adds `apply_outcome` (not_attempted, accepted, rejected, unknown), `step_outcome`
  (not_attempted, confirmed, not_executed, unknown), `observation_outcome` (not_available,
  failed, processing_failed, available), and `apply_recording` and `step_recording`
  (not_attempted, completed, failed). `failed` observation means a confirmed tick did not
  return a usable observation. `processing_failed` means it returned an observation but
  feedback could not finish processing it. The attempt retains a snapshot when one is
  available. Apply totals reflect confirmed replies before recording or stepping.
  `backend_step_count` counts only confirmed physics ticks, and `observation_count`
  counts the initial observation plus usable post-step states. Added totals cover unknown
  applies, unknown ticks, observation failures and recording failures. A version-1
  `diagnostic_reobserve` event records the tick, observation and recorder outcomes for
  recovery ticks with no associated apply. Each confirmed tick is charged once to the same
  1059-step episode ceiling. Execution or evidence processing exceptions fail-stop before
  another command. Ordinary safety and backend rejection boundaries retain their semantics.
- **Limitations:** the new request needs a server running this implementation. A protocol-v3
  ping alone establishes wire connectivity. Offline tests exercise injected devices, protocol
  messages and recorder faults. Live device readings, simulator behavior, and V5c remain
  unverified. Another read-only readiness audit and separate live authorization are required.
- **Offline verification:** 47 focused and 86 relevant tests passed. Approved-source parity
  matched all six candidate pools and 9,600 action payloads. Ruff lint and formatting and
  `git diff --check` passed. The repository-wide non-Webots suite excluded the Packet B scope
  module and returned 849 passed, 3 skipped and one failure. The failure was the pre-existing
  notebook-helper SHA mismatch (`071cf660…` expected, `88fd4db9…` present). Frozen config,
  definitions, the recomputed VALIDATION plan and all six historical evidence-tree hashes
  matched ADR-0079.

## ADR-0082 — Gate V5c on a session-bound ADR-0081 capability

- **Advertisement:** protocol-v3 ping preserves its existing fields. The server adds
  `capabilities.diagnostic_step: 1` and `diagnostic_backend: webots` only for the explicit Webots
  ADR-0081 backend opt-in with a callable diagnostic tick implementation.
- **Mandatory gate:** the Webots observed-state feedback runner uses the campaign client's ping
  before opening the writer or sending reset, apply or step. It validates the v3 ping reply type
  and success status, positive integer PID, finite positive start time, nonempty nonce, explicit
  integer capability version 1, and expected backend. Failure produces a structured
  `diagnostic_capability_rejected` error and leaves any existing evidence untouched.
- **Continuity:** the campaign client wraps reset, apply, diagnostic step and ground-truth reads
  in the new v3 `diagnostic_session_request` operation with the validated identity. The server
  checks capability and identity before dispatch and echoes identity in the reply. A replacement
  supported server rejects the stale identity. An older server rejects the new operation type.
  The guard also applies if a bound client requests shutdown. These checks bind requests to the
  server instance across transparent transport reconnects. They do not provide authentication
  against an impersonating endpoint.
- **Failure behavior:** a rejected, lost or invalid guarded reply latches the client closed to
  further campaign operations. A confirmed partial tick result remains available to feedback
  accounting. The client neither retries unknown operations nor falls back to ordinary step.
  Ordinary protocol-v3 clients, core interfaces, numerical settings, and evidence schemas retain
  their contracts. Offline-only change. A focused read-only readiness verification and separate
  live authorization are required.
- **Offline verification:** 64 focused and 103 relevant tests passed. Approved-source parity
  matched six candidate pools and 9,600 action payloads. Ruff lint and formatting and
  `git diff --check` passed. The repository-wide non-Webots suite excluded the Packet B scope
  module and returned 866 passed, 3 skipped and one failure. The only failure was the existing
  notebook-helper SHA mismatch (`071cf660…` expected, `88fd4db9…` present). The frozen
  configuration, definitions, regenerated VALIDATION plan and V1–V5b tree hashes all matched
  ADR-0079.

## ADR-0083 — Offline scripted-primary world-model comparison factory

- **Scope.** Adds a separate Python composition path for the frozen `vla_world_model` arm
  under the ADR-0029 scripted-primary allocation. The production `eval-worldmind` CLI,
  configuration schema, shared episode policy, and all existing arm gates keep their
  contracts. The new path runs a single fixture decision at a time on FakeBackend and
  has no scientific execution entry point.
- **Composition.** Reuse the scripted adapter binding, frozen seed derivation, K=32,
  H=50, analytic trajectory predictor, all four Eq. 3.8 inputs, the existing
  `ComparisonPlanner` ranking branch, and `execute_first` with SafetyMonitor. That branch
  does not call RiskGate selection or heuristic hard-check callbacks. Its prediction reasons
  do not become a predictive veto. The motor veto remains mandatory.
- **Inputs and provenance.** The factory reloads the report definitions and requires the
  ADR-0079 pin. With absent calibration it returns a non-executable scientific readiness
  result. An optional supplied candidate can be checked with the existing artifact loader
  for identity and hashes. Loading it establishes neither arm-specific applicability nor
  activation. An explicitly injected, in-memory fixture ranking can exercise the composition
  and is labeled fixture-only with no scientific approval or calibration ID. Fixture output
  carries arm, actual proposer, adapter and surrogate bindings, prediction and veto modes,
  definition and report hashes, draw seeds, candidate IDs, selection, and fake motor verdict.
  It creates no scored output or prediction-to-observation pairing.
- **Remaining gap.** The no-prediction arm is outside this implementation. ADR-0065's
  current objective factories require predicted trajectory inputs. A separate owner decision
  is required for the admissible no-prediction inputs, ranking and unavailable evidence
  representation before that arm can be composed. Live or scored execution, calibration
  selection, and full-system execution remain outside the authorization.
- **Verification.** Four focused and 102 relevant offline tests passed. The shipped production
  registry instantiated the analytic fixture without executing a decision. Ruff lint and format
  and `git diff --check` passed. The repository-wide non-Webots suite excluding the Packet B
  scope module returned 876 passed, 3 skipped, and one pre-existing notebook-helper SHA failure
  (`071cf660…` expected, `88fd4db9…` present). All four frozen input hashes, the regenerated
  VALIDATION18 plan hash, and V1–V5b evidence tree hashes matched ADR-0079.

## ADR-0084 — Prediction-free scripted-primary heuristic-veto ablation

- **Status and scope.** Owner-approved scientific contract and offline fixture implementation.
  The frozen arm is `vla_risk_veto`, with actual source `scripted` and adapter
  `scripted_chunk_adapter` under ADR-0029. This is an arm-specific exception to Eq. 3.8 ranking.
  The other objective definitions retain their approved semantics. Comparison to
  `vla_world_model` varies both prediction availability and selection policy. Scored or live
  execution remains unauthorized.
- **Candidate contract.** Reconcile the complete frozen K=32, H=50, 32 ms pool with the generator
  record, retaining the original draw seeds, indices, IDs and payloads. Pool-integrity failure
  stops before candidate evaluation. Evaluate `finite_chunk_v1` and first-command
  `SafetyMonitor.validate` against the current observation for every candidate. Require one
  explicit pass in each fixed check field. Record one disposition per candidate, count rejected
  candidates once, and select the lowest eligible draw index. No surrogate, Eq. 3.8 score,
  RiskGate ranking, or prediction-to-observation pairing is produced. A second independent
  motor validation immediately precedes apply. Record its outcome separately from candidate
  prefilter rejects. The checks provide no swept-clearance, future-command, or success guarantee.
- **Recovery accounting.** Keep the episode's empty-set counter separate from the verifier's
  retry counter, as in the shared production episode loop. The first empty set consumes the
  re-observation allowance and regenerates the primary family when budget permits. The next
  empty set records `alternate_candidate_family:unavailable` and aborts because the binding is
  null. Verification may request one primary-family retry, then aborts on another failure.
  That retry does not restore empty-set recovery. Both paths share the existing global decision
  and control ceilings. A rejected pre-apply motor verdict prevents apply and ends the fixture.
- **Verification.** The additive concrete `PlacementVerifier.check_observed(task, observed)`
  shares the existing calculation with `check(task, predicted, observed)`. The core protocol
  retains its signature. Target identity, pose and HOME finiteness, quaternion validation,
  placement distance, height and HOME inclusive bounds, and strict tilt bound remain unchanged.
- **Reporting.** The additive Table 5.1 v1 representation carries the approved name and unit,
  status, nullable value, reason, and applicable denominator. Prediction error for this arm is
  `not_applicable`, null, `arm_has_no_prediction`, with zero valid paired steps and no pairing
  rows. `unavailable` is reserved for applicable metrics lacking evidence. Measured zero requires
  valid evidence. A numeric comparison requires measured coverage for every expected arm;
  observed-outcome comparisons also retain their existing scientific pairing and denominator
  rules. Prediction-dependent contrasts including this arm are ineligible. The existing finite
  numeric export retains its contract.
- **Calibration and boundaries.** This arm adds no heuristic-score calibration. The world-model
  arm's calibration stays blocked pending an approved evidence manifest and applicability
  decision. The separate fixture uses FakeBackend only. The scientific CLI, frozen
  configurations, shared production policy, world-model fixture, full-system prerequisites,
  safety settings and report definitions retain their contracts.
- **Verifier source pin.** The ADR-0079 preservation test pins the additive verifier
  implementation at SHA-256 `823a1f2a2d4b0314f1481fbca95e9472cc5a589f71bfdbebbe53939cc8a5fe61`.
  Behavioral parity tests cover the legacy delegation and observed-only verdict, including the
  inclusive distance, height, and HOME bounds and strict tilt boundary.
- **Offline verification.** 18 focused and 138 relevant tests passed. Ruff lint and formatting
  and `git diff --check` passed. The repository-wide non-Webots suite excluded the Packet B
  scope module and returned 894 passed, 3 skipped and one pre-existing notebook-helper SHA
  failure (`071cf660…` expected, `88fd4db9…` present). All four frozen input hashes, the
  regenerated VALIDATION18 plan and the V1–V5b tree hashes matched ADR-0079.

### ADR-0030 addendum 2 — November 24 experiment scope

- Owner authorization sets a 0.0001 m **logging-only** finger telemetry allowance.
  The observed simulated finger coordinates can extend roughly 0.000001 m beyond
  motor targets during idle settling. The signed readings stay raw. Arm joint
  admission, the independent SafetyMonitor, motor targets, and task thresholds
  keep their original limits. No endpoint experiment is planned.
- Eq. 3.8 uses placement-stage tool distance in metres, path length in metres with
  zero time weight, a 0.03 m tool-envelope clearance proxy, and zero Ju weight
  because no justified scale bundle exists. The preregistered relative weights are
  equal and goal doubled, with risk doubled optional as capacity permits.
  Complete observed PlacementVerifier outcomes on the same frozen VALIDATION18
  identities select the policy by success fraction, then candidate ID.
- The November 24 arm-5 path uses joint-limit margin, waypoint IK feasibility and
  residual, verified table-box clearance, modeled-object tool-envelope clearance,
  and placement tolerance. The absence of a full-arm sweep certificate is outside
  this geometric-proxy scope. Arm links outside the tool envelope and unmodeled
  background geometry are excluded. The older Packet B result of zero positive
  controls in 13,766 expert steps remains attributable to the older sweep contract.
- Eligible trials finish on a frozen scene with the recorded configuration and no
  software or runner failure. Terminal task failures and vetoes count. An unreliable
  attempt is preserved and retried once on the same seed in a fresh session.
  The scripted-primary comparison does not establish that WorldMind improves a
  competent VLA.

### ADR-0030 addendum 3 — November 24 opening cadence and geometric replay

- Arm 3 VALIDATION uses the baseline's observed INIT opening criterion and
  timeout before PREGRASP. Idle ticks share the existing episode ceiling. The
  executor configuration SHA-256 is
  `98eb1ec19b3e322e32554cf9f6b6c004d9ebce96d1ea4849657c1d90360fb6e4`.
- The arm-5 geometric replay measures metres in the world frame. It uses the
  verified tabletop half-space, separate DESCEND and PLACE_DESCEND contact
  heights, and a 3.8 mm pad-derived tip radius against non-target oriented boxes.
  The target CAN is exempt. The 0th percentile of VALIDATION expert minima
  bounds the replay floors so every recorded geometric positive control remains
  represented. The fit and check share the same 15 successes. TRAIN provides a
  separate object-only check because the archived phase labels are missing.
- Geometric checks pass 15/15 VALIDATION successes. The two collision controls
  are vetoed. The full gate records zero verified passes: nine archived
  VALIDATION paths violate the existing joint margin and the full-pose waypoint
  IK inputs are absent. The geometric values are offline replay configuration
  SHA-256 `e35cf001b0fa02af16025fa4bbb1c5dfb83f9e5b56216f335537af99bb3ff87c`.
  Live arm 5 still requires a positive control with the existing joint, IK,
  and placement checks. Preserve the v1 spherical-veto results separately.

### ADR-0030 addendum 4 — November 24 RiskGate numerical tolerance

- The arm-5 v2 control admits signed predicted joint slack down to −0.0001 rad
  in the RiskGate joint-limit predicate only. Archived expert state crossed
  the original bound by up to 0.000000397 rad. The tolerance addresses numerical
  noise. Joint limits, independent SafetyMonitor, IK and placement gates retain
  their original values and behavior.
- The `equal` v2 campaign requires at least 14 of 15 successful VALIDATION
  expert-command controls to pass. The v2 geometric configuration hash is
  `704f143f49f5d3552ebccd7e965985fc482c0ac18ad1fbc4c392081abbd82420`.
  The prediction replay gives an upper bound of 6/15 because nine traces
  fail the geometric veto before any uncertainty about archived full-pose IK
  could affect the result. This configuration is retained as offline evidence.

### ADR-0030 addendum 5 — matched-prediction arm-5 clearance

Arm 5 equal VALIDATION uses the hash-pinned matched table half-space and
non-target oriented-box profile. Lower the affected fitted floors to the
corresponding minima of the archived expert commands propagated with the
registered velocity-limited predictor. Keep the physical tabletop floor and
box-penetration veto, the approved joint-slack allowance, live IK, placement,
and independent pre-apply motor checks. The same contacts and phase identities
apply during fitting and ranking, with the recorded expert timing in replay
and the frozen H=50 schedule in ranking. This is an in-sample control.

The expert-command replay passes all 15 recorded VALIDATION controls on
available inputs. An unexecuted full-pose solve on archived commands is not
a release condition. The deliberately crossing table and non-target controls
remain vetoed. The live executor uses the existing observed INIT opening
criterion. The arm-5 profile SHA-256 is
`6ea88763ad27a769943aa52132c60e785e4aa4e811899e34473c6f6cec17bed4`.

### ADR-0030 addendum 6 — stateful scripted WorldMind execution

Arm 3 binds the expert's observed waypoint completion state to WorldMind's seeded
chunk decisions. H=50 limits a candidate's prediction/command horizon. A newly
completed waypoint invalidates the held suffix. If it has not completed by H,
a fresh chunk starts from the resulting observation while retaining the active
target and expert settle timer. Final verification follows HOME. The motor veto,
global budgets, observed verifier and single recovery allowances are preserved.

The arm-3 executor profile SHA-256 is
`585ea6ef1ef850f2c0251b4fa797cb5a4032376a2ddca370144b0d35bf22c58b`.
Run the campaign from an isolated detached worktree. The candidate-00
demonstration is separately marked non-scored. The prior fixed-phase campaigns
retain their results and engineering interruptions.

Arm 5 still needs a binding for the mandatory placement-stage input when an
earlier chunk contains only the current waypoint. The matched-clearance
configuration and original placement definition are preserved.

### ADR-0030 addendum 7 — phase-aware WorldMind placement

For stateful arm 5, the terminal PLACE_DESCEND waypoint requires a predicted
settled state and the existing 0.10 m placement check. The state is the first
predicted joint observation satisfying the existing waypoint settle predicate.
Other chunks record placement and Jg as not applicable. Eq. 3.8 omits Jg from
both sides of such a ranking comparison. Jl and Jr use the predicted executable
prefix through waypoint completion or H, and Ju retains its approved zero weight.

The matched clearance profile, frames and thresholds are unchanged. Evaluate
contact height endpoints only at the corresponding predicted terminal DESCEND
or PLACE_DESCEND waypoint. All other motion retains its phase floor and
non-target clearance checks. The active suffix expires when the waypoint
changes or when the predicted completion boundary is reached. The next
prediction starts from the actual observation. Final placement verification
still follows observed HOME.

The arm-5 stateful profile SHA-256 is
`9373777118cfe9966569889e887453afa2c913cd5af7fe4d71673f49490b1d5f`.

### ADR-0030 addendum 8 — vetoed held continuation

Stateful arm 5 may draw up to three replacement pools for a vetoed held
continuation at the same waypoint, with the original jitter support, all
existing hard checks and an independent motor verdict. There is no simulator
tick between veto and replacement. The actual accepted replacement is the
setpoint for the existing settle predicate. Exhaustion is an eligible
NO_SAFE_SET outcome. The risk records identify each candidate, signed veto
margin when available, source pool, selected replacement and execution receipt.
Pool diversity counts use distinct command payloads, independent of draw IDs.

The revised executor SHA-256 is
`d716898ddd48dbfdee30f6a2d561c2e0e298c18c613e17a5c82f595e95f07538`.
The objective and clearance profile are unchanged. November 29 remains the
weight and threshold freeze date.

### ADR-0030 addendum 9 — stateful ranking comparisons

Compare the approved equal and goal-heavy vectors in stateful arm 5 on the
same VALIDATION18 identities. An archived equal-pool replay must establish
a changed selected command payload before launching goal-heavy. Missing Jg
outside terminal PLACE_DESCEND still omits that term under both vectors.

Arm 4 uses the same stateful scripted proposer, analytic prediction, active
prefix and approved equal weights for the pre-calibration VALIDATION comparison
against arm 5 equal. It ranks finite applicable Eq. 3.8 inputs without calling
RiskGate or enforcing predictive clearance and placement vetoes. It retains the
expert's waypoint checks, independent SafetyMonitor and final observed verifier.
Arm-5 veto-triggered replacement is absent when there is no predictive veto.
The arm-4 profile is specific to VALIDATION. It does not declare a selected
calibration or authorize TEST.

### ADR-0030 addendum 10 — terminal HOME reference alignment

On stateful VALIDATION, the optional verified-HOME profile requests the
frozen verifier's configured HOME vector for the expert's terminal HOME
waypoint. It waits for finite joint feedback within the same inclusive
tolerance used by the observed verifier. The waypoint predictor uses the
same terminal predicate. Intermediate waypoint completion, motor limits,
placement criteria, timeout and episode ceilings retain their values.
The baseline's HOME behavior is unchanged.

The prior `clear-i03-s2` trace was within 0.019699 rad of the selected
command and 0.021947 rad from the verifier's HOME. This mismatch triggered
a full task retry. The new stateful profile hash incorporates HOME profile
SHA-256 `2a014f1e38e28e56d43c5ec0127aa9030ef59e2af4dbf180ffc2640cba486e85`.
Do not mix old and corrected HOME profiles in policy selection.

### ADR-0030 addendum 11 — Matched VALIDATION LIFT entry, 2026-11-24

The owner approved a one-tick predictive-veto exception for the first LIFT
waypoint immediately after CLOSE. Three archived inherited shortfalls were
0.138–0.382 mm below the operating floor. All 192 archived predictions recovered
on tick one in the offline replay, with the clear and collision controls unchanged.

Only the starting endpoint may be below the unchanged operating floor. It must be
above the geometric plane, and every subsequent predicted endpoint must be
at or above the floor. Other predictive vetoes and the independent motor monitor
still apply. The first confirmed observation must recover the unchanged floor
and show no recorded contact. Failure terminates the episode, with no repeated
exception for that waypoint. Predicted and observed heights are recorded.
This is geometric-proxy evidence, with no independently certified safety claim.

Both matched arm-5 profiles use the same pinned exception. All four stateful
VALIDATION profiles use the configured HOME reference introduced in addendum 10.
The prior profiles and their results remain engineering-development evidence.

### ADR-0030 addendum 12 — Early frozen TEST, 2026-11-25

The owner moved the earliest TEST date to November 25 and made arm 4 mandatory.
Run the frozen 30 identities in arms 1, 3, 5 and 4. Arm 2 remains excluded.
The November 25 freeze, policy, thresholds, scene identities and seeds are unchanged.

### ADR-0085 — V2 learned TRAIN recovery profile, 2026-11-25

The recovery profile leaves V1 and the default historical VLA runner intact.
Learned proposals within 1 mm outside the per-finger target range may be
projected onto [0, 0.045] m. The envelope covers the observed 0.793 mm training
checkpoint overshoot. It applies to proposals alone. Larger or malformed
outputs fail closed. Raw and projected values and their differences are saved.
Independent SafetyMonitor and IK bounds are unchanged.

Returning to HOME becomes terminal only when the existing independent evaluator
confirms task success. That verdict gates episode termination. It supplies no
model input, phase commands or fallback actions. The original TRAIN horizon and
24-trial admission criterion are retained. A named TRAIN subset is diagnostic
evidence only. Its outcome cannot decide admission.

The V2 held-out commitment publishes a hash and trial count, with the fresh
scene and seed manifest stored privately and unopened for recovery evaluation.

ADR-0085 follow-up: the first V2 TRAIN diagnostic (three frozen seed-0 scenes)
passed 44 actions per scene, then the same original-pose-solver translation
residual exceeded 5 mm. A measured rejected target had a bounded feasibility
witness at 1.54 mm using residuals divided by the existing per-quantity limits.
The opt-in V2 IK fallback uses that normalization from the observed joint seed
only after a residual miss. It retains the existing trust region and motor
checks. This is an execution correction, with the first diagnostic evidence
preserved and the comparison repeated in a new directory.

### ADR-0086 — Prepared Eq. 3.2 conditioning interface, 2026-11-25

A minimal learned decision head is added as
`src/vlarobot/planning/prediction_conditioned.py`. It consumes explicit learned
observation features, a language task identifier, candidate action payloads and
a numeric context vector extracted from each candidate's predicted state
sequence. Seeded softmax weights are fitted on recorded TRAIN decisions; only
finite, valid predictions are accepted. Its rationale is numeric. No ground
truth, expert phase or scripted decision reaches the head, and no core public
interface is changed.

This is preparation, not activation: the head is not wired into any scored
campaign while learned TRAIN admission is pending. External Eq. 3.8 ranking
remains the evaluated V1 mechanism and is not claimed as prediction
conditioning. Controlled TRAIN evidence must show that changing only predicted
context, with other inputs fixed, changes the learned selection before any
Eq. 3.2 claim is made. Activation after admission is a separate decision.

`tools/v2_diagnostic_metrics.py` reproduces the recorded three-scene diagnostic
fields from raw episode artifacts with the same calculation used for the
endpoint-decoder baseline. It is a read-only measurement tool; it supplies no
policy input or admission criterion.

### Owner time-box for VLA V2 recovery — 2026-11-25

The owner authorized V2 recovery as a time-boxed side-track under these
controlling rules, which supersede the broader recovery schedule where they
differ:

1. V1 results (VLA-only TRAIN 0/24 and the frozen TEST campaign) are not
   modified. V2 is reported separately with its own receipts.
2. The admission criterion is unchanged: at least 10 successes out of the 24
   original TRAIN identities. No gate or threshold is lowered.
3. The fresh V2 holdout stays sealed until a checkpoint passes TRAIN admission.
   No held-out identity, plan or outcome is accessed before that gate.
4. The three-scene diagnostic is due by 2026-11-27. If it warrants admission,
   run the 24-trial admission by 2026-12-01. Otherwise stop the side-track.
5. Recovery evidence is an autonomous close, lift and transport verified by the
   independent task verifier. Model-query counts and accepted tick counts are
   not recovery evidence.
6. If the three-scene diagnostic is 0/3 with only open-endpoint gripper commands,
   stop and write the V2 negative result. No further decoder or safety changes.
7. The frozen-arm numbers in `docs/RESULTS.md` are not edited.

Daily status reports are five lines while this side-track is active.

### V2 recovered step-021000 diagnostic negative result — 2026-11-26

The recovered export `step-021000` was accepted by the production loader with
manifest SHA-256 `0d5059dc5051b12a0d19614cc93f4b6a8ec77887f7e1e97b6e830858399504d6`
and 1000 reloaded optimizer updates. The unchanged three-scene TRAIN diagnostic
returned `0/3`; all accepted gripper endpoint actions target `0.045` m open,
with no close command, no observed holding, micrometre-scale CAN motion and
`success=false`/`home_reached=false` from the independent verifier. Execution
profile, diagnostic indices, flags and all five controller/adapter hashes match
the step-020000 baseline; only the model manifest changed.

Decision: rule 6 above applies, so the V2 side-track stops here. The 24-trial
admission campaign is not run, the `>=10/24` threshold is not changed, the V2
holdout stays sealed, V1 results and frozen `docs/RESULTS.md` numbers are not
touched, and no decoder, IK or safety change is made. Evidence:
`docs/V2_NEGATIVE_RESULT.md`, `docs/evidence/v2-recovered-diagnostic-result.json`
and `docs/evidence/v2-recovered-step21000-receipt.json`.

### ADR-0087 — V3 separate arm and in-sample admission, 2026-11-28

The owner addendum of 2026-11-28 makes V3 a separate arm. Frozen V1/V2 results
are not modified. TRAIN admission is explicitly in-sample over the 8 base scene
identities; before any pass is reported as a working VLA, V3 must run an
object-position intervention (move the CAN with other inputs fixed apart from the
image change and require correctly directed waypoints) and a
leave-one-base-scene-out memorization check. If the intervention fails, the
result is reported as “memorized TRAIN scenes”.

Up to 80 additional TRAIN demonstrations are prioritized for randomized CAN
position/yaw inside the reachable envelope, starting in parallel with model
code. Language-varying demos are secondary; language conditioning remains
UNMEASURED without instruction-dependent destinations. Option B must avoid
privileged object pose in the prediction context where possible; any
oracle-derived field is labelled in every table and tested with zeroed/shuffled
prediction ablations.

V3 experiments stop on 2026-12-05 and report daily in five lines. Any V3
evaluation on VALIDATION/TEST requires explicit owner approval after the V3
configuration is frozen by commit. Evidence and plan:
`docs/VLA_ALTERNATIVES_ASSESSMENT.md`, `docs/V3_DAILY_REPORT.md` and
`docs/evidence/v3-demo-plan-80.json`.

## V4 skill experiment — oracle-assisted TRAIN milestone (2026-11-28)

The owner approved a separate skill-selection arm starting at V3 recovery commit
`33962ad`. Each dispatch runs one Cartesian or gripper skill with the existing
IK and settling logic and an independent command monitor. The utility prototype
uses NumPy ridge regression on candidate features and optionally the final
analytic state. Context interventions change the score inputs while the
original surrogate vetoes remain authoritative. Runtime privileged inputs
include the oracle CAN pose for world-state construction and CAN-rise logging.
Demonstration and simulator-observed utility labels are offline sources.
Synthetic fixture labels are used in unit tests.

The original TRAIN export contains 80 episodes and 63,081 frames. All 85
files in its integrity manifest matched SHA-256. The additional TRAIN collector
logged 99 completed attempts, with 81 successful and 18 quarantined. The
collector caps the sorted accepted manifest at its requested 80 episodes.
Attempt 98 (`train-reach-limited-i02-s212`) is the successful surplus. All 80
manifest entries had unique attempt IDs, successful independent audit flags
for grasp, lift, transport and placement, nonempty trace files, and readable
step arrays. This verifies the inputs at the collection level. V4 decision
extraction and training eligibility still require checks.

V4 honors `unverified-arm-sweep` from the existing analytic surrogate. The
shipped `configs/worldmind_arm_sweep.yaml` is a blocked template. A reviewed
transition checker must be bound before any live learned dispatch can pass
this hard check. The isolated expert-primitive diagnostic has its own
explicit command monitor and does not exercise learned selection.

The isolated Linux Webots probes both settled on `train-clear-i00-s0`.
OPEN dispatched one gripper command and reached 0.04310 m. PREGRASP
dispatched ten joint commands and kept the gripper closed. The first
PREGRASP attempt was vetoed before dispatch because the prototype omitted
velocity limits. The executor now uses the existing effective runtime
safety limits for joint positions, velocities, workspace and gripper.

The V4 extractor processed both accepted manifests and produced 1,440
decision boundaries from 160 distinct TRAIN episodes across eight scene
groups. It records the decision-time snapshot separately from simulator
measurements after the expert action. This export contains demonstrated
actions. Counterfactual candidate outcomes require separate collection.
