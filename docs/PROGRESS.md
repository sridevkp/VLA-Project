# Project progress

Current milestone: **M1 — Port controller behind interfaces**

Last updated: **2026-11-07**. M0 is committed; M1-01 through M1-05 are implemented. The M1-06
workspace conflict is addressed using measured original-controller waypoints and two safety
profiles; the brain-side seeded regression remains unverified.

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
- [x] Derive `configs/safety.yaml` workspace profiles from the original PickPlace waypoints with a
  0.10 m `original_envelope` margin and a tighter 0.02 m `strict` profile. The original controller
  had no workspace limit; the envelope is a behavioral baseline, not a validated safety limit.
- [x] Log validation rejection reasons and test waypoint acceptance under `original_envelope`,
  strict-profile containment, and out-of-box rejection under both profiles.
- [ ] Run the scripted CAN regression from the brain process and compare success with the original
  controller over at least ten seeded episodes.
  - **Actual result (Fedora/Webots R2025a):** `uv run python -m vlarobot.cli can-regression --seeds 10`
    produced `success=0/10` and `rejections=9`.
  - **Per-seed summary:** seed 0 final_distance=0.560m, seeds 1–9 terminal states were `INIT`,
    `place_descend`, or `pregrasp` with a final rejection. The run completed successfully, but it
    does not satisfy the M1-06 acceptance gate.
- **Local checks:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src
  tests`, `uv run pytest -m "not webots"`, and `uv build` passed; pytest reported **36 passed**.

- **Webots integration:** Fedora / Webots R2025a factory world, brain process:
  `uv run python -m vlarobot.cli brain --steps 10`

  ```text
  sim_time=0.416s image_shape=(480, 640, 3)
  ```

- **Webots RPC evidence:** ping, reset, apply, step, get_gt, and shutdown completed successfully;
  the returned ground-truth object key was `CAN`. This validates transport/backend operation, not
  the M1-06 pick-and-place regression.

- **M1-06 waypoint evidence:** an instrumented run of the original controller in
  `worlds/factory.wbt` collected home `(0.203672298, -0.985455309, 0.963048983)`, descend/grasp
  `(0.672736661, -0.213295065, 0.764821419)`, pregrasp/lift
  `(0.672736661, -0.213295065, 0.884821419)`, transit/retreat
  `(-1.077327982, -3.842045869, 0.936284755)` / `(-1.086885823, -3.858086446, 0.935119120)`,
  and place-descend/release `(-1.077327982, -3.842045869, 0.816284755)` /
  `(-1.086885823, -3.858086446, 0.815119120)`, all base-frame meters. These extrema include two
  runs; the latest original run terminated `FAILED`. See ADR-0002 for profile bounds and rationale.
- **M1-06 remaining:** The original behavior is observed through the waypoint flow, but the
  brain-process ≥10-seed success comparison did not meet the acceptance threshold. Webots reset
  replays the default scene, so repeated seeds are not varied scenarios. The latest original run
  terminated `FAILED`; the current brain-side scripted CAN policy is safety-compliant but still fails
  to reach the original controller's placement target across the ten-seed batch. No controller target
  was changed.
- **M1 gate status:** Blocked on M1-06 regression; Windows CI remains unverified.
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
