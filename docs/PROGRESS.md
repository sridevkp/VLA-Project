# Project progress

Current milestone: **M0 — Foundations**

Last updated: **2026-11-07**. Summary: M0 scaffold, documentation, environment helper, and Linux
Webots smoke test are implemented; no commit was created.

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
- **Local evidence:** `uv sync --locked --extra dev`, Ruff, ty, and non-Webots tests pass on Fedora
  with Python 3.11.18; `uv lock --check` passes and pytest reports 7 passed.
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
- **Gate status:** Pending the two Windows checks above. Stop here until the human reports their
  results and approves moving past M0.

## M1 — Port controller behind interfaces (P0)

- [ ] Move controller implementation into `src/vlarobot/backends/webots/` without behavior change.
- [ ] Keep `controllers/tiago_controller/tiago_controller.py` as the Webots-required thin shim.
- [ ] Implement core message types, protocols, and versioned wire serialization from §3.3–3.4.
- [ ] Add the Webots `SimServer` and lockstep `ZmqBackendClient`.
- [ ] Add `FakeBackend` and unit tests that do not require Webots.
- [ ] Move controller configuration to typed YAML-backed settings with identical defaults.
- [ ] Run the scripted CAN regression over at least ten seeded episodes.

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
