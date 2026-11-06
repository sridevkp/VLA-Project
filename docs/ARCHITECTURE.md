# Risk-Aware World-Model VLA Robotics — Validated Architecture & Build Plan

> Audience: human developer + coding agents (VS Code). Target: **Windows 10/11 and Linux (Ubuntu 22.04/24.04)**.
> Inputs reviewed: *Project Development Blueprint* (the plan) and *Current State Overview* (the repo).
> Companion file: `AGENTS.md` (copy to repo root; rules every agent must follow).

---

## 0. Verdict

The vision and research questions are sound and the module decomposition is good. The plan **cannot be built as written** on both OSes, and several choices conflict with what your repo already is. Main problems:

1. The plan is ROS 2-centric (ROS 2 + MoveIt 2 + Nav2 + RViz2 + OMPL). Your repo is a **direct-Webots Python controller using IKPy**, no ROS. ROS 2 with MoveIt 2/Nav2 is not a practical native-Windows stack.
2. It assumes SmolVLA can drive TIAGo. It was pretrained on single-arm tabletop data (SO-100/SO-101-style, ~6-DoF). **It will not work zero-shot on TIAGo.** It has to be fine-tuned on demonstrations you generate.
3. The world model needs to "simulate candidate actions", but **Webots cannot cheaply branch/rewind** to evaluate dozens of hypothetical futures. A lightweight surrogate simulator is required.
4. Evaluation is placed last (Phase 14). The research claims (ablations) depend on it, so it must come early.
5. Scope/timeline is optimistic and everything is flat-priority. It needs P0/P1/P2 tiers and acceptance tests.

All changes are listed in §2 with reasons. Everything not listed there is **kept as in your plan**.

---

## 1. Design Principles (derived from the cross-platform + agentic constraints)

1. **Core is pure Python with no simulator or ROS imports.** Only `backends/` touches Webots/ROS. This makes ~90% of the code testable on Windows and Linux CI without Webots.
2. **Process separation over in-process coupling.** Three processes talk over ZeroMQ + msgpack (works identically on Windows/Linux, and Windows↔WSL2):
   - **Sim process** — Webots + thin sim server (device I/O, stepping, ground truth).
   - **Brain process** — perception glue, language, world model, risk, planner, executor (CPU).
   - **Model server** — GPU inference (VLA, detector, segmenter). Local or remote; same protocol.
3. **Deterministic lockstep.** The brain advances the sim explicitly (`step(n)`); no wall-clock races. Same seed ⇒ same episode. Required for ablations.
4. **Interfaces first.** Every module implements a `Protocol` in `core/`; implementations are swappable (oracle vs real perception, surrogate vs learned dynamics, scripted vs VLA proposer).
5. **Ground truth is an oracle for metrics and debugging only** (after M3), never an input to planning in the "real" pipeline. Use a config flag `perception.mode = oracle | real`.
6. **Safety layer is independent and has veto power** over any proposer (VLA, scripted, human).
7. **Incremental delivery:** a runnable end-to-end pipeline at every milestone; no big-bang integration.

---

## 2. Corrections to the Original Plan (what / why)

| ID | Original plan | Correction | Why |
|---|---|---|---|
| C1 | ROS 2 is the middleware for all modules (Phase 2) | **ROS 2 becomes an optional adapter** (`backends/ros2/`, Linux/WSL2/Docker only). Primary path = ZMQ lockstep sim server + Python `RobotBackend` interface. | Existing code is not ROS-based. ROS 2 on native Windows is workable for basics but MoveIt 2 / Nav2 are not realistically supported there. ROS also adds sim-time/async complexity that hurts determinism. Your own "Robot Abstraction Layer" (§5.4) already enables this. |
| C2 | MoveIt 2 + OMPL for manipulation; Nav2 for navigation | **Keep existing IKPy stack** (`kinematics.py`, `motion.py`) + add own collision-aware planner (RRT-Connect / waypoint sampling) using the surrogate sim for collision checks. **Defer Nav2** (P2). Base positioning = simple pose-goal controller to pre-computed standoff poses. | Both are Linux-centric. The demo is tabletop manipulation: Nav2 needs maps/lidar/costmap tuning for little gain. Existing IK already works. MoveIt 2 stays as an optional Linux adapter (P2). |
| C3 | VLA (SmolVLA) generates actions directly on TIAGo | **VLA is a candidate-action proposer, fine-tuned on sim demos** produced by your scripted `PickPlace` state machine (it is your expert demonstrator). Zero-shot is only used as a smoke test (as `model/main.py` does now). | Embodiment/action-space mismatch (6-DoF single arm vs TIAGo torso+7-DoF arm+gripper). Fine-tuning is mandatory. The scripted expert makes data generation cheap and automatic. |
| C4 | World model "predicts S(t+1)=f(S(t),A(t))" using the physics simulation | **Two-tier dynamics:** (a) fast **surrogate simulator (MuJoCo)** for kinematic feasibility, collisions, coarse stability, batch rollouts; (b) **learned residual model** (object-centric MLP/GNN) trained on logged Webots outcomes to correct grasp/placement outcome predictions. | Webots is the "truth" but cannot cheaply fork/rewind for tens-hundreds of rollouts per decision. Surrogate+residual is also a defensible research contribution (sim-vs-sim gap correction). Video/latent world models stay P2. |
| C5 | YOLO closed-set detector; SAM 2 optional | **Open-vocabulary detector first** (OWLv2 or Grounding DINO, or YOLO-World) → SAM 2 for masks; optional fine-tuned small detector on Webots synthetic data later. Colour-threshold baseline for "red bottle" in M3 as a fast sanity path. | "Red bottle", "blue box" are attribute-grounded queries. Also: Ultralytics YOLO is **AGPL-3.0** — a licensing decision to make consciously. |
| C6 | Risk score R(a)=Σ wᵢ·riskᵢ | **Two-stage:** (1) *hard-constraint veto* (joint limits, collision with margin, unreachable, velocity limits); (2) *expected-utility ranking* using calibrated probabilities (success, collision) and separate perception/prediction/control uncertainties. Weights tuned on validation scenarios, not guessed. | A single weighted sum lets a "low average risk" hide a catastrophic constraint violation. Calibration is also one of your stated metrics. |
| C7 | Main loop is a single synchronous `while` | **Two-rate loop:** decision loop (1–5 Hz: perceive → predict → risk → select) + control loop (sim-step rate: execute/validate chunk). Lockstep with the sim. | VLA/detector latency ≫ control period. Needed for action chunking and watchdogs. |
| C8 | IKFast prep assets listed as part of the pipeline | **Park IKFast** (P2). Keep IKPy; profile first. | IKFast requires OpenRAVE-based codegen (painful on Windows) and the 7-DoF TIAGo arm needs a fixed "free joint" anyway. Only worth it if IKPy proves too slow. |
| C9 | Evaluation/benchmarking is Phase 14 | **Evaluation harness is M2** (right after porting the controller): seeded scenario generator, episode recorder, metrics. | Every later claim (ablations 1–6) needs it. Also doubles as the demonstration-data pipeline. |
| C10 | RViz2 for visualisation | **Rerun (rerun.io) or Open3D viewer + Streamlit** for dashboard; keep existing Webots RobotWindow UI. RViz2 only in the optional ROS adapter. | RViz2 is effectively Linux-only; Rerun is pip-installable on both OSes. |
| C11 | Timeline 5–8 months, flat scope | **P0/P1/P2 tiers + milestones with acceptance tests** (§5). P0 alone satisfies research questions 1–3. | Prevents the project from stalling on research extensions. |
| C12 | Supervisor ground-truth `CAN` as the target source | Keep it as an **oracle flag**, then replace with perception at M3. Rename concept: `OracleStateProvider`. | Matches principle 5; lets you measure perception error vs ground truth. |
| C13 | Phase ordering: VLA (Phase 5) before world model/risk | **Scripted-proposer + world model + risk comes BEFORE VLA** (M5–M7), VLA at M8. VLA-only baseline is still built for the ablation. | The structured pipeline works with no GPU/training, generates the demos the VLA needs, and de-risks the project. |
| C14 | Tooling unspecified | `uv` + `pyproject.toml` (src layout), `ruff`, `pytest`, `pre-commit`, `typer` CLI, GitHub Actions matrix `ubuntu-latest` + `windows-latest`. No bash/PowerShell scripts in core. | Single cross-platform workflow that agents can run identically. |

---

## 3. Target Architecture

### 3.1 Process topology

```text
┌──────────────────────────┐   ZMQ/msgpack (lockstep)   ┌────────────────────────────────────┐
│ SIM PROCESS              │ <────────────────────────> │ BRAIN PROCESS (CPU, Python)        │
│ Webots + SimServer       │  Observation, RobotState,  │ perception glue · language · world │
│ (existing controller     │  GroundTruth(oracle),      │ model · risk · planner · executor  │
│  code wrapped)           │  step(n), command(...)    │ memory · eval logging · dashboard  │
└──────────────────────────┘                            └───────────────┬────────────────────┘
                                                                        │ ZMQ/gRPC (async, request/response)
                                                        ┌───────────────▼────────────────────┐
                                                        │ MODEL SERVER (GPU, local or cloud) │
                                                        │ detector · SAM2 · VLA · LLM parser │
                                                        └────────────────────────────────────┘
```

Optional: `backends/ros2/` adapter exposes the same `RobotBackend` over ROS 2 topics/actions (Linux/WSL2/Docker).
Optional: `backends/mujoco/` is also a full `RobotBackend` used for fast CI tests and as the surrogate simulator.

### 3.2 Decision pipeline (per decision tick)

```text
instruction ──> TaskSpec (language) ─────────────────────────────┐
obs ──> Perception ──> SceneState ──> WorldModel.update ─────────┤
                                         │                        v
                                         ├─> CandidateGenerator (scripted variants + VLA proposals + memory)
                                         │            │
                                         │            v
                                         ├─> Predictor.rollout(candidate) x N  (surrogate + learned residual)
                                         │            │
                                         │            v
                                         └─> RiskEngine: veto -> utility ranking
                                                      │
                                                      v
                                    Planner picks best -> MotionPlanner -> Executor (control loop)
                                                      │
                                                      v
                              Verifier compares predicted vs observed -> WorldModel.correct / Recovery / Replan
```

### 3.3 Core interfaces (put in `src/vlarobot/core/`)

Agents must implement these first (M1) and never change signatures without updating this doc.

```python
# core/types.py  (pydantic or dataclasses; all numpy arrays in float32/float64 SI units)
@dataclass
class Observation:
    t: float                      # sim time (s)
    rgb: np.ndarray               # HxWx3 uint8
    depth: np.ndarray | None      # HxW float32 meters
    cam_info: CameraInfo          # K, extrinsics (camera->base), width/height
    robot: RobotState

@dataclass
class RobotState:
    joint_pos: dict[str, float]; joint_vel: dict[str, float]
    base_pose: Pose2D; ee_pose: Pose3D; gripper_opening: float
    gripper_holding: bool | None  # heuristic, may be None

# Action = discriminated union; every action is in the BASE frame, SI units
class Action: ...
class EEPoseTarget(Action): pose: Pose3D; gripper: float | None; duration: float
class EEDelta(Action):      dpos: Vec3; drot: Vec3; gripper: float
class JointTarget(Action):  q: dict[str, float]
class BaseGoal(Action):     pose: Pose2D
class GripperCmd(Action):   opening: float; force: float | None
class ActionChunk: actions: list[Action]; dt: float; source: str  # "scripted"|"vla"|"memory"|"human"

# core/interfaces.py
class RobotBackend(Protocol):
    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation: ...
    def step(self, n_steps: int = 1) -> Observation: ...
    def apply(self, action: Action) -> ApplyResult: ...      # validated command, non-blocking
    def get_ground_truth(self) -> GroundTruth: ...           # oracle; metrics only
    def stop(self) -> None: ...

class Detector(Protocol):   def detect(self, rgb, queries: list[str]) -> list[Detection]: ...
class Segmenter(Protocol):  def segment(self, rgb, boxes) -> list[Mask]: ...
class Localizer(Protocol):  def localize(self, dets, masks, depth, cam_info) -> list[ObjectState]: ...
class TaskParser(Protocol): def parse(self, text: str, scene: SceneState) -> TaskSpec | Clarification: ...
class WorldModel(Protocol):
    def update(self, scene: SceneState, robot: RobotState) -> None
    def snapshot(self) -> WorldState
    def correct(self, predicted: WorldState, observed: WorldState) -> PredictionError
class Predictor(Protocol):  def rollout(self, state: WorldState, chunk: ActionChunk, n_samples: int) -> list[Outcome]: ...
class RiskEngine(Protocol): def assess(self, chunk, outcomes, state) -> RiskReport   # veto flags + calibrated probs + uncertainties
class CandidateGenerator(Protocol): def propose(self, task, state, memory) -> list[ActionChunk]: ...
class Planner(Protocol):    def select(self, cands, outcomes, risks, task) -> Decision  # includes human-readable rationale built from real numbers
class Verifier(Protocol):   def check(self, task, predicted, observed) -> VerifyResult
```

### 3.4 Wire protocol (sim ↔ brain)

- ZeroMQ `REQ/REP` (lockstep) with `msgpack`; images as raw bytes with shape/dtype header (or JPEG for dashboard streaming only).
- Messages: `reset`, `step`, `apply`, `get_gt`, `shutdown`, `ping`; every reply carries `sim_time`, `seq`, and `error?`.
- Versioned: include `protocol_version`; mismatch ⇒ hard fail.
- Default `tcp://127.0.0.1:5555`; configurable so Windows-Webots ↔ WSL2-brain works.

### 3.5 World model contents (staged)

| Stage | Content | Priority |
|---|---|---|
| W0 | Object-centric state from oracle: poses, dims, support/contact relations, scene graph (`networkx`) | P0 |
| W1 | Surrogate rollouts in MuJoCo: reachability, collision (with margin), swept-volume checks, placement stability | P0 |
| W2 | Learned residual (small MLP/GNN) predicting grasp success, slip, placement error from features, trained on Webots logs; ensemble for uncertainty | P1 |
| W3 | Latent / video / Dreamer-style models | P2 (research extension only) |

Prediction-vs-observation error is logged at every step from W1 on; this is your headline metric.

### 3.6 Action space decision (for planner + VLA)

Use **end-effector pose deltas in the base frame + gripper (7-D)** as the canonical VLA/learned action, executed through the existing IKPy `move_linear` with tolerance checks. Reasons: transferable across embodiments, keeps the safety layer simple (check Cartesian + joint limits after IK), reuses current code. Torso and head are scripted; base is handled by `BaseGoal`. Joint-space VLA output is a P2 experiment.

---

## 4. Cross-Platform Plan

### 4.1 What runs where

| Component | Windows native | Linux / WSL2 |
|---|---|---|
| Webots + sim server (extern controller) | ✅ | ✅ |
| Brain process (core, planner, risk, memory, dashboard) | ✅ | ✅ |
| MuJoCo surrogate, IKPy, Open3D, OpenCV | ✅ | ✅ |
| Detector / SAM 2 inference (PyTorch CUDA) | ✅ | ✅ |
| LeRobot **training/fine-tuning** of SmolVLA | ⚠️ possible, video-decoding/ffmpeg issues likely | ✅ recommended (use WSL2 on Windows) |
| ROS 2 adapter, MoveIt 2, Nav2 | ❌ not supported as a goal | ✅ (Docker/WSL2) |
| IKFast codegen | ❌ | ⚠️ only if needed |

Rule: **nothing in P0 may require ROS or Linux-only tooling.** Training can be Linux/WSL2/cloud — it only has to produce a model the model server loads.

### 4.2 Webots on both OSes

- Use **extern controllers**: world file sets the robot's `controller "<extern>"`; run the sim server from your venv (VS Code debugger works). This avoids Webots' per-OS Python-command preferences and lets you pin the interpreter.
- Requires `WEBOTS_HOME` and `PYTHONPATH=$WEBOTS_HOME/lib/controller/python` (the repo's `webots_env` helper sets these per-OS; implement in `scripts/`/`vlarobot.env`).
- Windows host + WSL2 brain: set `WEBOTS_CONTROLLER_URL=tcp://<windows-host-ip>:<port>/<robot_name>` or just use the ZMQ link between processes (preferred).
- Keep a thin `controllers/tiago_controller/tiago_controller.py` shim (Webots requires the file name to match the folder) that imports `vlarobot.backends.webots.server`.
- Pin one Webots release for the whole project and record it in `docs/VERSIONS.md` (verify the TIAGo PROTO + Python version compatibility at M0).

### 4.3 Environment & tooling

- **Python 3.11** (verify against the pinned Webots + torch + lerobot at M0; change only in `pyproject.toml`).
- `uv` for env/lockfile; extras: `core`, `sim`, `perception`, `vla`, `dashboard`, `dev`. PyTorch CUDA wheels installed via extra index per-OS (documented command in `docs/SETUP.md`).
- `pathlib` everywhere; no hard-coded `/` or `\`; no symlinks; keep paths short (Windows MAX_PATH).
- `.gitattributes`: `* text=auto eol=lf` (and `*.bat eol=crlf` if any). `.editorconfig` committed.
- Multiprocessing: assume Windows **spawn** semantics (guard with `if __name__ == "__main__"`, picklable targets, no fork-only tricks). Prefer separate processes over threads+shared memory.
- No `bash`-only scripts. Tasks are `typer` CLI commands: `python -m vlarobot.cli sim-server | brain | eval | collect-demos | dashboard`.
- Logging with `structlog`/`logging` to files under `runs/<timestamp>_<seed>/`; configs via `pydantic-settings` + YAML in `configs/`.
- CI: GitHub Actions matrix (ubuntu, windows): lint + type-check + unit tests + MuJoCo/fake-backend integration test. Webots is **not** required in CI.
- Optional `.devcontainer/` (Ubuntu 24.04 + CUDA + Webots headless/xvfb) for Linux reproducibility; the project must still work without it.

### 4.4 Cross-platform pitfalls agents must avoid (checklist)

- File I/O: always `encoding="utf-8"`; open images/binaries explicitly in binary mode.
- Case-insensitive FS on Windows: no two files differing only by case.
- Signals: no reliance on `SIGKILL`/`SIGUSR*`; shutdown via ZMQ `shutdown` message + `atexit`.
- Subprocess: pass arg lists (no `shell=True`); use `sys.executable`.
- asyncio: don't assume `uvloop`; Windows event-loop policy must work (`pyzmq` with asyncio is OK on `SelectorEventLoop`).
- Timing: use `time.perf_counter()`, never rely on sleep precision < ~15 ms on Windows.
- Don't depend on `torchcodec`/system `ffmpeg` in core; only in the training extras.

---

## 5. Milestones (build order)

Priority: **P0** = required for research questions 1–3 · **P1** = strongly desirable · **P2** = optional extension.
Each milestone has an **acceptance test** an agent can run; a milestone is not done until it passes on Windows and Linux.

### M0 — Foundations (P0, ~1 week)
- Repo restructure to `src/vlarobot/` layout (§6), `pyproject.toml`, `uv.lock`, ruff, pytest, pre-commit, CI matrix, `.gitattributes`, `docs/SETUP.md`, `docs/VERSIONS.md`.
- "Hello Webots" extern-controller test: connect, read a camera frame, step 10 times, exit — on both OSes.
- **Acceptance:** `python -m vlarobot.cli smoke` prints sim time and image shape on Windows and Linux; CI green on both.

### M1 — Port controller behind interfaces (P0, ~1–2 weeks)
- Move `controllers/tiago_controller/*` into `backends/webots/` unchanged in behaviour; build `SimServer` + `ZmqBackendClient` implementing `RobotBackend`; add `FakeBackend` (no sim) for tests.
- Convert `config.py` constants to YAML-backed typed config (keep defaults identical).
- **Acceptance:** regression test — scripted pick-and-place of `CAN` via the brain process succeeds at the same rate as the original controller over ≥10 seeded runs; `FakeBackend` unit tests pass in CI.

### M2 — Evaluation & data harness (P0, ~1 week)
- `ScenarioSpec` + seeded scenario generator (object poses/colours/types, obstacles; Webots Supervisor spawns/moves nodes).
- `EpisodeRecorder` writing (obs, state, action, outcome, predicted, ground truth) to a LeRobot-compatible-convertible format (parquet/npz + mp4 or JPEG frames).
- Metrics module: success, grasp success, collisions (contact events from Supervisor), time, placement error, + CSV/JSON reports; mean/std/CI over N seeds.
- **Acceptance:** `eval --pipeline scripted --n 20 --seed 0` produces reproducible report (same seed ⇒ identical metrics).

### M3 — Perception (P0, ~2–3 weeks)
- `perception.mode=real`: RGB-D → detector (open-vocab) → mask → 3D localization → tracked `ObjectState` list; colour-threshold fallback.
- Calibration check against Supervisor ground truth (extrinsics, depth scale).
- **Acceptance:** localisation error ≤ 1.5 cm median on tabletop objects over 50 randomised scenes; scripted pipeline succeeds with `perception.mode=real` at ≥ 80% of the oracle rate.

### M4 — Language & task spec (P0, ~1 week)
- `TaskParser`: LLM (API or local) returning schema-validated JSON `TaskSpec` (targets, destination, constraints such as "avoid the cup"); rule-based fallback for the demo grammar; ambiguity ⇒ `Clarification`.
- **Acceptance:** 30-sentence test set parsed ≥ 95% correct; ambiguous "pick up the bottle" with 2 bottles returns a clarification.

### M5 — World state, scene graph, surrogate sim (P0, ~3 weeks)
- `WorldState`, scene graph relations (on, inside, next-to, blocking, reachable-from), MJCF/URDF import of TIAGo + objects into MuJoCo, rollout API, collision/reachability/stability checks.
- Prediction-error logging (predicted vs Webots outcome).
- **Acceptance:** surrogate reachability/collision agrees with Webots outcome ≥ 90% on 200 logged candidate actions; scene-graph unit tests; rollout of 32 candidates < 1 s on a laptop CPU.

### M6 — Candidates, risk engine, predictive planner (P0, ~3 weeks)
- Scripted `CandidateGenerator`: grasp approach directions, pregrasp offsets, base standoff poses, placement poses.
- `RiskEngine` (veto + utility), calibration plots, `Planner.select` with rationale generated from actual numbers.
- Receding-horizon execution: execute first segment, re-observe, replan.
- **Acceptance:** on the cluttered scenario set, collisions with the risk engine < without it (statistically, n ≥ 50 each) with task success not worse by > 10 pts.

### M7 — Closed-loop verification & recovery (P0, ~2 weeks)
- Grasp/placement verifiers, failure taxonomy, `RecoveryManager` (regrasp, alternate approach, reposition base, abort, request help).
- **Acceptance:** with injected failures (object nudged, grasp slip), recovery success ≥ 70% and no unsafe command passes the safety layer.

### M8 — VLA integration (P1, ~3–4 weeks)
1. Inspect `lerobot/smolvla_base` config (`input_features`, `output_features`, state/action dims, camera keys) — record in `docs/VLA_NOTES.md`.
2. `collect-demos`: run the scripted expert over randomised scenes → LeRobot dataset (EE-delta + gripper, 1–3 cameras, state, instruction text). Use the automatic generator for thousands of episodes; keep the ≥ docs-recommended minimum per task variant (verify in LeRobot docs).
3. Fine-tune SmolVLA (Linux/WSL2/cloud GPU) → checkpoint → `ModelServer`.
4. `ActionAdapter`: normalisation, EE→IK, joint/velocity limits, chunk execution at control rate, invalid-action rejection.
5. Use VLA as (a) **VLA-only baseline** and (b) **candidate proposer** inside the risk-aware planner.
- **Acceptance:** VLA-only reaches ≥ X% (set after first run) on seen scenes; adapter rejects 100% of injected out-of-limit actions; proposer mode runs end-to-end.

### M9 — Learned residual dynamics, memory, dynamic/uncertain envs (P1, ~4 weeks)
- W2 learned residual (ensemble) on Webots logs; calibration; compare W1 vs W1+W2 prediction error.
- Scene memory (confidence + expiry) and episodic memory used by `CandidateGenerator`.
- Dynamic scenarios: moving obstacle, nudged object, occlusion, sensor dropout.
- **Acceptance:** W2 reduces grasp-outcome prediction error vs W1 on held-out scenes; ablation on memory shows a measurable change (positive or negative — report honestly).

### M10 — Dashboard, full ablations, report (P0/P1, ~2–3 weeks)
- Dashboard (Streamlit + Rerun): camera/masks, scene graph, candidates, risks, timeline, metrics.
- Run the six ablations from the blueprint (§17.6) with ≥ 30 seeds each, CI/std reported; failure-category breakdown.
- **Acceptance:** `eval --suite ablation` regenerates all tables/figures from raw logs.

### P2 backlog (only after M10)
ROS 2 adapter · MoveIt 2/Nav2 adapters · IKFast · latent/video world models · RL fine-tuning · sim-to-real prep · multi-robot · CBF/chance-constrained planning · active perception.

> Timeline sanity check: P0 ≈ 3.5–4.5 months for one developer using agents; P1 adds ~2 months. The original 5–8 month estimate is plausible only if P2 is excluded.

---

## 6. Repository Layout (target)

```text
vlarobot/
├── AGENTS.md
├── pyproject.toml            # uv-managed, extras: core,sim,perception,vla,dashboard,dev
├── uv.lock
├── configs/                  # YAML: robot, perception, planner, risk, eval, scenarios
├── docs/                     # ARCHITECTURE_AND_PLAN.md, SETUP.md, VERSIONS.md, VLA_NOTES.md, DECISIONS.md (ADRs)
├── worlds/                   # factory.wbt (+ tabletop worlds), extern controller set
├── controllers/tiago_controller/tiago_controller.py   # thin shim -> vlarobot.backends.webots.server
├── src/vlarobot/
│   ├── core/                 # types, interfaces (Protocols), config, geometry, logging, wire protocol
│   ├── backends/
│   │   ├── webots/           # ported existing code: robot_io, kinematics, motion, gripper, frames, pick_place, server
│   │   ├── mujoco/           # surrogate sim + fast backend for CI
│   │   ├── fake/             # no-sim backend for unit tests
│   │   └── ros2/             # OPTIONAL (Linux/WSL2/Docker)
│   ├── perception/           # detector, segmenter, depth, localizer, tracker
│   ├── language/             # task_parser, decomposer, schemas
│   ├── world_model/          # state, scene_graph, surrogate predictor, learned residual, uncertainty
│   ├── risk/                 # constraints (veto), collision, grasp, uncertainty, calibration, scoring
│   ├── planning/             # candidate_generator, predictive_planner, motion_planner, recovery
│   ├── vla/                  # model_client, server, action_adapter, dataset_export, policy_manager
│   ├── execution/            # executor (two-rate loop), verifier, safety_monitor
│   ├── memory/               # scene_memory, episodic_memory
│   ├── eval/                 # scenarios, recorder, metrics, experiments, reports
│   ├── dashboard/            # streamlit app, rerun logging
│   └── cli.py                # typer entry point
├── tests/                    # unit/ (pure), integration/ (fake+mujoco), webots/ (marked, local-only)
├── scripts/                  # python-only helpers
└── runs/                     # gitignored experiment outputs
```

`model/main.py` (SmolVLA smoke test) moves to `src/vlarobot/vla/smoke.py` and is exposed as `cli vla-smoke`. `tiago_ikfast/` stays as a parked asset folder.

---

## 7. Risk Engine Specification (replaces §11.5–11.6 formula)

Per candidate `a`:

1. **Veto (hard constraints), any true ⇒ reject:** joint-limit violation, IK tolerance exceeded, predicted min-clearance < margin, unreachable, velocity/acceleration limit, target outside workspace box, perception confidence below floor for a safety-critical object.
2. **Calibrated probabilities:** `p_success`, `p_collision`, `p_drop` from W1/W2 (ensemble mean).
3. **Uncertainties, kept separate:** `u_perception` (localisation covariance), `u_prediction` (ensemble variance), `u_control` (tracking error history).
4. **Utility:** `U(a) = p_success·V_task − λc·p_collision − λd·p_drop − λt·time − λu·max(u_*)` with λ tuned on validation scenarios; report calibration curves (reliability diagram, ECE).
5. **Decision:** choose max-U among non-vetoed; if none, trigger recovery ladder (reposition → re-observe → ask human → abort).
6. **Rationale string** is built from the numbers above — never free-generated.

---

## 8. Evaluation Design (guards the research claims)

- Seeds fixed per scenario; same seed list across all pipelines. ≥ 30 seeds/condition (adjust from variance).
- Pipelines: `rule_based` · `vla_only` · `vla+scene` · `vla+world_model` · `vla+world_model+risk` · `full(+feedback/replan)`. (`scripted` is the M1 baseline; include it.)
- Report: mean, std, 95% CI (bootstrap), paired tests across identical seeds, failure categories, wall-time and compute cost.
- Perception evaluated against oracle; prediction evaluated against Webots outcome; both split into train/held-out scenes to avoid leakage.
- Raw logs are the source of truth; all tables/figures regenerate from them.

---

## 9. Open Items the Agents Must Verify at M0 (could not be confirmed from the documents)

1. Latest Webots release that supports your TIAGo PROTO and the chosen Python version on **both** OSes → write to `docs/VERSIONS.md`.
2. Compatible set of `torch` / `lerobot` / `transformers` versions and whether SmolVLA training works natively on Windows (default assumption: use WSL2/Linux).
3. SmolVLA base config: state/action dimensionality, required camera keys, normalisation — from `policy.config` (the existing `model/main.py` already prints these).
4. MuJoCo URDF import of `tiago.urdf` (mesh paths, collision geometry); fallback = hand-authored MJCF with primitive collision shapes.
5. Licences of chosen models (detector, SAM 2, VLA, LLM) — record in `docs/DECISIONS.md`.
6. Webots contact/collision event access for the collision metric (Supervisor contact points vs custom proximity checks).

---

## 10. Agent Task Card Template (use for every unit of work)

```markdown
### TASK <Mx-yy>: <title>
Context: <link to section of this doc / files involved>
Goal: <one sentence>
Interfaces touched: <Protocols / dataclasses>
Files to create/modify: <paths>
Constraints: cross-platform rules (§4.4), no sim imports in core/, typed, tests required
Acceptance test: <exact command + expected result>
Out of scope: <explicit>
Definition of done: tests pass on Windows+Linux CI, docs updated, DECISIONS.md entry if a choice was made
```

### Suggested first agent prompts

**M0-01** — "Read `docs/ARCHITECTURE_AND_PLAN.md` §1, §4, §6 and `AGENTS.md`. Restructure the repo to the target layout without changing behaviour. Create `pyproject.toml` (uv, Python 3.11, extras per §4.3), ruff/pytest/pre-commit config, `.gitattributes`, `.editorconfig`, and a GitHub Actions matrix (ubuntu-latest, windows-latest) running lint + unit tests. Do not modify controller logic. Acceptance: `uv sync && uv run pytest` passes on both OSes."

**M0-02** — "Implement an extern-controller smoke test (`cli smoke`) per §4.2 that connects to Webots, reads one RGB frame, steps 10 times and prints sim time + image shape. Document OS-specific setup in `docs/SETUP.md` and record versions in `docs/VERSIONS.md`."

**M1-01** — "Implement `core/types.py`, `core/interfaces.py`, `core/wire.py` exactly as specified in §3.3–3.4 with round-trip serialization tests."

**M1-02** — "Move the existing controller modules to `backends/webots/` behind `RobotBackend` per §6; add `SimServer` (ZMQ REP, lockstep) and `ZmqBackendClient`; add `FakeBackend`. Regression: scripted pick-and-place over 10 seeds matches the pre-refactor success rate."

For M2+ create task cards from §5 using the template above; implement one card per agent session, keep PRs small, and require the acceptance command output in the PR description.