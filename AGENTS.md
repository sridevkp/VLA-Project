# AGENTS.md — Rules for coding agents (VLA Robotics Project)

Read `docs/ARCHITECTURE_AND_PLAN.md` before any task. It is the source of truth for architecture, interfaces, milestones and acceptance tests. If a task conflicts with it, stop and report the conflict instead of improvising.

## Project in one paragraph
Risk-aware, world-model-driven VLA system for a TIAGo robot in Webots. Pure-Python core (no sim/ROS imports), three processes (sim, brain, model server) over ZeroMQ+msgpack in deterministic lockstep. Must run on **Windows and Linux**.

## Hard rules
1. **No Webots/ROS/MuJoCo imports outside `src/vlarobot/backends/`.** Core, perception, planning, risk, world_model, eval import only interfaces from `core/`.
2. **Cross-platform always.** Use `pathlib`, `sys.executable`, subprocess arg lists (never `shell=True`), `encoding="utf-8"`, `time.perf_counter()`. No bash-only or PowerShell-only scripts; add a `typer` command in `cli.py` instead. Assume Windows `spawn` multiprocessing. No symlinks. Keep paths short.
3. **Never change a public interface** in `core/types.py`, `core/interfaces.py`, `core/wire.py` without updating the plan doc and adding an entry to `docs/DECISIONS.md`.
4. **Determinism.** Every stochastic component takes an explicit seed. No wall-clock dependence in decision logic; the brain advances the sim via `step(n)`.
5. **Ground truth is an oracle.** `get_ground_truth()` may be used only by eval/metrics, debugging, and `perception.mode=oracle`. Never in the "real" pipeline.
6. **Safety layer has veto power** over every action source. No code path may send an action to a backend without passing `safety_monitor.validate()`.
7. **Typed and tested.** Type hints on public functions; each module ships unit tests that run without Webots (use `FakeBackend`/MuJoCo backend). Webots-dependent tests are marked `@pytest.mark.webots` and are local-only.
8. **Explanations come from real numbers.** Planner rationales are built from risk/prediction outputs, never free text invented after the fact.
9. **Don't add heavy dependencies silently.** New dependency ⇒ add to the right `pyproject.toml` extra, note licence in `docs/DECISIONS.md`, confirm it installs on Windows and Linux.
10. **Don't touch controller behaviour during refactors** (M0–M1). Move and wrap; prove equivalence with the regression test.

## Workflow per task
1. Restate the task card (goal, files, acceptance test) in your first message.
2. Make the smallest change that satisfies the acceptance test.
3. Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not webots"`.
4. Run the task's acceptance command and paste the output in the PR/summary.
5. Update docs (`SETUP.md`, `VERSIONS.md`, `DECISIONS.md`) when you make or discover a decision.
6. If something cannot be verified in your environment (e.g. Webots not installed), say so explicitly and list what remains unverified. Do not claim success for untested paths.

## Commands (all cross-platform)
```text
uv sync --extra dev
uv run pytest -m "not webots"
uv run python -m vlarobot.cli smoke            # Webots extern-controller hello world
uv run python -m vlarobot.cli sim-server       # Webots side
uv run python -m vlarobot.cli brain            # decision process
uv run python -m vlarobot.cli eval --pipeline scripted --n 20 --seed 0
uv run python -m vlarobot.cli collect-demos --n 500 --seed 0
uv run python -m vlarobot.cli dashboard
```

## Code style
- Python 3.11, `ruff` (lint+format), 100-col lines, dataclasses/pydantic for messages, `Protocol` for interfaces.
- Units: SI (m, rad, s). Frames: poses in the **robot base frame** unless the name says `_world`. Name frames explicitly in variables (`T_base_cam`).
- Logging via the project logger; no bare `print` in library code.
- Config in YAML under `configs/`, loaded into typed settings; no magic numbers in logic (move tuning constants such as `TOOL_OFFSET`, tolerances, margins to config).

## Priority tiers
P0 = required for core research questions · P1 = desirable · P2 = backlog (ROS 2, MoveIt 2, Nav2, IKFast, video/latent world models, RL, multi-robot). **Do not start P2 work unless explicitly instructed.**

## Things agents often get wrong here
- Assuming SmolVLA works zero-shot on TIAGo (it doesn't; fine-tune on scripted-expert demos).
- Using Webots to roll out many hypothetical futures (use the MuJoCo surrogate).
- Importing `controller` (Webots) in shared code; breaks CI on both OSes.
- Using a single weighted risk score instead of veto + calibrated utility.
- Hard-coding POSIX paths or relying on `fork`.