# Manual project invocation

This guide covers the commands currently available in the repository. For installation details,
Webots setup, and CUDA, see [SETUP.md](./SETUP.md). Run project commands from the repository root.

## 1. Prepare the Python environment

Install Python 3.11 and `uv`, then run these commands from the repository root. Create the local
`.venv` and install the simulator and development dependencies:

```text
uv venv --python 3.11
uv sync --extra sim --extra dev
```

`uv sync` installs the project into `.venv` and keeps it synchronized with the lockfile. Activate
it if you want to run `python` or other commands directly:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux**

```bash
source .venv/bin/activate
```

Activation is optional for commands prefixed with `uv run`; those automatically use the project
environment. Confirm the interpreter with `python --version` (should be Python 3.11.x).

## 2. Run the simulator and brain

Set `WEBOTS_HOME` for the terminal that will run the brain:

**Windows PowerShell**

```powershell
$env:WEBOTS_HOME = "C:\Program Files\Webots"
```

**Linux**

```bash
export WEBOTS_HOME="$HOME/.local/opt/webots"
```

Open `worlds/factory.wbt` in Webots. The `tiago_controller` shim starts the lockstep simulator
server. In a terminal at the repository root, run the brain transport smoke test:

```text
uv run python -m vlarobot.cli brain --steps 10
```

The command resets the default scene, advances ten simulation steps, and prints simulation time and
the RGB image shape. The endpoint defaults to `tcp://127.0.0.1:5555`; override it with
`--endpoint` if the server is configured to use another endpoint.

## 3. Run the CAN regression

With the factory world and its simulator server running, execute the brain-side scripted regression:

```text
uv run python -m vlarobot.cli can-regression --seeds 20 --trace-log runs/m1-07-brain-can.jsonl
```

For M1-07, require at least 16 successful episodes out of 20. Success means the policy reaches HOME
and the final CAN-to-target distance is below 0.10 m in the robot base frame. Each episode prints its
seed, success, safety rejection count, final target distance, and policy state/failure point. The
JSONL trace includes state boundaries, commanded/achieved base-frame pose, IK result,
joint/gripper state, and CAN pose in world and base frames.

The Webots reset mode is configured by `reset_mode` in `configs/robot.yaml`. Keep the default
`soft` for in-process episode resets. `reload` is reserved for a future Webots-controller restart
and ZeroMQ reconnect implementation; selecting it currently reports an explicit unsupported-mode
error.

**Current validation status:** M1-06a and M1-07 remain unverified until their live runs are
completed. The transit-only IK tolerance is 0.015 m; grasp-related moves retain 0.005 m and the
existing 15-degree tilt limit. Do not treat the earlier 0.560 m mixed-frame score as valid. See
[PROGRESS.md](./PROGRESS.md). Resets currently replay the default scene; different seeds do not
imply randomized scenarios.

### Watched single episode

Start Webots with the GUI and the factory world, then run one seeded brain episode with a trace:

```text
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-watched.jsonl
```

Inspect this episode before launching batch comparisons. It is diagnostic only and does not count
as M1-06a or M1-07 evidence.

### M1-06a refactor equivalence

Create a temporary original-controller worktree at the pre-refactor commit:

```text
git worktree add ../VLA-Project-m1-original 1b2e4da
```

From the current repository root, apply the comparison-only fixture:

```text
uv run python scripts/prepare_original_comparison.py ../VLA-Project-m1-original
```

The fixture applies the approved world-frame target and original `Frames.world_to_base()`
conversion, sets deterministic IK seed 0, stops after LIFT, switches the temporary factory world
to an extern controller, and emits state/waypoint JSONL records. It does not alter the original IK
or motion algorithms. For original runs, open the fixture worktree's factory world and run its
extern controller. For brain runs, close that world and open the current repository's
`worlds/factory.wbt`, whose TIAGo shim starts the ZMQ simulator server. Use the same headed factory
scene geometry; do not run both controllers simultaneously.

**Linux Webots launch**

```bash
export WEBOTS_HOME="$HOME/.local/opt/webots"
"$WEBOTS_HOME/webots" "$PWD/worlds/factory.wbt"
```

**Windows PowerShell Webots launch**

```powershell
$env:WEBOTS_HOME = "C:\Program Files\Webots"
& (Join-Path $env:WEBOTS_HOME "webots.exe") (Join-Path (Get-Location) "worlds\factory.wbt")
```

From the original worktree, run the original controller after each world restart. Use a new trace
file for every run:

**Linux**

```bash
mkdir -p runs
export WEBOTS_HOME="$HOME/.local/opt/webots"
export PYTHONPATH="$WEBOTS_HOME/lib/controller/python"
export WEBOTS_CONTROLLER_URL="<URL printed by Webots>"
export VLAROBOT_TRACE_LOG="$PWD/runs/m1-06a-original-01.jsonl"
uv run python controllers/tiago_controller/tiago_controller.py
```

**Windows PowerShell**

```powershell
New-Item -ItemType Directory -Force .\runs | Out-Null
$env:WEBOTS_HOME = "C:\Program Files\Webots"
$env:PYTHONPATH = Join-Path $env:WEBOTS_HOME "lib\controller\python"
$env:WEBOTS_CONTROLLER_URL = "<URL printed by Webots>"
$env:VLAROBOT_TRACE_LOG = Join-Path (Get-Location) "runs\m1-06a-original-01.jsonl"
uv run python .\controllers\tiago_controller\tiago_controller.py
```

Use suffixes `02` through `05` for the remaining original runs.

Run the current brain path five times with seed 0, restarting the simulator episode for each run:

```text
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-06a-brain-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-06a-brain-02.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-06a-brain-03.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-06a-brain-04.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-06a-brain-05.jsonl
```

Compare each original/brain run pair:

```text
uv run python -m vlarobot.cli compare-traces runs/m1-06a-original-01.jsonl runs/m1-06a-brain-01.jsonl --prefix INIT,PREGRASP,DESCEND,CLOSE,LIFT --tol-joint 1e-3 --tol-pos 1e-3
```

The command reports the first divergent state/waypoint and exits nonzero on mismatch. These prefix
runs do not assert full pick-and-place success.

## 4. Capture controller-side diagnostics

The simulator-server mode is the default. To run the scripted controller flow and append its
state/waypoint trace, close any active factory world and start Webots from a terminal with the
following environment. `runs/` is created automatically.

**Windows PowerShell**

```powershell
$env:VLAROBOT_WEBOTS_MODE = "legacy"
$env:VLAROBOT_WAYPOINT_LOG = Join-Path (Get-Location) "runs\controller-can.jsonl"
& (Join-Path $env:WEBOTS_HOME "webots.exe") (Join-Path (Get-Location) "worlds\factory.wbt")
```

**Linux**

```bash
export VLAROBOT_WEBOTS_MODE=legacy
export VLAROBOT_WAYPOINT_LOG="$PWD/runs/controller-can.jsonl"
"$WEBOTS_HOME/webots" "$PWD/worlds/factory.wbt"
```

The log records state entry/exit, simulation time, base/tool/CAN poses, joint/head positions, IK
error and waypoint, gripper opening/stall state, and finger-to-CAN contact. To use the brain-side
server instead, set `VLAROBOT_WEBOTS_MODE=server` before launching Webots and run the
`can-regression --trace-log` command above. Compare runs only after verifying that both use the same
target and world. Webots reset seeds do not randomize the current default scene.

## 5. Run the external-controller camera smoke test

Open `worlds/smoke.wbt` in Webots and note the external-controller URL shown by Webots. Set the
variables in the terminal that will run the command:

**Windows PowerShell**

```powershell
$env:WEBOTS_HOME = "C:\Program Files\Webots"
$env:WEBOTS_CONTROLLER_URL = "tcp://127.0.0.1:1234/TIAGo"
uv run python -m vlarobot.cli smoke
```

**Linux**

```bash
export WEBOTS_HOME="$HOME/.local/opt/webots"
export WEBOTS_CONTROLLER_URL="tcp://127.0.0.1:1234/TIAGo"
uv run python -m vlarobot.cli smoke
```

Replace the example URL with the one displayed by your Webots session. The smoke command connects
to the external controller, reads an RGBA camera frame, advances ten steps, and prints simulation
time and image dimensions.

## 6. Run checks without Webots

From the repository root:

```text
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not webots"
```

## Available CLI commands

To see command options, run:

```text
uv run python -m vlarobot.cli --help
uv run python -m vlarobot.cli brain --help
uv run python -m vlarobot.cli can-regression --help
```

`smoke`, `brain`, and `can-regression` are implemented. The standalone `sim-server` command is
intended to run inside Webots' controller environment; in the factory world, the controller shim
starts it automatically.
