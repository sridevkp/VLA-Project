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
uv run python -m vlarobot.cli can-regression --seeds 1 --trace-log runs/m1-watched-paced-01.jsonl
```

Inspect this episode before launching batch comparisons. It is diagnostic only and does not count
as M1-06a or M1-07 evidence.

The brain waits for each arm waypoint to settle before issuing the next command. Console output
reports per-state simulation seconds, accumulated waypoint-settle simulation seconds, and wall
seconds (including IK and transport). The original settle tolerances and timeouts are used; a
10-step no-progress watchdog fails the active state as `joint stall`. Trace records expose
`measured_joint_positions_rad` and `commanded_joint_target_rad`; the older
`commanded_joint_positions_rad` field is a deprecated measured-position alias.

### M1-06a refactor equivalence

The gate compares the original-completable prefix `INIT → PREGRASP → DESCEND → CLOSE → LIFT`.
The original trace must stop after LIFT and have exactly that entered-state sequence; the brain trace
must begin with that sequence (it may continue its full policy). Compare commanded and achieved EE
boundary positions and named DESCEND/LIFT Cartesian targets within `1e-3 m`. Every recorded IK tool tilt in
the prefix must be at or below the configured `15°` limit. Joint equality is withdrawn because the
original-only noise-floor study found a `4.139445 rad` arm_7 LIFT difference with effectively
identical task-space targets. `compare-traces` prints each trace's maximum per-joint step, minimum
joint-limit margin, and arm_7 steps over `1 rad` as diagnostics, not equivalence checks.

Fixture v2 is instrumentation-only relative to the comparison fixture. It adds CAN world/base pose,
world displacement and lift height, gripper opening/grasp/stall diagnostics, joint limits, IK error,
and tool tilt to boundaries and waypoints. Its original-controller batch uses new filenames so
existing evidence remains untouched. Keep the old
`m1-06a-original-04.jsonl` labelled **anomalous: CLOSE->RETRY, CAN displaced**. Failed or anomalous
traces are renamed with `-failed` or `-anomalous` and retained; never delete them. A clean batch is
five original runs that stop after LIFT with the required state sequence.

All M1 gates remain unverified until the live evidence is reviewed. Do not start M2 before M1-07 is
verified.

#### A. Capture five current-brain traces

Open the current worktree's `worlds/factory.wbt` in Webots and run these commands from the current
repository root. Each invocation uses seed 0 and resets the default scene. Preserve failed traces;
do not reuse an existing output path.

**Linux**

```bash
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-02.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-03.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-04.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-05.jsonl
```

**Windows PowerShell**

```powershell
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-02.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-03.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-04.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-06a-brain-paced-05.jsonl
```

`--no-pregrasp-realign` is required for M1-06a equivalence: it keeps the original semantics
(no PREGRASP re-align, no refreshed LIFT, no CLOSE guard). The comparer reports both traces'
`pregrasp_realign` flags and rejects a mixed ON/OFF pair as a documented deviation (ADR-0016).
The M1-07 full-flow runs use the default `pregrasp_realign: true` (or `--pregrasp-realign`).

#### B. Prepare and capture the original fixture-v2 batch

From the current repository root, make a clean worktree from the archived pre-refactor commit and
apply the fixture. Do not reuse the v1 worktree:

```text
git worktree add ../VLA-Project-m1-original-v2 1b2e4da
uv run python scripts/prepare_original_comparison.py ../VLA-Project-m1-original-v2
```

Open `../VLA-Project-m1-original-v2/worlds/factory.wbt` in Webots. For **each** run, start/restart
the world, note its external-controller URL, and launch the controller from the fixture worktree.
Use these new trace names:

- `m1-06a-original-v2-01.jsonl`
- `m1-06a-original-v2-02.jsonl`
- `m1-06a-original-v2-03.jsonl`
- `m1-06a-original-v2-04.jsonl`
- `m1-06a-original-v2-05.jsonl`

**Linux, from the fixture worktree**

```bash
cd ~/programs/VLA-Project-m1-original-v2
mkdir -p runs
export WEBOTS_HOME="$HOME/.local/opt/webots"
export PYTHONPATH="$WEBOTS_HOME/lib/controller/python"
export WEBOTS_CONTROLLER_URL="<URL printed by Webots>"
export VLAROBOT_TRACE_LOG="$PWD/runs/m1-06a-original-v2-01.jsonl"
uv run python controllers/tiago_controller/tiago_controller.py
```

After restarting Webots and refreshing its URL for each next run, set the next trace path and
launch the controller again:

```bash
export VLAROBOT_TRACE_LOG="$PWD/runs/m1-06a-original-v2-02.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
export WEBOTS_CONTROLLER_URL="<URL printed by Webots>"
uv run python controllers/tiago_controller/tiago_controller.py
export VLAROBOT_TRACE_LOG="$PWD/runs/m1-06a-original-v2-03.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
export WEBOTS_CONTROLLER_URL="<URL printed by Webots>"
uv run python controllers/tiago_controller/tiago_controller.py
export VLAROBOT_TRACE_LOG="$PWD/runs/m1-06a-original-v2-04.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
export WEBOTS_CONTROLLER_URL="<URL printed by Webots>"
uv run python controllers/tiago_controller/tiago_controller.py
export VLAROBOT_TRACE_LOG="$PWD/runs/m1-06a-original-v2-05.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
export WEBOTS_CONTROLLER_URL="<URL printed by Webots>"
uv run python controllers/tiago_controller/tiago_controller.py
```

**Windows PowerShell, from the fixture worktree**

```powershell
Set-Location "$HOME\programs\VLA-Project-m1-original-v2"
New-Item -ItemType Directory -Force .\runs | Out-Null
$env:WEBOTS_HOME = "C:\Program Files\Webots"
$env:PYTHONPATH = Join-Path $env:WEBOTS_HOME "lib\controller\python"
$env:WEBOTS_CONTROLLER_URL = "<URL printed by Webots>"
$env:VLAROBOT_TRACE_LOG = Join-Path (Get-Location) "runs\m1-06a-original-v2-01.jsonl"
uv run python .\controllers\tiago_controller\tiago_controller.py
```

Restart Webots before each next original run, then set the corresponding `-02` through `-05` path
and run these exact commands (refresh `WEBOTS_CONTROLLER_URL` if Webots reports a new URL):

```powershell
$env:VLAROBOT_TRACE_LOG = Join-Path (Get-Location) "runs\m1-06a-original-v2-02.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
$env:WEBOTS_CONTROLLER_URL = "<URL printed by Webots>"
uv run python .\controllers\tiago_controller\tiago_controller.py
$env:VLAROBOT_TRACE_LOG = Join-Path (Get-Location) "runs\m1-06a-original-v2-03.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
$env:WEBOTS_CONTROLLER_URL = "<URL printed by Webots>"
uv run python .\controllers\tiago_controller\tiago_controller.py
$env:VLAROBOT_TRACE_LOG = Join-Path (Get-Location) "runs\m1-06a-original-v2-04.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
$env:WEBOTS_CONTROLLER_URL = "<URL printed by Webots>"
uv run python .\controllers\tiago_controller\tiago_controller.py
$env:VLAROBOT_TRACE_LOG = Join-Path (Get-Location) "runs\m1-06a-original-v2-05.jsonl"
# Restart Webots before this run and refresh the URL if Webots reports a new one.
$env:WEBOTS_CONTROLLER_URL = "<URL printed by Webots>"
uv run python .\controllers\tiago_controller\tiago_controller.py
```

If a trace is failed or anomalous, keep it and rename it before retrying; never delete it or reuse
its output path. Example rename commands:

```bash
mv runs/m1-06a-original-v2-04.jsonl runs/m1-06a-original-v2-04-anomalous.jsonl
```

```powershell
Move-Item .\runs\m1-06a-original-v2-04.jsonl .\runs\m1-06a-original-v2-04-anomalous.jsonl
```

#### C. Compare matching traces

Close the original fixture world and open the current worktree's factory world. From the current
repository root, compare each pair:

```text
uv run python -m vlarobot.cli compare-traces ../VLA-Project-m1-original-v2/runs/m1-06a-original-v2-01.jsonl runs/m1-06a-brain-paced-01.jsonl --prefix INIT,PREGRASP,DESCEND,CLOSE,LIFT --tol-pos 1e-3 --max-tilt-deg 15
uv run python -m vlarobot.cli compare-traces ../VLA-Project-m1-original-v2/runs/m1-06a-original-v2-02.jsonl runs/m1-06a-brain-paced-02.jsonl --prefix INIT,PREGRASP,DESCEND,CLOSE,LIFT --tol-pos 1e-3 --max-tilt-deg 15
uv run python -m vlarobot.cli compare-traces ../VLA-Project-m1-original-v2/runs/m1-06a-original-v2-03.jsonl runs/m1-06a-brain-paced-03.jsonl --prefix INIT,PREGRASP,DESCEND,CLOSE,LIFT --tol-pos 1e-3 --max-tilt-deg 15
uv run python -m vlarobot.cli compare-traces ../VLA-Project-m1-original-v2/runs/m1-06a-original-v2-04.jsonl runs/m1-06a-brain-paced-04.jsonl --prefix INIT,PREGRASP,DESCEND,CLOSE,LIFT --tol-pos 1e-3 --max-tilt-deg 15
uv run python -m vlarobot.cli compare-traces ../VLA-Project-m1-original-v2/runs/m1-06a-original-v2-05.jsonl runs/m1-06a-brain-paced-05.jsonl --prefix INIT,PREGRASP,DESCEND,CLOSE,LIFT --tol-pos 1e-3 --max-tilt-deg 15
```

The comparator reports the first state/EE/Cartesian/tilt divergence and prints per-trace joint
diagnostics. M1-06a passes only when five clean matched pairs pass the task-space/state gate; joint
diagnostics are reported without an equality threshold.

## 3a. M1-07 realign ON/OFF batches and analysis

Every run below starts from a **fresh world** (close and reopen `worlds/factory.wbt`, or use the
documented reset procedure) and uses seed 0. Keep `hold_base: none`. Do not reuse an output path.
The ON batch is the M1-07 candidate; the OFF batch is the original-semantics control.

**Realign ON, three runs (default `pregrasp_realign: true`)**

```bash
uv run python -m vlarobot.cli can-regression --seeds 1 --pregrasp-realign --trace-log runs/m1-07-realign-on-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --pregrasp-realign --trace-log runs/m1-07-realign-on-02.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --pregrasp-realign --trace-log runs/m1-07-realign-on-03.jsonl
```

**Realign OFF, three runs (original semantics)**

```bash
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-07-realign-off-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-07-realign-off-02.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --no-pregrasp-realign --trace-log runs/m1-07-realign-off-03.jsonl
```

**Per-state report** (CAN displacement, lift, live-base tip→CAN xy error, realign before/after,
tilt, and the four final criteria). Run once per trace; replace the filename.

```bash
uv run python scripts/analyze_can_run.py runs/m1-07-realign-on-01.jsonl
uv run python scripts/analyze_can_run.py runs/m1-07-realign-on-02.jsonl
uv run python scripts/analyze_can_run.py runs/m1-07-realign-on-03.jsonl
uv run python scripts/analyze_can_run.py runs/m1-07-realign-off-01.jsonl
uv run python scripts/analyze_can_run.py runs/m1-07-realign-off-02.jsonl
uv run python scripts/analyze_can_run.py runs/m1-07-realign-off-03.jsonl
```

On Windows PowerShell use the same commands with `runs\...` paths; the script uses `pathlib` and
has no shell dependency.

**jq equivalents** (one line per state; `runs/m1-07-realign-on-01.jsonl`):

```bash
jq -r 'select(.event=="state_boundary" and .phase=="exit") | [.state, (.can_world_xyz_m|@json), (.can_displacement_world_xyz_m|@json), .lift_height_m, .pregrasp_realign, .base_tilt_roll_deg, .base_tilt_pitch_deg] | @tsv' runs/m1-07-realign-on-01.jsonl
jq -c 'select(.event=="pregrasp_realign" or .event=="pregrasp_realigned" or .event=="close_alignment_check" or .event=="close_realign" or .event=="close_misaligned")' runs/m1-07-realign-on-01.jsonl
jq -c 'select(.event=="state_boundary" and .state=="DONE") | {pregrasp_realign, distance_m, height_error_m, tilt_deg, home_reached, place_distance_ok, can_height_ok, can_tilt_ok, success}' runs/m1-07-realign-on-01.jsonl
```

`tip->can_xy_mm` is computed from the live base pose in the Python script, not stored directly in
the trace. The CLOSE guard's own measurement is `close_alignment_check.xy_error_m` (and
`close_realign.xy_error_after_m`). Report the ON and OFF batches separately; do not average them.

## 3b. Fresh-world episode runner (`run-episodes`)

`run-episodes` launches one Webots subprocess per episode, waits for the ZeroMQ server, runs one
brain-side episode, terminates the simulator, and prints one summary table row per episode
(outcome, the four success criteria, failure state, and timing). It resolves the Webots binary
from `--webots`, `WEBOTS_HOME`, then `PATH`; per-episode traces and Webots logs are written to
`runs/episode-runner/` by default and stale per-episode traces are deleted before each run.
Rendering stays on by default because the scripted policy reads the camera every step.

**Batch of three fresh worlds (one process per episode, seeds 0..2):**

```bash
uv run python -m vlarobot.cli run-episodes --n 3 --fresh-world
```

**Original-semantics control batch (closed-loop off) in fresh worlds:**

```bash
uv run python -m vlarobot.cli run-episodes --n 3 --fresh-world --no-closed-loop-waypoints
```

**Same process, soft reset per episode (like `can-regression`):**

```bash
uv run python -m vlarobot.cli run-episodes --n 3 --no-fresh-world
```

**Determinism check:** run the same seed 3 times, each in its own fresh world, and compare
outcomes, normalized trace digests, and numeric deltas (sim time, achieved EE, CAN pose,
commanded EE). The command exits 1 on a mismatch.

```bash
uv run python -m vlarobot.cli run-episodes --determinism-check 3
```

**Watched single state in a fresh world** (e.g. stop after DESCEND for tip telemetry):

```bash
uv run python -m vlarobot.cli run-episodes --n 1 --stop-after DESCEND
```

Useful extras: `--trace-dir runs/my-batch`, `--webots /path/to/webots`, `--endpoint tcp://...`,
`--startup-timeout-s 180`, `--pregrasp-realign/--no-pregrasp-realign`. All flags are standard
typer options and identical on Windows PowerShell (use `runs\episode-runner` style paths).

**Caveat — headless is UNVERIFIED.** `--headless` maps to the documented batch flags
(`webots --batch --mode=fast --no-rendering --stdout --stderr`). `--no-rendering` can blank
camera frames, and the scripted policy reads RGB every step, so an episode may fail for that
reason alone. Keep rendering (the default) until the headless path has been live-verified on the
target OS; the runner prints a warning whenever it is enabled.

_GT fingertip telemetry_ (`gt_tip_telemetry`, plus the merged boundary fields) appears in every
trace from this section's real runs; `scripts/analyze_can_run.py` prints it as a table after the
per-state rows, together with any `TOOL_OFFSET` calibration proposal (report-only, never
applied).

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

## 7. Diagnose PREGRASP table contact and base drift

The FK sweep below is Webots-free and uses the last watched paced trace as its start/goal example.
The table instance is verified in `worlds/factory.wbt` (4.5 x 1.5 m, top z=0.75 m); the hand
sample uses a conservative 0.08 m radius. It does not prove Webots collision behavior:

```text
uv run python scripts/analyze_pregrasp_table.py runs/m1-watched-paced-02.jsonl
```

For a watched live run, open `worlds/factory.wbt` in Webots and keep the world visible. From a
second terminal in the repository root run one command per fresh episode. Use distinct trace paths;
the CLI appends to a trace and will not overwrite an existing file.

**Linux**

```bash
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP --pregrasp-route interpolated --trace-log runs/m1-pregrasp-interpolated-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP --pregrasp-route safe_via --trace-log runs/m1-pregrasp-safe-via-01.jsonl
```

**Windows PowerShell**

```powershell
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP --pregrasp-route interpolated --trace-log .\runs\m1-pregrasp-interpolated-01.jsonl
uv run python -m vlarobot.cli can-regression --seeds 1 --stop-after PREGRASP --pregrasp-route safe_via --trace-log .\runs\m1-pregrasp-safe-via-01.jsonl
```

The factory world controller starts the simulator server. Do not start two brain commands at once.
The configured base invariant is 0.005 m planar drift or 0.5 degrees yaw from reset. A violation
stops the active state with `base moved`; contact events include node identity and world contact
points. The `safe_via` run is experimental and should remain watched.

Print base drift against simulation time and the torso readback with `jq`:

```bash
jq -r 'select(.state == "PREGRASP" and .event == "state_boundary") | [.sim_time_s, .phase, .base_drift_distance_m, .base_drift_yaw_deg, .base_drift_world_xyz_m] | @tsv' runs/m1-pregrasp-interpolated-01.jsonl
jq -r 'select(.state == "PREGRASP" and .event == "state_boundary") | [.sim_time_s, .phase, .torso_lift_joint_sensor_rad] | @tsv' runs/m1-pregrasp-interpolated-01.jsonl
jq -c 'select(.event == "unexpected_contact") | {sim_time_s, state, node_name, node_type, contact_points_world_xyz_m}' runs/m1-pregrasp-interpolated-01.jsonl
jq -r 'select(.state == "PREGRASP") | [.sim_time_s, .phase, .supervisor_min_table_clearance_m] | @tsv' runs/m1-pregrasp-interpolated-01.jsonl
jq -c 'select(.state == "PREGRASP") | {sim_time_s, phase, supervisor_arm_link_world_xyz_m, supervisor_min_table_clearance_m}' runs/m1-pregrasp-interpolated-01.jsonl
jq -r 'select(.state == "PREGRASP") | [.sim_time_s, .phase, .arm_7_joint_fk_ground_truth_error_m, .arm_7_joint_fk_ground_truth_mismatch] | @tsv' runs/m1-pregrasp-interpolated-01.jsonl
```

Table geometry and the clearance requirement come from `configs/robot.yaml`
(`table_top_height_m`, `table_footprint_centre_xy_m`, `table_footprint_size_xy_m`,
`table_clearance_required_m`). The table values were read from `worlds/factory.wbt` and the R2025a
`Table.proto`; the 0.05 m clearance requirement is a policy value. A `null`
`supervisor_min_table_clearance_m` means no Supervisor arm link was inside the table footprint, and
`null` FK fields mean the Supervisor position was unavailable. The FK error compares the arm_7 joint
anchor against the Supervisor position of the same joint node; a value above
`fk_ground_truth_tolerance_m` (0.01 m) sets `arm_7_joint_fk_ground_truth_mismatch` to `true`.

Equivalent `rg` filters for a quick line-level view:

```text
rg '"event":"state_boundary".*"state":"PREGRASP"' runs/m1-pregrasp-interpolated-01.jsonl
rg '"event":"unexpected_contact"' runs/m1-pregrasp-interpolated-01.jsonl
```

Use the same commands with `m1-pregrasp-safe-via-01.jsonl` for the second route. No M1 gate is
considered verified by these diagnostic runs.

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
