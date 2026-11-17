# Development setup

The project targets Windows 10/11 and Linux with Python 3.11. The repository pins its development
interpreter to Python 3.11.18 in `.python-version`. Install `uv`, then run these commands from the
repository root:

```text
uv sync --extra sim --extra dev
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv run pytest -m "not webots"
```

The `dev` extra contains Ruff, pytest, ty, and pre-commit (plus Pillow, used by the eval
`EpisodeRecorder` to write JPEG frames — see ADR-0031). The `sim` extra installs IKPy for the
Webots backend. The remaining optional runtime extras (`core`, `perception`, `vla`, `dashboard`,
and `eval`) are declared in `pyproject.toml`. The `gpu` and `webots` pytest markers identify tests
that require local hardware/simulator access.

> **Sync caveat (MEASURED 2026-11-14):** always sync with **both** extras —
> `uv sync --extra sim --extra dev`. A plain `uv sync --extra dev` prunes `ikpy`/`scipy` from the
> environment and the test suite then fails at import. The `eval` extra (`pillow>=10.4`) is what a
> production eval install would use; the `dev` group carries the same dependency so tests run
> without it.

## Windows native

Install Webots and Python 3.11.18, then open PowerShell in the repository root. Start the Webots
GUI, open `worlds/smoke.wbt`, and use the external-controller URL Webots displays:

```powershell
$env:WEBOTS_HOME = "C:\Program Files\Webots"
$env:WEBOTS_CONTROLLER_URL = "tcp://127.0.0.1:1234/TIAGo"
uv sync --extra sim --extra dev
uv run python -m vlarobot.cli smoke
```

Use the URL Webots actually prints/displays instead of the example URL above. The smoke command
sets the controller API path in `PYTHONPATH` and `sys.path` using the host OS path separator.

For the lockstep backend, open `worlds/factory.wbt` in Webots. The controller shim selects the
repository's `.venv` Python (override with `WEBOTS_PYTHON` if needed) and starts the simulator
server. In a second PowerShell window, set `WEBOTS_HOME` and run:

```powershell
uv run python -m vlarobot.cli brain --steps 10
```

The brain command resets the default scene, advances ten lockstep steps, and prints simulation time
and RGB shape. It is a transport smoke test, not the M1 CAN regression.

The simulator/brain wire protocol is version 3. Restart both processes after protocol changes;
peers with different versions fail fast. Backend reset initializes torso/head posture and traces
include read-only diagnostics. `pregrasp_route` in `configs/robot.yaml` selects `interpolated`
(default) or the experimental `safe_via` route.

### PyTorch CUDA on Windows

Choose **Windows / Pip / Python / CUDA** in the [official PyTorch selector](https://pytorch.org/get-started/locally/).
Use its current CUDA wheel index URL with `uv pip` from the repository's active environment. For
example, if the selector gives the CUDA 12.8 wheel index:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Change `cu128` to the index selected for the installed NVIDIA driver and supported PyTorch release.
No CUDA wheel or PyTorch version is pinned by this project yet.

## Linux native (Fedora/Ubuntu)

Install Webots and Python 3.11.18. Start Webots, open `worlds/smoke.wbt`, and set the variables in
the terminal from which the CLI will connect. Example for the validated Fedora install:

```bash
export WEBOTS_HOME="$HOME/.local/opt/webots"
export WEBOTS_CONTROLLER_URL="ipc://1234/TIAGo"
uv sync --extra sim --extra dev
uv run python -m vlarobot.cli smoke
```

The external-controller URL depends on the Webots session; use the URL it reports. `ipc://` is for
local processes. Use the displayed `tcp://` URL when connecting over a network.

For the lockstep backend, open `worlds/factory.wbt` in Webots, then run this in another terminal:

```bash
uv run python -m vlarobot.cli brain --steps 10
```

The controller shim uses `.venv/bin/python` when available. Set `WEBOTS_PYTHON` to a compatible
Python 3.11 executable if Webots is launched outside the repository environment.

### PyTorch CUDA on Linux

Choose **Linux / Pip / Python / CUDA** in the [official PyTorch selector](https://pytorch.org/get-started/locally/).
Use its current CUDA wheel index URL with `uv pip`:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Replace `cu128` with the selector's wheel index for the installed NVIDIA driver and supported
PyTorch release. Install the NVIDIA driver separately; the wheel does not replace it. PyTorch is
not part of the default project or development dependencies.

## WSL2 brain with Windows Webots

Install Python 3.11.18 and the Linux Webots controller Python API in WSL2. Start Webots on Windows
and open `worlds/smoke.wbt`. Use the Windows host IP reachable from WSL2 and the TCP external
controller URL displayed by Webots:

```bash
export WEBOTS_HOME="$HOME/.local/opt/webots"
export WEBOTS_CONTROLLER_URL="tcp://<windows-host-ip>:1234/TIAGo"
uv sync --extra sim --extra dev
uv run python -m vlarobot.cli smoke
```

Replace the example host address, port, and robot name with Webots' displayed URL. Allow the
external-controller port through the Windows firewall if prompted. Do not point WSL2 Python at a
Windows Webots installation's Python libraries; use Linux-compatible controller libraries.

For VLA training or GPU inference in WSL2, install the Linux PyTorch CUDA wheel using the Linux
instructions above, inside the WSL2 environment.

## What the Webots smoke command checks

The CLI connects to the running external-controller world, enables the `Astra rgb` camera, advances
ten basic simulation steps, validates the RGBA frame, and prints simulation time and image shape.
On Fedora/Linux with Webots R2025a it produced:

```text
sim_time=0.320s image_shape=(480, 640, 4)
```

The Webots installation must include `lib/controller/python`. The cross-platform environment setup
is provided by `vlarobot.env.configure_webots_environment`; it sets `WEBOTS_HOME` and adds the
controller API directory to both `PYTHONPATH` and the running Python process's import path.


Isolated WorldMind development commands and evidence: WORLDMIND_DEVELOPMENT_CHECKPOINT.md. Run pure-model tests without Webots; the recorded profile accepts the original project as an explicit read-only registered FK adapter. The TEST geometry generation unit is intentionally excluded before Week 4. Colab uses the approved main-worktree artifacts exclusively.

Proposed 017000 recovery is documented in TRAINED_EXPORT_RECOVERY.md. The separate
`tools/recovery` helpers add no dependency and do not edit approved source bundles.
Use audit/prepare before any opt-in training. Never rerun the original production
cell as part of this recovery, and never infer completeness from COMPLETE.json alone.

For the owner-confirmed Colab project shortcut, use the v3 bundle and the three
audit/prepare/restore-only cells in COLAB_SHORTCUT_FIX.md. Preserve old recovery
directories. The trusted project entrypoint is canonicalized before constructing
run paths; all internal symlink and hash checks remain active.

Durable recovery v4 supersedes the Drive-output training proposal. Follow
DURABLE_RECOVERY_PLAN.md for the owner-reviewed independent SSH transfer plan,
read-only Drive diagnostics and local-output restoration. Do not launch training
until the real endpoint canary and actual CUDA restoration pass. The Linux backup
is preserved and packaged separately; no original recovery directory is removed.

Current PASS/BLOCKED status, private receiver templates, owner key/host-key steps,
interruption limits and prepared import/TRAIN commands are in
DURABLE_READINESS_REPORT.md. No SSH listener or credential is provisioned by setup.

For a second Google account, use notebooks/smolvla_account_migration.ipynb and the
v5 copy-paste cells in DURABLE_RECOVERY_PLAN.md. Do not reuse the original account's
shortcut ID, cached authentication or canary receipt. The v4 plan is archived in
docs/history; the approved source/data ZIPs and original notebook remain unchanged.
# Current-session Drive recovery handoff

For an existing v6 canary rejected solely on historical version equality, use
[v7 reconnect cells](DRIVE_CANARY_V7_CURRENT_SESSION.md). They reuse the existing
folder/config/receipt, perform a stable full-content revalidation and restart the
credential refresh thread stopped by failed setup. No repeated migration or
training is needed. Do not manually rewrite receipt versions.

After second-account migration Cells 1–4, use
[the v6 replacement cells](DRIVE_HANDOFF_REPLACEMENT_CELLS.md) in the same kernel.
Only the small helper patch is uploaded; source, dataset and installed environment
are reused. Google authorization must permit read access and creation of app-owned
files in the second account. The worker needs no new SDK dependency: it uses stdlib
HTTPS and a private local access-token file refreshed by the existing Colab kernel.
The helper creates a new second-owned My Drive folder; no original evidence is
updated or deleted. Actual API persistence and CUDA restoration are owner-session
stop gates, not locally verified claims. Training remains commented out.

Recovered step20000 local inference validation and exact reproduction environment: see [LOCAL_MODEL_VALIDATION_STEP20000.md](LOCAL_MODEL_VALIDATION_STEP20000.md). CPU model loading/inference passed, but gripper-bound violations block evaluation. Do not rerun training or silently clamp outputs.

A-D1 numerical harness and executed commands are documented in [AD1_NUMERICAL_REPORT.md](AD1_NUMERICAL_REPORT.md). Its exact two-solve authorization is exhausted; no rerun or output replacement is authorized.

A-D2 offline infrastructure fixtures (no simulator/model/IK):
`uv run pytest -q tests/test_ad2_infrastructure.py` from the project root.
Read [AD2_CPU_INFRASTRUCTURE_REPORT.md](AD2_CPU_INFRASTRUCTURE_REPORT.md) for the hash-pinned
read-only source audit command. No A-D2 physical launch command is provided or authorized.

## Production WorldMind command (currently fail-closed)

The production command is additive to `eval` and `eval-vla`; it does not replace either command.
From the project root, the intended owner-reviewed invocation is a single cross-platform command:

```text
uv run python -m vlarobot.cli eval-worldmind --config configs/worldmind_production.yaml --out runs/worldmind/validation-s0-<NEW-ID> --backend webots --state-source oracle
```

The shipped config is intentionally `status: blocked`, so this command currently exits with
`eval-worldmind preflight failed` (code 1) and creates no output directory. That is the required
fail-closed behavior, not a setup error to bypass. The CLI registers the concrete existing
components through `production_worldmind_bindings()` and checks them in a structured preflight
(`eval/worldmind_preflight.py`); scientific bindings (Eq. 3.8 components/weights, ranking direction,
risk thresholds and the calibration evaluator) are deliberately absent and remain fail-closed.
Do not point it at TEST or TRAIN: only `--split validation` is accepted. Use a fresh `--out` path
for every run; prior evidence is never overwritten. A successful run requires owner-approved
report definitions and VALIDATION calibration candidate/evidence, the scientific component/check/
factory bindings, weights, ranking direction and constraints to be registered, a certified full-arm
sweep check, plus an already running sim server for `--backend webots`. The command never launches
Webots itself. Oracle-assisted results are labelled and are never presented as perception results.

Stage-1 diagnostic and parametric-selection review commands are additive and inert. From the
project root, the first diagnostic invocation is:

```text
uv run python -m vlarobot.cli worldmind-diagnostic --config configs/worldmind_diagnostic.yaml --out runs/worldmind/validation-s0-diagnostic --backend webots --state-source oracle
```

The shipped `configs/worldmind_diagnostic.yaml` is a blocked template, so this command currently
fails closed with code 1 and creates no output directory. Fill only owner-reviewed values; never
add Eq. 3.8 weights, ranking direction or a calibration result to this stage-1 config. Selection
statistics can be reviewed without writing or approving parameters:

```text
uv run python -m vlarobot.cli worldmind-calibration-proposal --draft draft.json --evaluator <REGISTERED_EVALUATOR>
```

A separate, execution-ready no-actuation Webots diagnostic is available but is **not authorized by
this setup note**. It must not be run until the owner explicitly authorizes the exact command below.
From the project root, that command is:

```text
uv run python -m vlarobot.cli worldmind-webots-diagnostic --config configs/worldmind_webots_diagnostic.yaml --out runs/worldmind/validation-s0-webots-readonly
```

It starts one Webots process in `mode: launch`, uses only the read-only `inspect` RPC, and creates a
new exclusive output directory containing `diagnostic.json`, `state.json` and `provenance.json`.
It never applies an action, steps, resets, exports kinematics, scores a campaign, accesses
TEST/TRAIN evidence or permits motion. A diagnostic `PASS` does not certify the arm sweep; the
`unverified-arm-sweep` veto remains hard. Use a fresh `--out` path for every authorized run.

A bounded geometry-only diagnostic is available without Webots authorization. From the project
root, use a fresh output path:

```text
uv run python -m vlarobot.cli worldmind-geometry-diagnostic --config configs/worldmind_geometry_diagnostic.yaml --out runs/worldmind/validation-s0-geometry-inert-<NEW-ID>
```

It reads only the pinned URDF, registration, robot config and world file. It does not launch a
simulator, open a socket, solve IK, apply/step/reset a backend, score an episode or read
TEST/TRAIN evidence. It writes `diagnostic.json`, `state.json` and `provenance.json`, and always
reports `motion_allowed=false` and `motion_readiness="NOT_READY"`. A `PASS` reports that evidence
was collected; unresolved external PROTOs, self-collision, floor/table/finger envelopes and the
missing owner-reviewed full-arm certificate keep geometric motion readiness blocked. See
`DECISIONS.md` ADR-0054 for the separate engineering and owner-safety gates.

Inert FakeBackend verification without Webots:

```text
uv run python -m pytest -q tests/test_worldmind_production.py
```

This uses synthetic fixtures only and does not authorize or produce a scored VALIDATION result.
See `WEEK3_DECISION_OBSERVABILITY.md` §10 for the reviewed inert node list.
