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
