# Development setup

The project targets Windows 10/11 and Linux with Python 3.11. The repository pins its development
interpreter to Python 3.11.18 in `.python-version`. Install `uv`, then run these commands from the
repository root:

```text
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv run pytest -m "not webots"
```

The `dev` extra contains Ruff, pytest, ty, and pre-commit. Optional runtime extras (`core`, `sim`,
`perception`, `vla`, and `dashboard`) are declared in `pyproject.toml`; dependencies are added when
their implementations are introduced. The `gpu` and `webots` pytest markers identify tests that
require local hardware/simulator access.

## Windows native

Install Webots and Python 3.11.18, then open PowerShell in the repository root. Start the Webots
GUI, open `worlds/smoke.wbt`, and use the external-controller URL Webots displays:

```powershell
$env:WEBOTS_HOME = "C:\Program Files\Webots"
$env:WEBOTS_CONTROLLER_URL = "tcp://127.0.0.1:1234/TIAGo"
uv sync --extra dev
uv run python -m vlarobot.cli smoke
```

Use the URL Webots actually prints/displays instead of the example URL above. The smoke command
sets the controller API path in `PYTHONPATH` and `sys.path` using the host OS path separator.

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
uv sync --extra dev
uv run python -m vlarobot.cli smoke
```

The external-controller URL depends on the Webots session; use the URL it reports. `ipc://` is for
local processes. Use the displayed `tcp://` URL when connecting over a network.

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
uv sync --extra dev
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
