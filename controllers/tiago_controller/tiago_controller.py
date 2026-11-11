"""Launch the packaged lockstep server with the project Python environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    venv_python = (
        project_root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else project_root / ".venv" / "bin" / "python"
    )
    default_interpreter = venv_python if venv_python.is_file() else Path(sys.executable)
    interpreter = Path(os.environ.get("WEBOTS_PYTHON", str(default_interpreter)))
    mode = os.environ.get("VLAROBOT_WEBOTS_MODE", "server")
    if mode not in {"server", "legacy"}:
        raise ValueError(f"Unknown VLAROBOT_WEBOTS_MODE: {mode!r}")
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    python_paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    if source_path not in python_paths:
        python_paths.insert(0, source_path)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
    module = (
        "vlarobot.backends.webots.server"
        if mode == "server"
        else "vlarobot.backends.webots.controller"
    )
    result = subprocess.run(
        [str(interpreter), "-m", module],
        cwd=project_root,
        env=environment,
        check=False,
    )
    # Crash forensics (ADR-0022): record the packaged server's exit status where the
    # episode runner can find it. Pure diagnostics; no controller behaviour change.
    stderr_path = os.environ.get("VLAROBOT_CONTROLLER_STDERR_PATH")
    if stderr_path:
        try:
            with open(stderr_path, "a", encoding="utf-8") as fh:
                fh.write(f"tiago_controller: server_exit_code={result.returncode}\n")
        except OSError:
            pass
    raise SystemExit(result.returncode)
