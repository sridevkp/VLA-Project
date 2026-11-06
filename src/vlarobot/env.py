"""Cross-platform environment setup for the Webots Python controller API."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _contains_path(entries: list[str], target: Path) -> bool:
    target_path = os.path.normcase(os.path.abspath(target))
    return any(
        entry and os.path.normcase(os.path.abspath(entry)) == target_path for entry in entries
    )


def configure_webots_environment(webots_home: str | os.PathLike[str] | None = None) -> Path:
    """Set WEBOTS_HOME and add Webots' controller API path to PYTHONPATH and sys.path."""
    configured_home = webots_home if webots_home is not None else os.environ.get("WEBOTS_HOME")
    if not configured_home:
        raise RuntimeError("Set WEBOTS_HOME to the Webots installation directory.")

    home = Path(configured_home).expanduser().resolve()
    if not home.is_dir():
        raise NotADirectoryError(f"Webots installation directory not found: {home}")

    controller_path = home / "lib" / "controller" / "python"
    if not controller_path.is_dir():
        raise FileNotFoundError(f"Webots controller Python API not found: {controller_path}")

    controller_path_str = str(controller_path)
    os.environ["WEBOTS_HOME"] = str(home)

    pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if not _contains_path(pythonpath_entries, controller_path):
        pythonpath_entries.insert(0, controller_path_str)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    if not _contains_path(sys.path, controller_path):
        sys.path.insert(0, controller_path_str)

    return controller_path
