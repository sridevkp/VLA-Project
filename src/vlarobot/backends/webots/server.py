"""Webots controller entry point for the lockstep simulator server."""

from __future__ import annotations

import faulthandler
import logging
import os
from pathlib import Path

from vlarobot.backends.webots.backend import WebotsBackend
from vlarobot.backends.zmq.server import (
    DEFAULT_ENDPOINT,
    SimServer,
    configured_port,
    endpoint_for_port,
)
from vlarobot.env import configure_webots_environment

# Kept open for the process lifetime so faulthandler can dump fatal-signal tracebacks.
_FAULT_FILE = None


def _diagnostics_file() -> Path | None:
    """Per-episode controller diagnostics path (set by the episode runner)."""
    raw = os.environ.get("VLAROBOT_CONTROLLER_STDERR_PATH")
    if not raw:
        return None
    try:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def run_webots_sim_server(endpoint: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    global _FAULT_FILE
    diagnostics_path = _diagnostics_file()
    if diagnostics_path is not None:
        # Crash forensics (ADR-0022): mirror controller logs to a per-episode file and dump
        # fatal-signal tracebacks (SIGSEGV/SIGABRT/...) there instead of losing them in the
        # Webots console pipe.
        try:
            file_handler = logging.FileHandler(diagnostics_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logging.getLogger().addHandler(file_handler)
        except OSError:
            file_handler = None
        try:
            _FAULT_FILE = diagnostics_path.open("a", encoding="utf-8")
            faulthandler.enable(file=_FAULT_FILE)
        except (OSError, RuntimeError):
            _FAULT_FILE = None
            faulthandler.enable()
    else:
        faulthandler.enable()
    configure_webots_environment()
    if endpoint is None:
        # Prefer explicit env so launchers (e.g. run-episodes) can pick a free port.
        endpoint = (
            endpoint_for_port(configured_port())
            if os.environ.get("VLAROBOT_ZMQ_PORT")
            else DEFAULT_ENDPOINT
        )
    SimServer(WebotsBackend()).serve_forever(endpoint)


def main() -> None:
    run_webots_sim_server()


if __name__ == "__main__":
    main()
