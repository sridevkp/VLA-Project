"""Webots controller entry point for the lockstep simulator server."""

from __future__ import annotations

import logging
import os

from vlarobot.backends.webots.backend import WebotsBackend
from vlarobot.backends.zmq.server import (
    DEFAULT_ENDPOINT,
    SimServer,
    configured_port,
    endpoint_for_port,
)
from vlarobot.env import configure_webots_environment


def run_webots_sim_server(endpoint: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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
