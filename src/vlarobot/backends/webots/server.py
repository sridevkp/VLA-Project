"""Webots controller entry point for the lockstep simulator server."""

from __future__ import annotations

import logging

from vlarobot.backends.webots.backend import WebotsBackend
from vlarobot.backends.zmq.server import DEFAULT_ENDPOINT, SimServer
from vlarobot.env import configure_webots_environment


def run_webots_sim_server(endpoint: str = DEFAULT_ENDPOINT) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_webots_environment()
    SimServer(WebotsBackend()).serve_forever(endpoint)


def main() -> None:
    run_webots_sim_server()


if __name__ == "__main__":
    main()
