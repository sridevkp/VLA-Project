"""Lockstep ZeroMQ REP server for a robot backend."""

from __future__ import annotations

import logging
from typing import Any

import zmq

from vlarobot.core.interfaces import RobotBackend
from vlarobot.core.types import Action, ScenarioSpec
from vlarobot.core.wire import ProtocolError, decode_message, encode_message, make_message

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
_LOGGER = logging.getLogger(__name__)


class SimServer:
    """Serve one request at a time and advance simulation only on explicit requests."""

    def __init__(self, backend: RobotBackend) -> None:
        self.backend = backend
        self._sim_time: float | None = None
        self._shutdown = False
        self.endpoint: str | None = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and dispatch one decoded protocol request."""
        seq = request["seq"]
        operation = request["type"]
        payload = request["payload"]
        try:
            if operation == "ping":
                response_payload: dict[str, Any] = {"status": "ok"}
            elif operation == "reset":
                scenario = payload.get("scenario", ScenarioSpec())
                seed = payload.get("seed", 0)
                if not isinstance(scenario, ScenarioSpec):
                    raise ProtocolError("reset scenario must be a ScenarioSpec.")
                if not isinstance(seed, int):
                    raise ProtocolError("reset seed must be an integer.")
                observation = self.backend.reset(scenario, seed)
                self._sim_time = observation.t
                response_payload = {"observation": observation}
            elif operation == "step":
                n_steps = payload.get("n_steps", 1)
                if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
                    raise ProtocolError("step n_steps must be a positive integer.")
                observation = self.backend.step(n_steps)
                self._sim_time = observation.t
                response_payload = {"observation": observation}
            elif operation == "apply":
                action = payload.get("action")
                if not isinstance(action, Action):
                    raise ProtocolError("apply action must be a supported Action.")
                result = self.backend.apply(action)
                response_payload = {"result": result}
            elif operation == "get_gt":
                response_payload = {"ground_truth": self.backend.get_ground_truth()}
            elif operation == "shutdown":
                self.backend.stop()
                self._shutdown = True
                response_payload = {"status": "stopped"}
            else:
                raise ProtocolError(f"Unknown simulator operation: {operation!r}.")
            return make_message(operation, seq, response_payload, sim_time=self._sim_time)
        except (ProtocolError, TypeError, ValueError, RuntimeError) as exc:
            _LOGGER.warning("Simulator request %s failed: %s", operation, exc)
            return make_message(
                operation,
                seq,
                error=str(exc),
                sim_time=self._sim_time,
            )

    def serve_forever(self, endpoint: str = DEFAULT_ENDPOINT) -> None:
        """Bind the REP socket and dispatch requests until a shutdown request arrives."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.linger = 0
        try:
            socket.bind(endpoint)
            self.endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            _LOGGER.info("Simulator REP server listening at %s", endpoint)
            while not self._shutdown:
                raw_request = socket.recv()
                try:
                    request = decode_message(raw_request)
                    response = self.handle(request)
                except ProtocolError as exc:
                    _LOGGER.warning("Rejected malformed simulator request: %s", exc)
                    response = make_message(
                        "error",
                        0,
                        error=str(exc),
                        sim_time=self._sim_time,
                    )
                socket.send(encode_message(response))
        finally:
            socket.close()
            context.term()
