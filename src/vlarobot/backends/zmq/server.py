"""Lockstep ZeroMQ REP server for a robot backend."""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any

import zmq

from vlarobot.core.interfaces import RobotBackend
from vlarobot.core.types import Action, ScenarioSpec
from vlarobot.core.wire import ProtocolError, decode_message, encode_message, make_message

DEFAULT_PORT = 5555
DEFAULT_HOST = "127.0.0.1"
_LOGGER = logging.getLogger(__name__)


def configured_port() -> int:
    """Port from ``VLAROBOT_ZMQ_PORT`` env or config default; fails loudly if invalid."""
    raw = os.environ.get("VLAROBOT_ZMQ_PORT")
    if raw is None or raw == "":
        # Lazy import to avoid hard dependency on webots config in pure-zmq contexts.
        try:
            from vlarobot.backends.webots.config import ZMQ_PORT

            return int(ZMQ_PORT)
        except Exception:
            return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"VLAROBOT_ZMQ_PORT must be an integer, got {raw!r}") from exc
    if not (1 <= port <= 65535):
        raise RuntimeError(f"VLAROBOT_ZMQ_PORT must be in [1,65535], got {port}")
    return port


def endpoint_for_port(port: int, host: str = DEFAULT_HOST) -> str:
    return f"tcp://{host}:{port}"


DEFAULT_ENDPOINT = endpoint_for_port(configured_port())


def session_identity() -> dict[str, object]:
    """Identity of this simulator process, carried in ping replies (ADR-0020 addendum)."""
    return {
        "pid": os.getpid(),
        "start_time": time.time(),
        "nonce": secrets.token_hex(8),
    }


def endpoint_for_port(port: int, host: str = "127.0.0.1") -> str:
    return f"tcp://{host}:{port}"


class SimServer:
    """Serve one request at a time and advance simulation only on explicit requests."""

    def __init__(self, backend: RobotBackend) -> None:
        self.backend = backend
        self._sim_time: float | None = None
        self._shutdown = False
        self.endpoint: str | None = None
        self.session_id = session_identity()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and dispatch one decoded protocol request."""
        seq = request["seq"]
        operation = request["type"]
        payload = request["payload"]
        try:
            if operation == "ping":
                response_payload = {"status": "ok", "session_id": self.session_id}
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
            elif operation == "export_kinematics":
                action = payload.get("action", "poll")
                if not isinstance(action, str):
                    raise ProtocolError("export_kinematics action must be a string.")
                exporter = getattr(self.backend, "export_kinematics", None)
                if exporter is None:
                    raise RuntimeError("Backend does not support export_kinematics.")
                response_payload = {"export": exporter(action)}
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
        except zmq.ZMQError as exc:
            socket.close()
            context.term()
            raise RuntimeError(
                f"Failed to bind simulator REP socket to {endpoint!r}: {exc}. "
                "Is another simulator still running? Use `vlarobot.cli doctor` to inspect."
            ) from exc
        # Resolve ephemeral ports (e.g. tcp://127.0.0.1:* -> actual address).
        bound = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.endpoint = bound
        _LOGGER.info("Simulator server listening on %s (session %s)", bound, self.session_id)
        try:
            while not self._shutdown:
                try:
                    raw = socket.recv()
                except zmq.ZMQError as exc:
                    if self._shutdown:
                        break
                    raise RuntimeError(f"Simulator socket recv failed: {exc}") from exc
                try:
                    request = decode_message(raw)
                    response = self.handle(request)
                    socket.send(encode_message(response))
                except ProtocolError as exc:
                    # Decode/validation failed before we know seq/type reliably.
                    _LOGGER.warning("Protocol error: %s", exc)
                    response = make_message("error", 0, error=str(exc))
                    try:
                        socket.send(encode_message(response))
                    except zmq.ZMQError:
                        pass
        finally:
            socket.close()
            context.term()
