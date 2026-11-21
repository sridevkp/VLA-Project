"""Lockstep ZeroMQ REP server for a robot backend."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
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
        self._status_lock = threading.Lock()
        self._current_request: str | None = None
        self._current_request_started: float | None = None
        self._last_completed: dict[str, object] | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._status_path: Path | None = None

    def _set_busy(self, operation: str | None) -> None:
        with self._status_lock:
            self._current_request = operation
            self._current_request_started = time.perf_counter() if operation else None

    def _write_status(self, extra: dict[str, object] | None = None) -> None:
        path = self._status_path
        if path is None:
            return
        with self._status_lock:
            op = self._current_request
            started = self._current_request_started
            age = (time.perf_counter() - started) if (op and started is not None) else None
            payload: dict[str, object] = {
                "session_id": self.session_id,
                "endpoint": self.endpoint,
                "busy": op is not None,
                "request": op,
                "request_age_s": age,
                "sim_time": self._sim_time,
                "last_completed": self._last_completed,
                "updated_at": time.time(),
            }
            if extra:
                payload.update(extra)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    def _watchdog_loop(self, interval_s: float = 2.0, warn_after_s: float = 5.0) -> None:
        while not self._watchdog_stop.wait(interval_s):
            with self._status_lock:
                op = self._current_request
                started = self._current_request_started
            if op and started is not None:
                age = time.perf_counter() - started
                if age >= warn_after_s:
                    _LOGGER.warning(
                        "Simulator busy: request=%s age=%.3fs endpoint=%s", op, age, self.endpoint
                    )
                self._write_status()
            else:
                self._write_status()

    def _diagnostic_step_capability(self) -> tuple[str, int] | None:
        capability = getattr(self.backend, "diagnostic_step_capability", None)
        if (
            isinstance(capability, tuple)
            and capability == ("webots", 1)
            and type(capability[1]) is int
            and callable(getattr(self.backend, "diagnostic_step", None))
        ):
            return capability
        return None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and dispatch one decoded protocol request."""
        seq = request["seq"]
        operation = request["type"]
        response_type = operation
        guarded = operation == "diagnostic_session_request"
        payload = request["payload"]
        started = time.perf_counter()
        self._set_busy(operation)
        self._write_status()
        _LOGGER.info("request_start op=%s seq=%s sim_time=%s", operation, seq, self._sim_time)
        try:
            if guarded:
                if self._diagnostic_step_capability() is None:
                    raise ProtocolError("diagnostic_session_unsupported_backend")
                if payload.get("session_id") != self.session_id:
                    raise ProtocolError("diagnostic_session_identity_mismatch")
                operation = payload.get("operation")
                payload = payload.get("arguments")
                if operation not in ("reset", "apply", "diagnostic_step", "get_gt", "shutdown"):
                    raise ProtocolError("diagnostic_session_unsupported_operation")
                if not isinstance(payload, dict):
                    raise ProtocolError("diagnostic_session_invalid_arguments")
            if operation == "ping":
                telemetry_enabled = bool(
                    getattr(self.backend, "export_sim_kinematics_enabled", None)
                )
                if not telemetry_enabled:
                    # Webots backend exposes config flag; fall back to settings attribute if present
                    try:
                        from vlarobot.backends.webots.config import EXPORT_SIM_KINEMATICS

                        telemetry_enabled = bool(EXPORT_SIM_KINEMATICS)
                    except Exception:
                        telemetry_enabled = False
                response_payload = {
                    "status": "ok",
                    "session_id": self.session_id,
                    "kinematics_telemetry_enabled": telemetry_enabled,
                }
                capability = self._diagnostic_step_capability()
                if capability is not None:
                    backend_kind, version = capability
                    response_payload["capabilities"] = {"diagnostic_step": version}
                    response_payload["diagnostic_backend"] = backend_kind
                world_hash = os.environ.get("VLAROBOT_WORLD_SHA256")
                if world_hash:
                    response_payload["world_sha256"] = world_hash
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
            elif operation == "diagnostic_step":
                executor = getattr(self.backend, "diagnostic_step", None)
                if not callable(executor):
                    raise RuntimeError("Backend does not support diagnostic step outcomes.")
                response_payload = executor()
                observation = response_payload.get("observation")
                if observation is not None:
                    self._sim_time = observation.t
            elif operation == "apply":
                action = payload.get("action")
                if not isinstance(action, Action):
                    raise ProtocolError("apply action must be a supported Action.")
                result = self.backend.apply(action)
                response_payload = {"result": result}
            elif operation == "get_gt":
                response_payload = {"ground_truth": self.backend.get_ground_truth()}
            elif operation == "inspect":
                from vlarobot.backends.read_only_state import capture_read_only_state

                snapshot = capture_read_only_state(self.backend)
                if not isinstance(snapshot, dict) or not {
                    "observation",
                    "ground_truth",
                    "identity",
                }.issubset(snapshot):
                    raise RuntimeError(
                        "Backend read-only inspection returned an incomplete snapshot."
                    )
                response_payload = dict(snapshot)
            elif operation == "inspect_geometry":
                from vlarobot.backends.read_only_geometry import capture_read_only_geometry

                if "table_targets" in payload:
                    table_targets = payload["table_targets"]
                    if not isinstance(table_targets, dict):
                        raise RuntimeError("inspect_geometry table_targets must be a mapping")
                    snapshot = capture_read_only_geometry(self.backend, table_targets)
                else:
                    snapshot = capture_read_only_geometry(self.backend)
                if not isinstance(snapshot, dict) or not {
                    "observation",
                    "ground_truth",
                    "identity",
                    "geometry",
                }.issubset(snapshot):
                    raise RuntimeError(
                        "Backend read-only geometry inspection returned an incomplete snapshot."
                    )
                response_payload = dict(snapshot)
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
            dur = time.perf_counter() - started
            _LOGGER.info(
                "request_end op=%s seq=%s duration_s=%.4f sim_time=%s",
                operation,
                seq,
                dur,
                self._sim_time,
            )
            if dur > 2.0:
                _LOGGER.warning("slow_request op=%s seq=%s duration_s=%.4f", operation, seq, dur)
            with self._status_lock:
                self._last_completed = {"op": operation, "seq": seq, "duration_s": dur, "ok": True}
            if guarded:
                response_payload = {**response_payload, "session_id": self.session_id}
            return make_message(response_type, seq, response_payload, sim_time=self._sim_time)
        except (ProtocolError, TypeError, ValueError, RuntimeError) as exc:
            dur = time.perf_counter() - started
            _LOGGER.warning(
                "request_error op=%s seq=%s duration_s=%.4f error=%s", operation, seq, dur, exc
            )
            with self._status_lock:
                self._last_completed = {
                    "op": operation,
                    "seq": seq,
                    "duration_s": dur,
                    "ok": False,
                    "error": str(exc),
                }
            return make_message(
                response_type,
                seq,
                error=str(exc),
                sim_time=self._sim_time,
            )
        finally:
            self._set_busy(None)
            self._write_status()

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
        # Status file for out-of-band liveness diagnosis (watchdog updates request/age).
        try:
            port = int(bound.rsplit(":", 1)[-1])
        except Exception:
            port = 0
        self._status_path = Path.cwd() / "runs" / "campaign-2026-11-11" / f"sim-status-{port}.json"
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="sim-watchdog", daemon=True
        )
        self._watchdog_thread.start()
        _LOGGER.info(
            "Simulator server listening on %s (session %s status=%s)",
            bound,
            self.session_id,
            self._status_path,
        )
        self._write_status({"event": "listening"})
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
            self._watchdog_stop.set()
            if self._watchdog_thread is not None:
                self._watchdog_thread.join(timeout=2.0)
            self._write_status({"event": "stopped"})
            socket.close()
            context.term()
