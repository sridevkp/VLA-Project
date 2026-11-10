"""RobotBackend client for a lockstep ZeroMQ simulator server."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

import zmq

from vlarobot.core.types import (
    Action,
    ApplyResult,
    GroundTruth,
    Observation,
    ScenarioSpec,
)
from vlarobot.core.wire import ProtocolError, decode_message, encode_message, make_message

DEFAULT_REQUEST_TIMEOUT_MS = 30_000
_LOGGER = logging.getLogger(__name__)

# Per-request-type default timeouts (ms). Export runs axis probes and may take minutes.
REQUEST_TIMEOUTS_MS: dict[str, int] = {
    "reset": 30_000,
    "step": 30_000,
    "apply": 30_000,
    "get_gt": 30_000,
    "ping": 5_000,
    "shutdown": 10_000,
    "export_kinematics": 300_000,
}


def _client_timings_enabled() -> bool:
    """Transport timing is opt-in (``VLAROBOT_BACKEND_TIMINGS=1``)."""
    return os.environ.get("VLAROBOT_BACKEND_TIMINGS") == "1"


class ZmqBackendClient:
    """Synchronous request/reply client; each call completes before the next is sent."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")
        self._endpoint = endpoint
        self._default_timeout_ms = int(timeout_ms)
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._seq = 0
        self._sim_time: float | None = None
        self._closed = False
        self._timings_enabled = _client_timings_enabled()
        self._timings: dict[str, float] = {}
        self._session_id: dict[str, object] | None = None
        self._expected_session_id: dict[str, object] | None = None
        self._ensure_socket(self._default_timeout_ms)

    # ------------------------------------------------------------------
    # Socket lifecycle (lazy-pirate: reset REQ state after a timeout).
    # ------------------------------------------------------------------

    def _ensure_socket(self, timeout_ms: int) -> zmq.Socket:
        if self._closed:
            raise RuntimeError("ZeroMQ backend client is closed.")
        if self._socket is None:
            sock = self._context.socket(zmq.REQ)
            sock.linger = 0
            sock.connect(self._endpoint)
            self._socket = sock
        self._socket.sndtimeo = int(timeout_ms)
        self._socket.rcvtimeo = int(timeout_ms)
        return self._socket

    def _reset_socket(self, timeout_ms: int | None = None) -> None:
        """Close and recreate the REQ socket so a timed-out request cannot poison state."""
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None
        if not self._closed:
            self._ensure_socket(timeout_ms or self._default_timeout_ms)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        response = self._request("reset", {"scenario": scenario, "seed": seed})
        return self._expect(response, "observation", Observation)

    def step(self, n_steps: int = 1) -> Observation:
        response = self._request("step", {"n_steps": n_steps})
        return self._expect(response, "observation", Observation)

    def apply(self, action: Action) -> ApplyResult:
        response = self._request("apply", {"action": action})
        return self._expect(response, "result", ApplyResult)

    def get_ground_truth(self) -> GroundTruth:
        response = self._request("get_gt", {})
        return self._expect(response, "ground_truth", GroundTruth)

    def ping(self) -> dict[str, Any]:
        return self._request("ping", {})

    def export_kinematics(
        self, action: str = "poll", timeout_ms: int | None = None
    ) -> dict[str, Any]:
        """Chunked export request (ADR-0020 addendum): start/poll/status/cancel."""
        response = self._request("export_kinematics", {"action": action}, timeout_ms=timeout_ms)
        return response.get("payload", {}).get("export", {})

    @property
    def session_id(self) -> dict[str, object] | None:
        return self._session_id

    def expect_session(self, session_id: dict[str, object]) -> None:
        """Record the server session identity this client is bound to."""
        self._expected_session_id = dict(session_id)

    def stop(self) -> None:
        if self._closed:
            return
        try:
            self._request("shutdown", {}, timeout_ms=REQUEST_TIMEOUTS_MS["shutdown"])
        except Exception:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            if self._socket is not None:
                try:
                    self._socket.close(0)
                except Exception:
                    pass
                self._socket = None
            self._context.term()
            self._closed = True

    def timings(self) -> dict[str, float]:
        """Copy of the opt-in transport counters; empty unless ``VLAROBOT_BACKEND_TIMINGS=1``."""
        return dict(self._timings)

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _track(self, key: str, mark: float | None) -> None:
        if mark is not None:
            self._timings[key] = self._timings.get(key, 0.0) + (perf_counter() - mark)

    def _timeout_for(self, operation: str, timeout_ms: int | None) -> int:
        if timeout_ms is not None:
            return int(timeout_ms)
        return REQUEST_TIMEOUTS_MS.get(operation, self._default_timeout_ms)

    def _request(
        self, operation: str, payload: dict[str, Any], timeout_ms: int | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ZeroMQ backend client is closed.")
        effective_timeout = self._timeout_for(operation, timeout_ms)
        socket = self._ensure_socket(effective_timeout)
        message = make_message(operation, self._seq, payload, self._sim_time)
        timed = self._timings_enabled
        mark = perf_counter() if timed else None
        encoded = encode_message(message)
        self._track("encode_s", mark)
        started = perf_counter()
        try:
            mark = perf_counter() if timed else None
            socket.send(encoded)
            self._track("send_s", mark)
            mark = perf_counter() if timed else None
            raw = socket.recv()
            self._track("recv_s", mark)
            mark = perf_counter() if timed else None
            response = decode_message(raw)
            self._track("decode_s", mark)
        except zmq.Again as exc:
            elapsed = perf_counter() - started
            _LOGGER.error(
                "Simulator did not reply to %r within %d ms (elapsed %.3fs, endpoint %s)",
                operation,
                effective_timeout,
                elapsed,
                self._endpoint,
            )
            # Lazy-pirate: drop the poisoned REQ socket so later calls can proceed.
            self._reset_socket(self._default_timeout_ms)
            raise TimeoutError(
                f"Simulator did not reply to {operation!r} within {effective_timeout} ms "
                f"(elapsed {elapsed:.3f}s)."
            ) from exc
        except zmq.ZMQError as exc:
            self._reset_socket(self._default_timeout_ms)
            raise RuntimeError(f"ZMQ error during {operation!r}: {exc}") from exc

        if timed:
            self._timings["requests"] = self._timings.get("requests", 0.0) + 1.0
            observation = response.get("payload", {}).get("observation")
            rgb = getattr(observation, "rgb", None)
            if rgb is not None:
                self._timings["image_transfer_bytes"] = self._timings.get(
                    "image_transfer_bytes", 0.0
                ) + float(rgb.nbytes)
        if response["seq"] != self._seq:
            raise ProtocolError(
                f"Simulator replied with sequence {response['seq']}; expected {self._seq}."
            )
        self._seq += 1
        if response["error"] is not None:
            raise RuntimeError(f"Simulator rejected {operation!r}: {response['error']}")

        # Session identity: ping advertises it; once known, every reply must match.
        payload_out = response.get("payload", {})
        session = payload_out.get("session_id") if isinstance(payload_out, dict) else None
        if isinstance(session, dict):
            self._session_id = session
            if self._expected_session_id is None:
                self._expected_session_id = dict(session)
        if self._expected_session_id is not None and self._session_id is not None:
            if self._session_id != self._expected_session_id:
                raise RuntimeError(
                    "Simulator session identity changed "
                    f"(expected {self._expected_session_id!r}, got {self._session_id!r}); "
                    "a stale server may be answering. Use `vlarobot.cli doctor`."
                )

        self._sim_time = response["sim_time"]
        return response

    @staticmethod
    def _expect(message: dict[str, Any], key: str, expected_type: type) -> Any:
        value = message.get("payload", {}).get(key)
        if not isinstance(value, expected_type):
            raise ProtocolError(f"Simulator response is missing a valid {key!r} value.")
        return value
