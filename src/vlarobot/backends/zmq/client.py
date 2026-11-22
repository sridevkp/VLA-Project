"""RobotBackend client for a lockstep ZeroMQ simulator server."""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping
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

# Per-request-type default timeouts (ms). Config-overridable; no hard-coded callers
# should rely on these values. Justification (bench-steps, 200 steps, server with
# RGB + kinematics_telemetry enabled): ~447 ms/step measured => a single lockstep
# step+observation RPC is well under 5 s in steady state, but reset/export span many
# steps and cold Webots startup can be slow, so reset/export/startup waits are longer.
REQUEST_TIMEOUTS_MS: dict[str, int] = {
    "reset": 120_000,  # many settle steps + possible cold-start work; not a fast RPC
    "step": 30_000,  # 1 step ~0.45 s measured; headroom for load/diagnostics
    "apply": 30_000,
    "get_gt": 30_000,
    "inspect_geometry": 30_000,  # read-only supervisor export; no physics or actuation
    "ping": 5_000,  # lightweight liveness/session probe
    "shutdown": 10_000,
    "export_kinematics": 300_000,  # per-poll chunk (one joint probe); bounded by probe caps
}


def _client_timings_enabled() -> bool:
    """Transport timing is opt-in (``VLAROBOT_BACKEND_TIMINGS=1``)."""
    return os.environ.get("VLAROBOT_BACKEND_TIMINGS") == "1"


# Bounded read-only diagnostic override: the default 30 s inspect_geometry ceiling is close to the
# ~13 s measured export time (Packet C timed out at 30 s). A separately authorized diagnostic may
# raise ONLY this operation's deadline, with strict validation and no change to the shipped default.
INSPECT_GEOMETRY_TIMEOUT_ENV = "VLAROBOT_INSPECT_GEOMETRY_TIMEOUT_MS"


def inspect_geometry_timeout_override() -> int | None:
    """Return the validated env override for inspect_geometry, or None when unset.

    Invalid values fail loudly instead of silently falling back to the default.
    """
    raw = os.environ.get(INSPECT_GEOMETRY_TIMEOUT_ENV)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{INSPECT_GEOMETRY_TIMEOUT_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{INSPECT_GEOMETRY_TIMEOUT_ENV} must be a positive integer")
    return value


class ZmqBackendClient:
    """Synchronous request/reply client; each call completes before the next is sent."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
        request_timeouts_ms: dict[str, int] | None = None,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")
        self._endpoint = endpoint
        self._default_timeout_ms = int(timeout_ms)
        self._request_timeouts_ms = {**REQUEST_TIMEOUTS_MS, **(request_timeouts_ms or {})}
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._seq = 0
        self._sim_time: float | None = None
        self._closed = False
        self._timings_enabled = _client_timings_enabled()
        self._timings: dict[str, float] = {}
        self._session_id: dict[str, object] | None = None
        self._expected_session_id: dict[str, object] | None = None
        self._diagnostic_session: dict[str, object] | None = None
        self._diagnostic_failed = False
        self._ever_responsive = False
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

    def diagnostic_step(self) -> dict[str, Any]:
        """Request one diagnostic tick exactly once."""
        return self._request("diagnostic_step", {}, timeout_ms=self._timeout_for("step", None))[
            "payload"
        ]

    def apply(self, action: Action) -> ApplyResult:
        response = self._request("apply", {"action": action})
        return self._expect(response, "result", ApplyResult)

    def get_ground_truth(self) -> GroundTruth:
        response = self._request("get_gt", {})
        return self._expect(response, "ground_truth", GroundTruth)

    def inspect_fingers(self) -> dict[str, Any]:
        """Read raw finger-device identity and coordinates from the pinned diagnostic session."""
        if self._diagnostic_session is None:
            raise ProtocolError("finger inspection requires a pinned diagnostic session")
        return self._request("inspect_fingers", {})["payload"]

    def inspect(self) -> dict[str, Any]:
        """Read a backend snapshot without applying actions or advancing simulation."""
        response = self._request("inspect", {})
        payload = response.get("payload", {})
        if not isinstance(payload, dict) or not {
            "observation",
            "ground_truth",
            "identity",
        }.issubset(payload):
            raise ValueError("inspect reply omitted observation, ground truth or identity")
        return payload

    def inspect_geometry(self, table_targets: Mapping[str, object] | None = None) -> dict[str, Any]:
        """Read the supervisor geometry export without advancing simulation.

        ``table_targets`` is a selector-only descriptor; expected geometry is never sent to the
        simulator side. Omitting it preserves the original direct boundingObject export.
        """
        if table_targets is None:
            payload: dict[str, Any] = {}
        elif isinstance(table_targets, Mapping):
            payload = {"table_targets": dict(table_targets)}
        else:
            raise TypeError("table_targets must be a mapping or None")
        response = self._request(
            "inspect_geometry", payload, timeout_ms=self._timeout_for("inspect_geometry", None)
        )
        payload = response.get("payload", {})
        if not isinstance(payload, dict) or not {
            "observation",
            "ground_truth",
            "identity",
            "geometry",
        }.issubset(payload):
            raise ValueError("inspect_geometry reply omitted state, identity or geometry")
        return payload

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

    def require_owned_diagnostic_session(
        self, *, session_id: dict[str, object], world_sha256: str
    ) -> None:
        """Require an independently established local controller identity and world pin."""
        if self._diagnostic_session is not None or self._diagnostic_failed:
            raise ProtocolError("diagnostic capability requires a fresh client")
        if (
            type(session_id.get("pid")) is not int
            or type(session_id.get("start_time")) not in (int, float)
            or not isinstance(session_id.get("nonce"), str)
            or not isinstance(world_sha256, str)
            or re.fullmatch("[0-9a-f]{64}", world_sha256) is None
        ):
            self._diagnostic_failed = True
            raise ProtocolError("invalid independently attested diagnostic pins")
        self.expect_session(session_id)
        self._require_diagnostic_session("webots", world_sha256)

    def require_diagnostic_session(self, *, expected_backend: str) -> None:
        """Validate capability before any campaign mutation and pin its server session."""
        self._require_diagnostic_session(expected_backend, None)

    def _require_diagnostic_session(self, expected_backend: str, world_sha256: str | None) -> None:
        if self._diagnostic_session is not None or self._diagnostic_failed:
            raise ProtocolError("diagnostic capability requires a fresh client")
        # Fail closed even when the ping or its decoding raises.
        self._diagnostic_failed = True
        reply = self.ping()
        payload = reply.get("payload")
        if (
            type(reply.get("protocol_version")) is not int
            or reply["protocol_version"] != 3
            or reply.get("type") != "ping"
            or not isinstance(payload, dict)
            or payload.get("status") != "ok"
        ):
            raise ProtocolError("diagnostic capability requires a successful protocol-v3 ping")
        session = payload.get("session_id")
        if not isinstance(session, dict):
            raise ProtocolError("diagnostic capability requires a valid session identity")
        pid, started, nonce = (session.get(name) for name in ("pid", "start_time", "nonce"))
        if (
            type(pid) is not int
            or pid <= 0
            or type(started) not in (int, float)
            or not math.isfinite(started)
            or started <= 0
            or not isinstance(nonce, str)
            or not nonce.strip()
        ):
            raise ProtocolError("diagnostic capability requires a valid session identity")
        if self._expected_session_id is not None and session != self._expected_session_id:
            raise ProtocolError("diagnostic controller differs from the expected session")
        if world_sha256 is not None and payload.get("world_sha256") != world_sha256:
            raise ProtocolError("diagnostic controller world hash mismatch")
        capabilities = payload.get("capabilities")
        version = capabilities.get("diagnostic_step") if isinstance(capabilities, dict) else None
        if type(version) is not int or version != 1:
            raise ProtocolError("diagnostic_step capability version 1 is required")
        if payload.get("diagnostic_backend") != expected_backend:
            raise ProtocolError(f"diagnostic backend must advertise {expected_backend!r}")
        self._diagnostic_session = dict(session)
        self._diagnostic_failed = False

    def shutdown_diagnostic_session(self, *, session_id: dict[str, object]) -> dict[str, Any]:
        """Return an owned guarded shutdown acknowledgement or raise, and close the socket."""
        try:
            if self._diagnostic_session is None or self._diagnostic_session != session_id:
                raise ProtocolError("shutdown requires the exact pinned diagnostic session")
            response = self._request("shutdown", {})
            if response["payload"].get("status") != "stopped":
                raise ProtocolError("diagnostic shutdown acknowledgement missing")
            return response["payload"]
        finally:
            self.close()

    def stop(self) -> None:
        if self._closed:
            return
        try:
            self._request("shutdown", {}, timeout_ms=self._timeout_for("shutdown", None))
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
        if operation == "inspect_geometry":
            override = inspect_geometry_timeout_override()
            if override is not None:
                return override
        return int(self._request_timeouts_ms.get(operation, self._default_timeout_ms))

    def _request(
        self, operation: str, payload: dict[str, Any], timeout_ms: int | None = None
    ) -> dict[str, Any]:
        if operation == "ping":
            return self._exchange(operation, payload, timeout_ms)
        if self._diagnostic_failed:
            raise ProtocolError("diagnostic session is closed after a failed operation")
        session = self._diagnostic_session
        if session is None:
            return self._exchange(operation, payload, timeout_ms)
        if operation not in (
            "reset",
            "apply",
            "diagnostic_step",
            "get_gt",
            "shutdown",
            "inspect_fingers",
        ):
            raise ProtocolError(f"operation {operation!r} is unavailable in a diagnostic session")

        # A server predating the gate rejects this new request type before dispatch.
        self._diagnostic_failed = True
        response = self._exchange(
            "diagnostic_session_request",
            {"session_id": session, "operation": operation, "arguments": payload},
            self._timeout_for(operation, timeout_ms),
        )
        if (
            response["type"] != "diagnostic_session_request"
            or response["payload"].get("session_id") != session
        ):
            raise ProtocolError("diagnostic reply session identity or type mismatch")
        if operation == "reset":
            self._expect(response, "observation", Observation)
        elif operation == "apply":
            result = self._expect(response, "result", ApplyResult)
            if type(result.accepted) is not bool:
                raise ProtocolError("diagnostic apply reply has an invalid verdict")
        elif operation == "get_gt":
            self._expect(response, "ground_truth", GroundTruth)
        elif operation == "diagnostic_step":
            step = response["payload"]
            ticks = step.get("confirmed_ticks")
            if (
                type(ticks) is not int
                or ticks != 1
                or not isinstance(step.get("observation"), Observation)
            ):
                # Preserve the authoritative partial reply for feedback evidence.
                return response
        self._diagnostic_failed = False
        return response

    def _exchange(
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
            self._ever_responsive = True
        except zmq.Again as exc:
            elapsed = perf_counter() - started
            state = (
                "busy/stalled (was responsive)" if self._ever_responsive else "not up/unreachable"
            )
            _LOGGER.error(
                "Simulator did not reply to %r within %d ms (elapsed %.3fs, endpoint %s, state=%s)",
                operation,
                effective_timeout,
                elapsed,
                self._endpoint,
                state,
            )
            # Lazy-pirate: drop the poisoned REQ socket so later calls can proceed.
            self._reset_socket(self._default_timeout_ms)
            raise TimeoutError(
                f"Simulator did not reply to {operation!r} within {effective_timeout} ms "
                f"(elapsed {elapsed:.3f}s, {state}, endpoint {self._endpoint})."
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
