"""RobotBackend client for a lockstep ZeroMQ simulator server."""

from __future__ import annotations

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


class ZmqBackendClient:
    """Synchronous request/reply client; each call completes before the next is sent."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.linger = 0
        self._socket.sndtimeo = timeout_ms
        self._socket.rcvtimeo = timeout_ms
        self._socket.connect(endpoint)
        self._seq = 0
        self._sim_time: float | None = None
        self._closed = False

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

    def stop(self) -> None:
        if self._closed:
            return
        try:
            self._request("shutdown", {})
        finally:
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._socket.close()
            self._context.term()
            self._closed = True

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ZeroMQ backend client is closed.")
        message = make_message(operation, self._seq, payload, self._sim_time)
        try:
            self._socket.send(encode_message(message))
            response = decode_message(self._socket.recv())
        except zmq.Again as exc:
            raise TimeoutError(
                f"Simulator did not reply to {operation!r} within the configured timeout."
            ) from exc
        if response["seq"] != self._seq:
            raise ProtocolError(
                f"Simulator replied with sequence {response['seq']}; expected {self._seq}."
            )
        self._seq += 1
        if response["error"] is not None:
            raise RuntimeError(f"Simulator rejected {operation!r}: {response['error']}")
        self._sim_time = response["sim_time"]
        return response

    @staticmethod
    def _expect(message: dict[str, Any], key: str, expected_type: type) -> Any:
        value = message["payload"].get(key)
        if not isinstance(value, expected_type):
            raise ProtocolError(f"Simulator response is missing a valid {key!r} value.")
        return value
