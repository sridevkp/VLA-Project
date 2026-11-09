from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import SimServer
from vlarobot.core.types import GripperCmd, ScenarioSpec


def test_zmq_backend_client_and_server_round_trip() -> None:
    server = SimServer(FakeBackend())
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"endpoint": "tcp://127.0.0.1:*"},
        daemon=True,
    )
    server_thread.start()
    deadline = time.monotonic() + 5.0
    while server.endpoint is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.endpoint is not None, "REP server did not bind in time"

    client = ZmqBackendClient(server.endpoint, timeout_ms=5_000)
    try:
        assert client.ping()["payload"]["status"] == "ok"
        initial = client.reset(ScenarioSpec(), seed=11)
        assert initial.t == 0.0
        assert client.apply(GripperCmd(opening=0.02)).accepted
        observation = client.step(2)
        assert observation.t == pytest.approx(0.064)
        assert observation.robot.gripper_opening == 0.02
        assert client.get_ground_truth().object_poses == {}
        client.close()

        reconnected_client = ZmqBackendClient(server.endpoint, timeout_ms=5_000)
        try:
            assert reconnected_client.ping()["payload"]["status"] == "ok"
        finally:
            reconnected_client.stop()
    finally:
        client.close()
        server_thread.join(timeout=5.0)
        assert not server_thread.is_alive()


def _wait_for_endpoint(server: SimServer, timeout_s: float = 5.0) -> str:
    deadline = time.monotonic() + timeout_s
    while server.endpoint is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.endpoint is not None, "REP server did not bind in time"
    return server.endpoint


def test_zmq_client_timings_are_off_by_default() -> None:
    server = SimServer(FakeBackend())
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"endpoint": "tcp://127.0.0.1:*"},
        daemon=True,
    )
    server_thread.start()
    endpoint = _wait_for_endpoint(server)
    try:
        client = ZmqBackendClient(endpoint, timeout_ms=5_000)
        try:
            client.ping()
            client.reset(ScenarioSpec(), seed=1)
            client.step(2)
            assert client.timings() == {}
            client.stop()
        finally:
            client.close()
    finally:
        server_thread.join(timeout=5.0)
        assert not server_thread.is_alive()


def test_zmq_client_timings_counters_are_accumulated_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("VLAROBOT_BACKEND_TIMINGS", "1")
    server = SimServer(FakeBackend())
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"endpoint": "tcp://127.0.0.1:*"},
        daemon=True,
    )
    server_thread.start()
    endpoint = _wait_for_endpoint(server)
    try:
        client = ZmqBackendClient(endpoint, timeout_ms=5_000)
        try:
            client.ping()
            client.reset(ScenarioSpec(), seed=1)
            client.step(2)
            client.get_ground_truth()
            timings = client.timings()
            assert timings["requests"] == pytest.approx(4.0)
            for key in ("encode_s", "send_s", "recv_s", "decode_s"):
                assert key in timings
                assert timings[key] > 0.0
            assert timings["image_transfer_bytes"] > 0.0
            # Two image-bearing responses (reset + step); each carries RGB raster bytes.
            observation_shape = client.step(1).rgb.shape
            expected_bytes_per_frame = int(np.prod(observation_shape))
            expected = pytest.approx(2.0 * expected_bytes_per_frame, rel=0.01)
            assert timings["image_transfer_bytes"] == expected
            client.stop()
        finally:
            client.close()
    finally:
        server_thread.join(timeout=5.0)
        assert not server_thread.is_alive()
