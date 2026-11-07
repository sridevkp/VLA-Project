from __future__ import annotations

import threading
import time

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
