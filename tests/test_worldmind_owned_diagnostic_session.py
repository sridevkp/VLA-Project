"""Test explicit launch pins and recorded shutdown acknowledgments without a socket."""

from __future__ import annotations

import pytest

from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.core.types import GripperCmd
from vlarobot.core.wire import make_message

SESSION = {"pid": 42, "start_time": 1.0, "nonce": "owned"}
WORLD = "a" * 64


@pytest.mark.parametrize("fault", [None, "session", "world", "lost", "bad_ack", "peer_changed"])
def test_owned_pins_and_guarded_shutdown(monkeypatch, fault):
    client = ZmqBackendClient()
    operations = []

    def exchange(operation, arguments, timeout_ms):
        operations.append(operation)
        if operation == "ping":
            return make_message(
                "ping",
                0,
                {
                    "session_id": SESSION if fault != "session" else {**SESSION, "nonce": "old"},
                    "world_sha256": WORLD if fault != "world" else "b" * 64,
                    "capabilities": {"diagnostic_step": 1},
                    "diagnostic_backend": "webots",
                    "status": "ok",
                },
            )
        assert arguments == {"session_id": SESSION, "operation": "shutdown", "arguments": {}}
        if fault == "lost":
            raise TimeoutError("shutdown reply lost")
        return make_message(
            operation,
            1,
            {
                "session_id": SESSION if fault != "peer_changed" else {**SESSION, "nonce": "new"},
                "status": "stopped" if fault != "bad_ack" else "ambiguous",
            },
        )

    monkeypatch.setattr(client, "_exchange", exchange)
    try:
        if fault in {"session", "world"}:
            with pytest.raises(ValueError, match="expected session|world hash"):
                client.require_owned_diagnostic_session(session_id=SESSION, world_sha256=WORLD)
            with pytest.raises(ValueError, match="closed"):
                client.apply(GripperCmd(0.0))
            assert operations == ["ping"]
            return
        client.require_owned_diagnostic_session(session_id=SESSION, world_sha256=WORLD)
        if fault == "lost":
            with pytest.raises(TimeoutError, match="lost"):
                client.shutdown_diagnostic_session(session_id=SESSION)
        elif fault in {"bad_ack", "peer_changed"}:
            with pytest.raises(ValueError, match="acknowledgement|identity"):
                client.shutdown_diagnostic_session(session_id=SESSION)
        else:
            assert client.shutdown_diagnostic_session(session_id=SESSION)["status"] == "stopped"
        assert operations == ["ping", "diagnostic_session_request"]
        assert client._closed
    finally:
        client.close()


def test_shutdown_without_owned_pin_never_sends(monkeypatch):
    client = ZmqBackendClient()
    monkeypatch.setattr(client, "_exchange", lambda *args: pytest.fail("unexpected request"))
    with pytest.raises(ValueError, match="exact pinned"):
        client.shutdown_diagnostic_session(session_id=SESSION)


def test_peer_discovered_only_by_ping_cannot_use_owned_shutdown(monkeypatch):
    client = ZmqBackendClient()
    operations = []

    def exchange(operation, arguments, timeout_ms):
        operations.append(operation)
        assert operation == "ping"
        return make_message(
            "ping",
            0,
            {
                "session_id": SESSION,
                "status": "ok",
                "capabilities": {"diagnostic_step": 1},
                "diagnostic_backend": "webots",
            },
        )

    monkeypatch.setattr(client, "_exchange", exchange)
    client.require_diagnostic_session(expected_backend="webots")
    with pytest.raises(ValueError, match="pinned owned"):
        client.shutdown_diagnostic_session(session_id=SESSION)
    assert operations == ["ping"]
