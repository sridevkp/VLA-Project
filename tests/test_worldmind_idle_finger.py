from types import SimpleNamespace

import pytest

from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.backends.zmq.server import SimServer
from vlarobot.core.wire import make_message


@pytest.mark.parametrize("fault", [None, "physics_error", "physics_stopped"])
def test_idle_step_server_allowlist_budget_and_stop(monkeypatch, fault):
    monkeypatch.setenv("VLAROBOT_FINGER_IDLE_ONLY", "1")

    class Robot:
        t = 0.096

        def getTime(self):
            return self.t

        def step(self, ms):
            calls.append(ms)
            if fault == "physics_error":
                raise RuntimeError("simulated ambiguous physics")
            if fault == "physics_stopped":
                return -1
            self.t += ms / 1000
            return 0

    calls = []
    backend = SimpleNamespace(
        io=SimpleNamespace(robot=Robot(), ts=32),
        _physics_step_count=0,
        diagnostic_step_capability=("webots", 1),
        diagnostic_step=lambda: pytest.fail("ordinary diagnostic step reached"),
    )
    server = SimServer(backend)

    def request(op, payload=None, *, guarded=True, session=None):
        args = payload or {}
        if guarded:
            return server.handle(
                make_message(
                    "diagnostic_session_request",
                    0,
                    {
                        "operation": op,
                        "arguments": args,
                        "session_id": server.session_id if session is None else session,
                    },
                )
            )
        return server.handle(make_message(op, 0, args))

    for op in ("reset", "apply", "step", "diagnostic_step", "shutdown"):
        assert request(op, guarded=False).get("error")
    for op in ("reset", "apply", "diagnostic_step"):
        assert request(op).get("error")
    assert request("idle_finger_step", guarded=False).get("error")
    assert request("idle_finger_step", {"n": 32}).get("error")
    assert request("idle_finger_step", session={}).get("error")
    assert calls == []
    if fault:
        assert request("idle_finger_step")["payload"]["confirmed_ticks"] is None
        assert request("idle_finger_step").get("error")
        assert calls == [32] and backend._physics_step_count == 0
    else:
        for count in range(1, 33):
            receipt = request("idle_finger_step")["payload"]
            assert receipt["confirmed_ticks"] == 1
            assert receipt["physics_step_count"] == count
        assert request("idle_finger_step").get("error")
        assert calls == [32] * 32


def test_idle_client_never_retries_lost_response(monkeypatch):
    client = ZmqBackendClient()
    session = {"pid": 42, "start_time": 1.0, "nonce": "x"}
    calls = []

    def exchange(operation, arguments, timeout_ms):
        calls.append(operation)
        if operation == "ping":
            return make_message(
                "ping",
                0,
                {
                    "status": "ok",
                    "session_id": session,
                    "diagnostic_backend": "webots",
                    "world_sha256": "a" * 64,
                    "capabilities": {"diagnostic_step": 1},
                },
            )
        assert arguments["operation"] == "idle_finger_step"
        raise TimeoutError("lost reply")

    monkeypatch.setattr(client, "_exchange", exchange)
    try:
        client.require_owned_diagnostic_session(session_id=session, world_sha256="a" * 64)
        with pytest.raises(TimeoutError):
            client.idle_finger_step()
        with pytest.raises(ValueError, match="closed"):
            client.idle_finger_step()
        assert calls == ["ping", "diagnostic_session_request"]
    finally:
        client.close()


@pytest.mark.parametrize("fault", ["intent", "receipt", "unknown", "counter"])
def test_idle_recorder_and_tick_faults_stop_the_schedule(tmp_path, monkeypatch, fault):
    from vlarobot.execution import worldmind_idle_fingers as probe

    session_id = {"pid": 42, "start_time": 1.0, "nonce": "x"}
    calls = []
    writes = []

    def record(path, value):
        writes.append(path.name)
        if fault == "intent" and path.name == "intent-01.json":
            raise OSError("intent write failed")
        if fault == "receipt" and path.name == "receipt-01.json":
            raise OSError("receipt write failed")

    def tick():
        calls.append("tick")
        if fault == "unknown":
            raise TimeoutError("step reply lost")
        return {
            "session_id": session_id,
            "confirmed_ticks": 1,
            "physics_step_count": 2 if fault == "counter" else 1,
            "simulator_time_s": 0.128,
            "previous_simulator_time_s": 0.096,
        }

    client = SimpleNamespace(
        require_owned_diagnostic_session=lambda **kw: None,
        ping=lambda: {
            "payload": {
                "finger_idle_only": 1,
                "world_sha256": probe.WORLD_SHA,
                "session_id": session_id,
            }
        },
        inspect_fingers=lambda: {"fixture": True},
        idle_finger_step=tick,
    )
    monkeypatch.setattr(probe, "check_sample", lambda sample, owned, index: None)
    monkeypatch.setattr(probe, "record", record)
    with pytest.raises((ValueError, OSError)):
        probe.capture(SimpleNamespace(client=client), tmp_path, session_id)
    assert calls == ([] if fault == "intent" else ["tick"])
    assert "intent-02.json" not in writes
