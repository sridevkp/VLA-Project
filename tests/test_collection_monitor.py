"""The recovery controller owns only its exact collector and preserves source identity."""

import json
from types import SimpleNamespace

import pytest

from vlarobot.execution import collection_watchdog as watchdog


def test_process_matching_excludes_other_collectors_and_unrelated_services(tmp_path):
    class Process:
        def __init__(self, args, status="running"):
            self.info = {"cmdline": args}
            self.value = status

        def is_running(self):
            return True

        def status(self):
            return self.value

    valid = Process(["python", "-m", "vlarobot.cli", "collect-demos", "--out", str(tmp_path)])
    unrelated = Process(["python", "-m", "vlarobot.cli", "model-serve", "--out", str(tmp_path)])
    other = Process(
        ["python", "-m", "vlarobot.cli", "collect-demos", "--out", str(tmp_path / "other")]
    )
    dead = Process(valid.info["cmdline"], "zombie")
    processes = SimpleNamespace(
        process_iter=lambda fields: [valid, unrelated, other, dead],
        STATUS_ZOMBIE="zombie",
        NoSuchProcess=RuntimeError,
        AccessDenied=ValueError,
    )
    assert watchdog.matching_collectors(processes, tmp_path) == [valid]


def setup(tmp_path):
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "attempts").mkdir()
    (collection / "attempts.jsonl").write_text("", encoding="utf-8")
    (collection / "collection-status.json").write_text(
        json.dumps({"status": "running", "target": 80}), encoding="utf-8"
    )
    command = ["python", "-m", "vlarobot.cli", "collect-demos", "--out", str(collection)]
    path = tmp_path / "command.json"
    path.write_text(json.dumps(command), encoding="utf-8")
    return collection, path, command


def test_ambiguous_multiple_collectors_never_launch_or_kill_anyone(tmp_path, monkeypatch):
    collection, command, _ = setup(tmp_path)
    monkeypatch.setattr(watchdog, "import_module", lambda name: object())
    monkeypatch.setattr(watchdog, "inspect_collection", lambda *args: {"admitted": 1})
    monkeypatch.setattr(watchdog, "matching_collectors", lambda *args: [object(), object()])

    def forbidden(*args, **kwargs):
        raise AssertionError("no process mutation allowed")

    monkeypatch.setattr(watchdog.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="multiple collectors"):
        watchdog.watch_collection(collection, tmp_path / "state", command, interval_s=2, stale_s=10)


def test_exited_collector_resumes_exact_command_without_reusing_raw_attempts(tmp_path, monkeypatch):
    collection, command_path, command = setup(tmp_path)
    raw = collection / "attempts/attempt-0001"
    raw.mkdir()
    (raw / "partial").write_bytes(b"preserve")
    monkeypatch.setattr(watchdog, "import_module", lambda name: object())
    health = {
        "admitted": 1,
        "attempted": 1,
        "admission_rate": 1.0,
        "verified_frames": 2,
        "disk_free_bytes": 1000,
    }
    monkeypatch.setattr(watchdog, "inspect_collection", lambda *args: health)
    monkeypatch.setattr(watchdog, "matching_collectors", lambda *args: [])
    launched = []
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda args, **kwargs: launched.append(args))

    class EndWatch(Exception):
        pass

    def sleep(seconds):
        if seconds == 2:
            raise EndWatch()

    monkeypatch.setattr(watchdog, "sleep", sleep)
    with pytest.raises(EndWatch):
        watchdog.watch_collection(
            collection, tmp_path / "state", command_path, interval_s=2, stale_s=10
        )
    assert launched == [[*command, "--resume"]]
    assert (raw / "partial").read_bytes() == b"preserve"
