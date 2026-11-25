"""Owner-authorized passive R2025a experiment. No reset, apply, or ordinary step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from vlarobot.backends.webots.runtime_attestation import read_owned_controller_session
from vlarobot.execution.episode_runner import WebotsSession, launch_webots_session
from vlarobot.execution.worldmind_offline_boundary import _verify_event, _write_event  # noqa: PLC2701

ROOT = Path(__file__).resolve().parents[3]
WORLD = ROOT / "worlds/factory.wbt"
RUNTIME = Path.home() / ".local/opt/webots/webots"
CONTROLLER = ROOT / "controllers/tiago_controller/tiago_controller.py"
VALVE = RUNTIME.parent / "projects/objects/factory/valves/controllers/valve_turner/valve_turner.py"
VALVE_SHA = "ca46815b5c59c9910674eb13f41cf0ef93a6ffbaab6e41fc15ee421ab5040ec5"
PRIOR = ROOT / "runs/worldmind/idle-finger-settling-v1"
PRIOR_TERMINAL_SHA = "7c83b98794b560e5ecc3412a0b321ade8312b80c22148baa40568f629953b28e"
WORLD_SHA = "35d87f3601813a47e932bb8558740443710ba15421f07e1f3c67c6e5fd027706"
RUNTIME_SHA = "d5a846c134dfdf9ff5984b4cf8a30c0aed8c943f022bf83f754ecb325cef86e6"
CONTROLLER_SHA = "567fd2421c27bce27af7827b3ccda2d41fcb20f7de07c0e204fb2de6f5728b7b"
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32)
START_TIME = 0.096


def record(path: Path, value: dict[str, Any]) -> None:
    _verify_event(path, _write_event(path, value))


def valid_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def check_sample(sample: dict[str, Any], owned: dict[str, object], tick: int) -> None:
    if (
        sample.get("scope") != "read_only_finger_admission"
        or sample.get("session_id") != owned
        or type(sample.get("physics_step_count")) is not int
        or sample["physics_step_count"] != tick
        or not valid_number(sample.get("simulator_time_s"))
        or not math.isclose(sample["simulator_time_s"], START_TIME + 0.032 * tick)
        or sample.get("reviewed_limits_m") != [0.0, 0.045]
    ):
        raise ValueError("inconsistent sample identity, limits, or physics clock")
    for key, size in (("base_position_world_m", 3), ("base_orientation_world", 9)):
        vector = sample.get(key)
        if (
            not isinstance(vector, list)
            or len(vector) != size
            or not all(valid_number(value) for value in vector)
        ):
            raise ValueError("missing or invalid TIAGo base pose")
    rows = sample.get("readings")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("two exact finger mappings required")
    for side, row in zip(("left", "right"), rows, strict=True):
        motor = f"gripper_{side}_finger_joint"
        sensor = f"gripper_{side}_sensor_finger_joint"
        raw = row.get("raw_coordinate_m")
        if (
            row.get("expected_motor") != motor
            or row.get("resolved_motor") != motor
            or row.get("resolved_sensor") != sensor
            or row.get("cached_sensor") != sensor
            or row.get("cached_pair_count") != 1
            or row.get("mapping") != "matches"
            or row.get("sampling_period_ms") != 32
            or row.get("motor_limits_m") != [0.0, 0.045]
            or not valid_number(raw)
            or row.get("cached_raw_coordinate_m") != raw
            or not valid_number(row.get("supervisor_position_field_m"))
            or not all(
                isinstance(row.get(key), list)
                and len(row[key]) == size
                and all(valid_number(v) for v in row[key])
                for key, size in (
                    ("endpoint_position_world_m", 3),
                    ("endpoint_orientation_world", 9),
                )
            )
        ):
            raise ValueError(f"inconsistent {side} finger mapping, coordinate, or geometry")


def capture(session: WebotsSession, path: Path, owned: dict[str, object]) -> None:
    client = session.client
    client.require_owned_diagnostic_session(session_id=owned, world_sha256=WORLD_SHA)
    ping = client.ping()["payload"]
    record(path / "ping.json", ping)
    if ping.get("session_id") != owned or type(ping.get("finger_idle_only")) is not int:
        raise ValueError("unverified idle controller")
    if ping["finger_idle_only"] != 1 or ping.get("world_sha256") != WORLD_SHA:
        raise ValueError("idle-only capability or world pin missing")
    for tick in range(33):
        if tick:
            record(path / f"intent-{tick:02d}.json", {"tick": tick, "session_id": owned})
            try:
                receipt = client.idle_finger_step()
            except Exception as exc:
                record(
                    path / f"receipt-{tick:02d}.json", {"confirmed_ticks": None, "error": str(exc)}
                )
                raise
            record(path / f"receipt-{tick:02d}.json", receipt)
            if (
                receipt.get("session_id") != owned
                or type(receipt.get("confirmed_ticks")) is not int
                or receipt["confirmed_ticks"] != 1
                or type(receipt.get("physics_step_count")) is not int
                or receipt["physics_step_count"] != tick
                or not valid_number(receipt.get("simulator_time_s"))
                or not valid_number(receipt.get("previous_simulator_time_s"))
                or not math.isclose(receipt["simulator_time_s"], START_TIME + 0.032 * tick)
                or not math.isclose(
                    receipt["previous_simulator_time_s"], START_TIME + 0.032 * (tick - 1)
                )
            ):
                raise ValueError("inconsistent or unknown idle tick receipt")
        if tick in CHECKPOINTS:
            sample = client.inspect_fingers()
            record(path / f"sample-{tick:02d}.json", sample)
            check_sample(sample, owned, tick)
    record(path / "shutdown.json", client.shutdown_diagnostic_session(session_id=owned))


def verify_prior_budget() -> None:
    prior = _verify_event(PRIOR / "terminal.json", PRIOR_TERMINAL_SHA)
    if prior["confirmed_additional_ticks"] != 32:
        raise ValueError("unexpected prior study budget")
    receipts = sorted((PRIOR / "session-1").glob("receipt-*.json"))
    if len(receipts) != 32 or any((PRIOR / "session-2").glob("receipt-*.json")):
        raise ValueError("prior evidence does not match the continuation allocation")
    for tick, path in enumerate(receipts, 1):
        wrapper = json.loads(path.read_bytes())
        receipt = _verify_event(path, wrapper["sha256"])
        if receipt["confirmed_ticks"] != 1 or receipt["physics_step_count"] != tick:
            raise ValueError("prior receipt sequence does not reconcile")


def record_background(session: WebotsSession, path: Path) -> None:
    launches = [
        line
        for line in session.log_path.read_text(errors="replace", encoding="utf-8").splitlines()
        if "valve_turner: Starting controller:" in line
    ]
    record(
        path / "background-controllers.json",
        {
            "controller": "valve_turner",
            "source_path": str(VALVE),
            "source_sha256": VALVE_SHA,
            "launch_messages": launches,
            "actual_motor_writes": "unobserved",
            "launch_instance_count": len(launches),
        },
    )
    if len(launches) != 2:
        raise ValueError("unexpected background valve controller launch count")


def run(output: Path, expected_head: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    if head != expected_head or status:
        raise ValueError("expected committed experiment source and clean tracked tree")
    verify_prior_budget()
    for path, digest in (
        (WORLD, WORLD_SHA),
        (RUNTIME, RUNTIME_SHA),
        (CONTROLLER, CONTROLLER_SHA),
        (VALVE, VALVE_SHA),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"provenance pin mismatch: {path}")
    output.mkdir(exist_ok=False)
    record(
        output / "protocol.json",
        {
            "schema_version": 2,
            "head": head,
            "runtime_sha256": RUNTIME_SHA,
            "world_sha256": WORLD_SHA,
            "controller_sha256": CONTROLLER_SHA,
            "max_sessions": 2,
            "prior_confirmed_additional_ticks": 32,
            "max_additional_ticks": 64,
            "max_study_confirmed_ticks": 96,
            "prior_terminal_sha256": PRIOR_TERMINAL_SHA,
            "background_valve_sha256": VALVE_SHA,
            "background_motor_writes": "unobserved",
            "max_ticks_per_session": 32,
            "tick_ms": 32,
            "checkpoints": list(CHECKPOINTS),
            "normal_startup_ticks": 3,
            "no_reset_or_apply": True,
        },
    )
    seen_sessions: set[str] = set()
    for number in (3, 4):
        path = output / f"session-{number}"
        path.mkdir()
        status_path = path / "controller-status.json"
        started = time.time()
        session = None
        try:
            with launch_webots_session(
                webots=str(RUNTIME),
                world=WORLD,
                trace_dir=path,
                trace_prefix="idle",
                runtime_attestation_required=True,
                rpc_shutdown_on_exit=False,
                extra_env={
                    "VLAROBOT_FINGER_IDLE_ONLY": "1",
                    "VLAROBOT_WEBOTS_MODE": "server",
                    "VLAROBOT_WORLD_SHA256": WORLD_SHA,
                    "VLAROBOT_DIAGNOSTIC_STATUS_PATH": str(status_path),
                },
            ) as session:
                runtime = dict(session.runtime_attestation)
                if (
                    runtime.get("reported_version") != "R2025a"
                    or runtime.get("executable_sha256") != RUNTIME_SHA
                    or not session.endpoint.startswith("tcp://127.0.0.1:")
                ):
                    raise ValueError("runtime or local endpoint mismatch")
                owned = read_owned_controller_session(
                    status_path,
                    endpoint=session.endpoint,
                    process=session.process,
                    runtime_attestation=runtime,
                    launch_started_at_s=started,
                )
                record(
                    path / "launch.json",
                    {
                        "endpoint": session.endpoint,
                        "launcher_pid": session.process.pid,
                        "session_id": owned,
                        "runtime": runtime,
                        "startup_s": session.startup_s,
                    },
                )
                nonce = str(owned["nonce"])
                if nonce in seen_sessions:
                    raise ValueError("reused controller session")
                seen_sessions.add(nonce)
                record_background(session, path)
                capture(session, path, owned)
        except Exception as exc:
            record(path / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
            raise
        finally:
            if session is not None:
                record(path / "exit.json", {"returncode": session.process.poll()})
        if session is None or session.process.returncode != 0:
            raise RuntimeError("owned process did not exit cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.expected_head)
