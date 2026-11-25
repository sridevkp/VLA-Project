"""Replay an expert command stream with the guarded WorldMind actuator and tick journal.

The timed candidate-00 control is diagnostic. It preserves the expert's command/hold
schedule and is deliberately distinct from the frozen H=50 ranked candidate pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import GRIPPER_OPEN
from vlarobot.backends.webots.runtime_attestation import read_owned_controller_session
from vlarobot.backends.worldmind_bindings import runtime_safety_limits
from vlarobot.core.types import Action, GripperCmd, JointTarget, Observation
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.vla_metrics import final_can_outcome
from vlarobot.eval.worldmind_config import FROZEN_CONTROL_DT_S
from vlarobot.execution.episode_runner import WebotsSession, launch_webots_session
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_can import GRIPPER_OPEN_TOLERANCE_M
from vlarobot.execution.worldmind_heuristic_qualification import (  # noqa: PLC2701
    _ROOT,
    _WORLD,
    _admit,
    _check_probe,
    _encode,
    _operation,
    _persist,
    _tick,
    _verified_head,
    _write_event,
)

SCOPE = "non_scored_expert_candidate_00_parity"
EPISODE = "validation-clear-i00-s0"
OPENING_PARITY_M = 0.001
REFERENCE_SHA = {
    "episode-00.actions.jsonl": "40c5dd9986efe05b7edc6540cc425aca2f4ae791909e0c47414b53d414c905d4",
    "episode-00.steps.npz": "46eaccc13a5e4f7b2d9c68fac82ca5ae44ae387e9efe1a6b2e2622052e6a7963",
    "episode-00.meta.json": "bd403a47b0ec9482ec1402646891328dbf87500d2294fbf6fa7f7080a60f9a55",
    "policy.jsonl": "e96b5043c514bc3bfaab2849f30f9b5c9a1a142255c34a1b2669519b3b59c91e",
}


@dataclass(frozen=True)
class ReplayPlan:
    episode_id: str
    seed: int
    origin_s: float
    ticks: int
    actions: tuple[tuple[int, Action], ...]
    phase_entries: tuple[tuple[int, str], ...]
    expert_before_close_m: float
    source_sha256: dict[str, str]


def _offset(t: float, origin: float) -> int:
    index = round((t - origin) / FROZEN_CONTROL_DT_S)
    if index < 0 or not math.isclose(t, origin + index * FROZEN_CONTROL_DT_S, abs_tol=1e-9):
        raise ValueError("expert timestamp is off the frozen control tick")
    return index


def load_plan(archive: Path) -> ReplayPlan:
    """Load the exact successful expert command stream and recorded phase boundaries."""
    commands = archive / "campaign/episode-00.actions.jsonl"
    observations = archive / "campaign/episode-00.steps.npz"
    metadata = archive / "campaign/episode-00.meta.json"
    phases = archive / "parallel-episodes/episode-00/policy.jsonl"
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (commands, observations, metadata, phases)
    }
    if hashes != REFERENCE_SHA:
        raise ValueError("canonical expert reference hash mismatch")
    meta = json.loads(metadata.read_bytes())
    if (
        meta["scenario"]["name"] != EPISODE
        or meta["seed"] != 0
        or meta["outcome"]["success"] is not True
    ):
        raise ValueError("the pinned successful clear VALIDATION trial is required")
    with np.load(observations, allow_pickle=False) as data:
        timestamps, gripper = data["t"], data["gripper"]
    origin = float(timestamps[0])
    offsets = tuple(_offset(float(timestamp), origin) for timestamp in timestamps)
    if offsets != tuple(range(len(timestamps))):
        raise ValueError("expert observation stream has a missing or duplicate tick")
    timed = []
    for line in commands.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["accepted"] is not True:
            raise ValueError("the expert reference contains a rejected apply")
        raw = row["action"]
        if set(raw) == {"q"}:
            action: Action = JointTarget(raw["q"])
        elif set(raw) == {"opening", "force"}:
            action = GripperCmd(raw["opening"], raw["force"])
        else:
            raise ValueError("unsupported command in expert reference")
        timed.append((_offset(row["t"], origin), action))
    indices = [tick for tick, _ in timed]
    if not indices or indices != sorted(set(indices)) or indices[-1] >= offsets[-1]:
        raise ValueError("ambiguous or unobserved expert apply time")
    phase_entries = tuple(
        (_offset(row["sim_time_s"], origin), row["state"])
        for row in map(json.loads, phases.read_text(encoding="utf-8").splitlines())
        if row.get("phase") == "enter" and row.get("event") == "state_boundary"
    )
    close = [
        tick for tick, action in timed if isinstance(action, GripperCmd) and action.opening == 0
    ]
    if len(close) != 1 or (close[0], "CLOSE") not in phase_entries:
        raise ValueError("expert CLOSE command and phase evidence do not reconcile")
    return ReplayPlan(
        EPISODE,
        0,
        origin,
        offsets[-1],
        tuple(timed),
        phase_entries,
        float(gripper[close[0]]),
        hashes,
    )


def execute_timed_candidate(
    plan: ReplayPlan,
    session: WebotsSession,
    owned: dict[str, object],
    directory: Path,
    initial: Observation,
    monitor: SafetyMonitor,
) -> tuple[Observation, float]:
    """Apply candidate 00's actual commands at their ticks using the normal guarded executor."""
    _admit(initial, finger_logging_only=True)
    if (
        plan.episode_id != EPISODE
        or plan.seed != 0
        or not 1 <= plan.ticks <= 1059
        or not math.isclose(initial.t, plan.origin_s, abs_tol=1e-9)
    ):
        raise ValueError("expert candidate has incompatible identity, tick budget, or reset time")
    by_tick = dict(plan.actions)
    if len(by_tick) != len(plan.actions) or not all(0 <= tick < plan.ticks for tick in by_tick):
        raise ValueError("expert candidate has invalid command ticks")
    close_ticks = [
        tick
        for tick, action in plan.actions
        if isinstance(action, GripperCmd) and action.opening == 0
    ]
    if len(close_ticks) != 1 or (close_ticks[0], "CLOSE") not in plan.phase_entries:
        raise ValueError("expected one CLOSE command and matching recorded phase boundary")
    snapshot_path = directory / "candidate-00.msgpack"
    snapshot = _persist(
        snapshot_path,
        _encode(
            {
                "candidate_id": "candidate-00",
                "episode_id": plan.episode_id,
                "ticks": plan.ticks,
                "actions": plan.actions,
                "phase_entries": plan.phase_entries,
                "archive_sha256": plan.source_sha256,
            },
            SCOPE,
        ),
    )
    observation = initial
    preclose = math.nan
    phase_by_tick: dict[int, list[str]] = {}
    for tick, phase in plan.phase_entries:
        phase_by_tick.setdefault(tick, []).append(phase)

    pregrasp_ticks = [tick for tick, phase in plan.phase_entries if phase == "PREGRASP"]
    if len(pregrasp_ticks) != 1:
        raise ValueError("expected a single archived PREGRASP entry")
    for tick in range(plan.ticks):
        tick_path = directory / f"tick-{tick:04d}"
        tick_path.mkdir()
        if tick in phase_by_tick:
            _write_event(
                tick_path / "phase.json",
                {
                    "reference_phases": phase_by_tick[tick],
                    "observed_time_s": observation.t,
                    "measured_opening_m": observation.robot.gripper_opening,
                },
            )
        if tick == pregrasp_ticks[0] and (
            not math.isfinite(observation.robot.gripper_opening)
            or observation.robot.gripper_opening < GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE_M
        ):
            raise RuntimeError("expert replay INIT did not reach the existing opening criterion")
        action = by_tick.get(tick)
        if action is None:
            status, after = _tick(
                tick_path, session, observation, {"expert_hold_tick": tick}, SCOPE
            )
        else:
            if tick == close_ticks[0]:
                preclose = observation.robot.gripper_opening
                if (
                    not math.isfinite(preclose)
                    or abs(preclose - plan.expert_before_close_m) > OPENING_PARITY_M
                ):
                    raise RuntimeError(
                        "measured pre-CLOSE opening differs from expert by more than 1 mm"
                    )
            verdict = monitor.validate(action, observation.robot)
            verdict_sha = _persist(
                tick_path / "motor-verdict.msgpack",
                _encode(
                    {
                        "action": action,
                        "motor_verdict": verdict,
                    },
                    SCOPE,
                ),
            )
            if verdict.accepted is not True:
                raise RuntimeError(f"expert replay motor veto at tick {tick}: {verdict.reason}")
            status, after = _operation(
                tick_path,
                session,
                owned,
                observation,
                snapshot_path,
                snapshot,
                action,
                tick,
                verdict_sha,
                SCOPE,
            )
        if after is None:
            raise RuntimeError(f"expert replay stopped at tick {tick}: {status}")
        observation = after
    return observation, preclose


def run_live(archive: Path, output: Path, expected_head: str) -> None:
    """One bounded, separately authorized clear seed-0 parity replay."""
    from vlarobot.backends.webots.config import (
        CAN_PLACE_Z_TOLERANCE_M,
        CAN_TILT_LIMIT_DEG,
        HOME_SETTLE_TOLERANCE_RAD,
        PLACE_TOLERANCE_M,
        PLACE_WORLD,
    )
    from vlarobot.execution.worldmind_idle_fingers import RUNTIME, RUNTIME_SHA

    head = _verified_head(expected_head)
    world = _ROOT / "worlds/factory.wbt"
    if (
        hashlib.sha256(world.read_bytes()).hexdigest() != _WORLD
        or hashlib.sha256(RUNTIME.read_bytes()).hexdigest() != RUNTIME_SHA
    ):
        raise ValueError("expert parity runtime or factory world pin mismatch")
    plan = load_plan(archive)
    entry = next(row for row in build_episode_plan("validation", 0) if row.scene.name == EPISODE)
    if entry.seed != plan.seed:
        raise ValueError("expert replay scenario seed mismatch")
    output.mkdir(exist_ok=False)
    status = output / "controller-status.json"
    started = time.time()
    monitor = SafetyMonitor(runtime_safety_limits())
    with launch_webots_session(
        webots=str(RUNTIME),
        world=world,
        trace_dir=output,
        trace_prefix="expert-parity",
        rpc_shutdown_on_exit=False,
        runtime_attestation_required=True,
        extra_env={
            "VLAROBOT_DIAGNOSTIC_STATUS_PATH": str(status),
            "VLAROBOT_WORLD_SHA256": _WORLD,
            "VLAROBOT_FINGER_IDLE_ONLY": "0",
            "VLAROBOT_WEBOTS_MODE": "server",
            "WEBOTS_PYTHON": sys.executable,
            "VLAROBOT_URDF_REGISTRATION_FILE": "calibration/urdf-base-train-v1.yaml",
        },
    ) as session:
        provenance = session.runtime_attestation
        if (
            provenance.get("reported_version") != "R2025a"
            or provenance.get("executable_sha256") != RUNTIME_SHA
        ):
            raise ValueError("expert parity execution runtime mismatch")
        owned = read_owned_controller_session(
            status,
            endpoint=session.endpoint,
            process=session.process,
            runtime_attestation=provenance,
            launch_started_at_s=started,
        )
        client = session.client
        client.require_owned_diagnostic_session(session_id=owned, world_sha256=_WORLD)
        _write_event(
            output / "launch.json",
            {
                "head": head,
                "runtime_attestation": dict(provenance),
                "world_sha256": _WORLD,
                "scope": SCOPE,
                "session": owned,
                "archive_sha256": plan.source_sha256,
                "episode_id": entry.scene.name,
                "seed": entry.seed,
            },
        )
        try:
            probe = client.inspect_fingers()
            _persist(output / "finger-probe.msgpack", _encode(probe, SCOPE))
            _check_probe(probe, owned, finger_logging_only=True)
            _write_event(output / "reset-intent.json", {"scene": EPISODE, "seed": entry.seed})
            initial = client.reset(entry.scene, entry.seed)
            initial_truth = client.get_ground_truth()
            last, before_close = execute_timed_candidate(
                plan, session, owned, output, initial, monitor
            )
            final_truth = client.get_ground_truth()
            home = all(
                abs(last.robot.joint_pos[j] - initial.robot.joint_pos[j])
                <= HOME_SETTLE_TOLERANCE_RAD
                for j in monitor.limits.joint_position
            )
            outcome = final_can_outcome(
                initial_truth.object_poses["CAN"],
                final_truth.object_poses["CAN"],
                place_world_m=PLACE_WORLD,
                distance_tolerance_m=PLACE_TOLERANCE_M,
                height_tolerance_m=CAN_PLACE_Z_TOLERANCE_M,
                tilt_limit_deg=CAN_TILT_LIMIT_DEG,
                home_reached=home,
                failure_state=None,
                failure_reason=None,
            )
            _write_event(
                output / "outcome.json",
                {
                    "candidate_id": "candidate-00",
                    "scored": False,
                    "ticks": plan.ticks,
                    "recorded_phase_sequence": [phase for _, phase in plan.phase_entries],
                    "observed_before_close_m": before_close,
                    "expert_before_close_m": plan.expert_before_close_m,
                    "outcome": outcome,
                },
            )
            _write_event(
                output / "shutdown.json", client.shutdown_diagnostic_session(session_id=owned)
            )
        except Exception as exc:
            _write_event(output / "failure.json", {"error": str(exc), "type": type(exc).__name__})
            raise
    _write_event(output / "exit.json", {"returncode": session.process.poll()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    run_live(args.archive.resolve(), args.out.resolve(), args.expected_head)
