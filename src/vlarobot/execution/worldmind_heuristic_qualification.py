"""Guarded, non-scored ADR-0084 Webots qualification and episode execution.

The caller owns the launch context and its process cleanup. This module registers no CLI.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlarobot.backends.webots.runtime_attestation import read_owned_controller_session
from vlarobot.backends.worldmind_bindings import production_worldmind_bindings
from vlarobot.core.types import Action, ApplyResult, GroundTruth, Observation, TaskSpec, WorldState
from vlarobot.core.wire import encode_message, make_message
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.worldmind_config import (
    FROZEN_CONTROL_DT_S,
    load_worldmind_diagnostic_config,
)
from vlarobot.eval.worldmind_offline_evidence import validate_offline_observed_state
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.worldmind_heuristic_comparison import build_offline_heuristic_comparison
from vlarobot.execution.worldmind_offline_boundary import _sync_dir, _verify_event, _write_event  # noqa: PLC2701
from vlarobot.execution.worldmind_pipeline import _world_state  # noqa: PLC2701
from vlarobot.execution.worldmind_stage1_pipeline import _plan_sha256  # noqa: PLC2701
from vlarobot.risk.gate import RecoveryState

if TYPE_CHECKING:
    from vlarobot.execution.episode_runner import WebotsSession

_ROOT = Path(__file__).resolve().parents[3]
_WORLD = "35d87f3601813a47e932bb8558740443710ba15421f07e1f3c67c6e5fd027706"
_ROBOT = "1a9f0c2b57d771e45b6908d05177eb64eb80634b7ab2d35c5dfbefb1f863ca1d"
_CONFIG = "9f9d4ebc328c4f05ade96be52d4aa1be26ff1df94fa077ffc1d2a24cc49c6c03"
_RUNTIME = "d5a846c134dfdf9ff5984b4cf8a30c0aed8c943f022bf83f754ecb325cef86e6"
_PLAN = "013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2"
_EPISODE = "validation-clear-i00-s0"
_FINGERS = ("gripper_left_finger_joint", "gripper_right_finger_joint")
_SCOPE = "non_scored_arm3_one_decision_qualification"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _persist(path: Path, raw: bytes) -> str:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)
    if path.read_bytes() != raw:
        raise ValueError("qualification bytes differ after durable write")
    return _sha(raw)


def _encode(payload: dict[str, Any], scope: str = _SCOPE) -> bytes:
    return encode_message(make_message(scope, 0, payload))


def _raw(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 0.045


def _check_probe(probe: dict[str, Any], session_id: dict[str, object]) -> None:
    if (
        probe.get("scope") != "read_only_finger_admission"
        or probe.get("session_id") != session_id
        or probe.get("reviewed_limits_m") != [0.0, 0.045]
        or type(probe.get("physics_step_count")) is not int
        or probe["physics_step_count"] < 0
        or type(probe.get("simulator_time_s")) not in (float, int)
        or not math.isfinite(probe["simulator_time_s"])
        or probe["simulator_time_s"] < 0
    ):
        raise ValueError("unverified guarded finger probe")
    rows = probe.get("readings")
    if not isinstance(rows, list) or [row.get("expected_motor") for row in rows] != list(_FINGERS):
        raise ValueError("finger probe lacks the two configured motor identities")
    for row in rows:
        name = row["expected_motor"]
        value = row.get("raw_coordinate_m")
        if not _raw(value):
            raise ValueError(f"invalid raw finger coordinate: {name}: {value!r}")
        if (
            row.get("resolved_motor") != name
            or row.get("resolved_sensor") != name.replace("_finger_joint", "_sensor_finger_joint")
            or row.get("cached_sensor") != row.get("resolved_sensor")
            or row.get("motor_limits_m") != [0.0, 0.045]
            or row.get("sampling_period_ms") != 32
            or row.get("mapping") != "matches"
            or row.get("cached_pair_count") != 1
            or not _raw(row.get("cached_raw_coordinate_m"))
            or row["cached_raw_coordinate_m"] != value
        ):
            raise ValueError(f"unreconciled raw finger mapping: {name}")


def _admit(obs: Observation) -> None:
    if (
        not isinstance(obs, Observation)
        or type(obs.t) not in (int, float)
        or not math.isfinite(obs.t)
        or obs.t < 0
    ):
        raise ValueError("invalid qualification observation timestamp")
    joints = {f"arm_{i}_joint" for i in range(1, 8)}
    if not joints <= obs.robot.joint_pos.keys() or not joints <= obs.robot.joint_vel.keys():
        raise ValueError("qualification observation lacks arm joint state")
    readings = obs.robot.diagnostics.get("finger_joint_sensor_rad")
    for name in _FINGERS:
        value = readings.get(name) if isinstance(readings, dict) else None
        if not _raw(value):
            raise ValueError(f"invalid raw finger coordinate: {name}: {value!r}")


def _row_data(row: Any) -> dict[str, Any]:
    return {field.name: getattr(row, field.name) for field in fields(row)}


def _operation(
    directory: Path,
    session: WebotsSession,
    session_id: dict[str, object],
    before: Observation,
    decision_path: Path,
    snapshot: str,
    action: Action,
    action_index: int,
    verdict_sha: str | None,
    scope: str,
) -> tuple[str, Observation | None]:
    client = session.client
    request = _encode({"action": action}, scope)
    intent = {
        "scope": scope,
        "session_id": session_id,
        "decision_sha256": snapshot,
        "command_sha256": _sha(request),
        "operation": "apply",
        "action_index": action_index,
        "motor_verdict_sha256": verdict_sha,
        "budget": 1,
    }
    intent_path = directory / "apply-intent.json"
    intent_sha = _write_event(intent_path, intent)
    _verify_event(intent_path, intent_sha)
    if (
        _sha(decision_path.read_bytes()) != snapshot
        or _encode({"action": action}, scope) != request
    ):
        raise ValueError("qualification intent no longer matches the candidate evidence")
    if verdict_sha is not None and (
        _sha((directory / "motor-verdict.msgpack").read_bytes()) != verdict_sha
    ):
        raise ValueError("qualification intent no longer matches the motor verdict")
    # Unknown outcomes leave the corresponding durable intent unresolved if its receipt cannot
    # be stored. The fresh directory gate prevents reuse or replay.
    try:
        result = client.apply(action)
    except Exception as exc:
        _write_event(directory / "apply-result.json", {"status": "unknown", "reason": str(exc)})
        raise
    if not isinstance(result, ApplyResult) or type(result.accepted) is not bool:
        _write_event(directory / "apply-result.json", {"status": "unknown"})
        raise ValueError("invalid backend apply receipt")
    apply_sha = _write_event(
        directory / "apply-result.json",
        {"status": "accepted" if result.accepted else "rejected", "reason": result.reason},
    )
    if not result.accepted:
        return "backend_rejected", None
    _verify_event(directory / "apply-result.json", apply_sha)
    return _tick(directory, session, before, {"accepted_apply_sha256": apply_sha}, scope)


def _tick(
    directory: Path,
    session: WebotsSession,
    before: Observation,
    intent: dict[str, Any],
    scope: str,
) -> tuple[str, Observation | None]:
    client = session.client
    tick_path = directory / "tick-intent.json"
    tick_sha = _write_event(tick_path, {**intent, "budget": 1})
    _verify_event(tick_path, tick_sha)
    reobserve_digest = intent.get("reobserve_decision_sha256")
    if reobserve_digest is not None and (
        _sha((directory / "decision.msgpack").read_bytes()) != reobserve_digest
    ):
        raise ValueError("re-observation intent no longer matches the candidate evidence")
    try:
        reply = client.diagnostic_step()
    except Exception as exc:
        _write_event(directory / "tick-result.json", {"confirmed_ticks": None, "reason": str(exc)})
        raise
    if not isinstance(reply, dict):
        reply = {"error": "invalid diagnostic step reply"}
    ticks = reply.get("confirmed_ticks")
    authoritative = ticks if type(ticks) is int and ticks in (0, 1) else None
    after = reply.get("observation")
    _write_event(
        directory / "tick-result.json",
        {
            "confirmed_ticks": authoritative,
            "observation_returned": isinstance(after, Observation),
            "reason": reply.get("error"),
        },
    )
    if authoritative is None:
        return "tick_unknown", None
    if authoritative == 0:
        return "tick_not_executed", None
    if not isinstance(after, Observation):
        return "tick_unobserved", None
    _persist(directory / "post-tick.msgpack", _encode({"observation": after}, scope))
    _admit(after)
    if not math.isclose(after.t - before.t, FROZEN_CONTROL_DT_S):
        raise ValueError("post-tick observation time does not match one control tick")
    return "one_tick_observed", after


def _verified_head(expected: str) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_ROOT, text=True
    )
    if head != expected or tracked:
        raise ValueError("qualification requires the authorized HEAD and clean tracked worktree")
    return head


def run_heuristic_qualification(
    session: WebotsSession,
    *,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
) -> str:
    """Run the fixed arm-3 qualification envelope in a separately authorized owned launch."""
    return _run_owned(
        session, status_path, launch_started_at_s, directory, expected_head, _EPISODE, _SCOPE
    )


def run_heuristic_episode(
    session: WebotsSession,
    *,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
    episode_id: str,
) -> str:
    """Run one separately authorized, non-scored ADR-0084 episode in an owned launch."""
    return _run_owned(
        session,
        status_path,
        launch_started_at_s,
        directory,
        expected_head,
        episode_id,
        "non_scored_arm3_guarded_episode",
    )


def _run_owned(
    session: WebotsSession,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
    episode_id: str,
    scope: str,
) -> str:
    qualification = scope == _SCOPE
    head = _verified_head(expected_head)
    config_path = _ROOT / "configs/worldmind_diagnostic.approved.yaml"
    for path, digest in (
        (config_path, _CONFIG),
        (_ROOT / "configs/robot.yaml", _ROBOT),
        (_ROOT / "worlds/factory.wbt", _WORLD),
    ):
        if _sha(path.read_bytes()) != digest:
            raise ValueError(f"qualification pin mismatch: {path}")
    plan = build_episode_plan("validation", 0)
    if len(plan) != 18 or _plan_sha256(plan) != _PLAN:
        raise ValueError("qualification VALIDATION plan pin mismatch")
    entry = next((row for row in plan if row.scene.name == episode_id), None)
    if entry is None:
        raise ValueError("episode is outside the frozen VALIDATION plan")
    attestation = session.runtime_attestation
    if (
        not session.endpoint.startswith("tcp://127.0.0.1:")
        or attestation.get("executable_sha256") != _RUNTIME
        or attestation.get("reported_version") != "R2025a"
    ):
        raise ValueError("qualification requires the pinned isolated R2025a launch")
    owned = read_owned_controller_session(
        status_path,
        endpoint=session.endpoint,
        process=session.process,
        runtime_attestation=attestation,
        launch_started_at_s=launch_started_at_s,
    )
    directory.mkdir(exist_ok=False)
    _sync_dir(directory.parent)
    pinned = False
    try:
        session.client.require_owned_diagnostic_session(session_id=owned, world_sha256=_WORLD)
        pinned = True
        _write_event(
            directory / "scope.json",
            {
                "scope": scope,
                "arm": "vla_risk_veto",
                "actual_proposer": "scripted_chunk_adapter",
                "selection_policy": "first_eligible_in_draw_order",
                "execution_policy": "first_action" if qualification else "selected_chunk",
                "source_commit": head,
                "episode": episode_id,
                "session": owned,
                "endpoint": session.endpoint,
                "runtime": dict(attestation),
                "world_sha256": _WORLD,
                "config_sha256": _CONFIG,
                "validation_plan_sha256": _PLAN,
                "robot_sha256": _ROBOT,
                "caller_sha256": _sha(Path(__file__).read_bytes()),
                "prediction_error": {
                    "status": "not_applicable",
                    "value": None,
                    "reason": "arm_has_no_prediction",
                    "valid_paired_steps": 0,
                },
            },
        )
        probe = session.client.inspect_fingers()
        _persist(directory / "pre-reset-fingers.msgpack", _encode(probe, scope))
        _check_probe(probe, owned)
        config = load_worldmind_diagnostic_config(config_path)
        bindings = production_worldmind_bindings()
        fixture = build_offline_heuristic_comparison(config, bindings)
        monitor = bindings.safety[config.safety.binding](config, bindings.inputs.get("safety", {}))
        if not isinstance(monitor, SafetyMonitor):
            raise TypeError("arm-3 qualification requires the independent SafetyMonitor")
        _write_event(directory / "reset-intent.json", {"episode": episode_id, "seed": entry.seed})
        observation = session.client.reset(entry.scene, entry.seed)
        _persist(
            directory / "reset-observation.msgpack", _encode({"observation": observation}, scope)
        )
        _admit(observation)
        task = TaskSpec(config.task.instruction, config.task.target, config.task.destination)
        required = {task.target} | {item["def"] for item in entry.scene.objects}

        def observe(obs: Observation, evidence_path: Path) -> tuple[WorldState, GroundTruth]:
            truth = session.client.get_ground_truth()
            if not required <= truth.object_poses.keys():
                raise ValueError("episode is missing a frozen scene object")
            state = _world_state(obs, truth, config, target=task.target)
            validate_offline_observed_state(state, (task.target,))
            _persist(evidence_path, _encode({"state": state, "oracle": truth}, scope))
            return state, truth

        state, truth = observe(observation, directory / "observed-state.msgpack")
        verifier = None if qualification else fixture.build_verifier(observation)
        no_safe = RecoveryState()
        confirmed = 0
        terminal = "WORLDMIND_BOUND_EXHAUSTED"
        max_decisions = 1 if qualification else config.bounds.max_decisions
        for decision_index in range(max_decisions):
            if confirmed >= config.bounds.max_steps:
                break
            decision_dir = directory
            if not qualification:
                decision_dir = directory / f"decision-{decision_index:04d}"
                decision_dir.mkdir()
                _sync_dir(directory)

            generation, pool = fixture._pool(  # noqa: PLC2701
                task,
                state,
                monitor,
                episode_id=episode_id,
                seed=entry.seed,
                decision_index=decision_index,
            )
            rows, selected = fixture.assess_pool(state, observation, monitor, generation, pool)
            record = {
                "state": state,
                "task": task,
                "observation": observation,
                "oracle": truth,
                "generation": asdict(generation),
                "selected": None,
            }
            if selected is None:
                record["pool"] = [_row_data(row) for row in rows]
                digest = _persist(decision_dir / "decision.msgpack", _encode(record, scope))
                terminal = "empty_eligible_set" if qualification else "NO_SAFE_SET"
                if qualification:
                    break
                transition = no_safe.no_safe()
                if transition != "reobserve":
                    _write_event(
                        decision_dir / "recovery.json",
                        {
                            "transition": "alternate_candidate_family:unavailable",
                            "empty_sets": no_safe.no_safe_count,
                        },
                    )
                    break
                if decision_index + 1 == max_decisions:
                    terminal = "WORLDMIND_BOUND_EXHAUSTED"
                    break
                _write_event(
                    decision_dir / "recovery.json",
                    {
                        "transition": transition,
                        "empty_sets": no_safe.no_safe_count,
                    },
                )
                tick_status, after = _tick(
                    decision_dir,
                    session,
                    observation,
                    {"reobserve_decision_sha256": digest},
                    scope,
                )
            else:
                picked = replace(rows[selected], disposition="selected")
                rows[selected] = picked
                record.update(pool=[_row_data(row) for row in rows], selected=picked.candidate_id)
                actions = picked.chunk.actions[:1] if qualification else picked.chunk.actions
                decision_path = decision_dir / "decision.msgpack"
                for action_index, action in enumerate(actions):
                    if confirmed >= config.bounds.max_steps:
                        terminal = "WORLDMIND_BOUND_EXHAUSTED"
                        break
                    verdict = monitor.validate(action, observation.robot)
                    if not isinstance(verdict, ApplyResult) or type(verdict.accepted) is not bool:
                        raise ValueError("invalid independent motor verdict")
                    action_dir = decision_dir
                    verdict_sha = None
                    if not qualification:
                        action_dir = decision_dir / f"action-{action_index:02d}"
                        action_dir.mkdir()
                        _sync_dir(decision_dir)
                        verdict_sha = _persist(
                            action_dir / "motor-verdict.msgpack",
                            _encode({"action": action, "motor_verdict": verdict}, scope),
                        )
                    if action_index == 0:
                        record["motor_verdict"] = verdict
                        snapshot = _persist(decision_path, _encode(record, scope))
                    if not verdict.accepted:
                        terminal = "motor_veto"
                        break
                    tick_status, after = _operation(
                        action_dir,
                        session,
                        owned,
                        observation,
                        decision_path,
                        snapshot,
                        action,
                        action_index,
                        verdict_sha,
                        scope,
                    )
                    if after is None:
                        terminal = tick_status
                        break
                    confirmed += 1
                    observation = after
                    state, truth = observe(after, action_dir / "post-state.msgpack")
                else:
                    if verifier is None:
                        terminal = tick_status
                        break
                    # Verification is meaningful only after the selected chunk has finished.
                    result = verifier.check_observed(task, state)
                    transition = verifier.next_action(result)
                    _write_event(
                        decision_dir / "verification.json",
                        {
                            "success": result.success,
                            "reason": result.reason,
                            "transition": transition,
                            "verification_retries": verifier.recovery.verification_retries,
                        },
                    )
                    if transition == "retry_once":
                        terminal = "WORLDMIND_BOUND_EXHAUSTED"
                        continue
                    terminal = "COMPLETE" if transition == "complete" else "VERIFICATION_ABORT"
                break

            if after is None:
                terminal = tick_status
                break
            confirmed += 1
            observation = after
            state, truth = observe(after, decision_dir / "post-state.msgpack")
            terminal = "WORLDMIND_BOUND_EXHAUSTED"
        if terminal in {"tick_unknown", "tick_unobserved", "tick_not_executed"}:
            pinned = False
        _write_event(
            directory / "terminal.json",
            {"status": terminal, "scored": False, "confirmed_observed_ticks": confirmed},
        )
        return terminal
    except BaseException as exc:
        # On failure use the owned launch context for cleanup without another RPC.
        pinned = False
        try:
            _write_event(
                directory / "failure.json", {"type": type(exc).__name__, "reason": str(exc)}
            )
        except OSError as record_error:
            exc.add_note(f"qualification failure recording failed: {record_error}")
        raise
    finally:
        try:
            if pinned:
                receipt = session.client.shutdown_diagnostic_session(session_id=owned)
                _write_event(
                    directory / "shutdown.json", {"acknowledged": True, "receipt": receipt}
                )
        except Exception as exc:
            _write_event(directory / "shutdown-failure.json", {"reason": str(exc)})
        finally:
            session.client.close()
