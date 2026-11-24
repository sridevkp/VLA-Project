"""Guarded, non-scored ADR-0084 Webots qualification and episode execution.

The caller owns the launch context and its process cleanup. This module registers no CLI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlarobot.backends.webots.config import GRIPPER_OPEN
from vlarobot.backends.webots.runtime_attestation import read_owned_controller_session
from vlarobot.backends.worldmind_bindings import production_worldmind_bindings
from vlarobot.core.types import (
    Action,
    ApplyResult,
    GripperCmd,
    GroundTruth,
    Observation,
    TaskSpec,
    WorldState,
)
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
_PROXY_CONFIG = "7c22ff211f007b9377762819f3fb6d9561ea88cc751091cb85cce4e47135a965"
_EVAL = "79e4dfa708baea8bb85e839cbcbcc6e48bb2ca84eac830ebdbaffd19b9daaf4f"
_SCENARIOS = "e519dc59d09012ad94fb2ba10a269bdcda847272a607113990c1f22ffdefebfa"
_EPISODE = "validation-clear-i00-s0"
_FINGERS = ("gripper_left_finger_joint", "gripper_right_finger_joint")
_SCOPE = "non_scored_arm3_one_decision_qualification"
VALIDATION_SCOPE = "scored_arm3_validation"
MATCHED_PROXY_PATH = "configs/worldmind_arm5_matched_clearance.json"
MATCHED_PROXY_SHA = "6ea88763ad27a769943aa52132c60e785e4aa4e811899e34473c6f6cec17bed4"
STATEFUL_DIAGNOSTIC_SCOPE = "non_scored_stateful_candidate_00"
STATEFUL_PROXY_DIAGNOSTIC_SCOPE = "non_scored_stateful_arm5_equal"
STATEFUL_PROXY_VALIDATION_SCOPE = "scored_arm5_stateful_policy"
STATEFUL_PROXY_PATH = "configs/worldmind_arm5_stateful.yaml"
STATEFUL_PROXY_SHA = "d716898ddd48dbfdee30f6a2d561c2e0e298c18c613e17a5c82f595e95f07538"
STATEFUL_PROXY_PROFILES = {
    "equal": (STATEFUL_PROXY_PATH, STATEFUL_PROXY_SHA),
    "goal_x2": (
        "configs/worldmind_arm5_stateful_goal_x2.yaml",
        "e740105921543a4895053d12bf796e0c185c54c5a3b51dc5fee8f0b63a004888",
    ),
}
STATEFUL_WORLD_MODEL_SCOPE = "scored_arm4_stateful_equal"
STATEFUL_WORLD_MODEL_PROFILE = (
    "configs/worldmind_arm4_stateful_equal.yaml",
    "9474404c16e1b75209cfea38b6e10d9641238257729dbd7913ddccf14528e21d",
)
STATEFUL_VALIDATION_SCOPE = "scored_arm3_validation_stateful"
STATEFUL_REGISTRATION = "calibration/urdf-base-train-v1.yaml"
STATEFUL_REGISTRATION_SHA = "dfc542e11f17e97dfb52f695f75d0787ca1a3ceace0b80715826bb97271b5ea0"
STATEFUL_CONFIG_PATH = "configs/worldmind_arm3_stateful.yaml"
STATEFUL_CONFIG_SHA = "585ea6ef1ef850f2c0251b4fa797cb5a4032376a2ddca370144b0d35bf22c58b"
LIFT_RECOVERY_PROFILE_PATH = "configs/worldmind_stateful_lift_recovery.yaml"
LIFT_RECOVERY_PROFILE_SHA = "1640526ec78e8a260fb58d905f54e65bed08a3ea08be98e80c61e4f4d9539e66"
VERIFIED_HOME_PROFILE_PATH = "configs/worldmind_stateful_verified_home.yaml"
VERIFIED_HOME_PROFILE_SHA = "2a014f1e38e28e56d43c5ec0127aa9030ef59e2af4dbf180ffc2640cba486e85"
SETTLED_VALIDATION_SCOPE = "scored_arm3_validation_settled_init"
SETTLED_CONFIG_SHA = "98eb1ec19b3e322e32554cf9f6b6c004d9ebce96d1ea4849657c1d90360fb6e4"
PROXY_VALIDATION_SCOPE = "scored_arm5_proxy_validation"
_ARM3_TEST_SCOPE = "scored_arm3_test"
_ARM5_TEST_SCOPE = "scored_arm5_proxy_test"
_SCORED_SCOPES = frozenset(
    {
        STATEFUL_VALIDATION_SCOPE,
        STATEFUL_PROXY_VALIDATION_SCOPE,
        STATEFUL_WORLD_MODEL_SCOPE,
        VALIDATION_SCOPE,
        SETTLED_VALIDATION_SCOPE,
        PROXY_VALIDATION_SCOPE,
        _ARM3_TEST_SCOPE,
        _ARM5_TEST_SCOPE,
    }
)
FINGER_LOG_TOLERANCE_M = 1e-4


def executor_profile_sha(base_sha: str, verified_home: bool, lift_recovery: bool = False) -> str:
    """Bind the independently pinned HOME change to the existing arm profile."""
    if not verified_home:
        return base_sha
    profiles = [base_sha, VERIFIED_HOME_PROFILE_SHA]
    if lift_recovery:
        profiles.append(LIFT_RECOVERY_PROFILE_SHA)
    return _sha(":".join(profiles).encode("ascii"))


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


def finger_log_status(value: float) -> str:
    if type(value) not in (int, float) or not math.isfinite(value):
        return "invalid"
    if _raw(value):
        return "in_motor_range"
    if -FINGER_LOG_TOLERANCE_M <= value <= 0.045 + FINGER_LOG_TOLERANCE_M:
        return "outside_motor_range_within_logging_tolerance"
    return "outside_logging_tolerance"


def _check_probe(
    probe: dict[str, Any], session_id: dict[str, object], *, finger_logging_only: bool = False
) -> None:
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
        if finger_log_status(value) == "invalid" or (not finger_logging_only and not _raw(value)):
            raise ValueError(f"invalid raw finger coordinate: {name}: {value!r}")
        if (
            row.get("resolved_motor") != name
            or row.get("resolved_sensor") != name.replace("_finger_joint", "_sensor_finger_joint")
            or row.get("cached_sensor") != row.get("resolved_sensor")
            or row.get("motor_limits_m") != [0.0, 0.045]
            or row.get("sampling_period_ms") != 32
            or row.get("mapping") != "matches"
            or row.get("cached_pair_count") != 1
            or finger_log_status(row.get("cached_raw_coordinate_m")) == "invalid"
            or row["cached_raw_coordinate_m"] != value
        ):
            raise ValueError(f"unreconciled raw finger mapping: {name}")


def _admit(obs: Observation, *, finger_logging_only: bool = False) -> None:
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
        if finger_log_status(value) == "invalid" or (not finger_logging_only and not _raw(value)):
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
    result, apply_sha = _journal_apply(
        directory,
        session,
        session_id,
        decision_path,
        snapshot,
        action,
        action_index,
        verdict_sha,
        scope,
    )
    if not result.accepted:
        return "backend_rejected", None
    return _tick(directory, session, before, {"accepted_apply_sha256": apply_sha}, scope)


def _journal_apply(
    directory: Path,
    session: WebotsSession,
    session_id: dict[str, object],
    decision_path: Path,
    snapshot: str,
    action: Action,
    action_index: int,
    verdict_sha: str | None,
    scope: str,
) -> tuple[ApplyResult, str]:
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
    _verify_event(directory / "apply-result.json", apply_sha)
    return result, apply_sha


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
    _admit(
        after,
        finger_logging_only=scope in _SCORED_SCOPES
        or scope
        in {
            "non_scored_expert_candidate_00_parity",
            STATEFUL_DIAGNOSTIC_SCOPE,
            STATEFUL_PROXY_DIAGNOSTIC_SCOPE,
        },
    )
    if not math.isclose(after.t - before.t, FROZEN_CONTROL_DT_S):
        raise ValueError("post-tick observation time does not match one control tick")
    return "one_tick_observed", after


def wait_for_expert_opening(
    path: Path,
    session: WebotsSession,
    observation: Observation,
    *,
    remaining_ticks: int,
    elapsed_open_ticks: int,
) -> tuple[str, Observation | None, int]:
    """Reuse the expert INIT measured opening criterion and shared tick ceiling."""
    from vlarobot.backends.webots.config import GRIPPER_OPEN_TIMEOUT_S
    from vlarobot.execution.scripted_can import GRIPPER_OPEN_TOLERANCE_M

    path.mkdir()
    cap = round(GRIPPER_OPEN_TIMEOUT_S / FROZEN_CONTROL_DT_S)
    consumed = 0
    while math.isfinite(observation.robot.gripper_opening) and (
        observation.robot.gripper_opening < GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE_M
    ):
        if consumed >= remaining_ticks:
            return "WORLDMIND_BOUND_EXHAUSTED", None, consumed
        if consumed + elapsed_open_ticks >= cap:
            return "GRIPPER_OPEN_TIMEOUT", None, consumed
        tick_path = path / f"hold-{consumed:03d}"
        tick_path.mkdir()
        status, after = _tick(
            tick_path,
            session,
            observation,
            {"phase": "INIT", "hold_tick": consumed},
            SETTLED_VALIDATION_SCOPE,
        )
        if after is None:
            return status, None, consumed + (status == "tick_unobserved")
        consumed += 1
        observation = after
    if not math.isfinite(observation.robot.gripper_opening):
        raise ValueError("invalid observed opening for expert INIT")
    _write_event(
        path / "opening-reached.json",
        {
            "opening_m": observation.robot.gripper_opening,
            "additional_ticks": consumed,
            "observed_time_s": observation.t,
        },
    )
    return "opening_reached", observation, consumed


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


def run_heuristic_validation_episode(
    session: WebotsSession,
    *,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
    episode_id: str,
    settled_init: bool = False,
    stateful: bool = False,
    verified_home: bool = False,
) -> str:
    """Run one scored episode from the frozen VALIDATION plan in an owned session."""
    from datetime import date

    if date.today() < date(2026, 11, 21):
        raise ValueError("arm-3 live VALIDATION is outside the owner execution dates")
    return _run_owned(
        session,
        status_path,
        launch_started_at_s,
        directory,
        expected_head,
        episode_id,
        STATEFUL_VALIDATION_SCOPE
        if stateful
        else (SETTLED_VALIDATION_SCOPE if settled_init else VALIDATION_SCOPE),
        verified_home=verified_home,
    )


def run_proxy_validation_episode(
    session: WebotsSession,
    *,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
    episode_id: str,
    policy_id: str,
    matched_clearance: bool = False,
    stateful: bool = False,
    verified_home: bool = False,
    world_model: bool = False,
) -> str:
    """Run one scoped arm-5 candidate policy on a frozen VALIDATION episode."""
    from datetime import date

    from vlarobot.eval.worldmind_proxy_policy import GRID, OPTIONAL_GRID

    if world_model and (not stateful or not matched_clearance or policy_id != "equal"):
        raise ValueError("arm-4 requires the matched equal VALIDATION profile")
    if policy_id not in GRID | OPTIONAL_GRID:
        raise ValueError("unknown November 24 candidate policy")
    if date.today() < date(2026, 11, 21):
        raise ValueError("arm-5 live VALIDATION is outside the owner execution dates")
    return _run_owned(
        session,
        status_path,
        launch_started_at_s,
        directory,
        expected_head,
        episode_id,
        STATEFUL_WORLD_MODEL_SCOPE
        if world_model
        else (STATEFUL_PROXY_VALIDATION_SCOPE if stateful else PROXY_VALIDATION_SCOPE),
        policy_id,
        matched_clearance=matched_clearance,
        verified_home=verified_home,
    )


def run_frozen_test_episode(
    session: WebotsSession,
    *,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
    episode_id: str,
    frozen_policy: Path | None = None,
) -> str:
    """Run the scheduled TEST only, with arm 5 bound to the committed freeze artifact."""
    from datetime import date

    from vlarobot.execution.worldmind_nov24_campaign import committed_test_policy

    if date.today() < date(2026, 11, 30):
        raise ValueError("live TEST is outside the owner execution dates")
    policy_id = committed_test_policy(frozen_policy) if frozen_policy else None
    return _run_owned(
        session,
        status_path,
        launch_started_at_s,
        directory,
        expected_head,
        episode_id,
        _ARM5_TEST_SCOPE if policy_id else _ARM3_TEST_SCOPE,
        policy_id,
        "test",
    )


def _run_owned(
    session: WebotsSession,
    status_path: Path,
    launch_started_at_s: float,
    directory: Path,
    expected_head: str,
    episode_id: str,
    scope: str,
    proxy_policy: str | None = None,
    split: str = "validation",
    *,
    matched_clearance: bool = False,
    verified_home: bool = False,
) -> str:
    qualification = scope == _SCOPE
    world_model = scope == STATEFUL_WORLD_MODEL_SCOPE
    proxy_stateful = world_model or scope in {
        STATEFUL_PROXY_DIAGNOSTIC_SCOPE,
        STATEFUL_PROXY_VALIDATION_SCOPE,
    }
    stateful_path, stateful_sha = STATEFUL_PROXY_PROFILES.get(
        proxy_policy, (STATEFUL_CONFIG_PATH, STATEFUL_CONFIG_SHA)
    )
    if world_model:
        stateful_path, stateful_sha = STATEFUL_WORLD_MODEL_PROFILE
    stateful = proxy_stateful or scope in {STATEFUL_DIAGNOSTIC_SCOPE, STATEFUL_VALIDATION_SCOPE}
    if scope == STATEFUL_PROXY_DIAGNOSTIC_SCOPE and episode_id != "validation-clear-i00-s0":
        raise ValueError("stateful arm-5 diagnostic requires the frozen clear identity")
    if scope == STATEFUL_DIAGNOSTIC_SCOPE and (
        episode_id != "validation-clear-i00-s0" or proxy_policy or split != "validation"
    ):
        raise ValueError("stateful candidate-00 diagnostic requires the frozen clear identity")
    scored = scope in _SCORED_SCOPES
    lift_recovery = bool(verified_home and proxy_policy and not world_model)
    if lift_recovery and (
        _sha((_ROOT / LIFT_RECOVERY_PROFILE_PATH).read_bytes()) != LIFT_RECOVERY_PROFILE_SHA
    ):
        raise ValueError("unapproved LIFT recovery profile")
    if verified_home and (
        not stateful
        or not scored
        or split != "validation"
        or _sha((_ROOT / VERIFIED_HOME_PROFILE_PATH).read_bytes()) != VERIFIED_HOME_PROFILE_SHA
    ):
        raise ValueError("verified HOME requires pinned stateful VALIDATION")
    settled_init = (scope == SETTLED_VALIDATION_SCOPE or matched_clearance) and not stateful
    head = _verified_head(expected_head)
    config_path = _ROOT / "configs/worldmind_diagnostic.approved.yaml"
    for path, digest in (
        (config_path, _CONFIG),
        (_ROOT / "configs/robot.yaml", _ROBOT),
        (_ROOT / "worlds/factory.wbt", _WORLD),
        (_ROOT / "configs/eval.yaml", _EVAL),
        (_ROOT / "configs/scenarios.yaml", _SCENARIOS),
    ):
        if _sha(path.read_bytes()) != digest:
            raise ValueError(f"qualification pin mismatch: {path}")
    proxy_path = _ROOT / (
        MATCHED_PROXY_PATH if matched_clearance else "configs/worldmind_proxy_nov24.yaml"
    )
    proxy_pin = MATCHED_PROXY_SHA if matched_clearance else _PROXY_CONFIG
    geometry = None
    if proxy_policy:
        raw = proxy_path.read_bytes()
        if _sha(raw) != proxy_pin:
            raise ValueError("November 24 proxy config pin mismatch")
        if matched_clearance:
            geometry = json.loads(raw)
    if (
        settled_init
        and _sha((_ROOT / "configs/worldmind_arm3_settled_init.yaml").read_bytes())
        != SETTLED_CONFIG_SHA
    ):
        raise ValueError("settled INIT profile pin mismatch")
    if stateful and (
        os.environ.get("VLAROBOT_URDF_REGISTRATION_FILE") != STATEFUL_REGISTRATION
        or _sha((_ROOT / "configs" / STATEFUL_REGISTRATION).read_bytes())
        != STATEFUL_REGISTRATION_SHA
        or split != "validation"
        or (world_model and proxy_policy != "equal")
        or (
            proxy_stateful
            and (proxy_policy not in STATEFUL_PROXY_PROFILES or not matched_clearance)
        )
        or (not proxy_stateful and proxy_policy is not None)
        or _sha((_ROOT / stateful_path).read_bytes()) != stateful_sha
    ):
        raise ValueError("stateful executor scope or profile mismatch")
    plan = build_episode_plan(split, 0, allow_test=split == "test")
    plan_hash = _plan_sha256(plan)
    if (split == "validation" and (len(plan) != 18 or plan_hash != _PLAN)) or (
        split == "test" and len(plan) != 30
    ):
        raise ValueError("frozen episode plan pin mismatch")
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
                "arm": "vla_world_model"
                if world_model
                else ("full_system" if proxy_policy else "vla_risk_veto"),
                "prediction_mode": "analytic" if proxy_policy else "none",
                "predictive_veto": bool(proxy_policy and not world_model),
                "policy_status": "approved_pre_calibration_equal" if world_model else None,
                "actual_proposer": "scripted_chunk_adapter",
                "selection_policy": proxy_policy or "first_eligible_in_draw_order",
                "safety_scope": (
                    "motor_safety"
                    if world_model
                    else "matched_clearance_proxy"
                    if matched_clearance
                    else "tool_envelope_proxy_v1"
                    if proxy_policy
                    else "motor_safety"
                ),
                "proxy_config_sha256": proxy_pin if proxy_policy else None,
                "execution_policy": "observed_expert_phase_and_waypoint_cursor"
                if stateful
                else "selected_chunk_with_expert_INIT_wait"
                if settled_init
                else ("first_action" if qualification else "selected_chunk"),
                "executor_config_sha256": executor_profile_sha(
                    stateful_sha, verified_home, lift_recovery
                )
                if stateful
                else (SETTLED_CONFIG_SHA if settled_init else None),
                "lift_recovery_profile_sha256": LIFT_RECOVERY_PROFILE_SHA
                if lift_recovery
                else None,
                "home_profile_sha256": VERIFIED_HOME_PROFILE_SHA if verified_home else None,
                "expert_registration_sha256": STATEFUL_REGISTRATION_SHA if stateful else None,
                "source_commit": head,
                "episode": episode_id,
                "session": owned,
                "endpoint": session.endpoint,
                "runtime": dict(attestation),
                "world_sha256": _WORLD,
                "config_sha256": _CONFIG,
                "split": split,
                "plan_sha256": plan_hash,
                "robot_sha256": _ROBOT,
                "caller_sha256": _sha(Path(__file__).read_bytes()),
                "prediction_error": None
                if proxy_policy
                else {
                    "status": "not_applicable",
                    "value": None,
                    "reason": "arm_has_no_prediction",
                    "valid_paired_steps": 0,
                },
            },
        )
        probe = session.client.inspect_fingers()
        _persist(directory / "pre-reset-fingers.msgpack", _encode(probe, scope))
        _check_probe(probe, owned, finger_logging_only=scored or stateful)
        if scored:
            _write_event(
                directory / "finger-telemetry.json",
                {
                    "logging_only_tolerance_m": FINGER_LOG_TOLERANCE_M,
                    "pre_reset": [
                        finger_log_status(row["raw_coordinate_m"]) for row in probe["readings"]
                    ],
                },
            )
        config = load_worldmind_diagnostic_config(config_path)
        bindings = production_worldmind_bindings()
        fixture = build_offline_heuristic_comparison(config, bindings)
        monitor = bindings.safety[config.safety.binding](config, bindings.inputs.get("safety", {}))
        if not isinstance(monitor, SafetyMonitor):
            raise TypeError("arm-3 qualification requires the independent SafetyMonitor")
        if proxy_policy:
            from vlarobot.execution.worldmind_proxy_binding import verified_table_world

            tables = verified_table_world(_ROOT)
            predictor = bindings.surrogates[config.surrogate.binding](
                config,
                {
                    k: v
                    for k, v in bindings.inputs.get("surrogate", {}).items()
                    if k != "arm_sweep_config"
                },
            )
        _write_event(directory / "reset-intent.json", {"episode": episode_id, "seed": entry.seed})
        observation = session.client.reset(entry.scene, entry.seed)
        _persist(
            directory / "reset-observation.msgpack", _encode({"observation": observation}, scope)
        )
        _admit(observation, finger_logging_only=scored or stateful)
        task = TaskSpec(config.task.instruction, config.task.target, config.task.destination)
        required = {task.target} | {item["def"] for item in entry.scene.objects}

        def observe(obs: Observation, evidence_path: Path) -> tuple[WorldState, GroundTruth]:
            truth = session.client.get_ground_truth()
            if not required <= truth.object_poses.keys():
                raise ValueError("episode is missing a frozen scene object")
            state = _world_state(obs, truth, config, target=task.target)
            validate_offline_observed_state(state, (task.target,))
            record = {"state": state, "oracle": truth}
            if scored:
                values = obs.robot.diagnostics["finger_joint_sensor_rad"]
                record["finger_logging_status"] = {
                    name: finger_log_status(values[name]) for name in _FINGERS
                }
            _persist(evidence_path, _encode(record, scope))
            return state, truth

        state, truth = observe(observation, directory / "observed-state.msgpack")
        verifier = None if qualification else fixture.build_verifier(observation)
        if stateful:
            from vlarobot.execution.worldmind_stateful_chunk import run_stateful_expert

            return run_stateful_expert(
                session,
                owned,
                directory,
                entry,
                config,
                fixture,
                monitor,
                observation,
                verifier,
                task,
                observe,
                scored=scored,
                proxy=(predictor, geometry) if proxy_stateful else None,
                evidence_scope=scope,
                ranking_policy=proxy_policy or "equal",
                predictive_veto=not world_model,
                lift_entry_recovery=lift_recovery,
                verified_home=verified_home,
            )
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

            try:
                generation, pool = fixture._pool(  # noqa: PLC2701
                    task,
                    state,
                    monitor,
                    episode_id=episode_id,
                    seed=entry.seed,
                    decision_index=decision_index,
                )
            except ValueError as exc:
                if not scored or str(exc) != "scripted pose residual outside reviewed tolerance":
                    raise
                _write_event(
                    decision_dir / "generation-veto.json",
                    {
                        "status": "IK_INFEASIBLE",
                        "reason": str(exc),
                        "decision_index": decision_index,
                        "partial_pool_executed": False,
                    },
                )
                terminal = "IK_INFEASIBLE"
                break
            rows, selected = fixture.assess_pool(state, observation, monitor, generation, pool)
            proxy_records = None
            if proxy_policy:
                from vlarobot.execution.worldmind_proxy_binding import evaluate_proxy_pool

                assessments, policy_choices = evaluate_proxy_pool(
                    state, truth, pool, monitor, predictor, tables, config, geometry
                )
                selected = policy_choices[proxy_policy][0]
                proxy_records = {
                    "policy_choices": policy_choices,
                    "assessments": [
                        {"checks": assessment.hard_checks, "components": assessment.components}
                        for assessment in assessments
                    ],
                }
                rows = [
                    replace(
                        row,
                        disposition="eligible"
                        if all(value for _, value in assessment.hard_checks)
                        else "rejected",
                    )
                    for row, assessment in zip(rows, assessments, strict=True)
                ]
            record = {
                "proxy": proxy_records,
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
                if settled_init and (
                    len(actions) < 19
                    or not all(
                        isinstance(action, GripperCmd) and action.opening == GRIPPER_OPEN
                        for action in actions[:2]
                    )
                ):
                    raise ValueError("settled INIT requires the frozen opening command prefix")
                decision_path = decision_dir / "decision.msgpack"
                for action_index, action in enumerate(actions):
                    if settled_init and action_index == 2:
                        terminal, opened, spent = wait_for_expert_opening(
                            decision_dir / "INIT-settle",
                            session,
                            observation,
                            remaining_ticks=config.bounds.max_steps - confirmed,
                            elapsed_open_ticks=action_index,
                        )
                        confirmed += spent
                        if opened is None:
                            break
                        observation = opened
                        state, truth = observe(opened, decision_dir / "post-INIT-state.msgpack")
                    if settled_init and action_index == 18:
                        _write_event(
                            decision_dir / "pre-CLOSE.json",
                            {
                                "opening_m": observation.robot.gripper_opening,
                                "sim_time_s": observation.t,
                            },
                        )
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
            {"status": terminal, "scored": scored, "confirmed_observed_ticks": confirmed},
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
