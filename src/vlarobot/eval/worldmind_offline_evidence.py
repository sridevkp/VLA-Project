"""Offline WORLDMIND evidence bundles. This module grants no calibration or execution authority."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.core.types import (
    ActionChunk,
    ApplyResult,
    BaseGoal,
    EEDelta,
    EEPoseTarget,
    GripperCmd,
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    TaskSpec,
    Vec3,
    WorldState,
)
from vlarobot.core.wire import decode_message, encode_message, make_message
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.worldmind_config import (
    FROZEN_BASE_SEED,
    FROZEN_CANDIDATE_COUNT,
    FROZEN_CONTROL_DT_S,
    FROZEN_HORIZON_STEPS,
)

VERSION = 1
EVIDENCE_CLASS = "OFFLINE_NON_SCIENTIFIC"
COMPONENTS = frozenset({"Jg", "Jl", "Jr", "Ju"})
FAILURES = frozenset(
    {
        "failed_generation",
        "partial_pool",
        "failed_rollout",
        "missing_prediction",
        "invalid_input",
        "incomplete_recording",
        "interrupted_execution",
        "unknown_backend_outcome",
        "non_comparable_pairing",
    }
)


@dataclass(frozen=True)
class OfflineEvidence:
    """Verified payload with independent structural and ranking completeness results."""

    digest: str
    records: dict[str, Any]
    pool_complete: bool
    ranking_evidence_complete: bool
    blockers: tuple[str, ...]


def _encode(data: dict[str, Any]) -> bytes:
    # Wire v3 preserves the typed values, tuple/list distinction, array bytes and map order.
    return encode_message(make_message("offline_evidence", 0, data))


def _decode(raw: bytes) -> dict[str, Any]:
    data = decode_message(raw)["payload"]
    if not isinstance(data, dict) or _encode(data) != raw:
        raise ValueError("noncanonical or duplicated evidence fields")
    return data


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def evidence_digest(value: dict[str, Any]) -> str:
    """Digest an ordered typed record at its capture boundary."""
    return _digest(_encode(value))


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _vec(value: Any) -> bool:
    return isinstance(value, Vec3) and all(_finite(x) for x in (value.x, value.y, value.z))


def _pose(value: Any) -> bool:
    return (
        isinstance(value, Pose3D)
        and _vec(value.position)
        and isinstance(value.orientation, tuple)
        and len(value.orientation) == 4
        and all(_finite(x) for x in value.orientation)
        and math.isclose(sum(x * x for x in value.orientation), 1.0, abs_tol=1e-8)
    )


def _planar(value: Any) -> bool:
    return isinstance(value, Pose2D) and all(_finite(x) for x in (value.x, value.y, value.yaw))


def _numeric_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(name, str) and name and _finite(number) for name, number in value.items()
    )


def _valid_action(action: Any) -> bool:
    if isinstance(action, JointTarget):
        return bool(action.q) and _numeric_map(action.q)
    if isinstance(action, GripperCmd):
        return _finite(action.opening) and (action.force is None or _finite(action.force))
    if isinstance(action, EEPoseTarget):
        return (
            _pose(action.pose)
            and _finite(action.duration)
            and (action.gripper is None or _finite(action.gripper))
        )
    if isinstance(action, EEDelta):
        return _vec(action.dpos) and _vec(action.drot) and _finite(action.gripper)
    return isinstance(action, BaseGoal) and _planar(action.pose)


def _robot(robot: Any) -> None:
    _require(isinstance(robot, RobotState), "RobotState required")
    _require(_pose(robot.base_pose_world), "full base_pose_world with unit quaternion required")
    _require(_pose(robot.ee_pose) and _planar(robot.base_pose), "invalid robot pose")
    _require(_numeric_map(robot.joint_pos) and _numeric_map(robot.joint_vel), "invalid joint data")
    _require(
        _finite(robot.gripper_opening)
        and (robot.gripper_holding is None or type(robot.gripper_holding) is bool)
        and isinstance(robot.diagnostics, dict),
        "invalid gripper or robot diagnostic state",
    )


def _state(state: Any, expected_objects: tuple[str, ...]) -> None:
    _require(isinstance(state, WorldState), "WorldState required")
    _robot(state.robot)
    objects = state.scene.objects
    _require(
        all(
            isinstance(obj, ObjectState)
            and _pose(obj.pose)
            and _vec(obj.dimensions)
            and min(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z) > 0
            and _finite(obj.confidence)
            for obj in objects
        ),
        "invalid scene object state",
    )
    identities = [obj.object_id for obj in objects]
    _require(
        len(identities) == len(set(identities)) and set(identities) == set(expected_objects),
        "missing, duplicate or unexpected scene objects",
    )


def _reference(value: dict[str, Any], blobs: dict[str, bytes]) -> str:
    raw = _encode(value)
    key = _digest(raw)
    blobs[key] = raw
    return key


def _resolve(key: str, blobs: dict[str, bytes], used: set[str]) -> dict[str, Any]:
    _require(_sha(key) and key in blobs, "missing evidence reference")
    raw = blobs[key]
    _require(isinstance(raw, bytes) and _digest(raw) == key, "corrupted evidence reference")
    used.add(key)
    return _decode(raw)


def _pack(records: dict[str, Any]) -> bytes:
    records = dict(records)
    blobs: dict[str, bytes] = {}
    records["input"] = _reference(records["input"], blobs) if records["input"] is not None else None
    records["candidates"] = [
        {**candidate, "payload": _reference(candidate["payload"], blobs)}
        for candidate in records["candidates"]
    ]
    return _encode(
        {
            "version": VERSION,
            "evidence_class": EVIDENCE_CLASS,
            "index": records,
            "payloads": dict(sorted(blobs.items())),
        }
    )


def _unpack(body: bytes) -> dict[str, Any]:
    envelope = _decode(body)
    _require(
        type(envelope["version"]) is int
        and envelope["version"] == VERSION
        and envelope["evidence_class"] == EVIDENCE_CLASS,
        "unsupported evidence version or authority",
    )
    records = envelope["index"]
    blobs = envelope["payloads"]
    used: set[str] = set()
    if records["input"] is not None:
        records["input"] = _resolve(records["input"], blobs, used)
    for candidate in records["candidates"]:
        candidate["payload"] = _resolve(candidate["payload"], blobs, used)
    _require(used == set(blobs), "unreferenced evidence payload")
    return records


def _decision_input(
    data: dict[str, Any], failures: set[str]
) -> tuple[WorldState | None, tuple[str, ...]]:
    if data is None:
        _require(
            bool(failures & {"invalid_input", "incomplete_recording"}), "missing decision input"
        )
        return None, ()
    state = data["state"]
    objects = data["expected_objects"]
    task = data["task"]
    _require(
        isinstance(objects, tuple)
        and bool(objects)
        and all(isinstance(x, str) and x for x in objects)
        and len(set(objects)) == len(objects)
        and isinstance(task, TaskSpec)
        and task.target in objects
        and data["scene_complete"] is True,
        "missing task binding or incomplete scene",
    )
    _require(
        data["frames"] == {"scene": "base", "goal": "base", "base_pose_world": "world"}
        and _pose(data["goal"])
        and _finite(data["time_s"]),
        "missing goal, timestamp, or frame binding",
    )
    _state(state, objects)
    return state, objects


def _candidate(
    candidate: dict[str, Any],
    index: int,
    seed: int,
    state: WorldState,
    objects: tuple[str, ...],
    failures: set[str],
) -> bool:
    _require(
        candidate["id"] == f"candidate-{index:02d}"
        and type(candidate["index"]) is int
        and candidate["index"] == index
        and type(candidate["seed"]) is int
        and candidate["seed"] == seed,
        "candidate ID, order or seed mismatch",
    )
    payload = candidate["payload"]
    chunk = payload["chunk"]
    _require(
        isinstance(chunk, ActionChunk)
        and chunk.source == "scripted"
        and chunk.dt == FROZEN_CONTROL_DT_S
        and isinstance(chunk.actions, list)
        and len(chunk.actions) == FROZEN_HORIZON_STEPS
        and all(map(_valid_action, chunk.actions)),
        "invalid ordered candidate action payload",
    )
    _require(
        candidate["draw_digest"]
        == evidence_digest({"id": candidate["id"], "seed": seed, "chunk": chunk}),
        "ordered draw payload does not match capture digest",
    )
    prediction = payload["states"]
    status = candidate["rollout"]
    _require(
        candidate["rollout_digest"]
        == evidence_digest(
            {
                "draw_digest": candidate["draw_digest"],
                "states": prediction,
            }
        ),
        "ordered rollout payload does not match capture digest",
    )
    _require(isinstance(prediction, list), "prediction states must be ordered")
    _require(len(prediction) <= FROZEN_HORIZON_STEPS + 1, "prediction exceeds frozen horizon")
    if prediction:
        _require(
            _encode({"state": prediction[0]}) == _encode({"state": state}),
            "prediction initial state does not match decision input",
        )
        for node in prediction:
            _state(node, objects)
    _require(
        status in {"complete", "failed_rollout", "missing_prediction"}
        and (
            (status == "complete" and len(prediction) == FROZEN_HORIZON_STEPS + 1)
            or (
                status == "failed_rollout"
                and status in failures
                and len(prediction) <= FROZEN_HORIZON_STEPS
            )
            or (status == "missing_prediction" and status in failures and not prediction)
        ),
        "inconsistent rollout status or horizon",
    )
    if status != "complete":
        _require(
            isinstance(candidate.get("rollout_reason"), str) and candidate["rollout_reason"],
            "rollout failure requires a candidate-specific reason",
        )
    checks = candidate["checks"]
    _require(
        isinstance(checks, dict)
        and all(isinstance(k, str) and k for k in checks)
        and all(isinstance(v, ApplyResult) and type(v.accepted) is bool for v in checks.values()),
        "invalid candidate checks",
    )
    components = candidate["components"]
    _require(
        components is None
        or (set(components) == COMPONENTS and all(map(_finite, components.values()))),
        "invalid objective components",
    )
    disposition = candidate["disposition"]
    _require(
        disposition in {"selected", "unselected", "hard_vetoed", "invalid", "unevaluable"},
        "invalid candidate disposition",
    )
    if disposition in {"selected", "unselected"}:
        _require(
            status == "complete" and all(v.accepted for v in checks.values()),
            "eligible disposition conflicts with candidate evidence",
        )
    if disposition == "hard_vetoed":
        _require(any(not v.accepted for v in checks.values()), "hard veto has no rejecting check")
    return status == "complete"


def _execution(
    records: dict[str, Any], state: WorldState | None, objects: tuple[str, ...], failures: set[str]
) -> None:
    selected = records["selected"]
    rows = records["candidates"]
    selections = [row for row in rows if row["disposition"] == "selected"]
    _require(
        [row["id"] for row in selections] == ([] if selected is None else [selected]),
        "selected identity does not match dispositions",
    )
    attempts = records["execution"]
    _require(not attempts or selected is not None, "execution lacks selected candidate")
    _require(
        selected is None or len(rows) == FROZEN_CANDIDATE_COUNT, "selection from an incomplete pool"
    )
    pairing = records["pairing"]
    paired = pairing["valid_steps"]
    _require(type(paired) is int and 0 <= paired <= FROZEN_HORIZON_STEPS, "invalid paired count")
    status = pairing["status"]
    if selections:
        actions = selections[0]["payload"]["chunk"].actions
        before = state
        for index, step in enumerate(attempts):
            _require(
                index < len(actions)
                and step["index"] == index
                and type(step["index"]) is int
                and _valid_action(step["action"]),
                "invalid execution command or index",
            )
            exact_action = _encode({"action": step["action"]}) == _encode(
                {"action": actions[index]}
            )
            _require(
                exact_action
                or (
                    status == "non_comparable"
                    and isinstance(step.get("deviation"), str)
                    and step["deviation"]
                ),
                "applied command does not match selected ordered payload",
            )
            _require(before is not None, "additional actuation after missing observation")
            _require(
                _encode({"state": before}) == _encode({"state": step["before"]}),
                "pre-apply observation does not match execution chain",
            )
            motor = step["motor"]
            _require(
                isinstance(motor, ApplyResult) and type(motor.accepted) is bool,
                "motor verdict required",
            )
            apply = step["apply"]
            tick = step["step"]
            _require(
                apply in {"accepted", "rejected", "unknown", "not_attempted"}
                and tick in {"confirmed", "unknown", "not_executed"},
                "invalid backend outcome",
            )
            _require(
                (apply != "not_attempted") == motor.accepted
                and (tick == "not_executed" or apply == "accepted"),
                "inconsistent motor/backend outcome",
            )
            if "unknown" in (apply, tick):
                _require(
                    "unknown_backend_outcome" in failures, "unknown backend outcome not declared"
                )
            _require(
                _finite(step["observation_time_s"])
                and math.isclose(
                    step["observation_time_s"],
                    records["input"]["time_s"] + index * FROZEN_CONTROL_DT_S,
                ),
                "execution observation time is misaligned",
            )
            before = step["after"]
            _require(
                before is None
                or (
                    _finite(step["result_time_s"])
                    and math.isclose(
                        step["result_time_s"],
                        records["input"]["time_s"] + (index + 1) * FROZEN_CONTROL_DT_S,
                    )
                ),
                "confirmed observation time is misaligned",
            )
            _require(
                before is not None or step["result_time_s"] is None,
                "timestamp claims a missing observation",
            )
            if before is None:
                _require(
                    index == len(attempts) - 1, "additional actuation after incomplete execution"
                )
                if apply == "accepted" and tick != "unknown":
                    _require(
                        bool(failures & {"incomplete_recording", "interrupted_execution"}),
                        "accepted apply with missing observation requires an explicit failure",
                    )
            if before is not None:
                _require(tick == "confirmed", "observation without confirmed step")
                _state(before, objects)
    aligned = sum(step["step"] == "confirmed" and step["after"] is not None for step in attempts)
    _require(
        (status == "paired" and paired == aligned == FROZEN_HORIZON_STEPS)
        or (status == "truncated" and 0 < paired == aligned < FROZEN_HORIZON_STEPS)
        or (status == "not_recorded" and paired == 0 and aligned == 0)
        or (status == "non_comparable" and paired == 0 and "non_comparable_pairing" in failures),
        "inconsistent prediction pairing",
    )
    if status in {"paired", "truncated"}:
        _require(pairing["candidate_id"] == selected, "pairing candidate mismatch")
    if status != "paired":
        _require(
            isinstance(pairing["boundary"], str) and pairing["boundary"],
            "pairing boundary required",
        )


def _ranking_record(records: dict[str, Any]) -> bool:
    ranking = records["ranking"]
    if ranking is None:
        return False
    scores = ranking["scores"]
    rows = records["candidates"]
    eligible = [row["id"] for row in rows if row["disposition"] in {"selected", "unselected"}]
    _require(
        isinstance(scores, dict)
        and set(scores) == set(eligible)
        and all(map(_finite, scores.values()))
        and type(ranking["maximize"]) is bool
        and _sha(ranking["parameter_ref"]),
        "invalid ranking evidence",
    )
    direction = -1 if ranking["maximize"] else 1
    ranked = sorted(eligible, key=lambda key: direction * scores[key])
    _require(
        ranking["order"] == ranked and records["selected"] == (ranked[0] if ranked else None),
        "ranking order or selected candidate inconsistent",
    )
    return True


def _validate(records: dict[str, Any], digest: str) -> OfflineEvidence:
    _require(
        records["arm"] == "vla_world_model"
        and records["split"] == "validation"
        and records["proposer"] == "scripted_chunk_adapter",
        "unexpected arm, split or proposer",
    )
    plan = build_episode_plan("validation", FROZEN_BASE_SEED, n=0, allow_test=False)
    episodes = {entry.scene.name: entry.seed for entry in plan}
    episode = records["episode"]
    index = records["decision_index"]
    _require(
        episode in episodes
        and type(index) is int
        and index >= 0
        and records["decision_id"] == f"{episode}-d{index:02d}",
        "invalid frozen decision identity",
    )
    seed = int(
        np.random.SeedSequence([FROZEN_BASE_SEED + episodes[episode], index]).generate_state(1)[0]
    )
    draws = np.random.default_rng(seed).integers(0, 2**31, FROZEN_CANDIDATE_COUNT).tolist()
    _require(
        type(records["decision_seed"]) is int
        and records["decision_seed"] == seed
        and records["draw_seeds"] == draws
        and all(type(x) is int for x in records["draw_seeds"]),
        "decision or draw seed mismatch",
    )
    source_hashes = records["source_hashes"]
    _require(
        {"source", "config", "plan", "definitions"} <= set(source_hashes)
        and all(map(_sha, source_hashes.values())),
        "missing source provenance hashes",
    )
    _require(isinstance(records["policy"], str) and records["policy"], "selection policy required")
    outcome = records["episode_outcome"]
    _require(
        isinstance(outcome, dict)
        and outcome["episode"] == episode
        and outcome["status"] in {"in_progress", "complete", "failed"}
        and (
            outcome["status"] == "in_progress"
            or (isinstance(outcome["classification"], str) and outcome["classification"])
        )
        and records["outcome_ref"] == evidence_digest(outcome),
        "missing or inconsistent episode outcome reference",
    )
    _require(
        isinstance(records["recovery"], list)
        and all(isinstance(x, str) for x in records["recovery"]),
        "invalid recovery transitions",
    )
    _require(
        _numeric_map(records["timings"]) and all(v >= 0 for v in records["timings"].values()),
        "invalid phase timing",
    )
    failure_rows = records["failures"]
    _require(
        isinstance(failure_rows, dict)
        and set(failure_rows) <= FAILURES
        and all(isinstance(v, str) and v for v in failure_rows.values()),
        "invalid failure reasons",
    )
    failures = set(failure_rows)
    state, objects = _decision_input(records["input"], failures)
    rows = records["candidates"]
    complete = len(rows) == FROZEN_CANDIDATE_COUNT
    _require(
        not complete or not failures & {"partial_pool", "failed_generation", "invalid_input"},
        "complete pool conflicts with declared failure",
    )
    _require(
        len(rows) <= FROZEN_CANDIDATE_COUNT
        and (not rows or state is not None)
        and (
            complete
            or bool(
                failures
                & {"failed_generation", "partial_pool", "invalid_input", "incomplete_recording"}
            )
        ),
        "inconsistent candidate pool",
    )
    rollout_complete = [
        _candidate(row, i, draws[i], state, objects, failures) for i, row in enumerate(rows)
    ]
    _execution(records, state, objects, failures)
    blockers = list(sorted(failures))
    if not complete or not all(rollout_complete):
        blockers.append("incomplete_pool_or_rollout")
    bindings = records["bindings"]
    _require(
        isinstance(bindings, dict)
        and set(bindings) <= COMPONENTS
        and all(map(_sha, bindings.values())),
        "invalid component binding references",
    )
    _require(
        set(bindings) == COMPONENTS or all(row["components"] is None for row in rows),
        "computed components lack binding provenance",
    )
    if set(bindings) != COMPONENTS or any(row["components"] is None for row in rows):
        blockers.append("missing_objective_binding_or_components")
    if not _ranking_record(records):
        blockers.append("ranking_not_recorded")
    if any(row["disposition"] in {"invalid", "unevaluable"} for row in rows):
        blockers.append("unevaluable_candidates")
    return OfflineEvidence(digest, records, complete, not blockers, tuple(blockers))


def validate_offline_observed_state(state: WorldState, expected_objects: tuple[str, ...]) -> None:
    """Validate a newly observed state without assigning an execution outcome."""
    _state(state, expected_objects)


def validate_offline_records(records: dict[str, Any]) -> None:
    """Check an in-memory index before allowing the offline fixture to apply an action."""
    _validate(records, "")


def read_evidence(path: Path, *, expected_digest: str | None = None) -> OfflineEvidence:
    """Validate the bundle, cross-references and semantic reconciliation without side effects."""
    try:
        envelope = _decode(path.read_bytes())
        body = envelope["body"]
        digest = envelope["sha256"]
        _require(isinstance(body, bytes) and _digest(body) == digest, "evidence digest mismatch")
        _require(expected_digest is None or digest == expected_digest, "unexpected evidence digest")
        return _validate(_unpack(body), digest)
    except (AttributeError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed offline decision evidence") from exc


def write_evidence(path: Path, records: Mapping[str, Any]) -> OfflineEvidence:
    """Exclusively create one verified bundle. An interrupted write fails reader validation."""
    try:
        body = _pack(dict(records))
        digest = _digest(body)
        result = _validate(_unpack(body), digest)
    except (AttributeError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed offline decision evidence") from exc
    with path.open("xb") as stream:
        stream.write(_encode({"body": body, "sha256": digest}))
    return result
