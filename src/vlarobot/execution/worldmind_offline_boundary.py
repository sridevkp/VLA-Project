"""One-decision, fail-closed acquisition boundary for owned FakeBackend sessions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import Action, ApplyResult, Observation, Pose3D, TaskSpec, WorldState
from vlarobot.eval.worldmind_offline_evidence import (
    EVIDENCE_CLASS,
    evidence_digest,
    read_evidence,
    write_evidence,
)
from vlarobot.execution.worldmind_offline_acquisition import (
    AcquisitionResult,
    acquire_fixture_decision,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vlarobot.execution.worldmind_world_model_comparison import WorldModelComparisonFixture

_CAPABILITY = "worldmind_fake_one_decision_v1"
_ROBOT_SHA256 = "1a9f0c2b57d771e45b6908d05177eb64eb80634b7ab2d35c5dfbefb1f863ca1d"


@dataclass(frozen=True)
class OfflineTick:
    status: Literal["confirmed", "not_executed", "unknown"]
    observation: Observation | None = None


class OwnedFakeSession:
    """Local capability and ownership model. The wrapped implementation must be FakeBackend."""

    def __init__(self, backend: FakeBackend, session_id: str) -> None:
        if (
            not isinstance(backend, FakeBackend)
            or not isinstance(session_id, str)
            or not session_id.strip()
        ):
            raise ValueError("owned offline session requires FakeBackend and an explicit identity")
        self.backend = backend
        self.session_id = session_id

    def describe(self) -> tuple[str, str]:
        return _CAPABILITY, self.session_id

    def require(self, expected: str) -> None:
        if self.describe() != (_CAPABILITY, expected):
            raise ValueError("offline capability or session mismatch")

    def apply(self, expected: str, action: Action) -> ApplyResult:
        self.require(expected)
        return self.backend.apply(action)

    def tick(self, expected: str) -> OfflineTick:
        self.require(expected)
        observation = self.backend.step(1)
        if not isinstance(observation, Observation):
            return OfflineTick("unknown")
        return OfflineTick("confirmed", observation)

    def shutdown(self, expected: str) -> None:
        self.require(expected)
        self.backend.stop()


@dataclass(frozen=True)
class BoundaryResult:
    acquisition: AcquisitionResult
    shutdown_error: str | None


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sync_dir(path: Path) -> None:
    # If directory durability is unavailable, fail before granting an operation.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_event(path: Path, event: dict[str, Any]) -> str:
    digest = _sha(_json(event))
    raw = _json({"payload": event, "sha256": digest})
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)
    if path.read_bytes() != raw:
        raise ValueError("event contents differ after durable write")
    return digest


def _verify_event(path: Path, digest: str) -> dict[str, Any]:
    raw = path.read_bytes()
    wrapper = json.loads(raw)
    event = wrapper["payload"]
    if (
        set(wrapper) != {"payload", "sha256"}
        or _json(wrapper) != raw
        or wrapper["sha256"] != digest
        or _sha(_json(event)) != digest
    ):
        raise ValueError("unverified operation intent or receipt")
    return event


class OfflineBoundary:
    """Exclusive, append-only decision journal. Existing journals never execute again."""

    def __init__(
        self,
        directory: Path,
        session: OwnedFakeSession,
        expected_session: str,
        expected_objects: tuple[str, ...],
        robot_config: Path,
    ) -> None:
        if type(session) is not OwnedFakeSession:
            raise TypeError("only an owned FakeBackend session is supported")
        self.directory = directory
        self.session = session
        self.expected_session = expected_session
        self.expected_objects = expected_objects
        if not expected_objects or len(set(expected_objects)) != len(expected_objects):
            raise ValueError("unique required scene identities must be declared")
        robot_bytes = robot_config.read_bytes()
        if _sha(robot_bytes) != _ROBOT_SHA256:
            raise ValueError("frozen robot configuration hash mismatch")
        config = yaml.safe_load(robot_bytes)
        self.finger_names = tuple(config["finger_joints"])
        self.minimum, self.maximum = config["gripper_opening_limits"]
        self.inverted = config["fingers_inverted"]
        self.arm_joints = set(config["arm_joints"])
        self.runtime = {
            "capability": _CAPABILITY,
            "session": expected_session,
            "backend": f"{type(session.backend).__module__}.{type(session.backend).__qualname__}",
            "robot_config": _ROBOT_SHA256,
            "boundary_source": _sha(Path(__file__).read_bytes()),
        }
        self.intent: str | None = None
        self.decision_digest: str | None = None
        self.apply_receipt: str | None = None
        self.tick_intent: str | None = None
        self.submitted = False
        self.ticked = False

    def admit(self, task: TaskSpec, state: WorldState, observation: Observation) -> None:
        self.session.require(self.expected_session)
        identities = [obj.object_id for obj in state.scene.objects]
        if len(identities) != len(self.expected_objects) or set(identities) != set(
            self.expected_objects
        ):
            raise ValueError("incomplete required fixture scene")
        if (
            task != self.task
            or task.target not in identities
            or type(observation.t) not in (float, int)
            or not math.isfinite(observation.t)
            or observation.t < 0
        ):
            raise ValueError("invalid task binding or observation timestamp")
        robot = state.robot
        if (
            not self.arm_joints <= robot.joint_pos.keys()
            or not self.arm_joints <= robot.joint_vel.keys()
        ):
            raise ValueError("missing configured arm joint state")
        if evidence_digest({"robot": robot}) != evidence_digest({"robot": observation.robot}):
            raise ValueError("observed and decision robot state differ")
        raw = robot.diagnostics.get("finger_joint_sensor_rad")
        values = []
        for name in self.finger_names:
            value = raw.get(name) if isinstance(raw, dict) else None
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"missing or invalid raw finger: {name}")
            if not self.minimum <= value <= self.maximum:
                raise ValueError(f"raw finger outside reviewed limits: {name}")
            values.append(value)
        left, right = values
        # Exact fixture consistency, with no physical noise allowance or inferred opening.
        opening = self.maximum - left if self.inverted else left - self.minimum
        if left != right or robot.gripper_opening != opening:
            raise ValueError("inconsistent raw finger readings or declared opening")

    def record(self, name: str, **detail: Any) -> str:
        if name == "apply_result":
            detail["intent_digest"] = self.intent
        if name == "tick_result":
            detail["intent_digest"] = self.tick_intent
        digest = _write_event(
            self.directory / f"{name}.json",
            {
                "evidence_class": EVIDENCE_CLASS,
                "event": name,
                "session": self.expected_session,
                **detail,
            },
        )
        if name == "apply_result":
            self.apply_receipt = digest
        return digest

    def prepare_apply(
        self, records: dict[str, Any] | None, action: Action, motor: ApplyResult | None
    ) -> None:
        self.session.require(self.expected_session)
        if (
            records is None
            or motor is None
            or motor.accepted is not True
            or self.intent is not None
        ):
            raise ValueError("unverified apply admission")
        if records["failures"] or len(records["candidates"]) != len(records["draw_seeds"]):
            raise ValueError("incomplete pre-apply candidate evidence")
        if any(row["rollout"] != "complete" for row in records["candidates"]):
            raise ValueError("incomplete pre-apply prediction evidence")
        self.admit(
            records["input"]["task"],
            records["input"]["state"],
            # The adapter already validated the observation. Verify command and input below.
            self.observation,
        )
        selected = next(row for row in records["candidates"] if row["id"] == records["selected"])
        command = evidence_digest({"action": action})
        if command != evidence_digest({"action": selected["payload"]["chunk"].actions[0]}):
            raise ValueError("actual command differs from selected candidate")
        snapshot = deepcopy(records)
        snapshot["boundary_runtime"] = self.runtime
        snapshot["motor_verdict_before_apply"] = motor
        snapshot["intended_action"] = action
        path = self.directory / "pre_apply.msgpack"
        evidence = write_evidence(path, snapshot)
        with path.open("rb+") as stream:
            os.fsync(stream.fileno())
        _sync_dir(self.directory)
        read_evidence(path, expected_digest=evidence.digest)
        self.decision_digest = evidence.digest
        self.intent = self.record(
            "apply_intent",
            decision_digest=evidence.digest,
            selected=records["selected"],
            command_digest=command,
            command_budget=1,
            tick_budget=1,
            runtime_digest=evidence_digest(self.runtime),
        )

    def submit(self, action: Action) -> ApplyResult:
        self.session.require(self.expected_session)
        if self.submitted or self.intent is None or self.decision_digest is None:
            raise ValueError("missing, unresolved or already spent apply intent")
        event = _verify_event(self.directory / "apply_intent.json", self.intent)
        if (
            event["command_digest"] != evidence_digest({"action": action})
            or event["decision_digest"] != self.decision_digest
        ):
            raise ValueError("intent differs from proposed operation")
        read_evidence(self.directory / "pre_apply.msgpack", expected_digest=self.decision_digest)
        self.submitted = True
        result = self.session.apply(self.expected_session, action)
        return result

    def prepare_tick(self, ticks: int) -> None:
        self.session.require(self.expected_session)
        if ticks != 1 or not self.submitted or self.ticked or self.apply_receipt is None:
            raise ValueError("durable accepted apply required before one tick")
        apply = _verify_event(self.directory / "apply_result.json", self.apply_receipt)
        if apply["outcome"] != "accepted" or apply["intent_digest"] != self.intent:
            raise ValueError("apply was not authoritatively accepted")
        self.tick_intent = self.record(
            "tick_intent", accepted_apply_digest=self.apply_receipt, ticks=1
        )

    def tick(self) -> OfflineTick:
        self.session.require(self.expected_session)
        if self.ticked or self.tick_intent is None:
            raise ValueError("tick budget exhausted or tick intent missing")
        _verify_event(self.directory / "tick_intent.json", self.tick_intent)
        self.ticked = True
        receipt = self.session.tick(self.expected_session)
        if not isinstance(receipt, OfflineTick) or receipt.status not in {
            "confirmed",
            "not_executed",
            "unknown",
        }:
            raise ValueError("invalid authoritative tick receipt")
        if receipt.status != "confirmed" and receipt.observation is not None:
            raise ValueError("unconfirmed tick cannot supply an observation")
        return receipt

    def run(
        self,
        fixture: WorldModelComparisonFixture,
        observation: Observation,
        *,
        episode_id: str,
        seed: int,
        goal_base: Pose3D,
        source_hashes: Mapping[str, str],
    ) -> BoundaryResult:
        """Start a fresh journal and perform at most one guarded offline command and tick."""
        self.session.require(self.expected_session)
        self.directory.mkdir()
        self.observation = observation
        task = fixture.config.task
        self.task = TaskSpec(task.instruction, task.target, task.destination)
        shutdown_error = None
        run_error = None
        try:
            _sync_dir(self.directory.parent)
            self.record("session_open", runtime=self.runtime, source_hashes=dict(source_hashes))
            acquired = acquire_fixture_decision(
                self.directory / "result.msgpack",
                fixture,
                self.session.backend,
                observation,
                episode_id=episode_id,
                seed=seed,
                goal_base=goal_base,
                source_hashes=source_hashes,
                boundary=self,
            )
            with (self.directory / "result.msgpack").open("rb+") as stream:
                os.fsync(stream.fileno())
            _sync_dir(self.directory)
            read_evidence(
                self.directory / "result.msgpack", expected_digest=acquired.evidence.digest
            )
            records = acquired.evidence.records
            attempts = records["execution"]
            if attempts:
                after = attempts[-1]["after"]
                self.record(
                    "observation_result",
                    outcome="available" if after is not None else "unavailable",
                    applied=attempts[-1]["apply"],
                    stepped=attempts[-1]["step"],
                    paired_steps=records["pairing"]["valid_steps"],
                    decision_result_digest=acquired.evidence.digest,
                )
            self.record("decision_result", digest=acquired.evidence.digest)
        except BaseException as exc:
            run_error = exc
            raise
        finally:
            try:
                self.session.shutdown(self.expected_session)
            except Exception as exc:
                shutdown_error = f"{type(exc).__name__}: {exc}"
            try:
                self.record(
                    "shutdown_result",
                    outcome="failed_or_unverified" if shutdown_error else "confirmed",
                    error=shutdown_error,
                )
            except Exception as exc:
                if run_error is None:
                    raise
                run_error.add_note(f"shutdown receipt could not be recorded: {exc}")
        return BoundaryResult(acquired, shutdown_error)
