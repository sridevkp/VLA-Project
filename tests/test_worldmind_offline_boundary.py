"""Offline fault injection for the ADR-0083 one-decision execution boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.worldmind_bindings import production_worldmind_bindings
from vlarobot.core.types import ApplyResult, GroundTruth, Pose3D, Vec3
from vlarobot.eval.report_contract import COMPONENTS
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config
from vlarobot.eval.worldmind_offline_evidence import read_evidence
from vlarobot.execution import worldmind_offline_boundary as boundary_module
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.execution.worldmind_offline_boundary import (
    OfflineBoundary,
    OfflineTick,
    OwnedFakeSession,
)
from vlarobot.execution.worldmind_world_model_comparison import (
    FixtureRanking,
    WorldModelComparisonFixture,
    build_offline_world_model_comparison,
)
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/worldmind_diagnostic.approved.yaml"
ROBOT = ROOT / "configs/robot.yaml"
LEFT = "gripper_left_finger_joint"
RIGHT = "gripper_right_finger_joint"
JOINT = "arm_7_joint"


class _Backend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.applies = 0
        self.ticks = 0
        self.shutdowns = 0
        self.fault = None

    def _observation(self):
        obs = super()._observation()
        obs.robot.diagnostics["finger_joint_sensor_rad"] = {LEFT: 0.045, RIGHT: 0.045}
        return obs

    def apply(self, action):
        self.applies += 1
        if self.fault == "apply_unknown":
            raise OSError("lost apply response")
        if self.fault == "apply_rejected":
            return ApplyResult(False, "injected rejection")
        return super().apply(action)

    def step(self, n_steps=1):
        self.ticks += 1
        if self.fault == "tick_unknown":
            raise OSError("lost tick response")
        return super().step(n_steps)

    def stop(self):
        self.shutdowns += 1
        if self.fault == "shutdown":
            raise OSError("lost shutdown response")
        super().stop()


@pytest.fixture
def composition():
    config = load_worldmind_diagnostic_config(CONFIG)
    bindings = production_worldmind_bindings()
    monitor = bindings.safety[config.safety.binding](config, bindings.inputs["safety"])

    class _Monitor(SafetyMonitor):
        veto = False

        def validate(self, action, state):
            return (
                ApplyResult(False, "injected motor veto")
                if self.veto
                else super().validate(action, state)
            )

    guard = _Monitor(monitor.limits)
    goal = Pose3D(Vec3(0.5, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))

    class _Predictor(AnalyticSurrogate):
        def predict_trajectory(self, state, chunk, **kwargs):
            result = super().predict_trajectory(state, chunk, **kwargs)
            if guard.fault == "motor":
                guard.veto = True
            if guard.fault == "prediction":
                result.states[-1].robot.base_pose_world = None
            return result

    guard.fault = None
    predictor = _Predictor(
        VelocityLimitedKinematics({JOINT: 1.0}, lambda joints: goal),
        inverse=lambda pose, robot: None,
        gripper_speed_m_s=0.03,
        gravity_m_s2=9.81,
        support_z_m={"CAN": 0.4},
        validate=lambda command, robot: monitor.validate(command, robot).accepted,
    )
    bindings = replace(
        bindings,
        surrogates={config.surrogate.binding: lambda *_: predictor},
        safety={config.safety.binding: lambda *_: guard},
        waypoint_factories={
            config.adapter.waypoints_binding: lambda *_: (
                ScriptedWaypoint("FIXTURE", 50, joints={JOINT: 0.02}),
            )
        },
    )
    # Existing fixture-only contract. Equal synthetic outputs exercise the draw-order tie.
    ranking = FixtureRanking(dict.fromkeys(COMPONENTS, lambda *_: 0.0), (1.0,) * 4, False)
    result = build_offline_world_model_comparison(config, bindings, fixture_ranking=ranking)
    hashes = {
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "plan": "013ca8ee314df6ca6b72505761c0ab7bc37caf4ec469f6696db80d7ce428fcb2",
        "definitions": result.provenance["objective_definitions_sha256"],
    }
    return result.fixture, guard, goal, hashes


def _event(path: Path, event: str):
    return json.loads((path / f"{event}.json").read_bytes())["payload"]


@pytest.mark.parametrize(
    ("fault", "applies", "ticks", "expected"),
    [
        ("success", 1, 1, ("accepted", "confirmed", 1)),
        ("negative", 0, 0, None),
        ("missing_finger", 0, 0, None),
        ("inconsistent_fingers", 0, 0, None),
        ("missing_pose", 0, 0, None),
        ("invalid_pose", 0, 0, None),
        ("incomplete_scene", 0, 0, None),
        ("incomplete_pool", 0, 0, None),
        ("prediction", 0, 0, None),
        ("prewrite", 0, 0, ("not_attempted", "not_executed", 0)),
        ("corrupt_intent", 0, 0, ("not_attempted", "not_executed", 0)),
        ("motor", 0, 0, ("not_attempted", "not_executed", 0)),
        ("apply_rejected", 1, 0, ("rejected", "not_executed", 0)),
        ("apply_unknown", 1, 0, ("unknown", "not_executed", 0)),
        ("tick_unknown", 1, 1, ("accepted", "unknown", 0)),
        ("observation", 1, 1, ("accepted", "confirmed", 0)),
        ("receipt_write", 1, 0, ("accepted", "not_executed", 0)),
        ("postwrite", 1, 1, None),
        ("wrong_session", 0, 0, None),
        ("changed_session", 0, 0, ("not_attempted", "not_executed", 0)),
        ("wrong_capability", 0, 0, None),
        ("shutdown", 1, 1, ("accepted", "confirmed", 1)),
    ],
)
def test_offline_boundary(composition, tmp_path, monkeypatch, fault, applies, ticks, expected):
    try:
        boundary_module._sync_dir(tmp_path)
    except OSError:
        pytest.skip("test filesystem cannot verify directory durability")
    fixture, guard, goal, hashes = composition
    backend = _Backend()
    backend.fault = fault
    guard.fault = fault
    observation = backend._observation()
    session = OwnedFakeSession(backend, "fixture-session")
    folder = tmp_path / "capture"
    boundary = OfflineBoundary(folder, session, "fixture-session", ("CAN",), ROBOT)
    params = dict(
        episode_id="validation-clear-i00-s0", seed=0, goal_base=goal, source_hashes=hashes
    )
    raw = observation.robot.diagnostics["finger_joint_sensor_rad"]

    if fault == "negative":
        raw.update({LEFT: -7.142594888992681e-07, RIGHT: 7.171095938981087e-07})
    elif fault == "missing_finger":
        del raw[RIGHT]
    elif fault == "inconsistent_fingers":
        raw[LEFT] = 0.04
    elif fault == "missing_pose":
        observation.robot.base_pose_world = None
    elif fault == "invalid_pose":
        observation.robot.base_pose_world = replace(
            observation.robot.base_pose_world, orientation=(0.0,) * 4
        )
    elif fault == "incomplete_scene":
        monkeypatch.setattr(backend, "get_ground_truth", GroundTruth)
    elif fault == "incomplete_pool":
        generate = WorldModelComparisonFixture.generate_fixture_pool

        def incomplete(self, *args, **kwargs):
            generation, rows = generate(self, *args, **kwargs)
            return generation, rows[:-1]

        monkeypatch.setattr(WorldModelComparisonFixture, "generate_fixture_pool", incomplete)
    elif fault in {"prewrite", "postwrite"}:
        write = boundary_module.write_evidence

        def disk_error(path, records):
            if path.name == ("pre_apply.msgpack" if fault == "prewrite" else "result.msgpack"):
                raise OSError("injected disk failure")
            return write(path, records)

        if fault == "prewrite":
            monkeypatch.setattr(boundary_module, "write_evidence", disk_error)
        else:
            monkeypatch.setattr(
                "vlarobot.execution.worldmind_offline_acquisition.write_evidence", disk_error
            )
    elif fault in {"corrupt_intent", "changed_session", "receipt_write"}:
        write = boundary_module._write_event

        def corrupt(path, record):
            if fault == "receipt_write" and path.name == "apply_result.json":
                raise OSError("injected receipt write failure")
            digest = write(path, record)
            if path.name == "apply_intent.json":
                if fault == "corrupt_intent":
                    path.write_bytes(
                        path.read_bytes().replace(b"command_budget", b"changed_budget")
                    )
                elif fault == "changed_session":
                    session.session_id = "replacement"
            return digest

        monkeypatch.setattr(boundary_module, "_write_event", corrupt)
    elif fault == "observation":

        def confirmed_without_observation(expected):
            session.require(expected)
            backend.step(1)
            return OfflineTick("confirmed")

        monkeypatch.setattr(session, "tick", confirmed_without_observation)
    elif fault == "wrong_session":
        session.session_id = "replacement"
    elif fault == "wrong_capability":
        monkeypatch.setattr(session, "describe", lambda: ("ordinary", "fixture-session"))

    if fault in {"wrong_session", "wrong_capability"}:
        with pytest.raises(ValueError, match="capability or session mismatch"):
            boundary.run(fixture, observation, **params)
        assert not folder.exists()
    elif fault == "postwrite":
        with pytest.raises(OSError, match="disk failure"):
            boundary.run(fixture, observation, **params)
        assert _event(folder, "apply_result")["outcome"] == "accepted"
        assert _event(folder, "tick_result")["outcome"] == "confirmed"
        assert not (folder / "decision_result.json").exists()
    else:
        result = boundary.run(fixture, observation, **params)
        archive = read_evidence(
            folder / "result.msgpack", expected_digest=result.acquisition.evidence.digest
        )
        attempts = archive.records["execution"]
        if expected is None:
            assert not attempts and archive.blockers
            assert not (folder / "apply_intent.json").exists()
        else:
            assert (
                attempts[0]["apply"],
                attempts[0]["step"],
                archive.records["pairing"]["valid_steps"],
            ) == expected
            if applies:
                assert archive.records["selected"] == "candidate-00"
                intent = _event(folder, "apply_intent")
                assert intent["session"] == "fixture-session"
                assert read_evidence(
                    folder / "pre_apply.msgpack", expected_digest=intent["decision_digest"]
                ).pool_complete
        assert bool(result.shutdown_error) == (fault in {"shutdown", "changed_session"})

    assert (backend.applies, backend.ticks) == (applies, ticks)
    assert backend.shutdowns == (
        0 if fault in {"wrong_session", "wrong_capability", "changed_session"} else 1
    )
    if folder.exists():
        # Even a session with an unresolved durable intent cannot use this directory again.
        fresh = OwnedFakeSession(_Backend(), "fixture-session")
        restarted = OfflineBoundary(folder, fresh, "fixture-session", ("CAN",), ROBOT)
        with pytest.raises(FileExistsError):
            restarted.run(fixture, observation, **params)
        assert (fresh.backend.applies, fresh.backend.ticks, fresh.backend.shutdowns) == (0, 0, 0)


def test_unsupported_storage_fails_before_actuation(composition, tmp_path, monkeypatch):
    fixture, _, goal, hashes = composition
    backend = _Backend()
    session = OwnedFakeSession(backend, "fixture-session")
    folder = tmp_path / "capture"
    boundary = OfflineBoundary(folder, session, "fixture-session", ("CAN",), ROBOT)

    def unsupported(path):
        raise OSError("directory synchronization unavailable")

    monkeypatch.setattr(boundary_module, "_sync_dir", unsupported)
    with pytest.raises(OSError, match="directory synchronization unavailable"):
        boundary.run(
            fixture,
            backend._observation(),
            episode_id="validation-clear-i00-s0",
            seed=0,
            goal_base=goal,
            source_hashes=hashes,
        )
    assert (backend.applies, backend.ticks, backend.shutdowns) == (0, 0, 1)
    assert not (folder / "apply_intent.json").exists()
