"""Inert tests for the no-actuation Webots diagnostic boundary.

No test in this module starts Webots, opens a ZMQ socket, plans motion or scores a campaign.
The CLI tests inject a synthetic snapshot so the exact production call path is exercised without
simulator access.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import vlarobot.backends.read_only_state as read_only_state
import vlarobot.backends.worldmind_webots_diagnostic as adapter
import vlarobot.eval.runner as runner_module
import vlarobot.eval.worldmind_calibration as calibration_module
import vlarobot.eval.worldmind_preflight as preflight_module
import vlarobot.execution.worldmind_pipeline as pipeline_module
from vlarobot.backends.worldmind_webots_diagnostic import ReadOnlyBackendClient, _capture
from vlarobot.backends.zmq.server import SimServer
from vlarobot.cli import app
from vlarobot.core.types import (
    Action,
    CameraInfo,
    GroundTruth,
    Observation,
    Pose2D,
    Pose3D,
    RobotState,
    ScenarioSpec,
    Vec3,
)
from vlarobot.eval.worldmind_webots_diagnostic import (
    DiagnosticSnapshot,
    WebotsDiagnosticConfig,
    failed_webots_diagnostic,
    finalize_webots_diagnostic,
    load_webots_diagnostic_config,
    preflight_webots_diagnostic,
)

_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics")
_UNSET = object()
_EXECUTABLE = "/opt/webots/webots"
_EXECUTABLE_SHA = "a" * 64
_AVAILABILITY = {
    "pose_solver": True,
    "safety_monitor": True,
    "placement_verifier": True,
    "collision_checker": True,
    "required_arm_links": True,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config_dir = tmp_path / "configs"
    world_dir = tmp_path / "worlds"
    geometry_dir = tmp_path / "src" / "vlarobot" / "backends" / "webots"
    config_dir.mkdir()
    world_dir.mkdir()
    geometry_dir.mkdir(parents=True)
    world = world_dir / "factory.wbt"
    urdf = geometry_dir / "tiago.urdf"
    registration = tmp_path / "configs" / "calibration.yaml"
    robot = config_dir / "robot.yaml"
    for path, payload in (
        (world, b"diagnostic world bytes\n"),
        (urdf, b"<robot><link name='arm_1_link'/><link name='arm_2_link'/></robot>"),
        (registration, b"registration bytes\n"),
        (robot, b"robot settings bytes\n"),
    ):
        path.write_bytes(payload)
    raw: dict[str, object] = {
        "schema_version": 1,
        "status": "diagnostic_uncalibrated",
        "split": "validation",
        "state_source": "oracle",
        "mode": "launch",
        "backend": {
            "kind": "webots",
            "declaration": "webots_factory_world_supervisor_oracle_v1",
        },
        "webots": "webots",
        "world": "../worlds/factory.wbt",
        "world_sha256": _sha256(world),
        "endpoint": "tcp://127.0.0.1:5555",
        "target_object": "CAN",
        "robot_identity": "TIAGO_ROBOT",
        "webots_version": "R2025a",
        "seed": 0,
        "scenario": "validation-clear-i00-s0",
        "observation_freshness_max_s": 5.0,
        "geometry": {
            "urdf": "../src/vlarobot/backends/webots/tiago.urdf",
            "registration": "calibration.yaml",
            "robot_config": "robot.yaml",
            "required_arm_links": ["arm_1_link", "arm_2_link"],
            "swept_arm_certificate": None,
        },
        "dependencies": {
            "pose_solver_binding": "corrected_bounded_pose_solver_v1",
            "safety_binding": "runtime_safety_monitor_v1",
            "verification_binding": "placement_verifier_v1",
            "geometry_binding": "urdf_collision_capsules",
        },
        "diagnostic": {
            "mode": "read_only_inspection",
            "no_actuation": True,
            "scored_evaluation_allowed": False,
            "evidence_collection_split": "validation",
        },
    }
    path = config_dir / "webots-diagnostic.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    return path, raw


def _write_raw(path: Path, raw: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")


def _launch_attestation(config: WebotsDiagnosticConfig, **overrides: object) -> dict[str, object]:
    attestation: dict[str, object] = {
        "executable_path": _EXECUTABLE,
        "executable_sha256": _EXECUTABLE_SHA,
        "reported_version": config.webots_version,
        "launch_command_executable": _EXECUTABLE,
        "version_probe_command": [_EXECUTABLE, "--version"],
        "version_probe_returncode": 0,
        "version_probe_error": None,
        "process_pid": 4242,
        "process_executable_path": _EXECUTABLE,
        "process_identity_source": "linux_proc_exe",
        "process_identity_verified": True,
        "verified_executable_version_pair": True,
        "evidence_source": "launch_time_executable_version_probe",
        "captured_at_utc": "2026-11-18T00:00:00+00:00",
    }
    attestation.update(overrides)
    return attestation


def _pose() -> Pose3D:
    return Pose3D(Vec3(0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))


def _observation() -> Observation:
    camera = CameraInfo(
        K=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        T_base_cam=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        width=4,
        height=4,
    )
    robot = RobotState(
        joint_pos={"arm_1_joint": 0.1},
        joint_vel={"arm_1_joint": 0.0},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=_pose(),
        gripper_opening=0.1,
    )
    return Observation(
        t=1.0,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        cam_info=camera,
        robot=robot,
    )


def _snapshot(
    config: WebotsDiagnosticConfig,
    *,
    age_s: float = 0.05,
    first_command: int | None = None,
    call_counts: dict[str, int] | None = None,
    availability: dict[str, bool] | None = None,
    session_id: object = _UNSET,
    world_hash: object = _UNSET,
    robot_def: object = _UNSET,
    robot_def_cross_checked: object = True,
    runtime_version: object = _UNSET,
    attestation: object = _UNSET,
) -> DiagnosticSnapshot:
    observed_world_hash = config.world_sha256 if world_hash is _UNSET else world_hash
    observed_version = config.webots_version if runtime_version is _UNSET else runtime_version
    observed_def = config.robot_identity if robot_def is _UNSET else robot_def
    observed_attestation = (
        _launch_attestation(config)
        if attestation is _UNSET
        else ({} if attestation is None else dict(attestation))
    )
    return DiagnosticSnapshot(
        ping={
            "status": "ok",
            "session_id": (
                {"pid": 42, "start_time": 1.0, "nonce": "nonce-42"}
                if session_id is _UNSET
                else session_id
            ),
            "world_sha256": observed_world_hash,
            "kinematics_telemetry_enabled": False,
            "endpoint": "tcp://127.0.0.1:5555",
        },
        observation=_observation(),
        ground_truth=GroundTruth(object_poses={"CAN": _pose()}),
        identity={
            "simulator": "webots",
            "robot_def": observed_def,
            "robot_def_cross_checked": robot_def_cross_checked,
            "robot_name": "tiago-node-name",
            "robot_type": "Tiago",
            "webots_version": observed_version,
            "webots_version_available": observed_version is not None,
            "webots_version_source": ("WEBOTS_VERSION" if observed_version is not None else None),
            "controller_webots_version": observed_version,
            "controller_webots_version_source": (
                "WEBOTS_VERSION" if observed_version is not None else None
            ),
            "first_command_step_count": first_command,
        },
        captured_at_utc="2026-11-17T00:00:00+00:00",
        age_s=age_s,
        dependency_availability=dict(availability or _AVAILABILITY),
        call_counts=dict(call_counts or {name: 0 for name in _FORBIDDEN_CALLS}),
        launch_attestation=observed_attestation,
    )


def _ping_envelope(ping: dict[str, object]) -> dict[str, object]:
    """Wrap a flattened snapshot ping back into the documented ZMQ reply envelope."""
    payload = {key: value for key, value in ping.items() if key not in {"endpoint", "launch_mode"}}
    return {
        "protocol_version": 3,
        "type": "ping",
        "seq": 1,
        "sim_time": 0.0,
        "error": None,
        "payload": payload,
    }


def test_config_loader_rejects_scoring_test_train_and_placeholders(tmp_path):
    path, raw = _write_config(tmp_path)
    loaded = load_webots_diagnostic_config(path)
    assert loaded.mode == "launch"
    assert loaded.state_source == "oracle"
    assert loaded.split == "validation"
    assert loaded.swept_arm_certificate is None

    for key in ("test", "train", "dataset", "scoring", "calibration", "components", "weights"):
        blocked = copy.deepcopy(raw)
        blocked[key] = {}
        blocked_path = tmp_path / f"{key}-config.yaml"
        _write_raw(blocked_path, blocked)
        with pytest.raises(ValueError, match="forbids keys"):
            load_webots_diagnostic_config(blocked_path)

    perception = copy.deepcopy(raw)
    perception["state_source"] = "perception"
    _write_raw(path, perception)
    with pytest.raises(ValueError, match="declared as oracle"):
        load_webots_diagnostic_config(path)

    scored = copy.deepcopy(raw)
    scored["diagnostic"]["scored_evaluation_allowed"] = True
    _write_raw(path, scored)
    with pytest.raises(ValueError, match="can never allow scored evaluation"):
        load_webots_diagnostic_config(path)

    placeholder = copy.deepcopy(raw)
    placeholder["backend"]["declaration"] = "<DECLARED BACKEND>"
    _write_raw(path, placeholder)
    with pytest.raises(ValueError, match="blocked template placeholder"):
        load_webots_diagnostic_config(path)


def test_missing_geometry_fails_closed_before_any_backend_collection(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    (tmp_path / "configs" / "calibration.yaml").unlink()
    config = load_webots_diagnostic_config(path)
    preflight = preflight_webots_diagnostic(config)
    assert any("missing geometry/config file" in blocker for blocker in preflight.blockers)
    report = failed_webots_diagnostic(config, preflight, "config_preflight_failed")
    assert report["outcome"] == "FAILED"
    assert report["motion_allowed"] is False
    assert report["no_actuation"] is True
    assert "unverified-arm-sweep" in report["motion_blockers"]

    called = []

    def forbidden_collect(*args, **kwargs):
        called.append(True)
        raise AssertionError("backend collection must not start after a geometry failure")

    monkeypatch.setattr(adapter, "collect_diagnostic_snapshot", forbidden_collect)
    out = tmp_path / "missing-geometry"
    result = CliRunner().invoke(
        app,
        ["worldmind-webots-diagnostic", "--config", str(path), "--out", str(out)],
    )
    assert result.exit_code == 1
    assert called == []
    written = json.loads((out / "diagnostic.json").read_text(encoding="utf-8"))
    assert written["outcome"] == "FAILED"
    assert written["backend_apply_calls"] == 0
    assert written["backend_step_calls"] == 0


def test_missing_sweep_certificate_permits_inspection_but_blocks_motion(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    preflight = preflight_webots_diagnostic(config)
    assert preflight.blockers == ()
    assert "unverified-arm-sweep" in preflight.motion_blockers
    report, state, provenance = finalize_webots_diagnostic(config, preflight, _snapshot(config))
    assert report["outcome"] == "PASS"
    assert report["read_only_inspection_allowed"] is True
    assert report["motion_allowed"] is False
    assert report["no_actuation"] is True
    assert "unverified-arm-sweep" in report["motion_blockers"]
    assert state["oracle_target_object"] == "CAN"
    assert provenance["frozen_campaign_modified"] is False


def test_stale_or_nonfinite_state_fails_closed(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    preflight = preflight_webots_diagnostic(config)
    stale = _snapshot(config, age_s=config.observation_freshness_max_s + 1.0)
    stale_report, _, _ = finalize_webots_diagnostic(config, preflight, stale)
    assert stale_report["outcome"] == "FAILED"
    assert "observation is stale" in stale_report["blockers"]

    command = _snapshot(config, first_command=7)
    command_report, _, _ = finalize_webots_diagnostic(config, preflight, command)
    assert command_report["outcome"] == "FAILED"
    assert any("actuator command was issued" in item for item in command_report["blockers"])

    counts = {name: 0 for name in _FORBIDDEN_CALLS}
    counts["step"] = 1
    step_report, _, _ = finalize_webots_diagnostic(
        config, preflight, _snapshot(config, call_counts=counts)
    )
    assert step_report["outcome"] == "FAILED"
    assert any("forbidden backend call" in item for item in step_report["blockers"])


def test_read_only_client_blocks_every_state_changing_call():
    class LowLevel:
        def ping(self):
            return {"status": "ok"}

        def inspect(self):
            return {}

        def close(self):
            return None

    client = ReadOnlyBackendClient(LowLevel())
    with pytest.raises(RuntimeError, match="apply is forbidden"):
        client.apply(Action())
    with pytest.raises(RuntimeError, match="step is forbidden"):
        client.step(1)
    with pytest.raises(RuntimeError, match="reset is forbidden"):
        client.reset(ScenarioSpec(), 0)
    with pytest.raises(RuntimeError, match="export_kinematics is forbidden"):
        client.export_kinematics()
    assert client.call_counts() == {name: 1 for name in _FORBIDDEN_CALLS}


def test_inspect_rpc_dispatch_never_calls_apply_step_or_reset(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    snapshot = _snapshot(config)

    class InspectBackend:
        def __init__(self):
            self.calls: list[str] = []

        def read_only_state(self):
            return {
                "observation": snapshot.observation,
                "ground_truth": snapshot.ground_truth,
                "identity": dict(snapshot.identity),
            }

        def apply(self, action):
            self.calls.append("apply")
            raise AssertionError("apply must not be called")

        def step(self, n_steps=1):
            self.calls.append("step")
            raise AssertionError("step must not be called")

        def reset(self, scenario, seed):
            self.calls.append("reset")
            raise AssertionError("reset must not be called")

        def stop(self):
            return None

    backend = InspectBackend()
    server = SimServer(backend)  # type: ignore[arg-type]
    response = server.handle({"seq": 1, "type": "inspect", "payload": {}})
    assert response["type"] == "inspect"
    assert response["error"] is None
    assert "observation" in response["payload"]
    assert backend.calls == []


def test_dependency_probe_fails_closed_on_unknown_geometry_binding(tmp_path):
    from dataclasses import replace

    from vlarobot.backends.worldmind_webots_diagnostic import dependency_availability

    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    availability = dependency_availability(config)
    assert availability["pose_solver"] is True
    assert availability["safety_monitor"] is True
    assert availability["placement_verifier"] is True
    assert availability["collision_checker"] is True  # synthetic URDF declares both links
    unknown = replace(config, geometry_binding="not_a_registered_checker")
    assert dependency_availability(unknown)["collision_checker"] is False


def test_cli_writes_read_only_evidence_without_scorer_or_split_access(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    snapshot = _snapshot(config)
    monkeypatch.setattr(
        adapter,
        "collect_diagnostic_snapshot",
        lambda resolved, trace_dir, **kwargs: snapshot,
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scored evaluator, calibration or split planner must not be called")

    monkeypatch.setattr(pipeline_module, "run_worldmind_evaluation", forbidden)
    monkeypatch.setattr(preflight_module, "run_worldmind_preflight", forbidden)
    monkeypatch.setattr(calibration_module, "evaluate_calibration", forbidden)
    monkeypatch.setattr(runner_module, "build_episode_plan", forbidden)

    out = tmp_path / "readonly-evidence"
    result = CliRunner().invoke(
        app,
        ["worldmind-webots-diagnostic", "--config", str(path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out / "diagnostic.json").read_text(encoding="utf-8"))
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "PASS"
    assert report["scored_evaluator"] is None
    assert report["scored_results_written"] is False
    assert report["test_split_accessed"] is False
    assert report["oracle_equivalent_to_perception"] is False
    assert report["backend_apply_calls"] == 0
    assert report["backend_step_calls"] == 0
    assert report["backend_reset_calls"] == 0
    assert report["backend_export_calls"] == 0
    assert report["actuator_commands"] == []
    assert report["motion_allowed"] is False
    assert state["oracle_target_pose"] is not None
    assert provenance["frozen_campaign_modified"] is False


def test_output_directory_is_exclusive_and_never_overwritten(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    out = tmp_path / "prior-evidence"
    out.mkdir()
    prior = out / "diagnostic.json"
    prior.write_text('{"outcome": "PRIOR"}', encoding="utf-8")

    def forbidden_collect(*args, **kwargs):
        raise AssertionError("backend collection must not start for an existing output directory")

    monkeypatch.setattr(adapter, "collect_diagnostic_snapshot", forbidden_collect)
    result = CliRunner().invoke(
        app,
        ["worldmind-webots-diagnostic", "--config", str(path), "--out", str(out)],
    )
    assert result.exit_code == 1
    assert "prior evidence directory" in result.stderr
    assert prior.read_text(encoding="utf-8") == '{"outcome": "PRIOR"}'
    assert sorted(path.name for path in out.iterdir()) == ["diagnostic.json"]


def test_capture_parses_realistic_ping_envelope_and_rejects_missing_payload(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    snapshot = _snapshot(config)

    class FakeLowLevel:
        def __init__(self, envelope: dict[str, object]):
            self.envelope = envelope

        def ping(self):
            return copy.deepcopy(self.envelope)

        def inspect(self):
            return {
                "observation": snapshot.observation,
                "ground_truth": snapshot.ground_truth,
                "identity": dict(snapshot.identity),
            }

        def close(self):
            return None

    envelope = _ping_envelope(snapshot.ping)
    captured = _capture(
        ReadOnlyBackendClient(FakeLowLevel(envelope)),
        config,
        "tcp://127.0.0.1:5555",
        _launch_attestation(config),
    )
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    assert captured.ping["session_id"] == payload["session_id"]
    assert captured.ping["world_sha256"] == config.world_sha256
    assert "payload" not in captured.ping
    report, _, _ = finalize_webots_diagnostic(config, preflight_webots_diagnostic(config), captured)
    assert report["outcome"] == "PASS"

    with pytest.raises(ValueError, match="omitted payload"):
        _capture(
            ReadOnlyBackendClient(FakeLowLevel({"status": "ok"})),
            config,
            "tcp://127.0.0.1:5555",
        )


def test_attestation_blockers_distinguish_missing_from_mismatch(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    preflight = preflight_webots_diagnostic(config)
    cases = [
        ({"session_id": None}, "simulator session identity is missing", None),
        ({"session_id": {"pid": 1}}, "simulator session identity is incomplete", None),
        ({"world_hash": None}, "running Webots world hash is missing", "mismatch"),
        ({"world_hash": "0" * 64}, "running Webots world hash mismatch:", "is missing"),
        ({"robot_def": None}, "robot DEF identity is missing", "mismatch"),
        ({"robot_def": "OTHER_ROBOT"}, "robot identity mismatch:", "is missing"),
        (
            {"robot_def_cross_checked": False},
            "robot DEF identity was not cross-checked",
            None,
        ),
        (
            {"attestation": _launch_attestation(config, reported_version=None)},
            "launch executable reported Webots version is unavailable",
            "launch executable Webots version mismatch",
        ),
        (
            {"attestation": _launch_attestation(config, reported_version="R2024b")},
            "launch executable Webots version mismatch:",
            "launch executable reported Webots version is unavailable",
        ),
        (
            {"runtime_version": "R2024b"},
            "controller-reported Webots version mismatch:",
            "launch executable Webots version mismatch",
        ),
    ]
    for kwargs, expected, forbidden in cases:
        report, _, _ = finalize_webots_diagnostic(config, preflight, _snapshot(config, **kwargs))
        blockers = report["blockers"]
        assert any(expected in item for item in blockers), (kwargs, blockers)
        if forbidden is not None:
            assert not any(forbidden in item for item in blockers), (kwargs, blockers)
        assert report["outcome"] == "FAILED"
        assert report["motion_allowed"] is False


@pytest.mark.parametrize(
    ("def_name", "node_name", "runtime_version", "expected_def", "expected_version"),
    [
        ("TIAGO_ROBOT", "tiago-node-name", "R2025a", "TIAGO_ROBOT", "R2025a"),
        ("OTHER_ROBOT", "TIAGO_ROBOT", "R2025a", "OTHER_ROBOT", "R2025a"),
    ],
)
def test_read_only_identity_uses_def_not_node_name(
    tmp_path, monkeypatch, def_name, node_name, runtime_version, expected_def, expected_version
):
    monkeypatch.delenv("WEBOTS_VERSION", raising=False)

    class FakeNode:
        def getDef(self):
            return def_name

        def getName(self):
            return node_name

        def getTypeName(self):
            return "Tiago"

        def getId(self):
            return 77

    node = FakeNode()

    class FakeSupervisor:
        def getFromDef(self, name):
            return node if name == node.getDef() else None

        def getVersion(self):
            return runtime_version

        def getSelf(self):
            return node

    class FakeIO:
        robot = FakeSupervisor()
        me = node

    class FakeBackend:
        io = FakeIO()
        _physics_step_count = 0
        _first_command_step_count = None
        _stopped = False

        def __init__(self):
            self.calls: list[str] = []

        def _observation(self):
            self.calls.append("observation")
            return _observation()

        def get_ground_truth(self):
            self.calls.append("ground_truth")
            return GroundTruth(object_poses={"CAN": _pose()})

        def apply(self, action):
            raise AssertionError("apply must not be called")

        def step(self, n_steps=1):
            raise AssertionError("step must not be called")

        def reset(self, scenario, seed):
            raise AssertionError("reset must not be called")

        def export_kinematics(self, action="poll"):
            raise AssertionError("export_kinematics must not be called")

    backend = FakeBackend()
    identity = read_only_state.capture_read_only_state(backend)["identity"]
    assert identity["robot_def"] == expected_def
    assert identity["robot_name"] == node_name
    assert identity["robot_type"] == "Tiago"
    assert identity["robot_def_cross_checked"] is True
    assert identity["webots_version"] == expected_version
    assert identity["webots_version_available"] is True
    assert identity["webots_version_source"] == "Webots controller API getVersion"
    assert identity["controller_webots_version"] == expected_version
    assert identity["controller_webots_version_source"] == "Webots controller API getVersion"
    assert backend.calls == ["observation", "ground_truth"]

    monkeypatch.setenv("WEBOTS_VERSION", "Webots R2024b")
    env_identity = read_only_state.capture_read_only_state(backend)["identity"]
    assert env_identity["controller_webots_version"] == "R2024b"
    assert env_identity["controller_webots_version_source"] == "WEBOTS_VERSION"
    assert env_identity["webots_version_source"] == "WEBOTS_VERSION"


def test_valid_launch_attestation_marks_runtime_version_verified(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    report, _, _ = finalize_webots_diagnostic(
        config,
        preflight_webots_diagnostic(config),
        _snapshot(config),
    )
    assert report["outcome"] == "PASS"
    assert report["runtime_webots_version_verified"] is True
    assert report["runtime_webots_version_source"] == "launch_executable"
    assert report["runtime_webots_version_observed"] == config.webots_version
    assert report["launch_webots_attestation"]["verified_executable_version_pair"] is True
    assert report["motion_allowed"] is False
    assert "unverified-arm-sweep" in report["motion_blockers"]


def test_launch_attestation_evidence_cases_fail_closed(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    preflight = preflight_webots_diagnostic(config)
    cases = [
        (None, "launch-time Webots version evidence is unavailable"),
        ({}, "launch-time Webots version evidence is unavailable"),
        (
            _launch_attestation(config, reported_version=None),
            "launch executable reported Webots version is unavailable",
        ),
        (
            _launch_attestation(config, executable_path="/opt/webots/other-webots"),
            "launch executable path mismatch:",
        ),
        (
            _launch_attestation(config, reported_version="R2024b"),
            "launch executable Webots version mismatch:",
        ),
        (
            _launch_attestation(
                config,
                process_identity_verified=False,
                process_executable_path="/opt/webots/other-webots",
            ),
            "launched Webots process executable does not match the attested executable",
        ),
    ]
    for observed_attestation, expected in cases:
        report, _, _ = finalize_webots_diagnostic(
            config, preflight, _snapshot(config, attestation=observed_attestation)
        )
        assert any(expected in item for item in report["blockers"]), (expected, report["blockers"])
        assert report["outcome"] == "FAILED"
        assert report["motion_allowed"] is False
        assert report["runtime_webots_version_verified"] is False


def test_missing_launch_version_is_not_substituted_from_config(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_webots_diagnostic_config(path)
    report, _, _ = finalize_webots_diagnostic(
        config,
        preflight_webots_diagnostic(config),
        _snapshot(
            config,
            runtime_version=None,
            attestation=_launch_attestation(config, reported_version=None),
        ),
    )
    assert config.webots_version == "R2025a"
    assert any(
        "launch executable reported Webots version is unavailable" in item
        for item in report["blockers"]
    )
    assert report["launch_webots_attestation"]["reported_version"] is None
    assert report["runtime_webots_version_observed"] is None
    assert report["runtime_webots_version_available"] is not True
    assert report["runtime_webots_version_verified"] is False
    assert report["runtime_webots_version_source"] is None
    assert report["simulator_version"] is None
