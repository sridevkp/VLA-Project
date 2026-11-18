"""Inert tests for the dedicated read-only Webots geometry inspection.

No test in this module starts Webots, opens a ZMQ socket, solves IK, applies an action, steps
physics, scores a campaign or reads TEST/TRAIN evidence. CLI tests inject a synthetic snapshot so
the exact production path is exercised without simulator access.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import vlarobot.backends.read_only_geometry as read_only_geometry
import vlarobot.backends.worldmind_webots_geometry_inspection as adapter
import vlarobot.eval.runner as runner_module
import vlarobot.eval.worldmind_calibration as calibration_module
import vlarobot.eval.worldmind_preflight as preflight_module
import vlarobot.execution.worldmind_pipeline as pipeline_module
from vlarobot.backends.worldmind_webots_geometry_inspection import (
    ReadOnlyGeometryBackendClient,
    _capture,
)
from vlarobot.backends.zmq.client import REQUEST_TIMEOUTS_MS
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
from vlarobot.eval.worldmind_webots_geometry_inspection import (
    TABLE_SOURCE_NAMES,
    GeometryInspectionConfig,
    GeometryInspectionSnapshot,
    _expected_table_geometry,
    failed_geometry_inspection,
    finalize_geometry_inspection,
    load_geometry_inspection_config,
    normalize_webots_sfbool,
    preflight_geometry_inspection,
    table_target_descriptor,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = ROOT / "configs" / "worldmind_webots_geometry_inspection.yaml"
_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics", "inspect", "get_ground_truth")
_ALLOWED_CALLS = ("ping", "inspect_geometry")
_UNSET = object()
_EXECUTABLE = "/opt/webots/webots"
_EXECUTABLE_SHA = "a" * 64
_STATUS = "MEASURED live expanded boundingObject export; collision safety NOT CERTIFIED"
_SCENE = "development copy of factory initial pose; no TRAIN/VALIDATION/TEST task rollout"
_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config_dir = tmp_path / "configs"
    geometry_dir = config_dir / "geometry"
    world_dir = tmp_path / "worlds"
    backend_dir = tmp_path / "src" / "vlarobot" / "backends" / "webots"
    for directory in (geometry_dir, world_dir, backend_dir):
        directory.mkdir(parents=True, exist_ok=True)

    world = world_dir / "factory.wbt"
    urdf = backend_dir / "tiago.urdf"
    registration = config_dir / "calibration.yaml"
    robot = config_dir / "robot.yaml"
    scene = geometry_dir / "scene-collision-v1.yaml"
    proto = geometry_dir / "webots-proto-sources-v1.yaml"
    sweep = config_dir / "worldmind_arm_sweep.yaml"
    # The source-expansion checks need byte-identical pinned geometry. Copying the reviewed
    # files into the temporary tree keeps the test self-contained and preserves relative paths.
    shutil.copytree(ROOT / "configs" / "geometry", geometry_dir, dirs_exist_ok=True)
    shutil.copytree(
        ROOT / "configs" / "calibration", config_dir / "calibration", dirs_exist_ok=True
    )
    shutil.copytree(
        ROOT / "assets" / "webots" / "protos" / "r2025a",
        tmp_path / "assets" / "webots" / "protos" / "r2025a",
        dirs_exist_ok=True,
    )
    shutil.copy2(ROOT / "worlds" / "factory.wbt", world)
    shutil.copy2(ROOT / "src" / "vlarobot" / "backends" / "webots" / "tiago.urdf", urdf)
    registration.write_bytes(b"registration bytes\n")
    robot.write_bytes(b"robot settings bytes\n")
    shutil.copy2(ROOT / "configs" / "worldmind_arm_sweep.yaml", sweep)

    raw: dict[str, object] = {
        "schema_version": 1,
        "status": "diagnostic_uncalibrated",
        "split": "validation",
        "state_source": "oracle",
        "mode": "launch",
        "backend": {
            "kind": "webots",
            "declaration": "webots_factory_world_supervisor_geometry_export_v1",
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
            "required_arm_links": [
                "TIAGo front arm",
                "TIAGo front arm_3",
                "arm_2_link",
                "arm_3_link",
                "arm_4_link",
                "arm_5_link",
                "arm_6_link",
                "arm_7_link",
                "wrist_ft_tool_link",
            ],
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
        "geometry_inspection": {
            "schema_version": 2,
            "status": "diagnostic_uncalibrated",
            "mode": "live_read_only_bounding_object_export",
            "scene_catalog": "geometry/scene-collision-v1.yaml",
            "scene_catalog_sha256": _sha256(scene),
            "proto_manifest": "geometry/webots-proto-sources-v1.yaml",
            "proto_manifest_sha256": _sha256(proto),
            "arm_sweep_contract": "worldmind_arm_sweep.yaml",
            "arm_sweep_contract_sha256": _sha256(sweep),
            "table_targets": {
                "parent_type": "Table",
                "nested_type": "SolidBox",
                "bounding_shape_type": "Box",
                "source_names": list(TABLE_SOURCE_NAMES),
                "translation_tolerance_m": 0.0001,
                "rotation_tolerance_rad": 0.0001,
                "size_tolerance_m": 0.0001,
                "transform_translation_tolerance_m": 0.0001,
                "transform_rotation_tolerance_rad": 0.0001,
            },
        },
    }
    path = config_dir / "geometry-inspection.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    return path, raw


def _write_raw(path: Path, raw: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")


def _launch_attestation(config: GeometryInspectionConfig, **overrides: object) -> dict[str, object]:
    attestation: dict[str, object] = {
        "executable_path": _EXECUTABLE,
        "executable_sha256": _EXECUTABLE_SHA,
        "reported_version": config.diagnostic.webots_version,
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
    return Pose3D(Vec3(0.1, -0.2, 0.3), _IDENTITY)


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


def _table_observation(config: GeometryInspectionConfig) -> dict[str, object]:
    expected = _expected_table_geometry(config)
    descriptor = table_target_descriptor(config)
    rows: list[dict[str, object]] = []
    for index, box in enumerate(expected["boxes"]):
        world_pose = np.eye(4)
        world_pose[:3, :3] = np.asarray(box["world_rotation_matrix"], dtype=float)
        world_pose[:3, 3] = np.asarray(box["world_translation_m"], dtype=float)
        base_pose = np.eye(4)
        base_pose[:3, :3] = np.asarray(box["base_rotation_matrix"], dtype=float)
        base_pose[:3, 3] = np.asarray(box["base_translation_m"], dtype=float)
        name = str(box["name"])
        rows.append(
            {
                "source_name": name,
                "status": "observed",
                "reason": None,
                "source_protos": copy.deepcopy(descriptor["source_protos"]),
                "runtime_node": {
                    "id": 100 + index,
                    "type": "SolidBox",
                    "base_type": "Solid",
                    "name": name,
                    "def": "",
                    "proto_type": "SolidBox",
                },
                "hierarchy": {
                    "reachable": True,
                    "table_ancestor": {
                        "id": 99,
                        "type": "Table",
                        "base_type": "Solid",
                        "name": "",
                        "def": "",
                        "proto_type": "Table",
                    },
                    "parent_chain": [{"type": "Table", "proto_type": "Table"}],
                    "children": [],
                    "bounding_object": {"type": "Box", "base_type": "Box"},
                },
                "evaluated": {
                    "translation_m": list(box["local_translation_m"]),
                    "rotation_axis_angle": list(box["local_rotation_axis_angle"]),
                    "size_m": list(box["size_m"]),
                    "enable_bounding_object": True,
                    "bounding_object_type": "Box",
                    "bounding_object_size_m": list(box["size_m"]),
                },
                "transforms": {
                    "T_world": world_pose.reshape(16).tolist(),
                    "T_base": base_pose.reshape(16).tolist(),
                },
                "unsupported_reasons": [],
            }
        )
    return {
        "schema_version": 1,
        "interface": copy.deepcopy(descriptor),
        "rows": rows,
        "errors": [],
        "robot_base_node_id": 77,
        "table_candidates": [{"type": "Table", "proto_type": "Table"}],
        "traversal": "supported getBaseNodeField(children) expansion of the runtime Table PROTO",
    }


def _geometry_export(
    robot_def: str = "TIAGO_ROBOT", config: GeometryInspectionConfig | None = None
) -> dict[str, object]:
    export: dict[str, object] = {
        "status": _STATUS,
        "robot_id": 77,
        "self_collision_enabled": False,
        "sim_time_s": 1.0,
        "objects": [
            {
                "id": 77,
                "name": robot_def,
                "def": robot_def,
                "type": "Tiago",
                "T_base_owner": np.eye(4).tolist(),
                "T_world_owner": np.eye(4).tolist(),
                "shapes": [
                    {
                        "type": "Box",
                        "parameters": {"size": [1.0, 1.0, 1.0]},
                        "T_owner_shape": np.eye(4).tolist(),
                        "supported": True,
                    }
                ],
            }
        ],
        "motor_commands_sent": 0,
        "devices": ["PositionSensor"],
        "joint_positions": {"arm_1_joint": 0.1},
        "motor_velocity_settings": [],
        "scene": _SCENE,
        "unsupported_shapes": [],
    }
    if config is not None:
        export["table_targets"] = _table_observation(config)
    return export


def _snapshot(
    config: GeometryInspectionConfig,
    *,
    age_s: float = 0.05,
    first_command: int | None = None,
    call_counts: dict[str, int] | None = None,
    robot_def: object = _UNSET,
    geometry: dict[str, object] | None = None,
    attestation: object = _UNSET,
) -> GeometryInspectionSnapshot:
    diagnostic = config.diagnostic
    observed_def = diagnostic.robot_identity if robot_def is _UNSET else robot_def
    observed_attestation = (
        _launch_attestation(config)
        if attestation is _UNSET
        else ({} if attestation is None else dict(attestation))
    )
    counts = {name: 0 for name in (*_ALLOWED_CALLS, *_FORBIDDEN_CALLS)}
    counts["ping"] = 1
    counts["inspect_geometry"] = 1
    if call_counts:
        counts.update(call_counts)
    return GeometryInspectionSnapshot(
        ping={
            "status": "ok",
            "session_id": {"pid": 42, "start_time": 1.0, "nonce": "nonce-42"},
            "world_sha256": diagnostic.world_sha256,
            "kinematics_telemetry_enabled": False,
            "endpoint": diagnostic.endpoint,
        },
        observation=_observation(),
        ground_truth=GroundTruth(object_poses={"CAN": _pose()}),
        identity={
            "simulator": "webots",
            "robot_def": observed_def,
            "robot_def_cross_checked": True,
            "robot_name": "tiago-node-name",
            "robot_type": "Tiago",
            "webots_version": diagnostic.webots_version,
            "webots_version_available": True,
            "webots_version_source": "WEBOTS_VERSION",
            "controller_webots_version": diagnostic.webots_version,
            "controller_webots_version_source": "WEBOTS_VERSION",
            "physics_step_count": 0,
            "first_command_step_count": first_command,
        },
        captured_at_utc="2026-11-18T00:00:00+00:00",
        age_s=age_s,
        dependency_availability={
            name: True
            for name in (
                "pose_solver",
                "safety_monitor",
                "placement_verifier",
                "collision_checker",
                "required_arm_links",
            )
        },
        call_counts=counts,
        launch_attestation=observed_attestation,
        geometry=(
            geometry if geometry is not None else _geometry_export(str(observed_def), config=config)
        ),
    )


def _ping_envelope(snapshot: GeometryInspectionSnapshot) -> dict[str, object]:
    payload = {
        key: value for key, value in snapshot.ping.items() if key not in {"endpoint", "launch_mode"}
    }
    return {
        "protocol_version": 3,
        "type": "ping",
        "seq": 1,
        "sim_time": 0.0,
        "error": None,
        "payload": payload,
    }


def test_shipped_config_and_synthetic_config_preflight_without_webots(tmp_path):
    shipped = load_geometry_inspection_config(SHIPPED_CONFIG)
    shipped_preflight = preflight_geometry_inspection(shipped)
    assert shipped.inspection_mode == "live_read_only_bounding_object_export"
    assert shipped_preflight.blockers == ()
    assert "unverified-arm-sweep" in shipped_preflight.motion_blockers
    assert "unresolved_forwarded_geometry" in shipped_preflight.motion_blockers

    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    preflight = preflight_geometry_inspection(config)
    assert preflight.blockers == ()
    assert preflight.hashes["config"] == _sha256(path)
    assert set(preflight.motion_blockers) == {
        "diagnostic_no_actuation_mode",
        "unverified-arm-sweep",
        "unresolved_forwarded_geometry",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(scoring={}), "forbids keys"),
        (lambda raw: raw["geometry_inspection"].update(mode="motion"), "geometry_inspection.mode"),
        (
            lambda raw: raw["geometry_inspection"].update(status="approved"),
            "geometry_inspection.status",
        ),
        (
            lambda raw: raw["geometry"].update(swept_arm_certificate="certificate.yaml"),
            "forbids a swept-arm certificate",
        ),
        (
            lambda raw: raw["geometry_inspection"].update(scene_catalog_sha256="not-a-digest"),
            "64-character SHA-256",
        ),
        (
            lambda raw: raw["geometry_inspection"].pop("table_targets"),
            "table_targets",
        ),
        (
            lambda raw: raw["geometry_inspection"]["table_targets"].update(source_names=["foot 1"]),
            "five Table box names",
        ),
        (
            lambda raw: raw["geometry_inspection"]["table_targets"].update(size_tolerance_m=-1),
            "nonnegative",
        ),
    ],
)
def test_geometry_config_loader_is_strict(tmp_path, mutate, message):
    path, raw = _write_config(tmp_path)
    blocked = copy.deepcopy(raw)
    mutate(blocked)
    _write_raw(path, blocked)
    with pytest.raises(ValueError, match=message):
        load_geometry_inspection_config(path)


@pytest.mark.parametrize(
    ("observed", "expected"),
    [(False, False), (True, True), (0, False), (1, True)],
)
def test_webots_sfbool_normalization_accepts_only_documented_representations(observed, expected):
    assert normalize_webots_sfbool(observed, field_name="selfCollision") is expected


@pytest.mark.parametrize(
    "observed",
    [-1, 2, -99, "0", "1", "false", "true", None, 0.0, 1.0, float("nan"), [], {}, (0,)],
)
def test_webots_sfbool_normalization_rejects_ambiguous_values(observed):
    with pytest.raises(ValueError, match="ambiguous Webots selfCollision"):
        normalize_webots_sfbool(observed, field_name="selfCollision")


def test_self_collision_integer_zero_passes_with_distinct_raw_and_normalized_report_fields(
    tmp_path,
):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    geometry = _geometry_export(config=config)
    geometry["self_collision_enabled"] = 0
    snapshot = _snapshot(config, geometry=geometry)
    report, _, _, geometry_artifact = finalize_geometry_inspection(
        config, preflight_geometry_inspection(config), snapshot
    )
    assert report["outcome"] == "PASS"
    assert report["self_collision_enabled"] == 0
    assert report["self_collision_enabled_normalized"] is False
    assert report["self_collision_enabled_observed_type"] == "int"
    assert geometry_artifact["export"]["self_collision_enabled"] == 0


@pytest.mark.parametrize("observed", [1, True, -1, 2, "0", None, 0.0])
def test_self_collision_mismatch_or_ambiguity_fails_closed(tmp_path, observed):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    geometry = _geometry_export(config=config)
    geometry["self_collision_enabled"] = observed
    snapshot = _snapshot(config, geometry=geometry)
    report, _, _, _ = finalize_geometry_inspection(
        config, preflight_geometry_inspection(config), snapshot
    )
    assert report["outcome"] == "FAILED"
    assert any("selfCollision" in item for item in report["geometry_blockers"])


def test_geometry_client_forbids_every_state_or_oracle_changing_call():
    class LowLevel:
        def ping(self):
            return {"status": "ok"}

        def inspect_geometry(self, table_targets=None):
            return {"observation": None}

        def close(self):
            return None

    client = ReadOnlyGeometryBackendClient(LowLevel())
    arguments = {
        "apply": (Action(),),
        "step": (1,),
        "reset": (ScenarioSpec(), 0),
    }
    for name in _FORBIDDEN_CALLS:
        with pytest.raises(RuntimeError, match=f"{name} is forbidden"):
            getattr(client, name)(*arguments.get(name, ()))
    assert client.ping() == {"status": "ok"}
    assert client.inspect_geometry() == {"observation": None}
    counts = client.call_counts()
    assert counts["ping"] == 1
    assert counts["inspect_geometry"] == 1
    assert all(counts[name] == 1 for name in _FORBIDDEN_CALLS)


def test_inspect_geometry_rpc_dispatch_never_calls_forbidden_backend_operations(
    tmp_path, monkeypatch
):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    snapshot = _snapshot(config)

    monkeypatch.setattr(
        read_only_geometry,
        "export_bounding_objects",
        lambda supervisor, table_targets=None: dict(snapshot.geometry),
    )
    assert REQUEST_TIMEOUTS_MS["inspect_geometry"] == 30_000

    class FakeSupervisor:
        pass

    class FakeIO:
        robot = FakeSupervisor()

    class FakeBackend:
        io = FakeIO()
        _physics_step_count = 0
        _first_command_step_count = None
        _stopped = False

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

        def export_kinematics(self, action="poll"):
            self.calls.append("export_kinematics")
            raise AssertionError("export_kinematics must not be called")

        def stop(self):
            self.calls.append("stop")
            raise AssertionError("stop must not be called")

    backend = FakeBackend()
    server = SimServer(backend)  # type: ignore[arg-type]
    response = server.handle({"seq": 1, "type": "inspect_geometry", "payload": {}})
    assert response["type"] == "inspect_geometry"
    assert response["error"] is None
    assert {"observation", "ground_truth", "identity", "geometry"}.issubset(response["payload"])
    assert response["payload"]["geometry"]["status"] == _STATUS
    assert backend.calls == []


def test_geometry_capture_blocks_commanded_or_stepped_backends(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    snapshot = _snapshot(config)
    monkeypatch.setattr(
        read_only_geometry,
        "export_bounding_objects",
        lambda supervisor, table_targets=None: dict(snapshot.geometry),
    )

    class FakeSupervisor:
        pass

    class FakeIO:
        robot = FakeSupervisor()

    class FakeBackend:
        io = FakeIO()
        _stopped = False

        def __init__(self, first_command=None, step_count=0):
            self._first_command_step_count = first_command
            self._physics_step_count = step_count

        def read_only_state(self):
            identity = dict(snapshot.identity)
            identity["first_command_step_count"] = self._first_command_step_count
            identity["physics_step_count"] = self._physics_step_count
            return {
                "observation": snapshot.observation,
                "ground_truth": snapshot.ground_truth,
                "identity": identity,
            }

    with pytest.raises(RuntimeError, match="actuator command"):
        read_only_geometry.capture_read_only_geometry(FakeBackend(first_command=7))
    with pytest.raises(RuntimeError, match="stepped physics"):
        read_only_geometry.capture_read_only_geometry(FakeBackend(step_count=1))

    monkeypatch.setattr(
        read_only_geometry,
        "export_bounding_objects",
        lambda supervisor, table_targets=None: {"status": _STATUS},
    )
    with pytest.raises(RuntimeError, match="incomplete snapshot"):
        read_only_geometry.capture_read_only_geometry(FakeBackend())


def test_finalize_reports_pass_without_motion_and_fails_closed_on_bad_evidence(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    preflight = preflight_geometry_inspection(config)
    snapshot = _snapshot(config)
    report, state, provenance, geometry = finalize_geometry_inspection(config, preflight, snapshot)
    assert report["outcome"] == "PASS"
    assert report["geometry_inspection_mode"] == "live_read_only_bounding_object_export"
    assert report["scored_evaluation_allowed"] is False
    assert report["test_split_accessed"] is False
    assert report["motion_allowed"] is False
    assert report["motion_readiness"] == "NOT_READY"
    assert report["backend_inspect_geometry_calls"] == 1
    assert report["geometry_object_count"] == 1
    assert report["unsupported_shape_count"] == 0
    assert state["oracle_target_object"] == "CAN"
    assert provenance["geometry_inspection_only"] is True
    assert provenance["authoritative_inert_report_sha256"]
    assert geometry["motion_ready"] is False

    commanded = replace(snapshot, geometry={**snapshot.geometry, "motor_commands_sent": 1})
    commanded_report, _, _, _ = finalize_geometry_inspection(config, preflight, commanded)
    assert commanded_report["outcome"] == "FAILED"
    assert any("reports motor commands" in item for item in commanded_report["blockers"])

    alt_oracle = replace(
        snapshot, call_counts={**snapshot.call_counts, "inspect": 1, "get_ground_truth": 1}
    )
    oracle_report, _, _, _ = finalize_geometry_inspection(config, preflight, alt_oracle)
    assert oracle_report["outcome"] == "FAILED"
    for name in ("inspect", "get_ground_truth"):
        assert any(f"attempted: {name}" in item for item in oracle_report["blockers"])

    wrong_def = _snapshot(
        config,
        robot_def="OTHER_ROBOT",
        geometry=_geometry_export("TIAGO_ROBOT", config=config),
    )
    identity_report, _, _, _ = finalize_geometry_inspection(config, preflight, wrong_def)
    assert identity_report["outcome"] == "FAILED"
    assert any(
        "robot identity is not cross-checked" in item for item in identity_report["blockers"]
    )

    failure = failed_geometry_inspection(config, preflight, "config_preflight_failed")
    assert failure["outcome"] == "FAILED"
    assert failure["motion_allowed"] is False
    assert failure["backend_inspect_geometry_calls"] == 0


def test_capture_flattens_ping_and_rejects_incomplete_replies(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    snapshot = _snapshot(config)
    envelope = _ping_envelope(snapshot)

    class LowLevel:
        def __init__(self, ping: dict[str, object], payload: dict[str, object] | None):
            self.envelope = ping
            self.payload = payload

        def ping(self):
            return copy.deepcopy(self.envelope)

        def inspect_geometry(self, table_targets=None):
            return dict(self.payload) if self.payload is not None else {}

        def close(self):
            return None

    payload = {
        "observation": snapshot.observation,
        "ground_truth": snapshot.ground_truth,
        "identity": dict(snapshot.identity),
        "geometry": dict(snapshot.geometry),
    }
    captured = _capture(
        ReadOnlyGeometryBackendClient(LowLevel(envelope, payload)),
        config,
        "tcp://127.0.0.1:5555",
        _launch_attestation(config),
    )
    assert captured.ping["session_id"] == snapshot.ping["session_id"]
    assert captured.ping["endpoint"] == "tcp://127.0.0.1:5555"
    assert captured.ping["launch_mode"] == "launch"
    assert "payload" not in captured.ping
    assert captured.geometry["status"] == _STATUS
    report, _, _, _ = finalize_geometry_inspection(
        config, preflight_geometry_inspection(config), captured
    )
    assert report["outcome"] == "PASS"

    with pytest.raises(ValueError, match="omitted payload"):
        _capture(
            ReadOnlyGeometryBackendClient(LowLevel({"status": "ok"}, payload)),
            config,
            "tcp://127.0.0.1:5555",
        )
    with pytest.raises(ValueError, match="omitted state, identity or geometry"):
        _capture(
            ReadOnlyGeometryBackendClient(LowLevel(envelope, None)),
            config,
            "tcp://127.0.0.1:5555",
        )


def test_cli_writes_exclusive_geometry_evidence_with_injected_snapshot(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    snapshot = _snapshot(config)
    monkeypatch.setattr(
        adapter,
        "collect_geometry_inspection_snapshot",
        lambda resolved, trace_dir, **kwargs: snapshot,
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scorer, calibration, preflight or planner must not be called")

    monkeypatch.setattr(pipeline_module, "run_worldmind_evaluation", forbidden)
    monkeypatch.setattr(preflight_module, "run_worldmind_preflight", forbidden)
    monkeypatch.setattr(calibration_module, "evaluate_calibration", forbidden)
    monkeypatch.setattr(runner_module, "build_episode_plan", forbidden)

    out = tmp_path / "geometry-evidence"
    result = CliRunner().invoke(
        app,
        ["worldmind-webots-geometry-inspection", "--config", str(path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert sorted(item.name for item in out.iterdir()) == [
        "diagnostic.json",
        "geometry.json",
        "provenance.json",
        "state.json",
    ]
    report = json.loads((out / "diagnostic.json").read_text(encoding="utf-8"))
    geometry = json.loads((out / "geometry.json").read_text(encoding="utf-8"))
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "PASS"
    assert report["scored_evaluation_allowed"] is False
    assert report["backend_apply_calls"] == 0
    assert report["backend_step_calls"] == 0
    assert report["backend_inspect_calls"] == 0
    assert report["backend_ground_truth_calls"] == 0
    assert report["motion_allowed"] is False
    assert geometry["motion_ready"] is False
    assert geometry["export"]["motor_commands_sent"] == 0
    assert provenance["frozen_campaign_modified"] is False

    prior = tmp_path / "prior-evidence"
    prior.mkdir()
    marker = prior / "geometry.json"
    marker.write_text('{"outcome": "PRIOR"}', encoding="utf-8")

    def no_collection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("backend collection must not start for an existing output directory")

    monkeypatch.setattr(adapter, "collect_geometry_inspection_snapshot", no_collection)
    result = CliRunner().invoke(
        app,
        ["worldmind-webots-geometry-inspection", "--config", str(path), "--out", str(prior)],
    )
    assert result.exit_code == 1
    assert marker.read_text(encoding="utf-8") == '{"outcome": "PRIOR"}'


def test_preflight_failure_writes_failed_report_and_never_collects(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    (path.parent / "geometry" / "scene-collision-v1.yaml").unlink()

    def no_collection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("backend collection must not start after a catalog failure")

    monkeypatch.setattr(adapter, "collect_geometry_inspection_snapshot", no_collection)
    out = tmp_path / "failed-evidence"
    result = CliRunner().invoke(
        app,
        ["worldmind-webots-geometry-inspection", "--config", str(path), "--out", str(out)],
    )
    assert result.exit_code == 1
    report = json.loads((out / "diagnostic.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "FAILED"
    assert report["motion_allowed"] is False
    assert not (out / "geometry.json").exists()


def test_collection_checks_dependencies_before_any_session_is_created(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    config = load_geometry_inspection_config(path)
    unknown = replace(
        config,
        diagnostic=replace(config.diagnostic, geometry_binding="not_a_registered_checker"),
    )

    def no_launch(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Webots session must not be created for unavailable dependencies")

    monkeypatch.setattr(adapter, "launch_webots_session", no_launch)
    with pytest.raises(RuntimeError, match="dependencies unavailable"):
        adapter.collect_geometry_inspection_snapshot(unknown, tmp_path)
