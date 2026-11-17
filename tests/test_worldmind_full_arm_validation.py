"""Inert validation of the concrete full-arm sweep adapter and local geometry diagnostic.

No test starts Webots, opens a socket, solves IK, applies an action, steps physics, scores a
campaign or reads TEST/TRAIN data. These cases verify conservative geometry behavior only and do
not certify physical safety.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import vlarobot.backends.worldmind_bindings as bindings
from vlarobot.backends.worldmind_geometry_diagnostic import (
    GeometryOnlyBackendGuard,
    collect_geometry_evidence,
)
from vlarobot.cli import app
from vlarobot.core.types import (
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_geometry_diagnostic import (
    finalize_geometry_diagnostic,
    load_geometry_diagnostic_config,
    preflight_geometry_diagnostic,
)
from vlarobot.world_model.arm_geometry import Capsule
from vlarobot.world_model.arm_sweep import (
    ArmSweepContract,
    IntermediateSweptArmCheck,
    SweepGeometryError,
    urdf_collision_capsules,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
CAPS = ("cap:a", "cap:b")


def _object(name: str, x: float, z: float = 0.55, size: tuple[float, float, float] = (0.02,) * 3):
    return ObjectState(name, name.lower(), Pose3D(Vec3(x, 0.0, z), IDENTITY), Vec3(*size), 1.0)


def _state(
    q: tuple[float, float] = (0.0, 0.0),
    *,
    objects: tuple[ObjectState, ...] = (_object("CAN", 10.0),),
) -> WorldState:
    robot = RobotState(
        joint_pos={"j1": q[0], "j2": q[1]},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(q[0], q[1], 0.55), IDENTITY),
        gripper_opening=0.045,
    )
    return WorldState(SceneState(list(objects)), robot)


def _capsules(state: WorldState, *, z: float = 0.55, radius: float = 0.01) -> tuple[Capsule, ...]:
    x = state.robot.joint_pos["j1"]
    return (
        Capsule(CAPS[0], (x, 0.0, z - 0.05), (x, 0.0, z + 0.05), radius, 1.0),
        Capsule(CAPS[1], (x + 0.25, 0.0, z - 0.05), (x + 0.25, 0.0, z + 0.05), radius, 1.0),
    )


def _contract(
    *,
    adjacent: frozenset[frozenset[str]] = frozenset(),
    objects: frozenset[str] = frozenset({"CAN"}),
    contacts: frozenset[str] = frozenset(),
    substeps: int = 4,
    margin: float = 0.01,
) -> ArmSweepContract:
    return ArmSweepContract(
        required_links=CAPS,
        required_objects=objects,
        adjacent_pairs=adjacent,
        dt_s=0.032,
        margin_m=margin,
        substeps=substeps,
        permitted_contact_objects=contacts,
    )


def _checker(
    geometry: Any = _capsules, *, contract: ArmSweepContract | None = None
) -> IntermediateSweptArmCheck:
    return IntermediateSweptArmCheck(geometry, contract or _contract())


def _synthetic_urdf(path: Path) -> Path:
    path.write_text(
        '<robot name="fixture">'
        '<link name="base_link"/>'
        '<link name="link_a"><collision><origin xyz="0.1 0 0" rpy="0 0 0"/>'
        '<geometry><cylinder radius="0.02" length="0.1"/></geometry></collision></link>'
        '<link name="link_b"><collision><origin xyz="0.1 0 0" rpy="0 0 1.5707963267948966"/>'
        '<geometry><box size="0.2 0.04 0.06"/></geometry></collision></link>'
        '<joint name="j1" type="revolute"><parent link="base_link"/><child link="link_a"/>'
        '<origin xyz="0.1 0 0"/><limit velocity="1.5"/></joint>'
        '<joint name="j2" type="revolute"><parent link="link_a"/><child link="link_b"/>'
        '<origin xyz="0.2 0 0"/><limit velocity="2.0"/></joint>'
        "</robot>",
        encoding="utf-8",
    )
    return path


class _FakeKinematics:
    def __init__(self, *, bad_registration: bool = False) -> None:
        self.idx = {"j1": 1, "j2": 2}
        self.chain = SimpleNamespace(links=(None, None, None))
        self.bad_registration = bad_registration

    def forward(self, q: np.ndarray) -> list[np.ndarray]:
        identity = np.eye(4)
        a = np.eye(4)
        a[:3, 3] = (float(q[1]), 0.0, 0.0)
        b = np.eye(4)
        b[:3, 3] = (float(q[1]), float(q[2]), 0.0)
        if self.bad_registration:
            a[:3, :3] *= 2.0
            b[:3, :3] *= 2.0
        return [identity, a, b]


def _adapter_config(urdf: Path, *, adjacent: frozenset[frozenset[str]] = frozenset()):
    certificate = bindings.ArmSweepCertificate(
        "fixture-certificate",
        "0" * 64,
        "fixture-owner",
        "approval-1",
        Path("approval.json"),
        "0" * 64,
        Path("cases.json"),
        "0" * 64,
        "0" * 64,
    )
    return bindings.ArmSweepConfig(
        source_path=Path("sweep.yaml"),
        status="owner_approved",
        urdf_path=urdf,
        registration_path=Path("registration.yaml"),
        robot_config_path=Path("robot.yaml"),
        world_path=Path("factory.wbt"),
        urdf_sha256="0" * 64,
        registration_sha256="0" * 64,
        robot_config_sha256="0" * 64,
        world_sha256="0" * 64,
        required_arm_links=("link_a", "link_b"),
        required_objects=frozenset({"CAN"}),
        adjacent_pairs=adjacent,
        permitted_contact_objects=frozenset(),
        dt_s=0.032,
        margin_m=0.01,
        substeps=4,
        speed_bound_m_s=100.0,
        certificate=certificate,
    )


def test_independent_capsule_containment_matches_pinned_shape_reduction(tmp_path):
    urdf = _synthetic_urdf(tmp_path / "fixture.urdf")
    parsed = urdf_collision_capsules(urdf, ("link_a", "link_b"), max_point_speed_m_s=3.0)
    cylinder = parsed["link_a"][0].capsule
    box = parsed["link_b"][0].capsule
    assert cylinder.start == pytest.approx((0.1, 0.0, -0.05))
    assert cylinder.end == pytest.approx((0.1, 0.0, 0.05))
    assert cylinder.radius_m == pytest.approx(0.02)
    assert box.start == pytest.approx((0.1, -0.1, 0.0))
    assert box.end == pytest.approx((0.1, 0.1, 0.0))
    assert box.radius_m == pytest.approx(math.hypot(0.02, 0.03))
    assert all(
        item.capsule.max_point_speed_m_s == 3.0 for items in parsed.values() for item in items
    )


def test_known_collision_clearance_and_independent_near_boundary_aabb():
    checker = _checker()
    # q displacement per sub-interval stays below the 1.0 m/s capsule speed certificate.
    assert checker(_state(), _state((0.004, 0.0))) is True
    assert (
        checker(
            _state(objects=(_object("CAN", 0.24),)),
            _state((0.004, 0.0), objects=(_object("CAN", 0.24),)),
        )
        is False
    )
    # dt/substep = 0.008 s; padding = radius 0.01 + margin 0.01 + v*dt/2 0.004 = 0.024.
    # Endpoints x=0 and x=0.03 give a swept AABB maximum x = 0.054.
    clear_object = _object("CAN", 0.065, size=(0.01, 0.02, 0.02))
    collide_object = _object("CAN", 0.055, size=(0.01, 0.02, 0.02))
    assert 0.054 < 0.065 - 0.005
    assert 0.054 > 0.055 - 0.005
    assert (
        checker(_state(objects=(clear_object,)), _state((0.03, 0.0), objects=(clear_object,)))
        is True
    )
    assert (
        checker(_state(objects=(collide_object,)), _state((0.03, 0.0), objects=(collide_object,)))
        is False
    )


def test_interpolation_samples_intermediate_states_and_rejects_moving_scene():
    seen: list[float] = []

    def recording(state: WorldState) -> tuple[Capsule, ...]:
        seen.append(state.robot.joint_pos["j1"])
        return _capsules(state)

    assert (
        _checker(recording, contract=_contract(substeps=3))(_state(), _state((0.003, 0.0))) is True
    )
    assert seen == pytest.approx([0.0, 0.001, 0.001, 0.002, 0.002, 0.003])
    moving = _state((0.3, 0.0), objects=(_object("CAN", 9.0),))
    assert _checker()(_state(), moving) is None


def test_self_collision_and_table_can_collision_cases():
    self_checker = _checker()

    def overlapping(state: WorldState) -> tuple[Capsule, ...]:
        x = state.robot.joint_pos["j1"]
        return (
            Capsule(CAPS[0], (x, 0.0, 0.5), (x, 0.0, 0.6), 0.02, 1.0),
            Capsule(CAPS[1], (x + 0.01, 0.0, 0.5), (x + 0.01, 0.0, 0.6), 0.02, 1.0),
        )

    assert self_checker(_state(), _state((0.004, 0.0))) is True
    assert (
        IntermediateSweptArmCheck(overlapping, _contract())(_state(), _state((0.004, 0.0))) is False
    )
    adjacent = frozenset({frozenset(CAPS)})
    assert (
        IntermediateSweptArmCheck(overlapping, _contract(adjacent=adjacent))(
            _state(), _state((0.004, 0.0))
        )
        is True
    )

    def table_geometry(state: WorldState) -> tuple[Capsule, ...]:
        x = state.robot.joint_pos["j1"]
        return (
            Capsule(CAPS[0], (x, 0.0, 0.70), (x, 0.0, 0.80), 0.01, 1.0),
            Capsule(CAPS[1], (x + 0.3, 0.0, 0.70), (x + 0.3, 0.0, 0.80), 0.01, 1.0),
        )

    objects = (_object("CAN", 10.0), _object("TABLE", 0.1, 0.74, (0.4, 0.4, 0.04)))
    contract = _contract(objects=frozenset({"CAN", "TABLE"}))
    assert (
        IntermediateSweptArmCheck(table_geometry, contract)(
            _state(objects=objects), _state((0.004, 0.0), objects=objects)
        )
        is False
    )


def test_missing_geometry_nonrigid_transform_and_unsupported_contact_fail_closed():
    assert _checker(lambda state: (_capsules(state)[0],))(_state(), _state((0.2, 0.0))) is None

    def invalid(state: WorldState) -> tuple[Capsule, ...]:
        del state
        return (
            Capsule(CAPS[0], (float("nan"), 0.0, 0.5), (0.0, 0.0, 0.6), 0.01, 1.0),
            Capsule(CAPS[1], (0.0, 0.0, 0.5), (0.1, 0.0, 0.6), 0.01, 1.0),
        )

    assert _checker(invalid)(_state(), _state((0.2, 0.0))) is None
    before = _state(objects=(_object("CAN", 10.0), _object("OTHER", 9.0)))
    after = _state((0.2, 0.0), objects=(_object("CAN", 10.0), _object("OTHER", 8.0)))
    assert _checker()(before, after) is None

    bad_pose = Pose3D(Vec3(10.0, 0.0, 0.55), (2.0, 0.0, 0.0, 0.0))
    bad_object = replace(_object("CAN", 10.0), pose=bad_pose)
    assert (
        _checker()(_state(objects=(bad_object,)), _state((0.004, 0.0), objects=(bad_object,)))
        is None
    )
    duplicate = (_object("CAN", 10.0), _object("CAN", 9.0))
    assert _checker()(_state(objects=duplicate), _state((0.004, 0.0), objects=duplicate)) is None


def test_concrete_adapter_uses_pinned_urdf_and_registered_fk(monkeypatch, tmp_path):
    urdf = _synthetic_urdf(tmp_path / "adapter.urdf")
    config = _adapter_config(urdf, adjacent=frozenset({frozenset({"link_a:0", "link_b:0"})}))
    monkeypatch.setattr(bindings, "corrected_kinematics", lambda: _FakeKinematics())
    monkeypatch.setattr(bindings, "UNVERIFIED_ARM_LINKS", ("link_a", "link_b"))
    monkeypatch.setattr(bindings, "UNVERIFIED_LINK_FRAMES", {"link_a": "j1", "link_b": "j2"})
    checker = bindings._build_configured_sweep_check(config)
    capsules = {capsule.name: capsule for capsule in checker.geometry(_state((0.11, 0.07)))}
    assert set(capsules) == {"link_a:0", "link_b:0"}
    assert capsules["link_a:0"].start == pytest.approx((0.21, 0.0, -0.05))
    assert capsules["link_b:0"].end == pytest.approx((0.21, 0.17, 0.0))
    assert checker(_state(), _state((0.05, 0.02))) is True

    monkeypatch.setattr(
        bindings, "corrected_kinematics", lambda: _FakeKinematics(bad_registration=True)
    )
    assert bindings._build_configured_sweep_check(config)(_state(), _state((0.05, 0.02))) is None


def test_blocked_sweep_contract_keeps_hard_veto_and_factory_absent(tmp_path):
    config = bindings.load_arm_sweep_config(bindings.ARM_SWEEP_CONFIG_PATH)
    assert config.status == "blocked"
    assert config.certificate is None
    assert bindings.arm_sweep_transition_check(bindings.ARM_SWEEP_CONFIG_PATH) is None
    production_inputs = bindings.production_worldmind_bindings().inputs["surrogate"]
    assert production_inputs["arm_sweep_config"] == bindings.ARM_SWEEP_CONFIG_PATH

    production_factory_config = SimpleNamespace(
        task=SimpleNamespace(target="CAN"),
        verification=SimpleNamespace(destination_world=(0.0, 0.0, 0.8)),
    )
    production_surrogate = bindings.analytic_surrogate_factory(
        production_factory_config,
        production_inputs,  # type: ignore[arg-type]
    )
    assert production_surrogate.check_transition is None

    raw = yaml.safe_load(bindings.ARM_SWEEP_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["geometry"].update(
        {
            "urdf": str(bindings.URDF_PATH),
            "registration": str(config.registration_path),
            "robot_config": str(bindings.SETTINGS_PATH),
            "world": str(config.world_path),
        }
    )
    raw["geometry"]["registration_sha256"] = "1" * 64
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(SweepGeometryError, match="registration hash mismatch"):
        bindings.load_arm_sweep_config(altered)

    approved = dict(raw)
    approved["status"] = "owner_approved"
    approved["certificate"] = {
        "id": "cert-1",
        "sha256": "2" * 64,
        "approved": True,
        "approver": "owner",
        "approval_id": "approval-1",
        "approval_record": "missing-record.json",
        "approval_record_sha256": "3" * 64,
        "reviewed_cases": "missing-cases.json",
        "reviewed_cases_sha256": "4" * 64,
        "contract_sha256": "5" * 64,
    }
    approved["geometry"]["registration_sha256"] = config.registration_sha256
    path = tmp_path / "approved.yaml"
    path.write_text(yaml.safe_dump(approved, sort_keys=True), encoding="utf-8")
    with pytest.raises(SweepGeometryError, match="review evidence hash mismatch"):
        bindings.load_arm_sweep_config(path)


def test_blocked_status_cannot_attach_a_certificate_callback(tmp_path):
    config = _adapter_config(
        _synthetic_urdf(tmp_path / "blocked.urdf"),
        adjacent=frozenset({frozenset({"link_a:0", "link_b:0"})}),
    )
    with pytest.raises(SweepGeometryError, match="owner_approved status"):
        bindings._build_configured_sweep_check(replace(config, status="blocked"))


def test_factory_attaches_only_an_explicit_reviewed_callback(monkeypatch):
    sentinel = object()
    config = SimpleNamespace(
        task=SimpleNamespace(target="CAN"),
        verification=SimpleNamespace(destination_world=(0.0, 0.0, 0.8)),
    )
    with monkeypatch.context() as patch:
        patch.setattr(bindings, "arm_sweep_transition_check", lambda path: sentinel)
        surrogate = bindings.analytic_surrogate_factory(
            config,
            {"arm_sweep_config": Path("reviewed.yaml")},  # type: ignore[arg-type]
        )
        assert surrogate.check_transition is sentinel
    assert bindings.analytic_surrogate_factory(config, {}).check_transition is None  # type: ignore[arg-type]


def test_geometry_only_guard_blocks_every_simulator_capability():
    guard = GeometryOnlyBackendGuard()
    calls = {
        "apply": (None,),
        "step": (1,),
        "reset": (None, 0),
        "export_kinematics": ("poll",),
        "inspect": (),
        "get_ground_truth": (),
    }
    for name, args in calls.items():
        with pytest.raises(RuntimeError, match="forbidden"):
            getattr(guard, name)(*args)
    assert all(count == 1 for count in guard.call_counts().values())


def test_geometry_diagnostic_is_passing_inert_and_never_authorizes_motion():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "worldmind_geometry_diagnostic.yaml"
    )
    config = load_geometry_diagnostic_config(config_path)
    preflight = preflight_geometry_diagnostic(config)
    assert preflight.blockers == ()
    evidence = collect_geometry_evidence(config)
    report, state, provenance = finalize_geometry_diagnostic(config, preflight, evidence)
    assert report["outcome"] == "PASS"
    assert report["missing_arm_links"] == []
    assert report["collision_capsule_count"] == 19
    assert report["unsupported_collision_shapes"] == []
    assert report["motion_allowed"] is False
    assert report["motion_readiness"] == "NOT_READY"
    assert report["test_split_accessed"] is False
    assert report["train_split_accessed"] is False
    assert all(
        report["forbidden_call_counts"][name] == 0 for name in report["forbidden_call_counts"]
    )
    assert "no owner-reviewed full-arm swept certificate" in report["coverage_gaps"]
    gates = report["engineering_gates"]
    assert gates["complete_required_arm_collision_capsules"] is True
    assert gates["complete_relevant_collision_geometry"] is False
    assert gates["reviewed_reference_cases"] is False
    assert gates["owner_certificate"] is False
    assert gates["motion_ready"] is False
    assert state["inspection_only"] is True
    assert provenance["local_file_inspection"] is True


def test_geometry_diagnostic_fails_closed_on_scoring_keys_and_forbidden_calls(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "worldmind_geometry_diagnostic.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["scoring"] = {"enabled": True}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids keys"):
        load_geometry_diagnostic_config(bad)

    config = load_geometry_diagnostic_config(config_path)
    preflight = preflight_geometry_diagnostic(config)
    evidence = collect_geometry_evidence(config)
    tampered = replace(evidence, call_counts={**evidence.call_counts, "step": 1})
    report, _, _ = finalize_geometry_diagnostic(config, preflight, tampered)
    assert report["outcome"] == "FAILED"
    assert any("forbidden backend capability" in blocker for blocker in report["blockers"])


def test_geometry_diagnostic_cli_help_and_inert_run(tmp_path):
    help_result = CliRunner().invoke(app, ["worldmind-geometry-diagnostic", "--help"])
    assert help_result.exit_code == 0
    assert "--config" in help_result.stdout
    assert "--out" in help_result.stdout
    assert "--split" not in help_result.stdout

    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "worldmind_geometry_diagnostic.yaml"
    )
    out = tmp_path / "geometry-evidence"
    result = CliRunner().invoke(
        app,
        ["worldmind-geometry-diagnostic", "--config", str(config_path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    report = json.loads((out / "diagnostic.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "PASS"
    assert (out / "state.json").is_file()
    assert (out / "provenance.json").is_file()
