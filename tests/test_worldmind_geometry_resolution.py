"""Independent inert checks for pinned Webots source resolution and conservative sweep coverage.

No test launches Webots, opens a socket, solves IK, applies an action, steps physics, scores a
campaign or reads TEST/TRAIN evidence. The assertions are structural and use only pinned files or
synthetic geometry.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from vlarobot.backends import worldmind_bindings as bindings
from vlarobot.backends.webots.proto_sources import (
    load_proto_source_manifest,
    manifest_report,
)
from vlarobot.core.types import (
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.world_model.arm_geometry import (
    Capsule,
    FloorHalfSpace,
    GripperEnvelope,
    StaticBox,
    StaticSceneGeometry,
    SweptArmCheck,
)
from vlarobot.world_model.arm_sweep import (
    ArmSweepContract,
    required_substeps,
    urdf_collision_capsules,
)
from vlarobot.world_model.scene_geometry import axis_angle_matrix, load_scene_collision_catalog

ROOT = Path(__file__).resolve().parents[1]
PROTO_MANIFEST = ROOT / "configs" / "geometry" / "webots-proto-sources-v1.yaml"
SCENE_CATALOG = ROOT / "configs" / "geometry" / "scene-collision-v1.yaml"
WORLD = ROOT / "worlds" / "factory.wbt"
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_proto_manifest_resolves_all_external_references_with_hashes(tmp_path):
    manifest = load_proto_source_manifest(PROTO_MANIFEST, world_path=WORLD)
    report = manifest_report(manifest)
    assert (report["external_count"], report["nested_count"]) == (27, 5)
    assert (report["relevant_count"], report["unrelated_count"]) == (29, 3)

    world_urls = set(re.findall(r'EXTERNPROTO\s+"([^"]+)"', WORLD.read_text(encoding="utf-8")))
    assert world_urls == set(manifest.world_urls)
    assert {item.name for item in manifest.unrelated} == {
        "TexturedBackground",
        "ThreadMetalPlate",
        "GlossyPaint",
    }
    for item in (*manifest.external, *manifest.nested):
        assert item.path.is_file(), item.name
        assert _sha256(item.path) == item.sha256, item.name
    raw = yaml.safe_load(PROTO_MANIFEST.read_text(encoding="utf-8"))
    provenance = raw["provenance"]
    for path_key, hash_key in (
        ("cache_log", "cache_log_sha256"),
        ("webots_version_file", "webots_version_sha256"),
    ):
        source = Path(provenance[path_key])
        assert source.is_file()
        assert _sha256(source) == provenance[hash_key]

    for name in ("Tiago", "Floor", "Table", "Can", "TiagoGripper", "TiagoFrontArm"):
        assert not manifest.inspections[name].unsupported_bounding_types, name
    assert set(manifest.inspections["Floor"].bounding_object_types) == {"Plane"}
    assert set(manifest.inspections["Can"].bounding_object_types) == {"Cylinder"}
    assert set(manifest.unresolved) == {"PipeSection", "StraightStairsRail"}
    for name in manifest.unresolved:
        assert set(manifest.inspections[name].unsupported_bounding_types) == {"USE"}

    altered = tmp_path / "altered-manifest.yaml"
    altered_data = yaml.safe_load(PROTO_MANIFEST.read_text(encoding="utf-8"))
    for row in (*altered_data["protos"], *altered_data["nested_sources"]):
        row["asset"] = str((PROTO_MANIFEST.parent / row["asset"]).resolve())
    altered_data["scene_catalog"] = str(
        (PROTO_MANIFEST.parent / altered_data["scene_catalog"]).resolve()
    )
    altered_data["unresolved_forwarded_geometry"].append("PipeSection")
    altered.write_text(yaml.safe_dump(altered_data), encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved_forwarded_geometry"):
        load_proto_source_manifest(altered, world_path=WORLD)


def test_scene_catalog_transforms_are_rigid_and_primitives_are_conservative():
    catalog = load_scene_collision_catalog(SCENE_CATALOG)
    assert catalog.motion_ready is False
    assert set(catalog.hashes) == {
        "world",
        "registration",
        "tiago",
        "floor",
        "table",
        "can",
        "gripper",
    }
    raw = yaml.safe_load(SCENE_CATALOG.read_text(encoding="utf-8"))
    for name, relative in raw["sources"].items():
        source = (SCENE_CATALOG.parent / relative).resolve()
        assert catalog.hashes[name] == _sha256(source), name
        assert re.fullmatch(r"[0-9a-f]{64}", catalog.hashes[name]), name

    rotation = catalog.T_world_base[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)
    assert np.allclose(catalog.T_base_world, np.linalg.inv(catalog.T_world_base))
    assert np.allclose(catalog.T_world_urdf, catalog.T_world_base @ catalog.T_base_urdf)
    world_pose = catalog.report()["expected_base_pose_world"]
    assert world_pose["position"] == pytest.approx([-0.53683, -3.70906, 0.0885438])

    table = catalog.geometry.boxes[0]
    assert table.name == "table"
    assert all(value > 0 for value in table.half_base)
    raw_box = raw["static_boxes"][0]
    world_center = np.asarray(raw_box["world_center"], dtype=float)
    world_rotation = axis_angle_matrix(raw_box["world_rotation"])
    half_world = np.asarray(raw_box["size"], dtype=float) / 2.0
    corners_world = [
        world_center + world_rotation @ np.asarray(offset, dtype=float) * half_world
        for offset in np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    ]
    center_base = np.asarray(table.center_base, dtype=float)
    half_base = np.asarray(table.half_base, dtype=float)
    offsets_base = np.asarray(
        [
            catalog.T_base_world[:3, :3] @ corner + catalog.T_base_world[:3, 3] - center_base
            for corner in corners_world
        ]
    )
    assert np.all(np.abs(offsets_base) <= half_base + 1e-9)
    assert np.allclose(np.max(np.abs(offsets_base), axis=0), half_base, atol=1e-9)
    for floor in catalog.geometry.floors:
        assert math.isclose(np.linalg.norm(floor.normal_base), 1.0, abs_tol=1e-9)
    assert len(catalog.geometry.floors) == 2
    assert catalog.geometry.gripper is not None
    assert catalog.geometry.gripper.radius_m == pytest.approx(0.1)
    assert catalog.geometry.gripper.end_tool[2] == pytest.approx(-0.21875)
    assert catalog.geometry.carried_objects[0].radius_m == pytest.approx(0.03175)
    assert catalog.geometry.catalog_sha256 == _sha256(SCENE_CATALOG)
    assert catalog.base_pose_tolerance["position_m"] == pytest.approx(0.003)
    assert catalog.base_pose_tolerance["orientation_rad"] == pytest.approx(0.005)

    joined = " ".join(catalog.unsupported)
    for marker in (
        "PipeSection",
        "StraightStairsRail",
        "20 types",
        "selfCollision FALSE",
        "not owner-reviewed",
    ):
        assert marker in joined


def test_urdf_primitives_conservatively_cover_every_required_arm_link():
    links = (
        "TIAGo front arm",
        "TIAGo front arm_3",
        "arm_2_link",
        "arm_3_link",
        "arm_4_link",
        "arm_5_link",
        "arm_6_link",
        "arm_7_link",
        "wrist_ft_tool_link",
    )
    speed = bindings.load_arm_sweep_config(bindings.ARM_SWEEP_CONFIG_PATH).speed_bound_m_s
    parsed = urdf_collision_capsules(bindings.URDF_PATH, links, max_point_speed_m_s=speed)
    assert set(parsed) == set(links)
    primitives = [item for group in parsed.values() for item in group]
    assert len(primitives) == 19
    assert all(item.certified_containment for item in primitives)
    assert all(item.capsule.radius_m > 0 for item in primitives)
    assert all(item.capsule.max_point_speed_m_s == speed for item in primitives)


def test_adaptive_subdivision_repairs_the_fixed_eight_step_gap():
    speed = bindings.load_arm_sweep_config(bindings.ARM_SWEEP_CONFIG_PATH).speed_bound_m_s
    dt_s, margin_m = 0.032, 0.02
    fixed_allowance = speed * dt_s / 8
    required = required_substeps(
        speed_bound_m_s=speed, dt_s=dt_s, margin_m=margin_m, minimum=8, maximum=4096
    )
    assert fixed_allowance == pytest.approx(0.1139726680109965)
    assert fixed_allowance > margin_m
    assert required == 46
    assert speed * dt_s / required <= margin_m
    assert speed * dt_s / required == pytest.approx(0.019821333567129826)

    fixed = ArmSweepContract(
        required_links=("arm",),
        required_objects=frozenset({"CAN"}),
        adjacent_pairs=frozenset(),
        dt_s=dt_s,
        margin_m=margin_m,
        substeps=8,
    )
    adaptive = ArmSweepContract(
        required_links=("arm",),
        required_objects=frozenset({"CAN"}),
        adjacent_pairs=frozenset(),
        dt_s=dt_s,
        margin_m=margin_m,
        substeps=8,
        adaptive_speed_bound_m_s=speed,
    )
    assert fixed.effective_substeps == 8
    assert adaptive.effective_substeps == 46
    with pytest.raises(Exception, match="exceeds the conservative cap"):
        required_substeps(
            speed_bound_m_s=speed, dt_s=dt_s, margin_m=margin_m, minimum=8, maximum=45
        )

    config = bindings.load_arm_sweep_config(bindings.ARM_SWEEP_CONFIG_PATH)
    assert config.status == "blocked"
    assert config.certificate is None
    assert config.adaptive_speed_bound_m_s == pytest.approx(speed)
    assert config.scene_catalog_sha256 == _sha256(SCENE_CATALOG)
    assert config.proto_manifest_sha256 == _sha256(PROTO_MANIFEST)
    assert bindings.arm_sweep_transition_check(bindings.ARM_SWEEP_CONFIG_PATH) is None


def _world(
    *,
    x: float = 0.0,
    objects: tuple[tuple[str, float], ...] = (("CAN", 2.0),),
    base_pose_world: Pose3D | None = None,
) -> WorldState:
    robot = RobotState(
        joint_pos={"arm_joint": x},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(x, 0.0, 0.5), IDENTITY),
        gripper_opening=0.045,
        base_pose_world=base_pose_world,
    )
    scene = SceneState(
        [
            ObjectState(
                name,
                "can",
                Pose3D(Vec3(position, 0.0, 0.5), IDENTITY),
                Vec3(0.0635, 0.0635, 0.1222),
                1.0,
            )
            for name, position in objects
        ]
    )
    return WorldState(scene, robot)


def _geometry(state: WorldState) -> tuple[Capsule, ...]:
    x = float(state.robot.joint_pos["arm_joint"])
    return (Capsule("arm", (x, 0.0, 0.4), (x, 0.0, 0.5), 0.02, 20.0),)


def _checker(scene: StaticSceneGeometry, *, links: tuple[str, ...] = ("arm",)) -> SweptArmCheck:
    return SweptArmCheck(
        _geometry,
        required_links=links,
        required_objects=frozenset({"CAN"}),
        scene_complete=True,
        dt_s=0.01,
        margin_m=0.01,
        adjacent_pairs=frozenset({frozenset({"arm", "gripper"})}),
        scene=scene,
    )


def _scene(
    *,
    box_center: tuple[float, float, float] = (1.0, 0.0, 0.45),
    floor_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    floor_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    gripper: bool = False,
    expected_pose: Pose3D | None = None,
) -> StaticSceneGeometry:
    return StaticSceneGeometry(
        boxes=(StaticBox("table", box_center, (0.1, 0.1, 0.1), "synthetic pinned-box proof"),),
        floors=(
            FloorHalfSpace("floor", floor_point, floor_normal, "synthetic pinned-plane proof"),
        ),
        gripper=GripperEnvelope(
            "gripper",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -0.2),
            0.1,
            20.0,
            "synthetic pinned-tool proof",
        )
        if gripper
        else None,
        supported_object_ids=frozenset({"CAN"}),
        catalog_sha256="0" * 64,
        base_pose_position_tolerance_m=0.003,
        base_pose_orientation_tolerance_rad=0.005,
    )


def test_static_scene_cases_cover_table_floor_gripper_and_fail_closed_paths():
    before, after = _world(), _world(x=0.1)
    assert _checker(_scene())(before, after) is True
    assert _checker(_scene(box_center=(0.1, 0.0, 0.45)))(before, after) is False
    assert _checker(_scene(floor_point=(0.0, 0.0, 0.48)))(before, after) is False
    assert (
        # Full-interval conservative padding extends the synthetic gripper sweep slightly below
        # z=0, so this case places the test-only floor beneath that explicit bound.
        _checker(_scene(gripper=True, floor_point=(0.0, 0.0, -0.05)), links=("arm", "gripper"))(
            _world(), _world(x=0.1)
        )
        is True
    )

    wrong_dimensions = _world()
    wrong_dimensions.scene.objects[0] = replace(
        wrong_dimensions.scene.objects[0], dimensions=Vec3(0.1, 0.1, 0.1)
    )
    assert _checker(_scene())(wrong_dimensions, _world(x=0.1)) is None
    unsupported = _world(objects=(("CAN", 2.0), ("OTHER", 3.0)))
    assert _checker(_scene())(unsupported, unsupported) is None
    moving = _world()
    moved = _world(x=0.1)
    moved.scene.objects[0] = replace(
        moved.scene.objects[0], pose=Pose3D(Vec3(3.0, 0.0, 0.5), IDENTITY)
    )
    assert _checker(_scene())(moving, moved) is None
    drifted = _world(base_pose_world=Pose3D(Vec3(0.01, 0.0, 0.0), IDENTITY))
    expected = Pose3D(Vec3(0.0, 0.0, 0.0), IDENTITY)
    scene = replace(_scene(), base_pose_position_tolerance_m=0.003)
    checker = SweptArmCheck(
        _geometry,
        required_links=("arm",),
        required_objects=frozenset({"CAN"}),
        scene_complete=True,
        dt_s=0.01,
        margin_m=0.01,
        adjacent_pairs=frozenset(),
        scene=scene,
        expected_base_pose_world=expected,
    )
    assert checker(drifted, drifted) is None
