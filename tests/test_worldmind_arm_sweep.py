"""Synthetic tests for the conservative full-arm sweep wrapper.

These tests use explicitly constructed capsules; they do NOT certify the TIAGo geometry.
Production still vetoes every prediction with `unverified-arm-sweep`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vlarobot.core.types import (
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.world_model.arm_geometry import Capsule
from vlarobot.world_model.arm_sweep import (
    ArmSweepContract,
    IntermediateSweptArmCheck,
    SweepGeometryError,
    interpolate_world_states,
    urdf_collision_capsules,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
LINKS = ("arm",)


def _state(x: float, *, objects: tuple[tuple[str, float], ...] = (("CAN", 10.0),)) -> WorldState:
    robot = RobotState(
        joint_pos={"arm_1_joint": x},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(x, 0.0, 0.55), IDENTITY),
        gripper_opening=0.045,
    )
    scene = SceneState(
        [
            ObjectState(
                name,
                "can",
                Pose3D(Vec3(position, 0.0, 0.55), IDENTITY),
                Vec3(0.02, 0.02, 0.02),
                1.0,
            )
            for name, position in objects
        ]
    )
    return WorldState(scene, robot)


def _geometry(state: WorldState, *, broken: bool = False) -> tuple[Capsule, ...]:
    if broken:
        return ()
    x = float(state.robot.joint_pos["arm_1_joint"])
    return (Capsule("arm", (x, 0.0, 0.5), (x, 0.0, 0.6), 0.02, 16.0),)


def _checker(
    *, substeps: int = 4, broken: bool = False, contract: ArmSweepContract | None = None
) -> IntermediateSweptArmCheck:
    return IntermediateSweptArmCheck(
        lambda state: _geometry(state, broken=broken),
        contract
        or ArmSweepContract(
            required_links=LINKS,
            required_objects=frozenset({"CAN"}),
            adjacent_pairs=frozenset(),
            dt_s=0.032,
            margin_m=0.01,
            substeps=substeps,
            adaptive_speed_bound_m_s=0.5,
            max_joint_displacement_bound_rad=0.0,
            joint_reach_bound_m=0.0,
            geometry_error_bound_m=0.0,
            registration_uncertainty_bound_m=0.0,
            numerical_error_bound_m=0.0,
            transform_error_bound_m=0.0,
        ),
    )


def test_interpolation_produces_inclusive_finite_states():
    states = interpolate_world_states(_state(0.0), _state(0.4), 4)
    assert len(states) == 5
    assert [s.robot.joint_pos["arm_1_joint"] for s in states] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )
    assert states[0].robot.joint_pos == _state(0.0).robot.joint_pos
    with pytest.raises(SweepGeometryError, match="base motion"):
        moving = _state(0.0)
        moving.robot.base_pose = Pose2D(1.0, 0.0, 0.0)
        interpolate_world_states(moving, _state(0.4), 4)
    moving_object = _state(0.4, objects=(("CAN", 9.0),))
    with pytest.raises(SweepGeometryError, match="moving scene"):
        interpolate_world_states(_state(0.0), moving_object, 4)


def test_sweep_detects_mid_interval_collision_and_clears_free_space():
    collision = _checker()(
        _state(0.0, objects=(("CAN", 0.2),)), _state(0.4, objects=(("CAN", 0.2),))
    )
    assert collision is False
    clear = _checker()(_state(0.0), _state(0.4))
    assert clear is True


def test_missing_or_invalid_geometry_fails_closed():
    assert _checker(broken=True)(_state(0.0), _state(0.4)) is None
    contract = ArmSweepContract(
        required_links=LINKS,
        required_objects=frozenset({"MISSING"}),
        adjacent_pairs=frozenset(),
        dt_s=0.032,
        margin_m=0.01,
        substeps=4,
    )
    assert _checker(contract=contract)(_state(0.0), _state(0.4)) is None

    def nonfinite(state: WorldState) -> tuple[Capsule, ...]:
        del state
        return (Capsule("arm", (float("nan"), 0.0, 0.5), (0.0, 0.0, 0.6), 0.02, 16.0),)

    assert (
        IntermediateSweptArmCheck(nonfinite, _checker().contract)(_state(0.0), _state(0.4)) is None
    )

    def raising(state: WorldState) -> tuple[Capsule, ...]:
        del state
        raise SweepGeometryError("missing transform")

    assert IntermediateSweptArmCheck(raising, _checker().contract)(_state(0.0), _state(0.4)) is None


def test_unsupported_contact_and_moving_obstacles_fail_closed():
    full_before = _state(0.0, objects=(("CAN", 10.0), ("OTHER", 9.0)))
    partial_after = _state(0.4, objects=(("CAN", 10.0),))
    moving_after = _state(0.4, objects=(("CAN", 10.0), ("OTHER", 8.5)))
    assert _checker()(full_before, partial_after) is None
    assert _checker()(full_before, moving_after) is None
    static_after = _state(0.4, objects=(("CAN", 10.0), ("OTHER", 9.0)))
    contract = ArmSweepContract(
        required_links=LINKS,
        required_objects=frozenset({"CAN", "OTHER"}),
        adjacent_pairs=frozenset(),
        dt_s=0.032,
        margin_m=0.01,
        substeps=4,
        permitted_contact_objects=frozenset({"OTHER"}),
        adaptive_speed_bound_m_s=0.5,
        max_joint_displacement_bound_rad=0.0,
        joint_reach_bound_m=0.0,
        geometry_error_bound_m=0.0,
        registration_uncertainty_bound_m=0.0,
        numerical_error_bound_m=0.0,
        transform_error_bound_m=0.0,
    )
    assert _checker(contract=contract)(full_before, static_after) is True


def test_urdf_capsule_parser_covers_all_arm_links_and_rejects_unsupported_shapes(tmp_path):
    geometry_dir = Path(__file__).resolve().parents[1] / "src" / "vlarobot" / "backends" / "webots"
    urdfs = sorted(geometry_dir.glob("*.urdf"))
    assert urdfs, "no URDF fixture is available for the parser test"
    urdf = urdfs[0]
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
    parsed = urdf_collision_capsules(urdf, links, max_point_speed_m_s=16.0)
    assert set(parsed) == set(links)
    assert all(primitives for primitives in parsed.values())
    assert all(
        primitive.certified_containment
        for primitives in parsed.values()
        for primitive in primitives
    )

    bad = tmp_path / "unsupported.urdf"
    bad.write_text(
        '<robot name="x"><link name="arm_2_link"><collision>'
        '<origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="x.stl"/>'
        "</geometry></collision></link></robot>",
        encoding="utf-8",
    )
    with pytest.raises(SweepGeometryError, match="unsupported collision geometry"):
        urdf_collision_capsules(bad, ("arm_2_link",), max_point_speed_m_s=16.0)
