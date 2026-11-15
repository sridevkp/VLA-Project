"""Copy mutable state containers while sharing only the core's frozen value objects."""

from __future__ import annotations

import copy

from vlarobot.core.types import RobotState, SceneState, WorldState


def copy_robot(state: RobotState) -> RobotState:
    """Preserve independent dictionaries/diagnostics; Pose2D/Pose3D/Vec3 are frozen."""
    return RobotState(
        dict(state.joint_pos),
        dict(state.joint_vel),
        state.base_pose,
        state.ee_pose,
        state.gripper_opening,
        state.gripper_holding,
        state.base_pose_world,
        copy.deepcopy(state.diagnostics) if state.diagnostics else {},
    )


def copy_world(state: WorldState) -> WorldState:
    """ObjectState and its pose/dimensions are frozen; scene lists remain independent."""
    return WorldState(SceneState(list(state.scene.objects)), copy_robot(state.robot))
