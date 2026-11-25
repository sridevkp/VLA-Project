"""Analytic tracking uses explicit speed/dt and never mutates observations or assumes grasp."""

import pytest

from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState, Vec3
from vlarobot.world_model.kinematic import VelocityLimitedKinematics


def test_joint_step_is_velocity_bounded_causal_and_nonmutating():
    def forward(joints):
        return Pose3D(Vec3(joints["arm"], 0, 0), (0, 0, 0, 1))

    model = VelocityLimitedKinematics({"arm": 0.5}, forward)
    state = RobotState({"arm": 0.0}, {}, Pose2D(0, 0, 0), forward({"arm": 0.0}), 0.031)
    predicted = model.step(state, JointTarget({"arm": 1}), 0.032)
    assert predicted.joint_pos["arm"] == pytest.approx(0.016)
    assert predicted.ee_pose.position.x == pytest.approx(0.016)
    assert predicted.gripper_opening == 0.031 and state.joint_pos["arm"] == 0
    assert model.step(state, JointTarget({"arm": 1}), 0.032) == predicted
    with pytest.raises(ValueError):
        model.step(state, JointTarget({"unknown": 1}), 0.032)


pytestmark = pytest.mark.safe_ci
