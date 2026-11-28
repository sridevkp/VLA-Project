from dataclasses import replace
from unittest.mock import Mock

from vlarobot.backends.fake.backend import FakeBackend
from vlarobot.backends.webots.config import TIME_STEP
from vlarobot.core.types import (
    ActionChunk,
    EEPoseTarget,
    GroundTruth,
    ObjectState,
    Pose3D,
    ScenarioSpec,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.execution.skill_proxy import assess_skill_proxy
from vlarobot.execution.skills import Skill
from vlarobot.world_model.rollout import AnalyticSurrogate, Trajectory


def test_proxy_detects_obstacle_between_samples_and_preserves_strict_unknown() -> None:
    backend = FakeBackend()
    initial = backend.reset(ScenarioSpec(), 0)
    rotation = (0.0, 0.0, 0.0, 1.0)
    start = Pose3D(Vec3(0.2, 0.0, 1.0), rotation)
    end = Pose3D(Vec3(0.8, 0.0, 1.0), rotation)
    can = Pose3D(Vec3(0.5, 0.3, 0.8), rotation)
    truth = GroundTruth(
        {
            "CAN": can,
            "HAM": Pose3D(Vec3(0.5, 0.0, 1.0), rotation),
            "APP": Pose3D(Vec3(1.2, 0.5, 1.0), rotation),
        }
    )
    robot = replace(
        initial.robot, ee_pose=start, base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), rotation)
    )
    state = WorldState(
        SceneState([ObjectState("CAN", "can", can, Vec3(0.06, 0.06, 0.1), 1.0)]), robot
    )
    future = replace(state, robot=replace(robot, ee_pose=end))
    dt = TIME_STEP / 1000.0
    chunk = ActionChunk([EEPoseTarget(end, None, dt)], dt, "vla")
    surrogate = Mock(spec=AnalyticSurrogate)
    surrogate.predict_trajectory.return_value = Trajectory(
        (state, future), dt, ("unverified-arm-sweep",), (), ()
    )
    result = assess_skill_proxy(
        state, truth, Skill.APPROACH, chunk, surrogate, backend.safety_monitor
    )
    assert result.strict_reasons == ("unverified-arm-sweep",)
    assert not result.full_arm_sweep_verified
    assert dict(result.checks)["table"]
    assert not dict(result.checks)["modeled_obstacles"]
    truth.object_poses["HAM"] = Pose3D(Vec3(0.5, -0.4, 1.0), rotation)
    result = assess_skill_proxy(
        state, truth, Skill.APPROACH, chunk, surrogate, backend.safety_monitor
    )
    assert all(dict(result.checks).values())
    surrogate.predict_trajectory.return_value = replace(
        surrogate.predict_trajectory.return_value, hard_reasons=("arm-sweep-collision",)
    )
    result = assess_skill_proxy(
        state, truth, Skill.APPROACH, chunk, surrogate, backend.safety_monitor
    )
    assert not dict(result.checks)["predictive_model_constraints"]
