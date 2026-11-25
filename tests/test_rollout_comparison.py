"""Chunk state prediction, conditional transport, arm ablations and safety/fallback integration."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from vlarobot.core.types import (
    ActionChunk,
    EEDelta,
    GripperCmd,
    JointTarget,
    ObjectState,
    Outcome,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.risk.comparison import ComparisonPlanner, sampled_candidates
from vlarobot.risk.gate import CandidateAssessment, RiskGate
from vlarobot.world_model.evaluation import prediction_errors
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate, Attachment, Trajectory, compose


def forward(q):
    return Pose3D(Vec3(q["x"], q["y"], q["z"]), (0, 0, 0, 1))


def world():
    q = dict(x=0.0, y=0.0, z=0.5)
    robot = RobotState(q, {}, Pose2D(0, 0, 0), forward(q), 0.02, True)
    obj = ObjectState(
        "can", "can", Pose3D(Vec3(0, 0, 0.45), (0, 0, 0, 1)), Vec3(0.1, 0.1, 0.1), 1.0
    )
    return WorldState(SceneState([obj]), robot)


def surrogate(**kwargs):
    return AnalyticSurrogate(
        VelocityLimitedKinematics(dict(x=1.0, y=1.0, z=1.0), forward),
        inverse=lambda pose, state: JointTarget(
            dict(zip(("x", "y", "z"), pose.position.as_array(), strict=True))
        ),
        gripper_speed_m_s=1.0,
        gravity_m_s2=9.81,
        support_z_m={"can": 0.4},
        validate=lambda action, state: (
            not isinstance(action, JointTarget) or action.q.get("z", 0) <= 0.65
        ),
        check_transition=lambda a, b: True,
        **kwargs,
    )


def test_observed_attachment_lift_release_gravity_and_nonalias():
    state = world()
    actions = [JointTarget({"z": 0.6}), GripperCmd(0.045), *[GripperCmd(0.045)] * 3]
    chunk = ActionChunk(actions, 0.1, "vla")
    attachment = Attachment("can", Pose3D(Vec3(0, 0, -0.05), (0, 0, 0, 1)), 0.04)
    prediction = surrogate().predict_trajectory(state, chunk, attachment=attachment)
    assert len(prediction.states) == 6 and not prediction.hard_reasons
    assert prediction.states[1].scene.objects[0].pose.position.z == pytest.approx(0.55)
    assert prediction.states[-1].scene.objects[0].pose.position.z == pytest.approx(0.45)
    assert prediction.released_objects == ("can",)
    assert not prediction.states[-1].robot.gripper_holding
    assert state.robot.joint_pos["z"] == 0.5
    prediction.states[-1].scene.objects.clear()
    assert len(prediction.states[0].scene.objects) == 1


def test_close_command_never_fabricates_grasp_and_unknown_sweep_fails_closed():
    model = surrogate()
    model.check_transition = None
    state = world()
    state.robot.gripper_holding = None
    chunk = ActionChunk([GripperCmd(0)], 0.1, "vla")
    prediction = model.predict_trajectory(state, chunk)
    assert prediction.states[-1].robot.gripper_holding is None
    assert "unverified-arm-sweep" in prediction.hard_reasons
    with pytest.raises(ValueError, match="observed"):
        model.predict_trajectory(
            state, chunk, attachment=Attachment("can", state.robot.ee_pose, 0.04)
        )


def test_ik_joint_veto_stops_before_unsafe_state_and_calibration_is_required():
    chunk = ActionChunk([EEDelta(Vec3(0, 0, 0.3), Vec3(0, 0, 0), 0.02)], 0.1, "vla")
    model = surrogate()
    prediction = model.predict_trajectory(world(), chunk)
    assert prediction.hard_reasons == ("ik-joint-safety-veto",)
    assert len(prediction.states) == 1
    with pytest.raises(ValueError, match="calibration"):
        model.rollout(world(), chunk, 1)
    calibrated = surrogate(
        calibrate=lambda t: Outcome(0.8, 0.1, 0.05), calibration_id="synthetic-test-only"
    )
    assert (
        calibrated.rollout(world(), ActionChunk([GripperCmd(0.02)], 0.1, "vla"), 2)
        == [Outcome(0.8, 0.1, 0.05)] * 2
    )


def test_base_frame_rotation_and_quantity_separated_errors():
    pose = compose(
        Pose3D(Vec3(0, 0, 0), (0, 0, np.sqrt(0.5), np.sqrt(0.5))),
        Pose3D(Vec3(0.1, 0, 0), (0, 0, 0, 1)),
    )
    assert pose.position.y == pytest.approx(0.1)
    prediction = surrogate().predict_trajectory(
        world(), ActionChunk([JointTarget({"z": 0.6})], 0.1, "vla")
    )
    observed = copy.deepcopy(prediction.states)
    observed[-1].robot.ee_pose = replace(observed[-1].robot.ee_pose, position=Vec3(0.03, 0.04, 0.6))
    error = prediction_errors(prediction, observed)
    assert error["robot_position_rmse_m"] == pytest.approx(0.05)
    assert error["object_position_rmse_m"] == 0


def planner(arm):
    def predict(state, chunk):
        unsafe = chunk.actions[0].q["x"] > 0.15
        return Trajectory(
            (state, copy.deepcopy(state)), chunk.dt, ("collision",) if unsafe else (), (), ()
        )

    def assess(state, chunk, trajectory):
        x = chunk.actions[0].q["x"]
        return CandidateAssessment((("geometry", x <= 0.15), ("ik", True)), (x, 0, 0, 0))

    return ComparisonPlanner(
        arm,
        RiskGate(weights=(1, 0, 0, 0), maximize=True, required_constraints=("geometry", "ik")),
        predict=predict,
        assess=assess,
    )


@pytest.mark.parametrize(
    "arm,expected",
    [("vla_only", 0), ("vla_risk_veto", 0), ("vla_world_model", 1), ("full_system", 0)],
)
def test_frozen_arm_switches_have_intended_ablation(arm, expected):
    choices = [ActionChunk([JointTarget({"x": x})], 0.1, "vla") for x in (0.1, 0.2)]
    result = planner(arm).select(world(), choices)
    assert result.index == expected
    assert result.prediction_count == (2 if arm in ("vla_world_model", "full_system") else 0)


def test_bounded_reobserve_then_scripted_alternate_and_abort():
    bad = [ActionChunk([JointTarget({"x": 0.2})], 0.1, "vla")]
    good = [ActionChunk([JointTarget({"x": 0.1})], 0.1, "scripted")]
    calls = []
    result = planner("full_system").with_recovery(
        world(),
        lambda s: bad,
        reobserve=lambda: calls.append("observe") or world(),
        alternate_family=lambda s: good,
    )
    assert result.chunk.source == "scripted"
    assert result.recovery_events == ("reobserve", "alternate_candidate_family")
    assert calls == ["observe"]
    failed = planner("full_system").with_recovery(
        world(), lambda s: bad, reobserve=world, alternate_family=lambda s: bad
    )
    assert failed.chunk is None and failed.recovery_events[-1] == "abort"
    with pytest.raises(ValueError, match="substitute"):
        planner("vla_only").select(world(), good)


def test_sample_seeds_are_explicit_repeatable_and_commands_not_perturbed():
    def sample(seed):
        return ActionChunk([GripperCmd(seed / 2**31 * 0.045)], 0.1, "vla")

    chunks, seeds = sampled_candidates(sample, seed=7, count=4)
    again, repeat = sampled_candidates(sample, seed=7, count=4)
    assert chunks == again and seeds == repeat and len(set(seeds)) == 4


def test_execution_veto_precedes_backend_apply_for_every_source():
    from vlarobot.backends.fake import FakeBackend
    from vlarobot.core.types import ScenarioSpec
    from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
    from vlarobot.planning.execution import execute_first
    from vlarobot.risk.comparison import Selection

    class CountingBackend(FakeBackend):
        calls = 0

        def apply(self, action):
            self.calls += 1
            return super().apply(action)

    backend = CountingBackend()
    obs = backend.reset(ScenarioSpec(), 0)
    monitor = SafetyMonitor(SafetyLimits(gripper_opening=(0, 0.045)))
    for source in ("vla", "scripted"):
        selection = Selection(ActionChunk([GripperCmd(0.2)], 0.032, source), 0, "test")
        _, verdict = execute_first(selection, obs, backend, monitor)
        assert not verdict.accepted and backend.calls == 0
    selection = Selection(ActionChunk([GripperCmd(0.04)], 0.032, "scripted"), 0, "fallback")
    after, verdict = execute_first(selection, obs, backend, monitor)
    assert verdict.accepted and backend.calls == 1 and after.t > obs.t


def test_five_arm_plans_share_pairs_and_test_guard_precedes_generation(monkeypatch):
    from vlarobot.eval import comparison

    plan = comparison.comparison_plan("validation", 0)
    assert len(plan["arms"]) == 5 and len(plan["episodes"]) == 18
    assert len({(e["scene"]["metadata"]["index"], e["seed"]) for e in plan["episodes"]}) == 18
    monkeypatch.setattr(
        comparison, "build_episode_plan", lambda *a, **k: pytest.fail("TEST generator accessed")
    )
    with pytest.raises(ValueError, match="TEST is sealed"):
        comparison.comparison_plan("test", 0)


def test_arm_swept_volume_certificate_and_unknown_coverage():
    from vlarobot.world_model.arm_geometry import Capsule, SweptArmCheck

    state = world()
    # Narrow stationary obstacle between safe arm endpoints must be rejected.
    state.scene.objects[0] = replace(
        state.scene.objects[0],
        pose=Pose3D(Vec3(0.5, 0, 0.5), (0, 0, 0, 1)),
        dimensions=Vec3(0.001, 0.1, 0.1),
    )

    def geometry(s):
        return (Capsule("arm", (0.0, 0.0, 0.5), (1.0, 0.0, 0.5), 0.02, 0.1),)

    check = SweptArmCheck(
        geometry,
        required_links=("arm",),
        required_objects=frozenset(),
        scene_complete=True,
        dt_s=0.032,
        margin_m=0.01,
        adjacent_pairs=frozenset(),
    )
    assert check(state, copy.deepcopy(state)) is False
    incomplete = SweptArmCheck(
        geometry,
        required_links=("arm", "missing"),
        required_objects=frozenset(),
        scene_complete=True,
        dt_s=0.032,
        margin_m=0.01,
        adjacent_pairs=frozenset(),
    )
    assert incomplete(state, state) is None
    state.scene.objects.clear()
    assert check(state, state) is True
    check.scene_complete = False
    assert check(state, state) is None


def test_common_recorder_all_five_arms_are_explicitly_simulated(tmp_path):
    from vlarobot.backends.fake import FakeBackend
    from vlarobot.eval.comparison import run_development_comparison
    from vlarobot.risk.comparison import ARMS

    class FixturePolicy:
        def __init__(self, scene):
            self.scene = scene

        def execute(self, backend, seed):
            backend.reset(self.scene, seed)
            backend.step(1)
            return {
                "success": False,
                "failure_state": "FIXTURE",
                "failure_reason": "synthetic integration only",
            }

    result = run_development_comparison(
        "validation",
        0,
        tmp_path / "five-arms",
        factories={arm: FixturePolicy for arm in ARMS},
        backend_factory=lambda arm: FakeBackend(),
        measurement="simulated",
    )
    assert set(result["arms"]) == set(ARMS)
    assert result["measurement"] == "simulated" and result["role_gate"] == "UNMEASURED"
    assert all(summary["n_episodes"] == 18 for summary in result["arms"].values())
    assert all(
        summary["week2_gate"]["threshold"] is None
        for arm, summary in result["arms"].items()
        if arm != "scripted_baseline"
    )


def test_linearized_ik_exact_residual_guard_and_reset_independence():
    from vlarobot.world_model.linearized_ik import LinearizedIK

    model = LinearizedIK(
        forward,
        bounds={n: (-1, 1) for n in ("x", "y", "z")},
        max_joint_step_rad=0.15,
        position_tolerance_m=0.001,
        orientation_tolerance_rad=0.01,
        difference_step_rad=1e-5,
        damping=1e-6,
    )
    state = world().robot
    model.reset(state)
    target = Pose3D(Vec3(0.03, 0.04, 0.5), (0, 0, 0, 1))
    solved = model.solve(target, state)
    assert solved is not None
    assert (
        np.linalg.norm(forward(solved.q).position.as_array() - target.position.as_array()) < 0.001
    )
    assert model.solve(Pose3D(Vec3(3, 0, 0.5), (0, 0, 0, 1)), state) is None
    model.reset(state)
    assert model.solve(target, state) == solved


def test_recorded_conditional_attachment_uses_flat_initial_oracle_not_future_states(tmp_path):
    import json

    from vlarobot.execution.attachment_validation import conditional_attachment_errors

    raw = tmp_path / "raw"
    raw.mkdir()
    entry = {"raw_dir": str(raw), "scene": {"metadata": {"split": "train", "family": "clear"}}}
    (tmp_path / "accepted.json").write_text(json.dumps({"episodes": [entry]}))
    initial = {
        "state": "TRANSIT",
        "phase": "enter",
        "sim_time_s": 0.0,
        "base_world_xyz_m": [0, 0, 0],
        "base_world_orientation_xyzw": [0, 0, 0, 1],
        "can_base_xyz_m": [0, 0, 0.45],
        "can_base_orientation_matrix": np.eye(3).reshape(-1).tolist(),
        "measured_joint_positions_rad": {"x": 0.0, "y": 0.0, "z": 0.5},
    }
    stop = {"state": "PLACE_DESCEND", "phase": "enter", "sim_time_s": 2.0}
    (raw / "policy.jsonl").write_text(json.dumps(initial) + "\n" + json.dumps(stop) + "\n")
    (raw / "episode-00.actions.jsonl").write_text(
        json.dumps({"t": 0.0, "action": {"q": {"z": 0.6}}}) + "\n"
    )
    xyz = np.array([[[0, 0, 0.45]], [[0, 0, 0.55]], [[0, 0, 0.55]]])
    np.savez(
        raw / "episode-00.steps.npz",
        t=[0.0, 0.1, 0.2],
        joint_names=["x", "y", "z"],
        joint_pos=[[0, 0, 0.5], [0, 0, 0.6], [0, 0, 0.6]],
        gripper=[0.02] * 3,
        gt_names=["CAN"],
        gt_xyz=xyz,
        gt_quat=np.tile([[[0, 0, 0, 1]]], (3, 1, 1)),
    )
    result = conditional_attachment_errors(
        tmp_path,
        forward,
        lambda m: (0.0, 0.0, 0.0, 1.0),
        joint_names=("x", "y", "z"),
        speed_rad_s=1.0,
        horizons=(1, 2),
    )
    assert result["families"]["clear"]["2"]["can_world_position_rmse_m"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result["not_grasp_success_prediction"] and result["brier_score"] is None


def test_candidate_generator_protocol_and_observed_placement_retry():
    from vlarobot.core.types import TaskSpec
    from vlarobot.planning.candidates import SeededCandidateGenerator
    from vlarobot.planning.verification import PlacementVerifier

    task = TaskSpec("place can", "can", "target")

    def draw(task, state, memory, seed):
        return ActionChunk([GripperCmd(0.02)], 0.032, "vla")

    generator = SeededCandidateGenerator(draw, seed=0, count=4, source="vla")
    assert len(generator.propose(task, world(), None)) == 4
    again = SeededCandidateGenerator(draw, seed=0, count=4, source="vla")
    again.propose(task, world(), None)
    assert generator.draw_log == again.draw_log
    state = world()
    goal = state.scene.objects[0].pose
    verifier = PlacementVerifier(
        destination_base=goal,
        rest_center_base=goal,
        up_axis_base=(0, 0, 1),
        home_joints=state.robot.joint_pos,
        distance_tolerance_m=0.1,
        height_tolerance_m=0.02,
        tilt_limit_rad=np.deg2rad(15),
        home_tolerance_rad=0.02,
    )
    observed = copy.deepcopy(state)
    observed.scene.objects[0] = replace(
        observed.scene.objects[0], pose=Pose3D(Vec3(0.3, 0, 0.45), (0, 0, 0, 1))
    )
    result = verifier.check(task, state, observed)
    assert not result.success  # HOME and a predicted placement are insufficient.
    assert verifier.next_action(result) == "retry_once"
    assert verifier.next_action(result) == "abort"
    assert verifier.check(task, state, state).success
