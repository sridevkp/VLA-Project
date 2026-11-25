"""Learned-only control stops on veto/home/budget, and the gate requires real paired evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vlarobot.core.types import (
    ApplyResult,
    CameraInfo,
    JointTarget,
    Observation,
    Pose2D,
    Pose3D,
    RobotState,
    ScenarioSpec,
    Vec3,
)
from vlarobot.eval.vla_metrics import final_can_outcome, paired_vla_gate
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.vla_episode import VLAEpisodePolicy
from vlarobot.vla.action_adapter import ActionAdapter, PoseSolution
from vlarobot.vla.checkpoint_export import require_trained_export

CAMERA = CameraInfo(
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    1,
    1,
)


class Backend:
    def __init__(self):
        self.q = 0.0
        self.commands = []
        self.steps = 0

    def observation(self):
        robot = RobotState(
            {"arm_1_joint": self.q},
            {},
            Pose2D(0, 0, 0),
            Pose3D(Vec3(0.5 + self.q * 0.1, 0, 0.5), (0, 0, 0, 1)),
            0.02,
        )
        return Observation(
            self.steps * 0.032, np.zeros((1, 1, 3), dtype=np.uint8), None, CAMERA, robot
        )

    def reset(self, scene, seed):
        return self.observation()

    def apply(self, action):
        self.commands.append(action)
        if isinstance(action, JointTarget):
            self.q = action.q["arm_1_joint"]
        return ApplyResult(True)

    def step(self, n=1):
        self.steps += n
        return self.observation()

    def get_ground_truth(self):
        raise AssertionError("policy control must not read oracle state")


class Solver:
    def solve(self, pose, state):
        q = 0.1 if pose.position.x > 0.505 else 0.0
        return PoseSolution({"arm_1_joint": q}, 0, 0)


class Model:
    def __init__(self, actions):
        self.actions = actions
        self.calls = 0

    def reset(self, seed):
        self.seed = seed

    def predict(self, obs, instruction):
        self.calls += 1
        return self.actions.copy(), {"inference_wall_s": 0.1, "roundtrip_wall_s": 0.11}


def policy(tmp_path, actions, max_steps=2):
    model = Model(np.asarray(actions, dtype=np.float32))
    adapter = ActionAdapter(
        Solver(),
        SafetyMonitor(
            SafetyLimits(joint_position={"arm_1_joint": (-1, 1)}, joint_velocity={"arm_1_joint": 1})
        ),
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.01,
    )

    def score(backend, obs, home, state, reason):
        return {"success": home, "failure_state": state}

    return VLAEpisodePolicy(
        ScenarioSpec(),
        model,
        adapter,
        instruction="place can",
        max_steps=max_steps,
        max_chunk_actions=50,
        home_tolerance_rad=0.02,
        arm_names=("arm_1_joint",),
        trace=tmp_path / "latency.jsonl",
        score=score,
    ), model


def test_vla_returns_home_only_using_learned_actions_and_explicit_steps(tmp_path):
    actions = [[0.01, 0, 0, 0, 0, 0, 0.02], [-0.01, 0, 0, 0, 0, 0, 0.02]]
    control, model = policy(tmp_path, actions)
    backend = Backend()
    result = control.execute(backend, 12)
    assert result["success"] and result["executed_actions"] == 2
    assert model.calls == 1 and model.seed == 12 and backend.steps == 2
    assert result["scripted_actions_sent"] == 0


def test_vla_invalid_chunk_and_safety_veto_do_not_become_fallback_actions(tmp_path):
    control, _ = policy(tmp_path, [[0, 0, 0, 0, 0, 0, 2.0]])
    backend = Backend()
    outcome = control.execute(backend, 0)
    assert not outcome["success"] and outcome["failure_state"] == "SAFETY"
    assert not backend.commands


def test_staying_at_initial_home_is_not_episode_completion(tmp_path):
    control, _ = policy(tmp_path, [[0, 0, 0, 0, 0, 0, 0.02]])
    result = control.execute(Backend(), 0)
    assert not result["success"] and result["failure_state"] == "BUDGET"


def test_frozen_final_physics_rejects_home_without_placement():
    initial = Pose3D(Vec3(0, 0, 0.8), (0, 0, 0, 1))
    result = final_can_outcome(
        initial,
        initial,
        place_world_m=(0.25, 0, 0.8),
        distance_tolerance_m=0.1,
        height_tolerance_m=0.02,
        tilt_limit_deg=15,
        home_reached=True,
        failure_state=None,
        failure_reason=None,
    )
    assert not result["success"] and result["failure_state"] == "PLACEMENT"


def summary(pipeline, successes):
    return {
        "pipeline": pipeline,
        "measurement": "measured",
        "split": "train",
        "n_episodes": 24,
        "seeds": [0, 1, 2],
        "success": {"successes": successes, "total": 24, "rate": successes / 24},
        "per_scene": [{"scene": f"train-slot-{i}", "seeds": [0, 1, 2]} for i in range(8)],
    }


def test_gate_literal_40_percent_needs_ten_of_24_and_no_zero_baseline_pass():
    baseline = summary("baseline", 23)
    assert paired_vla_gate(baseline, summary("vla_only", 9))["primary"] == "scripted"
    passed = paired_vla_gate(baseline, summary("vla_only", 10))
    assert passed["passed"] is True and passed["threshold"] == 0.4
    assert paired_vla_gate(summary("baseline", 0), summary("vla_only", 0))["passed"] is None
    fake = summary("vla_only", 24)
    fake["measurement"] = "fake"
    with pytest.raises(ValueError):
        paired_vla_gate(baseline, fake)
    unpaired = copy.deepcopy(summary("vla_only", 24))
    unpaired["per_scene"][0]["seeds"] = [3, 4, 5]
    with pytest.raises(ValueError):
        paired_vla_gate(baseline, unpaired)


def test_compatibility_export_cannot_be_evaluated_as_trained(tmp_path):
    (tmp_path / "export-manifest.json").write_text(
        json.dumps({"checkpoint_step": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="optimizer-step"):
        require_trained_export(tmp_path)


def test_recovery_projection_never_changes_pose_and_rejects_malformed_values():
    from vlarobot.execution.vla_episode import project_recovery_chunk

    values = np.array([[0.01, 0, 0, 0, 0, 0, 0.04579], [0, 0, 0, 0, 0, 0, -0.0003]])
    projected = project_recovery_chunk(values, (0.0, 0.045))
    np.testing.assert_array_equal(projected[:, :6], values[:, :6])
    np.testing.assert_array_equal(projected[:, 6], [0.045, 0.0])
    for bad in (2.0, float("nan"), 0.0461):
        values[0, 6] = bad
        with pytest.raises(ValueError):
            project_recovery_chunk(values, (0.0, 0.045))


def test_v2_ignores_unverified_home_and_requeries_model(tmp_path):
    actions = [[0.01, 0, 0, 0, 0, 0, 0.04579], [-0.01, 0, 0, 0, 0, 0, 0.045]]
    control, model = policy(tmp_path, actions, max_steps=4)
    control.recovery_v2 = True
    backend = Backend()
    control.score = lambda backend, obs, home, state, reason: {
        "success": home and backend.steps == 4,
        "failure_state": state,
    }
    result = control.execute(backend, 12)
    assert result["success"] and result["executed_actions"] == 4
    assert result["home_completion_candidates"] == 2 and model.calls == 2
    trace = [json.loads(line) for line in control.trace.read_text().splitlines()]
    assert trace[0]["projected_count"] == 2  # Includes the float32 representation of 45 mm.
    assert trace[0]["actions"][0][6] > 0.045
    assert trace[0]["projected_gripper_m"] == [0.045, 0.045]
    assert all(c.opening <= 0.045 for c in backend.commands if hasattr(c, "opening"))


def test_recovery_projection_rejection_preserves_the_model_input_without_apply(tmp_path):
    control, _ = policy(tmp_path, [[0, 0, 0, 0, 0, 0, 0.047]])
    control.recovery_v2 = True
    backend = Backend()
    result = control.execute(backend, 0)
    trace = json.loads(control.trace.read_text())
    assert result["failure_state"] == "MODEL"
    assert trace["projection_error"] and trace["actions"][0][6] > 0.046
    assert Path(trace["input_image"]).is_file()
    assert not backend.commands and backend.steps == 0
