"""Offline regression for ADR-0073: full-horizon endpoints and campaign fail-stop.

The preserved Stage 1 failure campaign is read-only evidence. The endpoint tests reconstruct
only the first observation of the two episodes that could not be solved by the old one-tick
trust region; they never launch Webots, resume, rerun or write into that campaign. All other
tests use inert stubs and FakeBackend.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_worldmind_stage1_runner import _one_episode_plan, _setup

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import ARM_JOINTS, MAX_JOINT_STEP_RAD
from vlarobot.backends.webots.pose_solver import quaternion_matrix
from vlarobot.backends.worldmind_bindings import (
    _rotation_to_quaternion,
    corrected_kinematics,
    corrected_pose_solver,
    corrected_scripted_waypoints,
)
from vlarobot.core.types import (
    ApplyResult,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    ScenarioSpec,
    SceneState,
    TaskSpec,
    Vec3,
    WorldState,
)
from vlarobot.eval.runner import (
    CampaignFailStop,
    EvalEpisode,
    EvalReport,
    EvalRunConfig,
    run_evaluation,
)
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import (
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.execution.worldmind_pipeline import _inverse_pose
from vlarobot.execution.worldmind_stage1_pipeline import run_stage1_evidence
from vlarobot.world_model.rollout import compose

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "worldmind_diagnostic.approved.yaml"
CAMPAIGN = ROOT / "runs" / "worldmind" / "stage1-validation-v1-2026-11-20"
GEOMETRY = ROOT / "runs" / "worldmind" / "geometry-inspection-v5-2026-11-19" / "geometry.json"
IDENTITY = (0.0, 0.0, 0.0, 1.0)
NEEDS_EVIDENCE = pytest.mark.skipif(
    not (CAMPAIGN / "episode-11.steps.npz").is_file() or not GEOMETRY.is_file(),
    reason="preserved offline Stage 1 evidence is not present in this checkout",
)


def _full_joint_vector(
    kinematics: Any, joints: dict[str, float], overrides: dict[str, float] | None = None
) -> np.ndarray:
    vector = np.zeros(len(kinematics.chain.links), dtype=float)
    for name, value in joints.items():
        if name in kinematics.idx:
            vector[kinematics.idx[name]] = float(value)
    for name, value in (overrides or {}).items():
        if name in kinematics.idx:
            vector[kinematics.idx[name]] = float(value)
    return vector


def _reviewed_residual(kinematics: Any, vector: np.ndarray, pose: Pose3D) -> tuple[float, float]:
    """Use the reviewed base-frame position and rotation residual formulas unchanged."""
    actual = kinematics.forward(vector)
    target_rotation = quaternion_matrix(pose.orientation)
    position_error = float(np.linalg.norm(actual[:3, 3] - pose.position.as_array()))
    cosine = (np.trace(target_rotation @ actual[:3, :3].T) - 1.0) / 2.0
    orientation_error = math.acos(float(np.clip(cosine, -1.0, 1.0)))
    return position_error, orientation_error


def _first_draw_seed(config: Any, episode_index: int) -> int:
    sequence = np.random.SeedSequence([config.pipeline.decision_seed_base, episode_index])
    decision_seed = int(sequence.generate_state(1)[0])
    draws = np.random.default_rng(decision_seed).integers(
        0, 2**31, size=config.pipeline.candidate_count
    )
    return int(draws[0])


def _preserved_first_state(episode_index: int) -> tuple[WorldState, Any, TaskSpec]:
    """Reconstruct one first observation without writing to preserved evidence."""
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    base_record = geometry["table_target_reconciliation"]["base_comparison"][
        "T_world_base_live_observed"
    ]
    transform = np.asarray(base_record, dtype=float)
    base_angles = Rotation.from_matrix(transform[:3, :3]).as_euler("ZYX")
    base_z = float(transform[2, 3])

    config = load_worldmind_diagnostic_config(CONFIG_PATH)
    task = TaskSpec(
        instruction=config.task.instruction,
        target=config.task.target,
        destination=config.task.destination,
    )
    with np.load(CAMPAIGN / f"episode-{episode_index:02d}.steps.npz", allow_pickle=False) as data:
        joints = {
            str(name): float(value)
            for name, value in zip(data["joint_names"], data["joint_pos"][0], strict=True)
        }
        x, y, yaw = (float(value) for value in data["base_pose"][0])
        names = [str(name) for name in data["gt_names"]]
        can_index = names.index("CAN")
        can_world = Pose3D(
            Vec3(*(float(value) for value in data["gt_xyz"][0, can_index])),
            tuple(float(value) for value in data["gt_quat"][0, can_index]),
        )
        ee_pose = Pose3D(
            Vec3(*(float(value) for value in data["ee_xyz"][0])),
            tuple(float(value) for value in data["ee_quat"][0]),
        )
        gripper_opening = float(data["gripper"][0])

    base_angles[0] = yaw
    base_world = Pose3D(
        Vec3(x, y, base_z),
        _rotation_to_quaternion(Rotation.from_euler("ZYX", base_angles).as_matrix()),
    )
    can_base = compose(_inverse_pose(base_world), can_world)
    scene = SceneState(
        [
            ObjectState(
                "CAN",
                "can",
                can_base,
                Vec3(*config.verification.object_dimensions_m),
                1.0,
            )
        ]
    )
    robot = RobotState(
        joint_pos=joints,
        joint_vel={},
        base_pose=Pose2D(x, y, yaw),
        ee_pose=ee_pose,
        gripper_opening=gripper_opening,
        base_pose_world=base_world,
    )
    return WorldState(scene, robot), config, task


@NEEDS_EVIDENCE
@pytest.mark.parametrize("episode_index", (11, 17))
def test_preserved_endpoint_solves_with_full_registration_and_reviewed_residuals(
    episode_index: int,
) -> None:
    state, config, task = _preserved_first_state(episode_index)
    waypoints = corrected_scripted_waypoints(config, state, task)
    waypoint = waypoints[1]
    assert waypoint.label == "PREGRASP"

    solver = corrected_pose_solver()
    assert solver.execution_max_joint_step_rad == MAX_JOINT_STEP_RAD
    solution = solver.solve(waypoint.pose, state, seed=_first_draw_seed(config, episode_index))

    kinematics = corrected_kinematics()
    vector = _full_joint_vector(kinematics, state.robot.joint_pos, solution.joints)
    position_error, orientation_error = _reviewed_residual(kinematics, vector, waypoint.pose)
    assert position_error == pytest.approx(float(solution.position_error_m), abs=1e-9)
    assert orientation_error == pytest.approx(float(solution.orientation_error_rad), abs=1e-9)
    assert position_error <= waypoint.position_tolerance_m
    assert orientation_error <= waypoint.orientation_tolerance_rad

    max_delta = max(
        abs(float(solution.joints[name]) - float(state.robot.joint_pos[name]))
        for name in ARM_JOINTS
    )
    assert max_delta > MAX_JOINT_STEP_RAD
    for name in ARM_JOINTS:
        lower, upper = kinematics.bounds(name)
        assert lower <= float(solution.joints[name]) <= upper


class _StubSolver:
    execution_max_joint_step_rad = MAX_JOINT_STEP_RAD

    def __init__(self, endpoint: dict[str, float]) -> None:
        self.endpoint = endpoint

    def solve(self, pose: Pose3D, state: WorldState, *, seed: int = 0) -> Any:
        del pose, state, seed
        return SimpleNamespace(
            joints=dict(self.endpoint), position_error_m=0.0, orientation_error_rad=0.0
        )


class _SpyMonitor:
    def __init__(self, inner: SafetyMonitor) -> None:
        self.inner = inner
        self.limits = inner.limits
        self.calls: list[Any] = []

    def validate(self, action: Any, robot: Any) -> ApplyResult:
        self.calls.append(action)
        return self.inner.validate(action, robot)


class _RejectingMonitor(_SpyMonitor):
    def __init__(self, inner: SafetyMonitor, *, reject_at: int) -> None:
        super().__init__(inner)
        self.reject_at = reject_at

    def validate(self, action: Any, robot: Any) -> ApplyResult:
        self.calls.append(action)
        if len(self.calls) == self.reject_at:
            return ApplyResult(False, "synthetic safety veto")
        return self.inner.validate(action, robot)


def _simple_state() -> WorldState:
    robot = RobotState(
        joint_pos={"arm_1_joint": 0.0},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY),
        gripper_opening=0.045,
        base_pose_world=Pose3D(Vec3(0.0, 0.0, 0.0), IDENTITY),
    )
    return WorldState(SceneState([]), robot)


def _adapter(waypoints: tuple[ScriptedWaypoint, ...], monitor: Any) -> ScriptedChunkAdapter:
    return ScriptedChunkAdapter(
        waypoints,
        solver=_StubSolver({"arm_1_joint": 1.0}),
        monitor=monitor,
        horizon_steps=sum(item.hold_steps for item in waypoints),
        control_dt_s=0.032,
        jitter=JointJitter(0.0, 0.0, 1),
        max_joint_step_rad=MAX_JOINT_STEP_RAD,
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.01,
    )


def _limits() -> SafetyLimits:
    return SafetyLimits(
        joint_position={"arm_1_joint": (-2.0, 2.0)},
        joint_velocity={"arm_1_joint": 1.0},
    )


def test_scripted_chunk_emits_bounded_transitions_and_validates_every_action() -> None:
    waypoints = (
        ScriptedWaypoint("move", 8, pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY)),
        ScriptedWaypoint("grip", 1, gripper_opening=0.045),
        ScriptedWaypoint("settle", 1, joints={"arm_1_joint": 1.0}),
    )
    monitor = _SpyMonitor(SafetyMonitor(_limits()))
    task = TaskSpec("offline regression", "CAN", "target")
    chunk = _adapter(waypoints, monitor).draw()(task, _simple_state(), None, 123)

    assert len(chunk.actions) == 10
    joint_actions = [action for action in chunk.actions if hasattr(action, "q")]
    assert len(joint_actions) == 9
    current = 0.0
    for action in joint_actions:
        target = float(action.q["arm_1_joint"])
        assert abs(target - current) <= MAX_JOINT_STEP_RAD + 1e-12
        current = target
    assert float(joint_actions[0].q["arm_1_joint"]) == pytest.approx(MAX_JOINT_STEP_RAD)
    assert current == pytest.approx(1.0)
    assert len(monitor.calls) == len(chunk.actions)


def test_scripted_chunk_fails_closed_when_safety_rejects_one_action() -> None:
    waypoints = (
        ScriptedWaypoint("move", 2, pose=Pose3D(Vec3(0.3, 0.0, 0.8), IDENTITY)),
        ScriptedWaypoint("grip", 1, gripper_opening=0.045),
    )
    monitor = _RejectingMonitor(SafetyMonitor(_limits()), reject_at=2)
    task = TaskSpec("offline regression", "CAN", "target")
    with pytest.raises(ValueError, match="synthetic safety veto"):
        _adapter(waypoints, monitor).draw()(task, _simple_state(), None, 7)
    assert len(monitor.calls) == 2


def test_campaign_fail_stop_records_only_the_first_excepted_episode(tmp_path: Path) -> None:
    plan = (
        EvalEpisode(
            0,
            ScenarioSpec(name="validation-clear-i00-s0", metadata={"family": "clear"}),
            0,
        ),
        EvalEpisode(
            1,
            ScenarioSpec(name="validation-clear-i00-s1", metadata={"family": "clear"}),
            1,
        ),
    )
    calls: list[int] = []

    class ExplodingPolicy:
        def execute(self, backend: Any, seed: int) -> dict[str, Any]:
            del backend
            calls.append(seed)
            raise RuntimeError("first episode boom")

    report = run_evaluation(
        plan,
        FakeBackend(),
        EvalRunConfig(
            out_dir=tmp_path / "partial",
            pipeline="baseline",
            split="validation",
            stop_on_exception=True,
        ),
        policy_factory=lambda _scene: ExplodingPolicy(),
    )

    assert calls == [0]
    assert len(report.rows) == 1
    assert report.summary["n_episodes"] == 1
    assert report.summary["planned_episodes"] == 2
    assert report.summary["campaign_status"] == "incomplete_fail_stop"
    fail_stop = report.summary["fail_stop"]
    assert fail_stop["episode_index"] == 0
    assert fail_stop["failure_state"] == "EXCEPTION"
    assert "first episode boom" in fail_stop["failure_reason"]

    written = json.loads(report.summary_json.read_text(encoding="utf-8"))
    assert written["campaign_status"] == report.summary["campaign_status"]
    assert written["fail_stop"] == fail_stop
    with report.episodes_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["failure_state"] == "EXCEPTION"
    assert not (tmp_path / "partial" / "episode-01.meta.json").exists()
    plan_payload = json.loads(report.plan_json.read_text(encoding="utf-8"))
    assert len(plan_payload["episodes"]) == 2


def test_stage1_fail_stop_finalizes_sidecars_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _source, config, runtime = _setup(tmp_path, monkeypatch)
    out = tmp_path / "stage1-partial"
    summary = {
        "campaign_status": "incomplete_fail_stop",
        "fail_stop": {
            "reason": "episode 0 recorded EXCEPTION: RuntimeError: first episode boom",
            "episode_index": 0,
            "failure_state": "EXCEPTION",
        },
    }
    report = EvalReport(
        summary=summary,
        rows=(),
        timings=(),
        plan_json=out / "plan.json",
        episodes_csv=out / "episodes.csv",
        summary_json=out / "summary.json",
        timings_json=out / "timings.json",
        total_wall_s=0.0,
    )
    monkeypatch.setattr(
        "vlarobot.execution.worldmind_stage1_pipeline.run_evaluation",
        lambda *_args, **_kwargs: report,
    )

    with pytest.raises(CampaignFailStop, match="first episode boom"):
        run_stage1_evidence(
            _one_episode_plan(),
            FakeBackend(),
            config,
            out,
            fixture.bindings,
            arm_mode="scripted_proposer",
            runtime_values=runtime,
        )

    sidecars = out / "worldmind-sidecars"
    for name in ("provenance.json", "instrumentation.json", "errors.jsonl"):
        assert (sidecars / name).is_file()
    written = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert written["campaign_status"] == "incomplete_fail_stop"
    assert written["stage1_diagnostic"]["scored_evaluation_allowed"] is False
