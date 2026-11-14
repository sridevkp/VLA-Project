"""EpisodeRecorder streams, frame capture, determinism, and backend equivalence (ADR-0031)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_scripted_can_policy import RateLimitedCanBackend

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.core.types import (
    GripperCmd,
    GroundTruth,
    JointTarget,
    Pose3D,
    ScenarioSpec,
    Vec3,
)
from vlarobot.eval.recorder import EpisodeRecorder, RecordingBackend
from vlarobot.execution.scripted_can import ScriptedCanPolicy

_CAN_GT = GroundTruth(
    object_poses={"CAN": Pose3D(Vec3(-0.799879, -4.38998, 0.810949), (0.0, 0.0, 0.0, 1.0))}
)


class _ScheduleGtBackend(FakeBackend):
    """FakeBackend whose ground truth follows a fixed per-read schedule (last entry repeats)."""

    def __init__(self, schedule: list[GroundTruth]) -> None:
        super().__init__()
        self._gt_schedule = list(schedule)
        self._gt_reads = 0

    def get_ground_truth(self) -> GroundTruth:
        index = min(self._gt_reads, len(self._gt_schedule) - 1)
        self._gt_reads += 1
        return self._gt_schedule[index]


def _drive(
    recorder: EpisodeRecorder,
    backend: FakeBackend,
    *,
    scenario: ScenarioSpec | None = None,
) -> None:
    """One deterministic episode: reset, steps, one accepted + one rejected action."""
    wrapped = RecordingBackend(backend, recorder)
    recorder.open_episode(0, seed=7, pipeline="scripted", provenance="measured")
    wrapped.reset(scenario or ScenarioSpec(name="train-clear-i00-s7"), seed=7)
    wrapped.step(1)
    wrapped.step(1)
    wrapped.apply(GripperCmd(0.04))
    wrapped.apply(JointTarget({"arm_1_joint": 99.0}))  # beyond FakeBackend limits: rejected
    wrapped.step(1)
    recorder.record_outcome({"success": True, "final_state": "home"})
    recorder.close_episode()


def test_recorder_roundtrip_writes_all_streams(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "run", frame_hz=0.0)
    _drive(recorder, FakeBackend())
    paths = recorder.paths
    assert paths is not None

    for path in (paths.meta, paths.steps, paths.actions, paths.predicted, paths.contacts):
        assert path.is_file(), path
    assert paths.contacts.read_text(encoding="utf-8") == ""  # FakeBackend reports no contacts
    assert not paths.frames_dir.exists()  # frame_hz=0 disables frame capture

    with np.load(paths.steps) as data:
        assert set(data.files) == {
            "t",
            "joint_names",
            "joint_pos",
            "joint_vel",
            "gripper",
            "ee_xyz",
            "base_pose",
            "gt_names",
            "gt_xyz",
            "gt_quat",
        }
        np.testing.assert_allclose(data["t"], [0.0, 0.032, 0.064, 0.096])
        assert list(data["joint_names"]) == [f"arm_{index}_joint" for index in range(1, 8)]
        assert data["joint_pos"].shape == (4, 7)
        assert data["gripper"].shape == (4,)
        assert data["ee_xyz"].shape == (4, 3)
        assert data["base_pose"].shape == (4, 3)
        # FakeBackend reports no object poses: the arrays exist with zero columns.
        assert list(data["gt_names"]) == []
        assert data["gt_xyz"].shape == (4, 0, 3)
        assert data["gt_quat"].shape == (4, 0, 4)

    actions = [json.loads(line) for line in paths.actions.read_text(encoding="utf-8").splitlines()]
    assert len(actions) == 2
    assert actions[0]["accepted"] is True
    assert actions[0]["action"]["opening"] == pytest.approx(0.04)
    assert actions[1]["accepted"] is False
    assert actions[1]["reason"]
    # Both actions were applied after the last recorded step (t=0.064).
    assert [row["t"] for row in actions] == [0.064, 0.064]
    assert paths.predicted.read_text(encoding="utf-8") == ""  # empty until M5

    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert meta["index"] == 0
    assert meta["seed"] == 7
    assert meta["reset_seed"] == 7
    assert meta["pipeline"] == "scripted"
    assert meta["state_source"] == "measured"
    assert meta["scenario"]["name"] == "train-clear-i00-s7"
    assert meta["step_count"] == 4
    assert meta["frame_count"] == 0
    assert meta["frame_hz"] == 0.0
    assert meta["contact_steps"] == 0
    assert meta["outcome"] == {"success": True, "final_state": "home"}


def test_recorder_captures_ground_truth_rows_and_pads_missing_with_nan(tmp_path: Path) -> None:
    empty = GroundTruth()
    # Reset and first step report CAN; later steps report nothing.
    backend = _ScheduleGtBackend([_CAN_GT, _CAN_GT, empty])
    recorder = EpisodeRecorder(tmp_path / "run", frame_hz=0.0)
    wrapped = RecordingBackend(backend, recorder)
    recorder.open_episode(0, seed=0, pipeline="scripted", provenance="measured")
    wrapped.reset(ScenarioSpec(), seed=0)
    wrapped.step(1)
    wrapped.step(1)
    wrapped.step(1)
    recorder.close_episode()

    paths = recorder.paths
    assert paths is not None
    with np.load(paths.steps) as data:
        assert list(data["gt_names"]) == ["CAN"]
        gt_xyz = data["gt_xyz"]
        assert gt_xyz.shape == (4, 1, 3)
        np.testing.assert_allclose(gt_xyz[0, 0], [-0.799879, -4.38998, 0.810949])
        np.testing.assert_allclose(gt_xyz[1, 0], [-0.799879, -4.38998, 0.810949])
        assert np.isnan(gt_xyz[2, 0]).all()
        assert np.isnan(gt_xyz[3, 0]).all()
        # Orientation rides along (needed for the tilt half of the collision proxy).
        gt_quat = data["gt_quat"]
        assert gt_quat.shape == (4, 1, 4)
        np.testing.assert_allclose(gt_quat[0, 0], [0.0, 0.0, 0.0, 1.0])
        assert np.isnan(gt_quat[2, 0]).all()


def test_recorder_captures_frames_at_the_configured_rate(tmp_path: Path) -> None:
    # FakeBackend step_duration is 0.032 s; frame_hz=10 -> frames at t=0 and t=0.128.
    recorder = EpisodeRecorder(tmp_path / "run", frame_hz=10.0)
    wrapped = RecordingBackend(FakeBackend(), recorder)
    recorder.open_episode(0, seed=7, pipeline="scripted", provenance="measured")
    wrapped.reset(ScenarioSpec(), seed=7)
    for _ in range(5):  # t = 0.032 .. 0.160
        wrapped.step(1)
    recorder.close_episode()
    paths = recorder.paths
    assert paths is not None

    frames = sorted(paths.frames_dir.glob("*.jpg"))
    assert [frame.name for frame in frames] == ["000000.jpg", "000001.jpg"]
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert meta["frame_count"] == 2
    assert meta["frame_hz"] == 10.0
    assert meta["step_count"] == 6  # reset observation + 5 steps

    # Pillow round-trip: the captured frame is a valid 48x64 RGB JPEG.
    from PIL import Image

    with Image.open(frames[0]) as image:
        assert image.size == (64, 48)
        assert image.mode == "RGB"


def test_recorder_record_frames_flag_suppresses_the_frames_dir(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "run", frame_hz=10.0, record_frames=False)
    _drive(recorder, FakeBackend())
    paths = recorder.paths
    assert paths is not None
    assert not paths.frames_dir.exists()
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert meta["frame_count"] == 0
    assert meta["frame_hz"] == 0.0


def test_recorder_is_deterministic_across_identical_runs(tmp_path: Path) -> None:
    def run(directory: Path) -> None:
        _drive(
            EpisodeRecorder(directory, frame_hz=40.0),
            FakeBackend(),
            scenario=ScenarioSpec(
                name="cluttered-fixture",
                metadata={"family": "cluttered", "split": "train"},
            ),
        )

    first = tmp_path / "first"
    second = tmp_path / "second"
    run(first)
    run(second)

    first_files = {path.name: path for path in sorted(first.rglob("*")) if path.is_file()}
    second_files = {path.name: path for path in sorted(second.rglob("*")) if path.is_file()}
    assert first_files.keys() == second_files.keys()

    for name, first_path in first_files.items():
        second_path = second_files[name]
        if name.endswith(".steps.npz"):
            # npz is a zip container with wall-clock entry timestamps: compare arrays.
            with np.load(first_path) as left, np.load(second_path) as right:
                assert left.files == right.files
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key], err_msg=name)
        else:
            assert first_path.read_bytes() == second_path.read_bytes(), name


def test_recording_backend_is_policy_transparent(tmp_path: Path) -> None:
    """The decorator must not change controller behaviour (AGENTS rule 10)."""
    plain_kin = Kinematics()
    plain_result = ScriptedCanPolicy(plain_kin, stop_after="PREGRASP").execute(
        RateLimitedCanBackend(plain_kin), seed=0
    )

    recorder = EpisodeRecorder(tmp_path / "wrapped", frame_hz=4.0)
    wrapped_kin = Kinematics()
    wrapped_backend = RecordingBackend(RateLimitedCanBackend(wrapped_kin), recorder)
    recorder.open_episode(0, seed=0, pipeline="scripted", provenance="measured")
    wrapped_result = ScriptedCanPolicy(wrapped_kin, stop_after="PREGRASP").execute(
        wrapped_backend, seed=0
    )
    recorder.record_outcome(
        {"success": wrapped_result.success, "final_state": wrapped_result.final_state}
    )
    recorder.close_episode()

    assert wrapped_result == plain_result

    paths = recorder.paths
    assert paths is not None
    with np.load(paths.steps) as data:
        assert data["t"].size > 1  # the recording captured the policy's steps
    actions = paths.actions.read_text(encoding="utf-8").splitlines()
    assert actions, "the recording captured no applied actions"
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert meta["outcome"] == {
        "success": plain_result.success,
        "final_state": plain_result.final_state,
    }


def test_recorder_captures_actual_contact_nodes_separately(tmp_path: Path) -> None:
    class _ContactBackend(FakeBackend):
        def step(self, n_steps: int = 1):
            observation = super().step(n_steps)
            observation.robot.diagnostics = {"contact_nodes": [{"node_name": "TABLE"}]}
            return observation

    recorder = EpisodeRecorder(tmp_path / "run", frame_hz=0.0)
    wrapped = RecordingBackend(_ContactBackend(), recorder)
    recorder.open_episode(0, seed=0, pipeline="scripted", provenance="measured")
    wrapped.reset(ScenarioSpec(), seed=0)  # no diagnostics on the reset observation
    wrapped.step(1)
    wrapped.step(1)
    recorder.close_episode()

    paths = recorder.paths
    assert paths is not None
    rows = [json.loads(line) for line in paths.contacts.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"t": 0.032, "nodes": ["TABLE"]},
        {"t": 0.064, "nodes": ["TABLE"]},
    ]
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert meta["contact_steps"] == 2
    # The proxy inputs (gt arrays) stay empty: the two channels never merge.
    with np.load(paths.steps) as data:
        assert data["gt_xyz"].shape == (3, 0, 3)


def test_recorder_fails_loud_when_misused(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "run", frame_hz=0.0)
    observation = FakeBackend().reset(ScenarioSpec(), seed=0)

    with pytest.raises(RuntimeError, match="no episode is open"):
        recorder.record_step(observation, None)
    with pytest.raises(RuntimeError, match="no episode is open"):
        recorder.close_episode()

    recorder.open_episode(0, seed=0, pipeline="scripted", provenance="measured")
    with pytest.raises(RuntimeError, match="already open"):
        recorder.open_episode(1, seed=0, pipeline="scripted", provenance="measured")

    recorder.on_reset(ScenarioSpec(), seed=0)
    with pytest.raises(RuntimeError, match="reset seen twice"):
        recorder.on_reset(ScenarioSpec(), seed=0)

    recorder.record_step(observation, None)
    changed = FakeBackend().step(1)
    changed.robot.joint_pos = {"other_joint": 0.0}
    with pytest.raises(ValueError, match="joint key set changed"):
        recorder.record_step(changed, None)
    recorder.close_episode()

    with pytest.raises(ValueError, match="frame_hz"):
        EpisodeRecorder(tmp_path / "bad", frame_hz=-1.0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        EpisodeRecorder(tmp_path / "bad", jpeg_quality=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        EpisodeRecorder(tmp_path / "bad", jpeg_quality=101)
