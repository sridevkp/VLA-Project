"""Metrics primitives, collision proxy vs contacts, FK clearance, paired analysis (ADR-0029)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_eval_recorder import _CAN_GT, _ScheduleGtBackend

from vlarobot.core.types import GroundTruth, Pose3D, ScenarioSpec, Vec3
from vlarobot.eval.metrics import (
    PAIRING_UNIT,
    EpisodeMetrics,
    EpisodeRecord,
    brier_score,
    collision_proxy,
    compute_episode_metrics,
    load_episode,
    mae,
    mcnemar_exact,
    mean_std,
    metrics_to_row,
    min_fk_clearance_m,
    paired_comparison,
    rmse,
    scene_id,
    summarize_success,
    tilt_angle_deg,
    wilson_interval,
    write_csv,
    write_json,
)
from vlarobot.eval.recorder import EpisodeRecorder, RecordingBackend

_HOME = (-0.799879, -4.38998, 0.810949)
_MOVED_6MM = (_HOME[0] + 0.006, _HOME[1], _HOME[2])
_MOVED_1MM = (_HOME[0] + 0.001, _HOME[1], _HOME[2])


def _gt(xyz: tuple[float, float, float], quat: tuple[float, float, float, float]) -> GroundTruth:
    return GroundTruth(object_poses={"CAN": Pose3D(Vec3(*xyz), quat)})


def _record(
    gt_xyz: np.ndarray,
    gt_quat: np.ndarray,
    *,
    names: tuple[str, ...] = ("CAN",),
    meta: dict | None = None,
) -> EpisodeRecord:
    steps = gt_xyz.shape[0]
    return EpisodeRecord(
        meta=meta or {},
        t=np.arange(steps, dtype=np.float64),
        joint_names=(),
        joint_pos=np.zeros((steps, 0)),
        joint_vel=np.zeros((steps, 0)),
        gripper=np.zeros(steps),
        ee_xyz=np.zeros((steps, 3)),
        base_pose=np.zeros((steps, 3)),
        gt_names=names,
        gt_xyz=gt_xyz,
        gt_quat=gt_quat,
        contacts=(),
    )


_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def test_wilson_interval_known_values_and_validation() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)

    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)  # rule-of-three upper bound ~0.3

    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high <= 1.0

    with pytest.raises(ValueError, match="total"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(11, 10)
    with pytest.raises(ValueError, match="z"):
        wilson_interval(1, 2, z=0.0)


def test_mean_std_and_continuous_error_metrics() -> None:
    assert mean_std([1.0, 2.0, 3.0]) == (2.0, 1.0)
    mean, std = mean_std([5.0])
    assert mean == 5.0
    assert math.isnan(std)  # a single sample has no defined spread
    with pytest.raises(ValueError, match="empty"):
        mean_std([])

    assert rmse([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert mae([3.0, 4.0]) == pytest.approx(3.5)
    with pytest.raises(ValueError, match="empty"):
        rmse([])
    with pytest.raises(ValueError, match="empty"):
        mae([])


def test_brier_is_for_probabilities_and_stays_separate_from_rmse() -> None:
    probabilities = [0.8, 0.4, 0.55]
    outcomes = [1, 0, 1]
    score = brier_score(probabilities, outcomes)
    assert score == pytest.approx(0.13416666666666666)
    # Brier is the mean squared probability error: related to RMSE, reported separately.
    errors = [prob - outcome for prob, outcome in zip(probabilities, outcomes, strict=True)]
    assert score == pytest.approx(rmse(errors) ** 2)

    with pytest.raises(ValueError, match="length mismatch"):
        brier_score([0.5], [1, 0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.5], [1])
    with pytest.raises(ValueError, match="binary"):
        brier_score([0.5], [2])


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(8, 1) == pytest.approx(0.0390625)
    assert mcnemar_exact(0, 0) == 1.0  # no discordant pairs -> no evidence
    assert mcnemar_exact(5, 5) == 1.0  # perfectly symmetric -> capped at 1
    assert mcnemar_exact(2, 0) == pytest.approx(0.5)
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact(-1, 0)


def test_tilt_angle_deg_known_values() -> None:
    half = math.radians(22.5)  # quaternion half-angle -> 45 deg rotation about z
    rotated_z = (0.0, 0.0, math.sin(half), math.cos(half))
    assert tilt_angle_deg(_IDENTITY, _IDENTITY) == pytest.approx(0.0)
    assert tilt_angle_deg(_IDENTITY, rotated_z) == pytest.approx(45.0)
    assert tilt_angle_deg(rotated_z, _IDENTITY) == pytest.approx(45.0)


def test_collision_proxy_flags_displacement_and_tilt_separately() -> None:
    # 6 mm displacement over three steps -> displacement hit, no tilt.
    xyz = np.array([[list(_HOME)], [list(_HOME)], [list(_MOVED_6MM)]])
    quats = np.tile(np.array([list(_IDENTITY)]), (3, 1, 1))
    proxy = collision_proxy(_record(xyz, quats), max_displacement_m=0.005, max_tilt_deg=5.0)
    assert proxy.hit is True
    assert proxy.objects[0].displacement_hit is True
    assert proxy.objects[0].tilt_hit is False
    assert proxy.max_displacement_m == pytest.approx(0.006)
    assert proxy.objects[0].final_displacement_m == pytest.approx(0.006)
    assert proxy.max_tilt_deg == pytest.approx(0.0)

    at_threshold = np.array([[list(_HOME)], [list((_HOME[0] + 0.005, _HOME[1], _HOME[2]))]])
    # Strict ">": a threshold equal to the computed max (same float) must NOT hit.
    # (Literal 0.005 can't be reproduced exactly after float addition, so compare
    # against the value the function itself produced.)
    proxy = collision_proxy(
        _record(at_threshold, quats[:1]), max_displacement_m=0.005, max_tilt_deg=5.0
    )
    assert proxy.hit is True  # 0.0050000000000000044 > 0.005
    exact = collision_proxy(
        _record(at_threshold, quats[:1]),
        max_displacement_m=proxy.max_displacement_m,
        max_tilt_deg=5.0,
    )
    assert exact.hit is False
    assert exact.objects[0].displacement_hit is False

    # 6 deg tilt with zero displacement -> tilt hit only.
    half = math.radians(3.0)
    tilted = (0.0, 0.0, math.sin(half), math.cos(half))
    tilted_quats = np.array([[list(_IDENTITY)], [list(_IDENTITY)], [list(tilted)]])
    flat_xyz = np.tile(np.array([list(_HOME)]), (3, 1, 1))
    proxy = collision_proxy(
        _record(flat_xyz, tilted_quats), max_displacement_m=0.005, max_tilt_deg=5.0
    )
    assert proxy.hit is True
    assert proxy.objects[0].displacement_hit is False
    assert proxy.objects[0].tilt_hit is True
    assert proxy.max_tilt_deg == pytest.approx(6.0, abs=1e-9)

    # NaN-padded leading rows are skipped when picking the reset pose.
    nan_row = [[math.nan, math.nan, math.nan]]
    xyz = np.array([nan_row, [list(_HOME)], [list(_MOVED_6MM)]])
    proxy = collision_proxy(_record(xyz, quats), max_displacement_m=0.005, max_tilt_deg=5.0)
    assert proxy.max_displacement_m == pytest.approx(0.006)

    # No tracked objects at all -> proxy cannot fire and reports nothing.
    empty_xyz = np.zeros((3, 0, 3))
    empty_quats = np.zeros((3, 0, 4))
    proxy = collision_proxy(
        _record(empty_xyz, empty_quats, names=()), max_displacement_m=0.005, max_tilt_deg=5.0
    )
    assert proxy.hit is False
    assert proxy.max_displacement_m is None
    assert proxy.objects == ()

    with pytest.raises(ValueError, match="positive"):
        collision_proxy(_record(xyz, quats), max_displacement_m=0.0, max_tilt_deg=5.0)


def test_min_fk_clearance_is_a_standalone_measurement() -> None:
    points = [("tool_tip", (0.0, 0.0, 0.0), 0.02)]
    objects = [("CAN", (0.1, 0.0, 0.0), 0.03)]
    assert min_fk_clearance_m(points, objects) == pytest.approx(0.05)

    # 0.1 m apart, radii 0.05 + 0.03: gap 0.02. Object 0.01 m away -> 0.01-0.05-0.03 = -0.07.
    penetrating = [("elbow", (0.0, 0.0, 0.0), 0.05)]
    near = [("CAN", (0.01, 0.0, 0.0), 0.03)]
    assert min_fk_clearance_m(penetrating, near) == pytest.approx(-0.07)

    assert min_fk_clearance_m([], objects) is None
    assert min_fk_clearance_m(points, []) is None
    with pytest.raises(ValueError, match="xyz triple"):
        min_fk_clearance_m([("bad", (0.0, 0.0), 0.0)], objects)


def test_paired_comparison_preserves_episode_pairs_and_clusters_by_scene() -> None:
    a = {
        ("sA", 0): True,
        ("sA", 1): True,
        ("sA", 2): True,
        ("sB", 0): True,
        ("sB", 1): False,
        ("sB", 2): False,
        ("sC", 0): False,
        ("sC", 1): False,
        ("sC", 2): False,
    }
    b = {
        ("sA", 0): True,
        ("sA", 1): True,
        ("sA", 2): False,  # A-only success
        ("sB", 0): False,  # A-only success
        ("sB", 1): False,
        ("sB", 2): False,
        ("sC", 0): True,  # B-only success
        ("sC", 1): False,
        ("sC", 2): False,
    }
    result = paired_comparison(a, b)

    assert result.pairing_unit == PAIRING_UNIT == "(scene, seed)"
    # Episode-level pairs are preserved (9 pairs, not collapsed away)...
    assert result.episode_pairs == 9
    assert result.episode_b == 2
    assert result.episode_c == 1
    assert result.episode_mcnemar_p == pytest.approx(mcnemar_exact(2, 1))

    # ...but the independent units are the 3 scenes, never the 9 episode pairs.
    assert result.independent_units == 3
    assert len(result.rows) == 3
    assert result.scene_b == 2  # sA and sB favour A
    assert result.scene_c == 1  # sC favours B
    assert result.scene_sign_p == pytest.approx(mcnemar_exact(2, 1))

    s_a = next(row for row in result.rows if row.scene == "sA")
    assert s_a.seeds == (0, 1, 2)
    assert s_a.n == 3
    assert s_a.a_successes == 3
    assert s_a.b_successes == 2
    assert s_a.rate_a == 1.0
    assert s_a.rate_b == pytest.approx(2.0 / 3.0)
    assert s_a.std_a == pytest.approx(0.0)
    assert s_a.std_b == pytest.approx(math.sqrt(1.0 / 3.0))
    assert (s_a.b_discordant, s_a.c_discordant) == (1, 0)

    with pytest.raises(ValueError, match="identical"):
        paired_comparison(a, {**b, ("sD", 0): True})
    with pytest.raises(ValueError, match="at least one"):
        paired_comparison({}, {})


def test_summarize_success_uses_wilson_not_a_bare_rate() -> None:
    summary = summarize_success([True, True, False, True])
    assert summary.successes == 3
    assert summary.total == 4
    assert summary.rate == 0.75
    assert 0.0 <= summary.ci_low < summary.rate < summary.ci_high <= 1.0
    with pytest.raises(ValueError, match="empty"):
        summarize_success([])


def test_scene_id_strips_the_seed_suffix() -> None:
    meta = {"scenario": {"name": "train-clear-i00-s7"}, "reset_seed": 7}
    assert scene_id(meta) == "train-clear-i00"
    assert scene_id({"scenario": {"name": "plain"}, "seed": 0}) == "plain"
    assert scene_id({}) == ""


class _MovingContactBackend(_ScheduleGtBackend):
    """CAN moves 6 mm (or 1 mm) after reset AND reports a TABLE contact each step."""

    def __init__(self, schedule: list[GroundTruth]) -> None:
        super().__init__(schedule)
        self.contact_label = "TABLE"

    def step(self, n_steps: int = 1):
        observation = super().step(n_steps)
        observation.robot.diagnostics = {"contact_nodes": [{"node_name": self.contact_label}]}
        return observation


def _recorded_episode(
    tmp_path: Path,
    name: str,
    schedule: list[GroundTruth],
    outcome: dict,
) -> Path:
    recorder = EpisodeRecorder(tmp_path / name, frame_hz=0.0)
    wrapped = RecordingBackend(_MovingContactBackend(schedule), recorder)
    recorder.open_episode(0, seed=7, pipeline="scripted", provenance="measured")
    wrapped.reset(ScenarioSpec(name="train-clear-i00-s7"), seed=7)
    wrapped.step(1)
    wrapped.step(1)
    recorder.record_outcome(outcome)
    paths = recorder.close_episode()
    return paths.meta


def test_compute_episode_metrics_keeps_proxy_contacts_and_fk_separate(tmp_path: Path) -> None:
    outcome = {
        "success": False,
        "failure_state": "PLACE",
        "failure_reason": "distance",
        "final_target_distance": 0.12,
        "can_height_error_m": 0.01,
        "can_tilt_deg": 2.5,
        "rejections": 0,
        "fk_min_clearance_m": 0.05,
    }
    meta_path = _recorded_episode(
        tmp_path, "hit", [_gt(_HOME, _IDENTITY), _gt(_MOVED_6MM, _IDENTITY)], outcome
    )
    metrics = compute_episode_metrics(load_episode(meta_path))

    assert metrics.scene == "train-clear-i00"  # seed suffix stripped for pairing
    assert metrics.seed == 7
    assert metrics.arm == "scripted"
    assert metrics.success is False
    assert metrics.failure_state == "PLACE"
    assert metrics.failure_reason == "distance"
    assert metrics.evaluation_failure_category == "PLACE"
    assert metrics.final_target_distance_m == pytest.approx(0.12)
    assert metrics.can_height_error_m == pytest.approx(0.01)
    assert metrics.can_tilt_deg == pytest.approx(2.5)
    assert metrics.rejections == 0

    # Channel 1: displacement/tilt PROXY (labelled, thresholds from the frozen config).
    assert metrics.proxy_hit is True
    assert metrics.proxy_max_displacement_m == pytest.approx(0.006)
    assert metrics.proxy_max_tilt_deg == pytest.approx(0.0)

    # Channel 2: ACTUAL contact detection — present independently of the proxy.
    assert metrics.contact_steps == 2
    assert metrics.contact_nodes == ("TABLE",)

    # Channel 3: FK clearance, its own measurement supplied by the caller.
    assert metrics.fk_min_clearance_m == pytest.approx(0.05)

    # Contacts do not imply a proxy hit and vice versa: 1 mm motion + contacts.
    quiet = _recorded_episode(
        tmp_path, "quiet", [_gt(_HOME, _IDENTITY), _gt(_MOVED_1MM, _IDENTITY)], outcome
    )
    quiet_metrics = compute_episode_metrics(load_episode(quiet))
    assert quiet_metrics.proxy_hit is False
    assert quiet_metrics.contact_steps == 2  # contacts still reported
    assert quiet_metrics.contact_nodes == ("TABLE",)


def test_compute_episode_metrics_requires_a_recorded_outcome(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path / "no-outcome", frame_hz=0.0)
    wrapped = RecordingBackend(_ScheduleGtBackend([_CAN_GT]), recorder)
    recorder.open_episode(0, seed=0, pipeline="scripted", provenance="measured")
    wrapped.reset(ScenarioSpec(), seed=0)
    paths = recorder.close_episode()

    with pytest.raises(ValueError, match="outcome"):
        compute_episode_metrics(load_episode(paths.meta))


def test_metrics_to_row_and_writers_roundtrip(tmp_path: Path) -> None:
    metrics = EpisodeMetrics(
        scene="train-clear-i00",
        seed=7,
        arm="scripted",
        success=True,
        failure_state=None,
        failure_reason=None,
        evaluation_failure_category="success",
        final_target_distance_m=0.01,
        can_height_error_m=None,
        can_tilt_deg=None,
        rejections=0,
        proxy_hit=False,
        proxy_max_displacement_m=0.001,
        proxy_max_tilt_deg=0.5,
        contact_steps=1,
        contact_nodes=("TABLE", "CAN"),
        fk_min_clearance_m=0.04,
    )
    row = metrics_to_row(metrics)
    assert row["contact_nodes"] == "TABLE;CAN"

    csv_path = tmp_path / "out" / "episodes.csv"
    write_csv([row, {**row, "scene": "train-cluttered-i01"}], csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("scene,seed,arm,success")
    assert len(lines) == 3
    assert "TABLE;CAN" in lines[1]

    json_path = tmp_path / "out" / "summary.json"
    write_json({"metrics": metrics, "array": np.array([1.0, 2.0])}, json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["scene"] == "train-clear-i00"
    assert payload["metrics"]["contact_nodes"] == ["TABLE", "CAN"]
    assert payload["array"] == [1.0, 2.0]

    with pytest.raises(ValueError, match="no rows"):
        write_csv([], tmp_path / "empty.csv")
