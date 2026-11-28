"""V3 offline waypoint tooling: grouping, splits, ridge baselines and persistence."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from vlarobot.v3.model import CompactEncoder, benchmark_encoder
from vlarobot.v3.offline import (
    WaypointSample,
    _constant_result,
    _persistence_result,
    _ridge_result,
    base_scene_name,
    choose_validation_bases,
    fit_ridge,
    image_feature,
    loso_folds,
    mean_abs_error,
    predict_ridge,
    split_by_bases,
)


def _sample(
    episode: int,
    base: str,
    family: str,
    t: float,
    target: tuple[float, float, float],
    *,
    frame: int | None = None,
) -> WaypointSample:
    return WaypointSample(
        episode_index=episode,
        scene=f"{base}-s100",
        base_scene=base,
        family=family,
        t=t,
        frame_index=episode * 10 + int(t * 10) if frame is None else frame,
        state=np.asarray([t, 0.0, 0.0], dtype=float),
        action=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02], dtype=float),
        target_xyz=np.asarray(target, dtype=float),
        gripper_target=0.02,
        phase="TRANSPORT",
    )


def test_base_scene_name_strips_only_seed_suffix() -> None:
    assert base_scene_name("train-clear-i00-s100") == "train-clear-i00"
    assert base_scene_name("train-clear-i00") == "train-clear-i00"
    assert base_scene_name("train-clear-s12-s100") == "train-clear-s12"


def test_split_groups_all_seeds_of_a_base_scene_together() -> None:
    samples = [
        _sample(0, "a", "clear", 0.0, (0.0, 0.0, 0.0)),
        _sample(1, "a", "clear", 1.0, (0.0, 0.0, 0.1)),
        _sample(2, "b", "clear", 0.0, (0.0, 0.0, 0.2)),
        _sample(3, "c", "cluttered", 0.0, (0.0, 0.0, 0.3)),
    ]
    train, held_out = split_by_bases(samples, ("a", "c"))
    assert {sample.base_scene for sample in held_out} == {"a", "c"}
    assert {sample.base_scene for sample in train} == {"b"}
    assert len(train) + len(held_out) == len(samples)


def test_choose_validation_bases_is_one_per_family_and_deterministic() -> None:
    samples = [
        _sample(0, "clear-b", "clear", 0.0, (0.0, 0.0, 0.0)),
        _sample(1, "clear-a", "clear", 0.0, (0.0, 0.0, 0.0)),
        _sample(2, "clut-z", "cluttered", 0.0, (0.0, 0.0, 0.0)),
    ]
    assert choose_validation_bases(samples) == ("clear-a", "clut-z")


def test_loso_folds_hold_out_every_seed_of_one_base_scene() -> None:
    samples = [
        _sample(0, "a", "clear", 0.0, (0.0, 0.0, 0.0)),
        _sample(1, "a", "clear", 1.0, (0.0, 0.0, 0.1)),
        _sample(2, "b", "clear", 0.0, (0.0, 0.0, 0.2)),
    ]
    folds = loso_folds(samples)
    assert [name for name, _, _ in folds] == ["a", "b"]
    by_name = {name: (train, held_out) for name, train, held_out in folds}
    assert {sample.base_scene for sample in by_name["a"][1]} == {"a"}
    assert len(by_name["a"][1]) == 2
    assert {sample.base_scene for sample in by_name["a"][0]} == {"b"}


def test_ridge_recovers_a_linear_waypoint_map() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    weights = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]])
    targets = features @ weights + np.asarray([0.1, -0.2, 0.3])
    model = fit_ridge(features, targets, alpha=1e-6)
    prediction = predict_ridge(model, features)
    np.testing.assert_allclose(prediction, targets, atol=1e-6)


def test_ridge_rejects_misaligned_shapes() -> None:
    with pytest.raises(ValueError):
        fit_ridge(np.zeros((4, 2)), np.zeros((3, 3)))


def test_ridge_result_reports_train_and_validation_errors() -> None:
    features = np.arange(12, dtype=float).reshape(6, 2)
    targets = np.column_stack([features[:, 0], features[:, 1], features.sum(axis=1)])
    train = np.asarray([0, 1, 2, 3])
    validation = np.asarray([4, 5])
    result = _ridge_result("state", features, targets, train, validation)
    assert result["name"] == "state"
    assert result["train"]["overall_m"] <= result["validation"]["overall_m"] + 1.0


def test_constant_and_persistence_baselines_use_only_train_statistics() -> None:
    samples = [
        _sample(0, "a", "clear", 0.0, (0.0, 0.0, 0.0)),
        _sample(0, "a", "clear", 1.0, (0.1, 0.0, 0.0)),
        _sample(1, "b", "clear", 0.0, (5.0, 5.0, 5.0)),
        _sample(1, "b", "clear", 1.0, (5.0, 5.0, 5.0)),
    ]
    targets = np.stack([sample.target_xyz for sample in samples])
    train = np.asarray([0, 1])
    validation = np.asarray([2, 3])
    constant = _constant_result(train, validation, targets)
    assert constant["train"]["overall_m"] == pytest.approx(0.05 / 3.0, abs=1e-6)
    assert constant["validation"]["overall_m"] == pytest.approx((4.95 + 5.0 + 5.0) / 3.0, abs=1e-6)

    persistence = _persistence_result(samples, targets, train, validation)
    # First samples fall back to the train mean; later samples reuse the previous waypoint.
    assert persistence["train"]["overall_m"] == pytest.approx((0.05 + 0.1) / 3.0 / 2.0, abs=1e-6)
    assert persistence["validation"]["overall_m"] == pytest.approx(
        (4.95 + 5.0 + 5.0) / 3.0 / 2.0, abs=1e-6
    )


def test_mean_abs_error_is_per_axis() -> None:
    result = mean_abs_error(np.asarray([[1.0, 2.0, 3.0]]), np.asarray([[0.0, 2.0, 5.0]]))
    assert result == {"overall_m": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 2.0}


def test_image_feature_is_a_normalised_gray_square() -> None:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(255, 0, 0)).save(buffer, format="JPEG")
    feature = image_feature(buffer.getvalue(), size=8)
    assert feature.shape == (64,)
    assert float(feature.min()) >= 0.0
    assert float(feature.max()) <= 1.0


def test_compact_encoder_is_deterministic_and_shape_checked() -> None:
    first = CompactEncoder(input_dim=16, hidden_dim=8, output_dim=4, seed=3)
    second = CompactEncoder(input_dim=16, hidden_dim=8, output_dim=4, seed=3)
    batch = np.ones((5, 16), dtype=np.float32)
    np.testing.assert_array_equal(first(batch), second(batch))
    assert first(batch).shape == (5, 4)
    with pytest.raises(ValueError):
        first(np.ones((5, 15), dtype=np.float32))


def test_encoder_benchmark_reports_positive_throughput() -> None:
    encoder = CompactEncoder(input_dim=16, hidden_dim=8, output_dim=4, seed=0)
    rows = benchmark_encoder(encoder, batch_sizes=(1, 4), iterations=5, seed=1)
    assert [row.batch_size for row in rows] == [1, 4]
    for row in rows:
        assert row.mean_ms > 0.0
        assert row.p95_ms >= row.mean_ms
        assert row.images_per_s > 0.0
