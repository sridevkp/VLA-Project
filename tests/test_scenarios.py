"""Scene sets, CAN pose perturbation, and scenario threading (ADR-0025, M1-07(b)/M2)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import PLACE_WORLD
from vlarobot.backends.webots.frames import (
    _axis_angle_to_matrix,
    compose_can_reset_pose,
)
from vlarobot.cli import app
from vlarobot.core.types import ScenarioSpec
from vlarobot.core.wire import decode_message, encode_message, make_message
from vlarobot.eval.scenarios import (
    GRID_OFFSETS_M,
    YAW_SCENE_RAD,
    generate_scenes,
    get_scenarios,
    load_scenario_library,
    mini_grid10,
)
from vlarobot.eval.settings import load_eval_settings
from vlarobot.execution.episode_runner import RunnerSettings, scenario_for_index
from vlarobot.execution.scripted_can import ScriptedCanPolicy


def _settings(scenes: tuple[ScenarioSpec, ...] = ()) -> RunnerSettings:
    return RunnerSettings(
        webots="webots",
        world=Path("worlds") / "factory.wbt",
        endpoint="tcp://127.0.0.1:5555",
        trace_dir=Path("runs") / "scenario-test",
        scenario=scenes,
    )


def test_mini_grid10_is_nine_grid_cells_plus_one_yaw_scene() -> None:
    scenes = mini_grid10()
    assert len(scenes) == 10
    assert len({scene.name for scene in scenes}) == 10

    grid = [scene for scene in scenes if scene.can_xy_yaw_offset is not None]
    yaw_zero = [
        scene
        for scene in grid
        if scene.can_xy_yaw_offset is not None and scene.can_xy_yaw_offset[2] == 0.0
    ]
    cells = {
        (scene.can_xy_yaw_offset[0], scene.can_xy_yaw_offset[1])
        for scene in yaw_zero
        if scene.can_xy_yaw_offset is not None
    }
    expected = {(dx, dy) for dx in GRID_OFFSETS_M for dy in GRID_OFFSETS_M}
    assert cells == expected
    assert len(yaw_zero) == 9
    assert len(grid) == 10

    yaw_scene = scenes[-1]
    assert yaw_scene.name == "yaw90_center"
    assert yaw_scene.can_xy_yaw_offset == (0.0, 0.0, YAW_SCENE_RAD)

    assert get_scenarios("mini-grid10") == scenes
    with pytest.raises(KeyError):
        get_scenarios("does-not-exist")


def test_scenario_spec_wire_roundtrip_preserves_offset() -> None:
    spec = ScenarioSpec(name="yaw90_center", can_xy_yaw_offset=(0.0, 0.0, math.pi / 2))
    message = make_message("reset", 0, {"scenario": spec, "seed": 7})
    decoded = decode_message(encode_message(message))
    assert decoded["payload"]["scenario"] == spec

    default = ScenarioSpec()
    message = make_message("reset", 1, {"scenario": default, "seed": 0})
    decoded = decode_message(encode_message(message))
    assert decoded["payload"]["scenario"] == default


def test_compose_can_reset_pose_shifts_translation_and_keeps_rotation_without_yaw() -> None:
    translation, rotation = compose_can_reset_pose(
        [1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 0.0], (0.05, -0.05, 0.0)
    )
    assert translation == [1.05, 1.95, 3.0]
    assert rotation == [0.0, 0.0, 1.0, 0.0]


def test_compose_can_reset_pose_applies_yaw_about_world_z() -> None:
    initial = [1.0, 0.0, 0.0, math.pi / 2]  # 90 deg about world x
    _, rotation = compose_can_reset_pose([1.0, 2.0, 3.0], initial, (0.0, 0.0, math.pi / 2))
    # Yaw about world Z on an identity initial pose is exact.
    identity_case = compose_can_reset_pose(
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], (0.0, 0.0, math.pi / 2)
    )
    np.testing.assert_allclose(identity_case[1], [0.0, 0.0, 1.0, math.pi / 2], atol=1e-12)

    rotation_z = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    expected = rotation_z @ _axis_angle_to_matrix(initial)
    np.testing.assert_allclose(_axis_angle_to_matrix(rotation), expected, atol=1e-9)


def test_compose_can_reset_pose_handles_pi_rotation() -> None:
    # Rz(pi/2) @ Rx(pi) has angle pi, exercising the singular axis recovery branch.
    initial = [1.0, 0.0, 0.0, math.pi]
    _, rotation = compose_can_reset_pose([1.0, 2.0, 3.0], initial, (0.0, 0.0, math.pi / 2))
    rotation_z = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    expected = rotation_z @ _axis_angle_to_matrix(initial)
    np.testing.assert_allclose(_axis_angle_to_matrix(rotation), expected, atol=1e-7)


def test_fake_backend_stores_scenario_on_reset() -> None:
    backend = FakeBackend()
    spec = ScenarioSpec(name="grid_dx+0.05_dy+0.00", can_xy_yaw_offset=(0.05, 0.0, 0.0))
    backend.reset(spec, seed=0)
    assert backend._scenario == spec
    backend.reset(ScenarioSpec(), seed=1)
    assert backend._scenario == ScenarioSpec()


def test_scenario_for_index_cycles_configured_set_and_defaults_otherwise() -> None:
    scenes = mini_grid10()
    settings = _settings(scenes)
    assert scenario_for_index(settings, 0) is scenes[0]
    assert scenario_for_index(settings, 9) is scenes[9]
    assert scenario_for_index(settings, 10) is scenes[0]

    plain = scenario_for_index(_settings(), 3)
    assert plain.name == "default"
    assert plain.can_xy_yaw_offset is None


class _StopScenarioProbe(RuntimeError):
    """Raised by the recording backend right after it captures the reset scenario."""


class _RecordingResetBackend:
    def __init__(self) -> None:
        self.scenarios: list[ScenarioSpec] = []

    def reset(self, scenario: ScenarioSpec, seed: int) -> None:
        self.scenarios.append(scenario)
        raise _StopScenarioProbe(seed)


def test_policy_passes_scenario_to_backend_reset() -> None:
    spec = ScenarioSpec(name="grid_dx-0.05_dy+0.05", can_xy_yaw_offset=(-0.05, 0.05, 0.0))
    policy = ScriptedCanPolicy(scenario=spec)
    backend = _RecordingResetBackend()
    with pytest.raises(_StopScenarioProbe):
        policy.execute(backend, seed=0)
    assert backend.scenarios == [spec]

    plain_backend = _RecordingResetBackend()
    with pytest.raises(_StopScenarioProbe):
        ScriptedCanPolicy().execute(plain_backend, seed=0)
    assert plain_backend.scenarios == [ScenarioSpec()]


def test_run_episodes_cli_rejects_unknown_scenario() -> None:
    # Fails during validation, before any Webots resolution or launch.
    result = CliRunner().invoke(app, ["run-episodes", "--n", "1", "--scenario", "does-not-exist"])
    assert result.exit_code == 2
    try:
        stderr = result.stderr
    except ValueError:  # click < 8.2 merges stderr into output
        stderr = ""
    assert "unknown scenario" in (result.output + stderr)


# ---------------------------------------------------------------------------
# Seeded split-by-seed generator (ADR-0029 week 1).
# ---------------------------------------------------------------------------

_ROUND_ROBIN_TRAIN = [
    "clear",
    "cluttered",
    "reach-limited",
    "clear",
    "cluttered",
    "reach-limited",
    "clear",
    "cluttered",
]


def test_generate_scenes_counts_families_and_test_split_guard() -> None:
    sizes = load_eval_settings().sizes

    train = generate_scenes("train", 0)
    assert len(train) == sizes.train == 8
    assert [scene.metadata["family"] for scene in train] == _ROUND_ROBIN_TRAIN

    validation = generate_scenes("validation", 0)
    assert len(validation) == sizes.validation == 6
    assert {scene.metadata["family"] for scene in validation} == {
        "clear",
        "cluttered",
        "reach-limited",
    }

    # The test split is fail-closed until week 4 days 1-3 (ADR-0029).
    with pytest.raises(ValueError, match="week 4"):
        generate_scenes("test", 0)
    test_scenes = generate_scenes("test", 0, allow_test=True)
    assert len(test_scenes) == sizes.test == 10

    with pytest.raises(ValueError, match="unknown split"):
        generate_scenes("dev", 0)
    with pytest.raises(TypeError, match="integer"):
        generate_scenes("train", "0")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        generate_scenes("train", True)  # type: ignore[arg-type]


def test_generate_scenes_is_deterministic_per_split_and_seed() -> None:
    first = generate_scenes("train", 7)
    second = generate_scenes("train", 7)
    assert first == second
    assert first != generate_scenes("train", 8)

    # The same seed yields independent streams per split (SeedSequence salts).
    assert [scene.name for scene in first] != [
        scene.name for scene in generate_scenes("validation", 7)
    ]
    assert first[0].name == "train-clear-i00-s7"
    assert all(scene.metadata["seed"] == 7 for scene in first)
    assert len({scene.name for scene in first}) == len(first)


def test_generate_scenes_family_geometry_respects_pre_registered_ranges() -> None:
    library = load_scenario_library()
    for split in ("train", "validation"):
        for scene in generate_scenes(split, 7):
            family = library.families[scene.metadata["family"]]
            dx, dy, yaw = scene.can_xy_yaw_offset
            radius = math.hypot(dx, dy)
            radius_low, radius_high = family.can_radius_m
            assert radius_low - 1e-12 <= radius <= radius_high + 1e-12
            yaw_low, yaw_high = library.can_yaw_range_rad
            assert yaw_low <= yaw <= yaw_high

            mode = library.obstacle_modes[family.obstacle_mode]
            assert [entry["def"] for entry in scene.objects] == list(library.obstacle_names)
            can_x = library.world_can_xyz[0] + dx
            can_y = library.world_can_xyz[1] + dy
            for entry in scene.objects:
                assert set(entry) == {"def", "xyz", "yaw_offset"}
                obstacle_x, obstacle_y, obstacle_z = entry["xyz"]
                distance = math.hypot(obstacle_x - can_x, obstacle_y - can_y)
                distance_low, distance_high = mode.radius_m
                assert distance_low - 1e-9 <= distance <= distance_high + 1e-9
                theta = math.atan2(obstacle_y - can_y, obstacle_x - can_x)
                theta_low, theta_high = mode.theta_rad
                assert theta_low - 1e-9 <= theta <= theta_high + 1e-9
                assert obstacle_z == library.obstacle_xyz[entry["def"]][2]
                if mode.yaw_random:
                    assert -math.pi <= entry["yaw_offset"] <= math.pi
                else:
                    assert entry["yaw_offset"] == 0.0


def test_cluttered_obstacles_never_land_on_the_place_target() -> None:
    # Cone-exclusion design (configs/scenarios.yaml): MEASURED worst-case clearance over
    # seeds 0..59 x train+validation is 0.0815 m; assert the 0.06 m design floor with margin
    # so placement stays physically possible for every experiment arm.
    place_x, place_y = PLACE_WORLD[0], PLACE_WORLD[1]
    worst = float("inf")
    for seed in range(20):
        for split in ("train", "validation"):
            for scene in generate_scenes(split, seed):
                if scene.metadata["family"] != "cluttered":
                    continue
                for entry in scene.objects:
                    obstacle_x, obstacle_y, _ = entry["xyz"]
                    worst = min(worst, math.hypot(obstacle_x - place_x, obstacle_y - place_y))
    assert worst >= 0.06, f"cluttered obstacle reached {worst:.4f} m from PLACE_WORLD"


def test_scenario_library_is_pre_registered_and_matches_factory_world() -> None:
    library = load_scenario_library()
    # Pre-registration values (week 1, before any baseline/VLA result exists).
    assert library.world_can_xyz == (-0.799879, -4.38998, 0.810949)
    assert library.obstacle_xyz == {
        "HAM": (-0.544929, -4.5081, 0.754298),
        "APP": (-0.41, -4.44, 0.83),
    }
    assert library.split_salts == {"train": 0, "validation": 1, "test": 2}
    assert library.families_order == ("clear", "cluttered", "reach-limited")
    assert library.families["clear"].can_radius_m == (0.0, 0.05)
    assert library.families["cluttered"].can_radius_m == (0.0, 0.05)
    assert library.families["reach-limited"].can_radius_m == (0.06, 0.1)
    assert library.families["clear"].obstacle_mode == "parked"
    assert library.families["reach-limited"].obstacle_mode == "parked"
    assert library.families["cluttered"].obstacle_mode == "near_can"
    parked = library.obstacle_modes["parked"]
    assert parked.radius_m == (0.55, 0.7)
    assert parked.theta_rad == (-0.5235987755982988, 0.17453292519943295)
    assert parked.yaw_random is False
    near_can = library.obstacle_modes["near_can"]
    assert near_can.radius_m == (0.13, 0.2)
    assert near_can.theta_rad == (-2.9670597283903604, 2.0943951023931953)
    assert near_can.yaw_random is True

    # Drift guard: the world constants must equal the actual factory.wbt node poses.
    world_text = (Path("worlds") / "factory.wbt").read_text(encoding="utf-8")

    def translation_of(def_name: str) -> tuple[float, float, float]:
        match = re.search(
            rf"DEF {re.escape(def_name)}\s+\w+\s*\{{\s*translation\s+"
            rf"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            world_text,
        )
        assert match is not None, f"DEF {def_name} with translation not found in factory.wbt"
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

    assert translation_of("CAN") == library.world_can_xyz
    for name, xyz in library.obstacle_xyz.items():
        assert translation_of(name) == xyz


def test_scenario_library_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    source = load_scenario_library().source_path
    drifted = tmp_path / "scenarios.yaml"
    drifted.write_text(
        source.read_text(encoding="utf-8") + "\nsurprise_key: true\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly"):
        load_scenario_library(drifted)


def test_generated_scenario_objects_survive_the_wire_roundtrip() -> None:
    spec = generate_scenes("train", 3)[1]  # index 1 is the cluttered family (round-robin)
    assert spec.metadata["family"] == "cluttered"
    message = make_message("reset", 0, {"scenario": spec, "seed": 7})
    decoded = decode_message(encode_message(message))
    roundtripped = decoded["payload"]["scenario"]
    assert roundtripped == spec
    assert isinstance(roundtripped.objects, tuple)
    assert all(isinstance(entry, dict) for entry in roundtripped.objects)
