"""Scene sets, CAN pose perturbation, and scenario threading (ADR-0025, M1-07(b)/M2)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from vlarobot.backends.fake import FakeBackend
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
    get_scenarios,
    mini_grid10,
)
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
