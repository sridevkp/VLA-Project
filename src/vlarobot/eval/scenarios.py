"""Seeded scene generation for evaluation.

First piece of M2's ``ScenarioGenerator`` (ADR-0025): named scene sets that perturb the CAN
pose at reset through ``ScenarioSpec.can_xy_yaw_offset`` (world-frame ``(dx, dy, yaw)`` added
to the world-file pose by the backend). The grid constants live here, not in
``configs/robot.yaml``: they describe an evaluation scene set, not robot tuning, and the
robot config schema is strict (exact keys).
"""

from __future__ import annotations

import math

from vlarobot.core.types import ScenarioSpec

# Mini-generalization grid (ADR-0025): 3x3 CAN xy offsets at +/-5 cm in the world frame.
GRID_OFFSETS_M: tuple[float, ...] = (-0.05, 0.0, 0.05)
# One rotated-yaw scene on the center cell. The CAN is a cylinder, so yaw is likely
# physically inert; it exercises orientation plumbing end to end (stated in the report).
YAW_SCENE_RAD: float = math.pi / 2


def mini_grid10() -> tuple[ScenarioSpec, ...]:
    """The 10-scene mini-generalization set: 9 xy-grid scenes + 1 rotated-yaw scene.

    Episode ``i`` of a ``--scenario mini-grid10`` batch runs scene ``i``; names are stable
    so per-position results can be reported from the episode summaries.
    """
    scenes = [
        ScenarioSpec(
            name=f"grid_dx{dx:+.2f}_dy{dy:+.2f}",
            can_xy_yaw_offset=(dx, dy, 0.0),
        )
        for dy in GRID_OFFSETS_M
        for dx in GRID_OFFSETS_M
    ]
    scenes.append(ScenarioSpec(name="yaw90_center", can_xy_yaw_offset=(0.0, 0.0, YAW_SCENE_RAD)))
    return tuple(scenes)


SCENARIOS: dict[str, tuple[ScenarioSpec, ...]] = {"mini-grid10": mini_grid10()}


def get_scenarios(name: str) -> tuple[ScenarioSpec, ...]:
    """Return the scene set registered under ``name`` (KeyError when unknown)."""
    return SCENARIOS[name]
