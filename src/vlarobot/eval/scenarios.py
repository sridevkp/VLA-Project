"""Seeded scene generation for evaluation.

First piece of M2's ``ScenarioGenerator`` (ADR-0025): named scene sets that perturb the CAN
pose at reset through ``ScenarioSpec.can_xy_yaw_offset`` (world-frame ``(dx, dy, yaw)`` added
to the world-file pose by the backend). The grid constants live here, not in
``configs/robot.yaml``: they describe an evaluation scene set, not robot tuning, and the
robot config schema is strict (exact keys).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from vlarobot.core.types import ScenarioSpec
from vlarobot.eval.settings import CampaignSizes, load_eval_settings

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


# ---------------------------------------------------------------------------
# Seeded split-by-seed generator (ADR-0029 week 1): can xy/yaw + obstacle positions.
# ---------------------------------------------------------------------------

SPLITS: tuple[str, ...] = ("train", "validation", "test")

_SOURCE_SCENARIOS_PATH = Path(__file__).resolve().parents[3] / "configs" / "scenarios.yaml"
_INSTALLED_SCENARIOS_PATH = Path(sys.prefix) / "configs" / "scenarios.yaml"


@dataclass(frozen=True)
class ObstacleMode:
    """Where one family parks or crowds the world's obstacle DEF nodes."""

    name: str
    radius_m: tuple[float, float]
    theta_rad: tuple[float, float]
    yaw_random: bool


@dataclass(frozen=True)
class FamilySpec:
    """Scene family: CAN reach band + obstacle placement mode (pre-registered)."""

    name: str
    can_radius_m: tuple[float, float]
    obstacle_mode: str


@dataclass(frozen=True)
class ScenarioLibrary:
    """Parsed ``configs/scenarios.yaml`` (world layout facts + family geometry)."""

    world_can_xyz: tuple[float, float, float]
    obstacle_xyz: dict[str, tuple[float, float, float]]
    split_salts: dict[str, int]
    families_order: tuple[str, ...]
    can_yaw_range_rad: tuple[float, float]
    families: dict[str, FamilySpec]
    obstacle_modes: dict[str, ObstacleMode]
    source_path: Path

    @property
    def obstacle_names(self) -> tuple[str, ...]:
        """Obstacle DEF names in config order (draw order is part of determinism)."""
        return tuple(self.obstacle_xyz)


def scenario_config_path() -> Path:
    """Resolve ``configs/scenarios.yaml`` (source tree first, then the installed prefix)."""
    if _SOURCE_SCENARIOS_PATH.is_file():
        return _SOURCE_SCENARIOS_PATH
    if _INSTALLED_SCENARIOS_PATH.is_file():
        return _INSTALLED_SCENARIOS_PATH
    raise FileNotFoundError(
        f"scenario config not found at {_SOURCE_SCENARIOS_PATH} or {_INSTALLED_SCENARIOS_PATH}"
    )


def _pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a [min, max] pair.")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{label} min must be <= max.")
    return (low, high)


def _triple(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be an [x, y, z] triple.")
    return (float(value[0]), float(value[1]), float(value[2]))


def load_scenario_library(path: Path | None = None) -> ScenarioLibrary:
    """Load and validate the scene-geometry config (strict keys, referential integrity)."""
    resolved = path if path is not None else scenario_config_path()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved}: top level must be a mapping.")
    expected = {
        "world",
        "split_salts",
        "families_order",
        "can_yaw_range_rad",
        "families",
        "obstacle_modes",
    }
    if set(raw) != expected:
        raise ValueError(
            f"{resolved}: keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}."
        )

    world = raw["world"]
    if not isinstance(world, dict) or set(world) != {"can_xyz", "obstacles"}:
        raise ValueError(f"{resolved}: world must map exactly {{can_xyz, obstacles}}.")
    obstacle_raw = world["obstacles"]
    if not isinstance(obstacle_raw, dict) or not obstacle_raw:
        raise ValueError(f"{resolved}: world.obstacles must be a non-empty mapping.")
    obstacle_xyz = {
        str(name): _triple(xyz, f"world.obstacles.{name}") for name, xyz in obstacle_raw.items()
    }

    salts_raw = raw["split_salts"]
    if not isinstance(salts_raw, dict) or set(salts_raw) != set(SPLITS):
        raise ValueError(f"{resolved}: split_salts must cover exactly {list(SPLITS)}.")
    split_salts = {str(name): int(salt) for name, salt in salts_raw.items()}

    families_order = tuple(raw["families_order"])
    if not families_order:
        raise ValueError(f"{resolved}: families_order must not be empty.")

    modes_raw = raw["obstacle_modes"]
    if not isinstance(modes_raw, dict) or not modes_raw:
        raise ValueError(f"{resolved}: obstacle_modes must be a non-empty mapping.")
    modes: dict[str, ObstacleMode] = {}
    for name, body in modes_raw.items():
        if not isinstance(body, dict) or set(body) != {"radius_m", "theta_rad", "yaw_random"}:
            raise ValueError(
                f"{resolved}: obstacle mode {name!r} must map exactly "
                "{{radius_m, theta_rad, yaw_random}}."
            )
        modes[str(name)] = ObstacleMode(
            name=str(name),
            radius_m=_pair(body["radius_m"], f"obstacle_modes.{name}.radius_m"),
            theta_rad=_pair(body["theta_rad"], f"obstacle_modes.{name}.theta_rad"),
            yaw_random=bool(body["yaw_random"]),
        )

    families_raw = raw["families"]
    if not isinstance(families_raw, dict) or set(families_raw) != set(families_order):
        raise ValueError(
            f"{resolved}: families must be defined for exactly "
            f"families_order={list(families_order)}."
        )
    families: dict[str, FamilySpec] = {}
    for name in families_order:
        body = families_raw[name]
        if not isinstance(body, dict) or set(body) != {"can_radius_m", "obstacles"}:
            raise ValueError(
                f"{resolved}: family {name!r} must map exactly {{can_radius_m, obstacles}}."
            )
        mode_name = str(body["obstacles"])
        if mode_name not in modes:
            raise ValueError(f"{resolved}: family {name!r} references unknown mode {mode_name!r}.")
        families[str(name)] = FamilySpec(
            name=str(name),
            can_radius_m=_pair(body["can_radius_m"], f"families.{name}.can_radius_m"),
            obstacle_mode=mode_name,
        )

    return ScenarioLibrary(
        world_can_xyz=_triple(world["can_xyz"], "world.can_xyz"),
        obstacle_xyz=obstacle_xyz,
        split_salts=split_salts,
        families_order=families_order,
        can_yaw_range_rad=_pair(raw["can_yaw_range_rad"], "can_yaw_range_rad"),
        families=families,
        obstacle_modes=modes,
        source_path=resolved,
    )


def generate_scenes(
    split: str,
    seed: int,
    *,
    allow_test: bool = False,
    library: ScenarioLibrary | None = None,
    sizes: CampaignSizes | None = None,
) -> tuple[ScenarioSpec, ...]:
    """Deterministic split-by-seed scenes: same ``(split, seed)`` ⇒ byte-identical scenes.

    Counts come from the FROZEN ``configs/eval.yaml`` (ADR-0029); geometry from
    ``configs/scenarios.yaml`` (pre-registered week 1). Families are assigned round-robin
    by scene index. The ``test`` split is fail-closed until week 4 (ADR-0029).
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; available: {', '.join(SPLITS)}")
    if split == "test" and not allow_test:
        raise ValueError(
            "the test split is frozen until week 4 days 1-3 (ADR-0029); "
            "pass allow_test=True only from the week-4 campaign."
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer.")

    campaign_sizes = sizes or load_eval_settings().sizes
    counts = {
        "train": campaign_sizes.train,
        "validation": campaign_sizes.validation,
        "test": campaign_sizes.test,
    }
    lib = library or load_scenario_library()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), lib.split_salts[split]]))
    can_yaw_low, can_yaw_high = lib.can_yaw_range_rad
    can_home = np.asarray(lib.world_can_xyz, dtype=float)
    count = counts[split]

    scenes: list[ScenarioSpec] = []
    for index in range(count):
        family_name = lib.families_order[index % len(lib.families_order)]
        family = lib.families[family_name]
        radius = float(rng.uniform(*family.can_radius_m))
        angle = float(rng.uniform(-math.pi, math.pi))
        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        yaw = float(rng.uniform(can_yaw_low, can_yaw_high))
        can_xy = can_home[:2] + np.asarray((dx, dy))

        mode = lib.obstacle_modes[family.obstacle_mode]
        objects: list[dict[str, object]] = []
        for obstacle_name in lib.obstacle_names:
            obstacle_home = np.asarray(lib.obstacle_xyz[obstacle_name], dtype=float)
            obstacle_radius = float(rng.uniform(*mode.radius_m))
            obstacle_angle = float(rng.uniform(*mode.theta_rad))
            obstacle_yaw = float(rng.uniform(-math.pi, math.pi)) if mode.yaw_random else 0.0
            objects.append(
                {
                    "def": obstacle_name,
                    "xyz": [
                        float(can_xy[0] + obstacle_radius * math.cos(obstacle_angle)),
                        float(can_xy[1] + obstacle_radius * math.sin(obstacle_angle)),
                        float(obstacle_home[2]),
                    ],
                    "yaw_offset": obstacle_yaw,
                }
            )

        scenes.append(
            ScenarioSpec(
                name=f"{split}-{family_name}-i{index:02d}-s{seed}",
                can_xy_yaw_offset=(float(dx), float(dy), yaw),
                objects=tuple(objects),
                metadata={
                    "family": family_name,
                    "split": split,
                    "seed": int(seed),
                    "index": index,
                    "obstacle_mode": family.obstacle_mode,
                },
            )
        )
    return tuple(scenes)
