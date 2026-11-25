"""Typed loader for the frozen evaluation-campaign config (``configs/eval.yaml``).

Sizes, arms, repetitions, and the week-2 gate are FROZEN by owner decision ADR-0029
(2026-11-14). The gate comparator is the literal ``min`` of the written criterion
(owner answer, ADR-0029): the threshold is the weaker of the two bounds.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

_SOURCE_EVAL_PATH = Path(__file__).resolve().parents[3] / "configs" / "eval.yaml"
_INSTALLED_EVAL_PATH = Path(sys.prefix) / "configs" / "eval.yaml"

_REQUIRED_KEYS = {
    "campaign",
    "splits",
    "families",
    "arms",
    "repetitions",
    "week2_gate",
    "collision_proxy",
}
_SPLITS = ("train", "validation", "test")
_ARMS = (
    "scripted_baseline",
    "vla_only",
    "vla_risk_veto",
    "vla_world_model",
    "full_system",
)
_FAMILIES = ("clear", "cluttered", "reach-limited")


@dataclass(frozen=True)
class CampaignSizes:
    """Frozen scene counts per split (ADR-0029)."""

    train: int
    validation: int
    test: int


@dataclass(frozen=True)
class Week2Gate:
    """Frozen week-2 VLA gate: ``success >= min(min_rate, fraction * baseline_rate)``."""

    min_success_rate: float
    baseline_fraction: float
    comparator: str

    def threshold(self, baseline_rate: float) -> float:
        """Pass threshold for a measured baseline success rate (literal ``min``, ADR-0029)."""
        if self.comparator != "min":
            raise ValueError(f"unsupported gate comparator: {self.comparator!r}")
        if not 0.0 <= baseline_rate <= 1.0:
            raise ValueError("baseline_rate must be in [0, 1].")
        return min(self.min_success_rate, self.baseline_fraction * baseline_rate)


@dataclass(frozen=True)
class EvalSettings:
    """Frozen campaign configuration (ADR-0029)."""

    start_date: str
    deadline: str
    weeks: int
    sizes: CampaignSizes
    families: tuple[str, ...]
    arms: tuple[str, ...]
    vla_seeds_per_scene: int
    baseline_seeds_per_scene: int
    gate: Week2Gate
    collision_proxy_max_displacement_m: float
    collision_proxy_max_tilt_deg: float
    source_path: Path


def eval_config_path() -> Path:
    """Resolve ``configs/eval.yaml`` (source tree first, then the installed prefix)."""
    if _SOURCE_EVAL_PATH.is_file():
        return _SOURCE_EVAL_PATH
    if _INSTALLED_EVAL_PATH.is_file():
        return _INSTALLED_EVAL_PATH
    raise FileNotFoundError(
        f"eval config not found at {_SOURCE_EVAL_PATH} or {_INSTALLED_EVAL_PATH}"
    )


def load_eval_settings(path: Path | None = None) -> EvalSettings:
    """Load and validate the frozen campaign config (strict: exact keys, exact arms)."""
    resolved = path if path is not None else eval_config_path()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved}: top level must be a mapping.")
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"{resolved}: missing keys: {sorted(missing)}")
    extra = set(raw) - _REQUIRED_KEYS
    if extra:
        raise ValueError(f"{resolved}: unknown keys: {sorted(extra)}")

    campaign = raw["campaign"]
    splits = raw["splits"]
    repetitions = raw["repetitions"]
    gate = raw["week2_gate"]
    collision = raw["collision_proxy"]
    for mapping, label in (
        (campaign, "campaign"),
        (splits, "splits"),
        (repetitions, "repetitions"),
        (gate, "week2_gate"),
        (collision, "collision_proxy"),
    ):
        if not isinstance(mapping, dict):
            raise ValueError(f"{resolved}: {label} must be a mapping.")

    if set(splits) != set(_SPLITS):
        raise ValueError(f"{resolved}: splits must be exactly {list(_SPLITS)}.")
    if set(gate) != {"min_success_rate", "baseline_fraction", "comparator"}:
        raise ValueError(f"{resolved}: week2_gate keys must be exactly as frozen.")
    if set(collision) != {"max_displacement_m", "max_tilt_deg"}:
        raise ValueError(f"{resolved}: collision_proxy keys must be exactly as frozen.")
    if set(repetitions) != {"vla_seeds_per_scene", "baseline_seeds_per_scene"}:
        raise ValueError(f"{resolved}: repetitions keys must be exactly as frozen.")
    if set(campaign) != {"start_date", "deadline", "weeks"}:
        raise ValueError(f"{resolved}: campaign keys must be exactly as frozen.")

    arms = tuple(raw["arms"])
    if arms != _ARMS:
        raise ValueError(f"{resolved}: arms must be exactly {list(_ARMS)} (ADR-0029).")
    families = tuple(raw["families"])
    if families != _FAMILIES:
        raise ValueError(f"{resolved}: families must be exactly {list(_FAMILIES)} (ADR-0029).")

    sizes = CampaignSizes(
        train=int(splits["train"]),
        validation=int(splits["validation"]),
        test=int(splits["test"]),
    )
    if (sizes.train, sizes.validation, sizes.test) != (8, 6, 10):
        raise ValueError(
            f"{resolved}: frozen sizes are train=8, validation=6, test=10 (ADR-0029); got {sizes}."
        )
    vla_seeds = int(repetitions["vla_seeds_per_scene"])
    baseline_seeds = int(repetitions["baseline_seeds_per_scene"])
    if vla_seeds != 3:
        raise ValueError(f"{resolved}: vla_seeds_per_scene must be 3 (ADR-0029); got {vla_seeds}.")
    if baseline_seeds != 3:
        raise ValueError(
            f"{resolved}: baseline_seeds_per_scene must be 3 (ADR-0029 owner answer); "
            f"got {baseline_seeds}."
        )

    min_rate = float(gate["min_success_rate"])
    fraction = float(gate["baseline_fraction"])
    comparator = str(gate["comparator"])
    if comparator != "min":
        raise ValueError(
            f"{resolved}: week2_gate.comparator must be the literal 'min' "
            f"(owner answer, ADR-0029); got {comparator!r}."
        )
    if (min_rate, fraction) != (0.40, 0.60):
        raise ValueError(
            f"{resolved}: frozen gate values are 0.40/0.60 (ADR-0029); got {min_rate}/{fraction}."
        )

    return EvalSettings(
        start_date=str(campaign["start_date"]),
        deadline=str(campaign["deadline"]),
        weeks=int(campaign["weeks"]),
        sizes=sizes,
        families=families,
        arms=arms,
        vla_seeds_per_scene=vla_seeds,
        baseline_seeds_per_scene=baseline_seeds,
        gate=Week2Gate(
            min_success_rate=min_rate, baseline_fraction=fraction, comparator=comparator
        ),
        collision_proxy_max_displacement_m=float(collision["max_displacement_m"]),
        collision_proxy_max_tilt_deg=float(collision["max_tilt_deg"]),
        source_path=resolved,
    )
