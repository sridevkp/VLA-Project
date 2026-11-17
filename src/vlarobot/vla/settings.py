"""Typed VLA collection/execution settings; numerical tuning stays in YAML."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from vlarobot.vla.schema import CAMERA_KEY


@dataclass(frozen=True)
class VLASettings:
    schema_version: int
    camera_key: str
    sample_steps: int
    max_chunk_actions: int
    ik_position_tolerance_m: float
    ik_orientation_tolerance_rad: float
    successful_demo_target: int
    max_collection_attempts: int
    collection_workers: int

    @classmethod
    def from_yaml(cls, path: Path) -> VLASettings:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError("VLA settings keys do not match the data contract")
        if data["schema_version"] != 1 or data["camera_key"] != CAMERA_KEY:
            raise ValueError("VLA schema version or camera key mismatch")
        for key in [
            "sample_steps",
            "max_chunk_actions",
            "successful_demo_target",
            "max_collection_attempts",
            "collection_workers",
        ]:
            if type(data[key]) is not int or data[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ["ik_position_tolerance_m", "ik_orientation_tolerance_rad"]:
            if (
                not isinstance(data[key], (int, float))
                or not math.isfinite(data[key])
                or data[key] <= 0
            ):
                raise ValueError(f"{key} must be positive and finite")
        if data["collection_workers"] > 3:
            raise ValueError("At most three isolated collection workers are supported")
        return cls(**data)


def load_vla_settings() -> VLASettings:
    source = Path(__file__).resolve().parents[3] / "configs/vla.yaml"
    return VLASettings.from_yaml(
        source if source.is_file() else Path(sys.prefix) / "configs/vla.yaml"
    )
