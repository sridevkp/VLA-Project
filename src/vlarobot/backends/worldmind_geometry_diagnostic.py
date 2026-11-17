"""Backend-boundary collector for the bounded local-only geometry diagnostic.

This module reads pinned files only. It never imports a Webots controller, opens a socket,
solves IK, applies a backend action, steps or resets a simulator, or reads campaign splits.
A guard object makes every state-changing backend capability fail loudly if touched.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, NoReturn

from vlarobot.backends.webots.geometry_inventory import proto_inventory, urdf_inventory
from vlarobot.backends.webots.urdf_registration import load_registration
from vlarobot.backends.worldmind_bindings import ARM_SWEEP_CONFIG_PATH, load_arm_sweep_config
from vlarobot.eval.worldmind_geometry_diagnostic import (
    GeometryDiagnosticConfig,
    GeometryEvidence,
)

_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics", "inspect", "get_ground_truth")


class GeometryOnlyBackendGuard:
    """No-op boundary guard: every state-changing or simulator call is blocked."""

    def __init__(self) -> None:
        self._calls = {name: 0 for name in _FORBIDDEN_CALLS}

    def call_counts(self) -> dict[str, int]:
        return dict(self._calls)

    def _blocked(self, name: str) -> NoReturn:
        self._calls[name] += 1
        raise RuntimeError(f"{name} is forbidden in geometry-only diagnostic mode")

    def apply(self, action: Any) -> Any:
        del action
        self._blocked("apply")

    def step(self, n_steps: int = 1) -> Any:
        del n_steps
        self._blocked("step")

    def reset(self, scenario: Any, seed: int) -> Any:
        del scenario, seed
        self._blocked("reset")

    def export_kinematics(self, action: str = "poll") -> Any:
        del action
        self._blocked("export_kinematics")

    def inspect(self) -> Any:
        self._blocked("inspect")

    def get_ground_truth(self) -> Any:
        self._blocked("get_ground_truth")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_proto_count(world_path: Path) -> int:
    return len(re.findall(r"(?m)^\s*EXTERNPROTO\b", world_path.read_text(encoding="utf-8")))


def _self_collision_setting(config: GeometryDiagnosticConfig, world_path: Path) -> bool | None:
    roots: list[Path] = [world_path.parent]
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        roots.append(Path(webots_home) / "projects" / "robots" / "pal_robotics" / "tiago")
    for root in roots:
        for candidate in (*root.glob("*Tiago*.proto"), root / "protos" / "Tiago.proto"):
            if not candidate.is_file():
                continue
            match = re.search(
                r"\bselfCollision\s+(TRUE|FALSE)\b", candidate.read_text(encoding="utf-8")
            )
            if match:
                return match.group(1) == "TRUE"
    return None


def _capsule_summary(
    config: GeometryDiagnosticConfig, speed_bound_m_s: float, margin_m: float, substeps: int
) -> dict[str, object]:
    from vlarobot.world_model.arm_sweep import urdf_collision_capsules

    primitives = urdf_collision_capsules(
        config.urdf_path, config.required_arm_links, max_point_speed_m_s=speed_bound_m_s
    )
    rows: list[dict[str, object]] = []
    for link in config.required_arm_links:
        items = primitives[link]
        rows.append(
            {
                "link": link,
                "collision_count": len(items),
                "shapes": sorted({item.shape for item in items}),
                "capsule_names": [item.capsule.name for item in items],
                "radii_m": [item.capsule.radius_m for item in items],
                "unsupported_shapes": sorted(
                    item.shape for item in items if not item.certified_containment
                ),
            }
        )
    return {
        "links": rows,
        "capsule_count": sum(len(items) for items in primitives.values()),
        "speed_bound_m_s": speed_bound_m_s,
        "margin_m": margin_m,
        "margin_positive": margin_m > 0.0,
        "substeps": substeps,
        "substeps_valid": substeps >= 1,
    }


def collect_geometry_evidence(config: GeometryDiagnosticConfig) -> GeometryEvidence:
    """Collect bounded source-geometry facts; no simulator process or backend mutation."""
    guard = GeometryOnlyBackendGuard()
    sweep_config = load_arm_sweep_config(ARM_SWEEP_CONFIG_PATH)
    world_inventory = proto_inventory(config.world_path)
    urdf = urdf_inventory(config.urdf_path)
    try:
        load_registration(config.registration_path)
        registration_valid = True
    except Exception:  # noqa: BLE001 - malformed registration must fail closed in evaluation rules
        registration_valid = False
    hashes = {
        "urdf": _sha256_file(config.urdf_path),
        "registration": _sha256_file(config.registration_path),
        "robot_config": _sha256_file(config.robot_config_path),
        "world": _sha256_file(config.world_path),
    }
    return GeometryEvidence(
        world_inventory=dict(world_inventory),
        urdf_inventory=dict(urdf),
        collision_capsules=_capsule_summary(
            config,
            sweep_config.speed_bound_m_s,
            sweep_config.margin_m,
            sweep_config.substeps,
        ),
        hashes=dict(hashes),
        registration_valid=registration_valid,
        external_proto_count=_external_proto_count(config.world_path),
        self_collision_enabled=_self_collision_setting(config, config.world_path),
        call_counts=guard.call_counts(),
    )


__all__ = ["GeometryOnlyBackendGuard", "collect_geometry_evidence"]
