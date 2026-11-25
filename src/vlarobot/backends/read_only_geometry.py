"""Backend-internal read-only Webots geometry capture.

This helper reuses the existing read-only state capture and the supervisor-side boundingObject
export. It never calls ``apply``, ``step``, ``reset``, the kinematics-export probe, IK, a planner or
a scorer, and it changes no simulator state. It is deliberately separate from the frozen Webots
backend source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vlarobot.backends.read_only_state import capture_read_only_state
from vlarobot.backends.webots.geometry_probe import snapshot as export_bounding_objects

_GEOMETRY_KEYS = frozenset(
    {
        "status",
        "robot_id",
        "self_collision_enabled",
        "sim_time_s",
        "objects",
        "motor_commands_sent",
        "devices",
        "joint_positions",
        "motor_velocity_settings",
        "scene",
        "unsupported_shapes",
    }
)


def capture_read_only_geometry(
    backend: Any, table_targets: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Return observation, oracle state, identity and geometry without touching simulator state.

    The supervisor bounding-object export can take seconds. It runs *before* the state capture
    so the observation's server-side capture timestamp is stamped at the point of use, rather
    than being aged by the export that happens after it. The export is read-only and cannot
    advance physics; commanded/stepped backends are still rejected before any result is
    returned.
    """
    io = getattr(backend, "io", None)
    supervisor = getattr(io, "robot", None)
    if supervisor is None:
        raise RuntimeError("Backend does not expose a Webots supervisor for geometry inspection.")
    geometry = export_bounding_objects(supervisor, table_targets)
    if not isinstance(geometry, dict) or not _GEOMETRY_KEYS.issubset(geometry):
        raise RuntimeError("Backend read-only geometry export returned an incomplete snapshot.")
    if table_targets is not None and "table_targets" not in geometry:
        raise RuntimeError("Backend read-only geometry export omitted table target observations.")
    if geometry.get("motor_commands_sent") != 0:
        raise RuntimeError("Geometry export reported motor commands; inspection is blocked.")

    state = capture_read_only_state(backend)
    identity = state.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Backend read-only geometry inspection returned no identity.")
    if identity.get("first_command_step_count") is not None:
        raise RuntimeError(
            "Backend has issued an actuator command; geometry inspection is blocked."
        )
    if identity.get("physics_step_count") != 0:
        raise RuntimeError("Backend has stepped physics; geometry inspection is blocked.")
    return {**state, "geometry": dict(geometry)}


__all__ = ["capture_read_only_geometry"]
