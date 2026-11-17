"""Backend-internal read-only state capture for the no-actuation diagnostic.

This helper deliberately lives outside the frozen Webots backend source file. It reuses the
backend's existing observation builder and oracle reader; it never calls ``step``, ``apply``,
``reset`` or the kinematics-export probe, and it changes no simulator state.
"""

from __future__ import annotations

from typing import Any

from vlarobot.backends.webots.runtime_attestation import (
    controller_runtime_version,
    normalize_webots_version,
)
from vlarobot.core.types import GroundTruth, Observation


def _call(target: Any, name: str, *args: object) -> Any:
    getter = getattr(target, name, None)
    return getter(*args) if callable(getter) else None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _webots_identity(backend: Any) -> dict[str, object]:
    io = getattr(backend, "io", None)
    robot = getattr(io, "me", None)
    if robot is None:
        raise RuntimeError("backend does not expose a Webots robot identity")
    from vlarobot.backends.webots.supervisor_kinematics import webots_version

    robot_def = _string_or_none(_call(robot, "getDef"))
    supervisor = getattr(io, "robot", None)
    def_node = _call(supervisor, "getFromDef", robot_def) if robot_def else None
    robot_id = _call(robot, "getId")
    def_node_id = _call(def_node, "getId") if def_node is not None else None
    def_cross_checked = bool(
        robot_def is not None
        and def_node is not None
        and robot_id is not None
        and def_node_id is not None
        and int(robot_id) == int(def_node_id)
    )

    controller_version, version_source = controller_runtime_version()
    if controller_version is None:
        api_version = webots_version(io)
        controller_version = normalize_webots_version(api_version)
        if controller_version is not None:
            version_source = "Webots controller API getVersion"

    return {
        "simulator": "webots",
        "robot_def": robot_def,
        "robot_def_cross_checked": def_cross_checked,
        "robot_name": _string_or_none(_call(robot, "getName")),
        "robot_type": _string_or_none(_call(robot, "getTypeName")),
        "webots_version": controller_version,
        "webots_version_available": controller_version is not None,
        "webots_version_source": version_source,
        "controller_webots_version": controller_version,
        "controller_webots_version_source": version_source,
        "physics_step_count": int(getattr(backend, "_physics_step_count", 0)),
        "first_command_step_count": getattr(backend, "_first_command_step_count", None),
    }


def capture_read_only_state(backend: Any) -> dict[str, object]:
    """Return observation, oracle state and identity without advancing or commanding the sim."""
    if getattr(backend, "_stopped", False):
        raise RuntimeError("Cannot inspect a stopped backend.")

    inspector = getattr(backend, "read_only_state", None)
    if callable(inspector):
        snapshot = inspector()
        if not isinstance(snapshot, dict):
            raise RuntimeError("Backend read-only inspection returned a non-mapping snapshot.")
        return snapshot

    observation_builder = getattr(backend, "_observation", None)
    oracle_reader = getattr(backend, "get_ground_truth", None)
    if not callable(observation_builder) or not callable(oracle_reader):
        raise RuntimeError("Backend does not support read-only inspection.")

    observation: Observation = observation_builder()
    ground_truth: GroundTruth = oracle_reader()
    return {
        "observation": observation,
        "ground_truth": ground_truth,
        "identity": _webots_identity(backend),
    }


__all__ = ["capture_read_only_state"]
