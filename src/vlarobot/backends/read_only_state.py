"""Backend-internal read-only state capture for the no-actuation diagnostic.

This helper deliberately lives outside the frozen Webots backend source file. It reuses the
backend's existing observation builder and oracle reader; it never calls ``step``, ``apply``,
``reset`` or the kinematics-export probe, and it changes no simulator state.
"""

from __future__ import annotations

from typing import Any

from vlarobot.core.types import GroundTruth, Observation


def _webots_identity(backend: Any) -> dict[str, object]:
    io = getattr(backend, "io", None)
    robot = getattr(io, "me", None)
    if robot is None:
        raise RuntimeError("backend does not expose a Webots robot identity")
    from vlarobot.backends.webots.supervisor_kinematics import webots_version

    return {
        "simulator": "webots",
        "robot_name": str(getattr(robot, "getName", lambda: "")()),
        "robot_type": str(getattr(robot, "getTypeName", lambda: "")()),
        "webots_version": webots_version(io),
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
