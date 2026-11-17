"""Backend-boundary adapter for the no-actuation Webots diagnostic.

This is the only module allowed to open a Webots session for the new diagnostic. It wraps the
ZeroMQ client so that ``apply``, ``step``, ``reset`` and ``export_kinematics`` raise instead of
reaching the simulator, and it collects a read-only snapshot through the dedicated ``inspect``
RPC. No planner, scored evaluator or split accessor is imported here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, NoReturn

from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.core.types import Action, ApplyResult, GroundTruth, Observation, ScenarioSpec
from vlarobot.eval.worldmind_webots_diagnostic import DiagnosticSnapshot, WebotsDiagnosticConfig
from vlarobot.execution.episode_runner import launch_webots_session

_TRACE_PREFIX = "worldmind-webots-diagnostic"
_WORLD_HASH_ENV = "VLAROBOT_WORLD_SHA256"
_FORBIDDEN_CALLS = ("apply", "step", "reset", "export_kinematics")


class ReadOnlyBackendClient:
    """Forward only identity/inspection calls; every state-changing call fails loudly."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._calls = {name: 0 for name in _FORBIDDEN_CALLS}

    def ping(self) -> dict[str, Any]:
        return self._client.ping()

    def inspect(self) -> dict[str, Any]:
        return self._client.inspect()

    def close(self) -> None:
        self._client.close()

    def call_counts(self) -> dict[str, int]:
        return dict(self._calls)

    def _blocked(self, name: str) -> NoReturn:
        self._calls[name] += 1
        raise RuntimeError(f"{name} is forbidden in read-only Webots diagnostic mode")

    def apply(self, action: Action) -> ApplyResult:
        self._blocked("apply")

    def step(self, n_steps: int = 1) -> Observation:
        self._blocked("step")

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        self._blocked("reset")

    def export_kinematics(self, action: str = "poll") -> dict[str, object]:
        self._blocked("export_kinematics")


def _binding_available(binding: str, registry: Mapping[str, Any]) -> bool:
    return bool(binding) and registry.get(binding) is not None


def dependency_availability(config: WebotsDiagnosticConfig) -> dict[str, bool]:
    """Check registered solver/safety/verification objects and local geometry coverage.

    This is an importability and declaration check. It does not solve IK, run inference, plan
    motion, inspect TEST/TRAIN evidence or contact the simulator.
    """
    try:
        from vlarobot.backends.worldmind_bindings import production_worldmind_bindings

        bindings = production_worldmind_bindings()
        solver_ok = _binding_available(config.pose_solver_binding, bindings.pose_solvers)
        safety_ok = _binding_available(config.safety_binding, bindings.safety)
        verification_ok = _binding_available(config.verification_binding, bindings.verification)
    except Exception:  # noqa: BLE001 - dependency readiness must fail closed with a boolean
        solver_ok = safety_ok = verification_ok = False

    geometry_ok = False
    try:
        if config.geometry_binding == "urdf_collision_capsules":
            from vlarobot.world_model.arm_sweep import (
                IntermediateSweptArmCheck,
                urdf_collision_capsules,
            )

            geometry_ok = callable(urdf_collision_capsules) and callable(IntermediateSweptArmCheck)
            tree = ET.parse(config.urdf_path)
            declared_links = {node.attrib.get("name") for node in tree.findall("link")}
            geometry_ok = geometry_ok and set(config.required_arm_links).issubset(declared_links)
    except Exception:  # noqa: BLE001 - malformed URDF must fail closed
        geometry_ok = False

    return {
        "pose_solver": solver_ok,
        "safety_monitor": safety_ok,
        "placement_verifier": verification_ok,
        "collision_checker": geometry_ok,
        "required_arm_links": geometry_ok,
    }


def _capture(
    client: ReadOnlyBackendClient, config: WebotsDiagnosticConfig, endpoint: str
) -> DiagnosticSnapshot:
    ping = dict(client.ping())
    ping["endpoint"] = endpoint
    ping["launch_mode"] = config.mode
    started = perf_counter()
    payload = client.inspect()
    age_s = perf_counter() - started
    if not {"observation", "ground_truth", "identity"}.issubset(payload):
        raise ValueError("inspect reply omitted observation, ground truth or identity")
    observation = payload["observation"]
    ground_truth = payload["ground_truth"]
    identity = payload["identity"]
    if not isinstance(observation, Observation):
        raise TypeError("inspect reply observation has the wrong type")
    if not isinstance(ground_truth, GroundTruth):
        raise TypeError("inspect reply ground truth has the wrong type")
    if not isinstance(identity, Mapping):
        raise TypeError("inspect reply identity has the wrong type")
    return DiagnosticSnapshot(
        ping=ping,
        observation=observation,
        ground_truth=ground_truth,
        identity=dict(identity),
        captured_at_utc=datetime.now(UTC).isoformat(),
        age_s=age_s,
        dependency_availability=dependency_availability(config),
        call_counts=client.call_counts(),
    )


def collect_diagnostic_snapshot(
    config: WebotsDiagnosticConfig,
    trace_dir: Path,
    *,
    headless: bool = False,
    startup_timeout_s: float = 180.0,
) -> DiagnosticSnapshot:
    """Collect one read-only snapshot, launching or connecting exactly once as configured."""
    availability = dependency_availability(config)
    if not all(availability.values()):
        unavailable = sorted(name for name, ok in availability.items() if not ok)
        raise RuntimeError(f"required diagnostic dependencies unavailable: {unavailable}")

    if config.mode == "launch":
        with launch_webots_session(
            webots=config.webots,
            world=config.world,
            trace_dir=Path(trace_dir),
            trace_prefix=_TRACE_PREFIX,
            headless=headless,
            startup_timeout_s=startup_timeout_s,
            extra_env={_WORLD_HASH_ENV: config.world_sha256},
        ) as session:
            return _capture(ReadOnlyBackendClient(session.client), config, session.endpoint)

    client = ZmqBackendClient(config.endpoint)
    read_only = ReadOnlyBackendClient(client)
    try:
        return _capture(read_only, config, config.endpoint)
    finally:
        read_only.close()


__all__ = [
    "ReadOnlyBackendClient",
    "collect_diagnostic_snapshot",
    "dependency_availability",
]
