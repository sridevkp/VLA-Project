"""Backend-boundary adapter for the dedicated read-only Webots geometry inspection.

This is the only module allowed to open a Webots session for the geometry-inspection command. It
wraps the ZeroMQ client so that state-changing and alternate oracle calls fail loudly, and it
collects the supervisor boundingObject export through the dedicated ``inspect_geometry`` RPC. No
planner, IK solver, scored evaluator or split accessor is imported here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, NoReturn

from vlarobot.backends.worldmind_webots_diagnostic import (
    _ping_payload,
    dependency_availability,
)
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.core.types import (
    Action,
    ApplyResult,
    GroundTruth,
    Observation,
    ScenarioSpec,
)
from vlarobot.eval.worldmind_webots_diagnostic import (
    capture_age_s,
    validate_capture_timestamp,
)
from vlarobot.eval.worldmind_webots_geometry_inspection import (
    GeometryInspectionConfig,
    GeometryInspectionSnapshot,
    table_target_descriptor,
)
from vlarobot.execution.episode_runner import launch_webots_session

_TRACE_PREFIX = "worldmind-webots-geometry-inspection"
_WORLD_HASH_ENV = "VLAROBOT_WORLD_SHA256"
_FORBIDDEN_CALLS = (
    "apply",
    "step",
    "reset",
    "export_kinematics",
    "inspect",
    "get_ground_truth",
)
_ALLOWED_CALLS = ("ping", "inspect_geometry")


class ReadOnlyGeometryBackendClient:
    """Forward only ping/inspect_geometry; every other backend call fails loudly."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._calls = {name: 0 for name in (*_ALLOWED_CALLS, *_FORBIDDEN_CALLS)}

    def ping(self) -> dict[str, Any]:
        self._calls["ping"] += 1
        return self._client.ping()

    def inspect_geometry(self, table_targets: Mapping[str, object] | None = None) -> dict[str, Any]:
        self._calls["inspect_geometry"] += 1
        return self._client.inspect_geometry(table_targets)

    def close(self) -> None:
        self._client.close()

    def call_counts(self) -> dict[str, int]:
        return dict(self._calls)

    def _blocked(self, name: str) -> NoReturn:
        self._calls[name] += 1
        raise RuntimeError(f"{name} is forbidden in read-only geometry inspection mode")

    def apply(self, action: Action) -> ApplyResult:
        self._blocked("apply")

    def step(self, n_steps: int = 1) -> Observation:
        self._blocked("step")

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        self._blocked("reset")

    def export_kinematics(self, action: str = "poll") -> dict[str, object]:
        self._blocked("export_kinematics")

    def inspect(self) -> dict[str, Any]:
        self._blocked("inspect")

    def get_ground_truth(self) -> GroundTruth:
        self._blocked("get_ground_truth")


def _capture(
    client: ReadOnlyGeometryBackendClient,
    config: GeometryInspectionConfig,
    endpoint: str,
    launch_attestation: Mapping[str, object] | None = None,
) -> GeometryInspectionSnapshot:
    ping = _ping_payload(client)
    ping["endpoint"] = endpoint
    ping["launch_mode"] = config.diagnostic.mode
    payload = client.inspect_geometry(table_target_descriptor(config))
    if not {"observation", "ground_truth", "identity", "geometry"}.issubset(payload):
        raise ValueError("inspect_geometry reply omitted state, identity or geometry")
    observation = payload["observation"]
    ground_truth = payload["ground_truth"]
    identity = payload["identity"]
    geometry = payload["geometry"]
    if not isinstance(observation, Observation):
        raise TypeError("inspect_geometry observation has the wrong type")
    if not isinstance(ground_truth, GroundTruth):
        raise TypeError("inspect_geometry ground truth has the wrong type")
    if not isinstance(identity, Mapping):
        raise TypeError("inspect_geometry identity has the wrong type")
    if not isinstance(geometry, Mapping):
        raise TypeError("inspect_geometry geometry has the wrong type")
    now_wall_time_s = time()
    capture, capture_blockers = validate_capture_timestamp(payload, now_wall_time_s=now_wall_time_s)
    if capture is not None:
        captured_at_utc = capture.captured_at_utc
        capture_wall_time_s = capture.capture_wall_time_s
        age_s = capture_age_s(capture.capture_wall_time_s, now_wall_time_s=now_wall_time_s)
    else:
        raw_utc = payload.get("captured_at_utc")
        captured_at_utc = raw_utc if isinstance(raw_utc, str) else ""
        raw_wall = payload.get("capture_wall_time_s")
        capture_wall_time_s = (
            float(raw_wall)
            if isinstance(raw_wall, (int, float))
            and not isinstance(raw_wall, bool)
            and math.isfinite(float(raw_wall))
            else None
        )
        age_s = float("inf")
    return GeometryInspectionSnapshot(
        ping=ping,
        observation=observation,
        ground_truth=ground_truth,
        identity=dict(identity),
        captured_at_utc=captured_at_utc,
        age_s=age_s,
        dependency_availability=dependency_availability(config.diagnostic),
        call_counts=client.call_counts(),
        launch_attestation=dict(launch_attestation or {}),
        geometry=dict(geometry),
        capture_wall_time_s=capture_wall_time_s,
        capture_timestamp_blockers=capture_blockers,
    )


def collect_geometry_inspection_snapshot(
    config: GeometryInspectionConfig,
    trace_dir: Path,
    *,
    headless: bool = False,
    startup_timeout_s: float = 180.0,
) -> GeometryInspectionSnapshot:
    """Collect one read-only geometry snapshot, launching or connecting exactly once."""
    availability = dependency_availability(config.diagnostic)
    if not all(availability.values()):
        unavailable = sorted(name for name, ok in availability.items() if not ok)
        raise RuntimeError(f"required geometry dependencies unavailable: {unavailable}")

    if config.diagnostic.mode == "launch":
        with launch_webots_session(
            webots=config.diagnostic.webots,
            world=config.diagnostic.world,
            trace_dir=Path(trace_dir),
            trace_prefix=_TRACE_PREFIX,
            headless=headless,
            startup_timeout_s=startup_timeout_s,
            extra_env={_WORLD_HASH_ENV: config.diagnostic.world_sha256},
            runtime_attestation_required=True,
        ) as session:
            return _capture(
                ReadOnlyGeometryBackendClient(session.client),
                config,
                session.endpoint,
                session.runtime_attestation,
            )

    client = ZmqBackendClient(config.diagnostic.endpoint)
    read_only = ReadOnlyGeometryBackendClient(client)
    try:
        return _capture(read_only, config, config.diagnostic.endpoint)
    finally:
        read_only.close()


__all__ = [
    "ReadOnlyGeometryBackendClient",
    "collect_geometry_inspection_snapshot",
]
