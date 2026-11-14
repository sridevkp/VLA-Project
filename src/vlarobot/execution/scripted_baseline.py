"""Scripted CAN expert as an eval-runner policy (ADR-0029 week 1, ADR-0033).

The eval package deliberately imports only ``core`` and ``eval`` modules (AGENTS rule 1,
ADR-0032), so it cannot know about :class:`ScriptedCanPolicy`. This adapter is the composition
seam: it runs the scripted expert on one scene and returns the plain mapping of outcome keys
that ``eval.metrics`` reads from the episode metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vlarobot.core.types import ScenarioSpec
from vlarobot.execution.scripted_can import ScriptedCanPolicy, ScriptedCanResult


def outcome_payload(result: ScriptedCanResult) -> dict[str, Any]:
    """Map one ``ScriptedCanResult`` onto the eval outcome keys (MEASURED values only)."""
    return {
        "success": bool(result.success),
        "final_state": str(result.final_state),
        "failure_state": result.failure_state,
        "failure_reason": result.failure_reason,
        "final_target_distance": float(result.final_target_distance),
        "home_reached": bool(result.home_reached),
        "place_distance_ok": bool(result.place_distance_ok),
        "can_height_ok": bool(result.can_height_ok),
        "can_tilt_ok": bool(result.can_tilt_ok),
        "can_height_error_m": result.can_height_error_m,
        "can_tilt_deg": result.can_tilt_deg,
        "rejections": int(result.rejections),
        "pregrasp_realign": result.pregrasp_realign,
        "closed_loop_waypoints": result.closed_loop_waypoints,
        "max_base_drift_m": result.max_base_drift_m,
        "max_base_drift_yaw_deg": result.max_base_drift_yaw_deg,
        "max_base_tilt_deg": result.max_base_tilt_deg,
        # FK clearance is a standalone geometry measurement (ADR-0032). The scripted baseline
        # has no FK-point source wired yet, so it is recorded as unmeasured, never invented.
        "fk_min_clearance_m": None,
    }


class ScriptedBaselinePolicy:
    """Eval-runner policy: run the scripted expert on one scene, return outcome keys."""

    def __init__(self, scenario: ScenarioSpec, *, trace_path: Any | None = None) -> None:
        self.scenario = scenario
        self.trace_path = trace_path

    def execute(self, backend: Any, seed: int) -> Mapping[str, Any]:
        policy = ScriptedCanPolicy(scenario=self.scenario, trace_path=self.trace_path)
        return outcome_payload(policy.execute(backend, seed))


def scripted_policy_factory(scenario: ScenarioSpec) -> ScriptedBaselinePolicy:
    """Factory matching ``eval.runner.PolicyFactory``."""
    return ScriptedBaselinePolicy(scenario)
