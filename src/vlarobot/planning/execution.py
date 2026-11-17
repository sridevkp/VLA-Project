"""Receding-horizon selection execution: every source crosses the same live safety veto."""

from __future__ import annotations

from vlarobot.core.interfaces import RobotBackend
from vlarobot.core.types import ApplyResult, Observation
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.risk.comparison import Selection


def execute_first(
    selection: Selection, observation: Observation, backend: RobotBackend, monitor: SafetyMonitor
) -> tuple[Observation, ApplyResult]:
    """Execute at most one accepted control tick, then re-observe before replanning."""
    if selection.chunk is None or not selection.chunk.actions:
        return observation, ApplyResult(False, selection.rationale)
    action = selection.chunk.actions[0]
    verdict = monitor.validate(action, observation.robot)
    if not verdict.accepted:
        return observation, verdict
    applied = backend.apply(action)
    return (backend.step(1) if applied.accepted else observation), applied
