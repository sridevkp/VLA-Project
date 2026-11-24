"""WorldMind's stateful candidate-00 manipulation diagnostic.

The live expert owns waypoint/subphase control flow. The chunk boundary owns
candidate evidence, motor admission, simulator operations, and final verification.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import ndtr, ndtri

from vlarobot.core.types import (
    Action,
    ActionChunk,
    ApplyResult,
    JointTarget,
    Observation,
    ScenarioSpec,
)
from vlarobot.eval.decision_adapters import CandidateGenerationProbe
from vlarobot.execution.scripted_can import ScriptedCanPolicy
from vlarobot.execution.worldmind_heuristic_qualification import (  # noqa: PLC2701
    STATEFUL_DIAGNOSTIC_SCOPE,
    STATEFUL_VALIDATION_SCOPE,
    _encode,
    _journal_apply,
    _persist,
    _row_data,
    _tick,
    _write_event,
)
from vlarobot.planning.candidates import SeededCandidateGenerator

if TYPE_CHECKING:
    from pathlib import Path


def bounded_normal_target(
    value: float, lower: float, upper: float, std: float, radius: float, rng: np.random.Generator
) -> float:
    """Draw from the same conditional normal without finite rejection exhaustion."""
    if not lower <= value <= upper:
        raise ValueError("reference waypoint violates absolute joint bounds")
    if std == 0 or radius == 0 or lower == upper:
        return value
    low_cdf = ndtr(max(lower - value, -radius) / std)
    high_cdf = ndtr(min(upper - value, radius) / std)
    drawn = value + std * float(ndtri(rng.uniform(low_cdf, high_cdf)))
    if not lower <= drawn <= upper or abs(drawn - value) > radius:
        raise ValueError("conditional waypoint draw outside reviewed jitter support")
    return drawn


class TaskStop(RuntimeError):
    """An observed bounded task termination, separate from a runner exception."""

    def __init__(self, terminal: str) -> None:
        super().__init__(terminal)
        self.terminal = terminal


class StatefulChunkBackend:
    """Bound one expert coroutine to observation-driven held-command chunks."""

    def __init__(
        self,
        initial: Observation,
        scenario: ScenarioSpec,
        seed: int,
        *,
        horizon: int,
        max_ticks: int,
        decide: Any,
        advance: Any,
        truth: Any,
        reobserve: Any = None,
    ) -> None:
        self.observation = initial
        self.scenario, self.seed = scenario, seed
        self.horizon, self.max_ticks = horizon, max_ticks
        self.decide, self.advance, self.truth = decide, advance, truth
        self.reobserve = reobserve
        self.empty_recovery_used = False
        self.verification_retry_used = False
        self.ticks = self.decisions = self.used = 0
        self.phase = "INIT"
        self.target: Action | None = None
        self.armed = False
        self.initial_delivered = False

    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation:
        """Supply the already owned and durably recorded reset exactly once."""
        if self.initial_delivered or scenario != self.scenario or seed != self.seed:
            raise ValueError("stateful episode attempted an inconsistent reset")
        self.initial_delivered = True
        return self.observation

    def retry_verification(self) -> None:
        """Explicit primary-family retry, without a backend reset or budget reset."""
        if self.phase != "HOME" or self.verification_retry_used:
            raise ValueError("invalid verification retry boundary")
        self.verification_retry_used = True
        self.initial_delivered = False
        self.armed = False
        self.phase = "INIT"

    def get_ground_truth(self) -> Any:
        return self.truth()

    def apply(self, action: Action) -> ApplyResult:
        return self._choose(action, continuation=False)

    def _choose(self, action: Action, *, continuation: bool) -> ApplyResult:
        if not self.initial_delivered or self.ticks >= self.max_ticks:
            raise TaskStop("WORLDMIND_BOUND_EXHAUSTED")
        self.armed = False
        result, selected = self.decide(self, action, continuation)
        self.decisions += 1
        if (
            result.reason == "no eligible stateful candidate"
            and self.reobserve is not None
            and not self.empty_recovery_used
            and self.ticks < self.max_ticks
        ):
            self.empty_recovery_used = True
            self.observation = self.reobserve(self)
            self.ticks += 1
            if self.ticks >= self.max_ticks:
                raise TaskStop("WORLDMIND_BOUND_EXHAUSTED")
            result, selected = self.decide(self, action, continuation)
            self.decisions += 1
        if result.reason == "no eligible stateful candidate" and self.reobserve is not None:
            raise TaskStop("NO_SAFE_SET")
        self.used = 0
        if result.accepted:
            self.target = selected
            self.armed = True
        return result

    def step(self, count: int) -> Observation:
        if count != 1 or not self.armed or self.target is None:
            raise ValueError("stateful caller requires one tick under an accepted command")
        if self.ticks >= self.max_ticks:
            raise TaskStop("WORLDMIND_BOUND_EXHAUSTED")
        if self.used == self.horizon:
            result = self._choose(self.target, continuation=True)
            if not result.accepted:
                raise RuntimeError(f"stateful continuation rejected: {result.reason}")
        self.observation = self.advance(self)
        self.ticks += 1
        self.used += 1
        return self.observation


class ChunkExpert(ScriptedCanPolicy):
    """Expose the baseline's actual phase cursor to the chunk backend."""

    def _accepted_joint_waypoint(
        self, backend: StatefulChunkBackend, requested: dict[str, float]
    ) -> dict[str, float]:
        if not isinstance(backend.target, JointTarget):
            raise RuntimeError("accepted waypoint has no reconciled joint target")
        return backend.target.q

    def _trace_boundary(
        self,
        backend: StatefulChunkBackend,
        observation: Observation,
        state: str,
        phase: str,
        target: np.ndarray | None = None,
        failed: bool = False,
        **details: object,
    ) -> None:
        if phase in {"enter", "command"}:
            backend.phase = state
        super()._trace_boundary(backend, observation, state, phase, target, failed, **details)


def run_stateful_expert(
    session: Any,
    owned: dict[str, object],
    directory: Path,
    entry: Any,
    config: Any,
    fixture: Any,
    monitor: Any,
    observation: Observation,
    verifier: Any,
    task: Any,
    observe: Any,
    *,
    scored: bool,
) -> str:
    """Execute the real WorldMind selection/receipt path with the live expert cursor."""
    scope = STATEFUL_VALIDATION_SCOPE if scored else STATEFUL_DIAGNOSTIC_SCOPE
    active: dict[str, Any] = {}
    peak = float("-inf")
    rejected_terminal = "PLACEMENT_FAILED"
    start_z = session.client.get_ground_truth().object_poses[task.target].position.z
    limits = monitor.limits.joint_position
    settings = config.adapter.jitter

    def decide(
        cursor: StatefulChunkBackend, action: Action, continuation: bool
    ) -> tuple[ApplyResult, Action]:
        nonlocal rejected_terminal
        if cursor.decisions >= config.bounds.max_decisions:
            raise TaskStop("WORLDMIND_BOUND_EXHAUSTED")
        folder = directory / f"chunk-{cursor.decisions:04d}"
        folder.mkdir()
        state, _ = observe(cursor.observation, folder / "observed-state.msgpack")
        index = 0

        def draw(_task: Any, _state: Any, _memory: Any, seed: int) -> ActionChunk:
            nonlocal index
            chosen = action
            if index and not continuation and isinstance(action, JointTarget):
                rng = np.random.default_rng(seed)
                joints = {
                    name: bounded_normal_target(
                        value, *limits[name], settings.joint_std_rad, settings.joint_limit_rad, rng
                    )
                    for name, value in action.q.items()
                }
                chosen = JointTarget(joints)
            index += 1
            return ActionChunk(
                [chosen] * config.pipeline.horizon_steps, config.pipeline.control_dt_s, "scripted"
            )

        generator = SeededCandidateGenerator(
            draw,
            seed=config.pipeline.decision_seed_base + entry.seed,
            count=config.pipeline.candidate_count,
            source="scripted",
        )
        generator.decision_index = cursor.decisions
        probe = CandidateGenerationProbe(
            generator, episode_id=entry.scene.name, decision_id=folder.name
        )
        pool = tuple(probe.propose(task, state, None))
        (generation,) = probe.observations
        rows, selected = fixture.assess_pool(state, cursor.observation, monitor, generation, pool)
        if selected is not None and (scored or selected == 0):
            rows[selected] = replace(rows[selected], disposition="selected")
        else:
            selected = None
        path = folder / "decision.msgpack"
        snapshot = _persist(
            path,
            _encode(
                {
                    "generation": asdict(generation),
                    "pool": [_row_data(row) for row in rows],
                    "selected": rows[selected].candidate_id if selected is not None else None,
                    "phase": cursor.phase,
                    "continuation": continuation,
                    "episode_tick": cursor.ticks,
                    "observation": cursor.observation,
                    "expiration": "earliest expert waypoint completion or horizon",
                },
                scope,
            ),
        )
        if selected is None:
            return ApplyResult(False, "no eligible stateful candidate"), action
        action = pool[selected].actions[0]
        motor_verdict = monitor.validate(action, cursor.observation.robot)
        verdict_sha = _persist(
            folder / "motor-verdict.msgpack",
            _encode(
                {
                    "action": action,
                    "motor_verdict": motor_verdict,
                },
                scope,
            ),
        )
        if not motor_verdict.accepted:
            rejected_terminal = "motor_veto"
            return motor_verdict, action
        if continuation:
            # The persistent target was already submitted. Reconcile a fresh chunk without
            # resubmitting the unchanged command (and without resetting its settle timer).
            receipt_sha = active["receipt"]
            _write_event(
                folder / "held-target.json",
                {
                    "original_apply_receipt_sha256": receipt_sha,
                    "decision_sha256": snapshot,
                },
            )
            result = ApplyResult(True)
        else:
            result, receipt_sha = _journal_apply(
                folder,
                session,
                owned,
                path,
                snapshot,
                action,
                0,
                verdict_sha,
                scope,
            )
        if result.accepted:
            active.update(folder=folder, receipt=receipt_sha, decision=snapshot)
        else:
            rejected_terminal = "backend_rejected"
        return result, action

    def advance(cursor: StatefulChunkBackend) -> Observation:
        nonlocal peak
        folder = active["folder"] / f"tick-{cursor.used:02d}"
        folder.mkdir()
        status, after = _tick(
            folder,
            session,
            cursor.observation,
            {
                "accepted_apply_sha256": active["receipt"],
                "decision_sha256": active["decision"],
                "phase": cursor.phase,
                "held_target_tick": cursor.used,
            },
            scope,
        )
        if after is None:
            raise RuntimeError(f"stateful chunk stopped: {status}")
        truth = session.client.get_ground_truth()
        height = truth.object_poses[task.target].position.z
        peak = max(peak, height)
        _persist(folder / "truth.msgpack", _encode({"oracle": truth}, scope))
        return after

    def reobserve(cursor: StatefulChunkBackend) -> Observation:
        folder = directory / f"empty-set-reobserve-{cursor.decisions:04d}"
        folder.mkdir()
        status, after = _tick(
            folder, session, cursor.observation, {"reason": "one empty-set recovery tick"}, scope
        )
        if after is None:
            raise RuntimeError(f"stateful empty-set re-observation stopped: {status}")
        return after

    cursor = StatefulChunkBackend(
        observation,
        entry.scene,
        entry.seed,
        horizon=config.pipeline.horizon_steps,
        max_ticks=config.bounds.max_steps,
        decide=decide,
        advance=advance,
        truth=session.client.get_ground_truth,
        reobserve=reobserve if scored else None,
    )
    result = None
    verdict = None
    terminal = "PLACEMENT_FAILED"
    try:
        for attempt in range(2 if scored else 1):
            expert = ChunkExpert(
                scenario=entry.scene, trace_path=directory / f"expert-phases-{attempt}.jsonl"
            )
            result = expert.execute(cursor, entry.seed)
            if not result.home_reached or cursor.phase != "HOME":
                if result.rejections:
                    terminal = rejected_terminal
                break
            last, _ = observe(cursor.observation, directory / f"final-state-{attempt}.msgpack")
            verdict = verifier.check_observed(task, last)
            transition = (
                verifier.next_action(verdict)
                if scored
                else ("complete" if result.success and verdict.success else "abort")
            )
            _write_event(
                directory / f"verification-{attempt}.json",
                {
                    "tick": cursor.ticks,
                    "phase": cursor.phase,
                    "verdict": asdict(verdict),
                    "transition": transition,
                },
            )
            if transition == "complete":
                terminal = "COMPLETE"
                break
            terminal = "VERIFICATION_ABORT"
            if transition != "retry_once":
                break
            cursor.retry_verification()
    except TaskStop as stop:
        terminal = stop.terminal
    _write_event(
        directory / "terminal.json",
        {
            "status": terminal,
            "scored": scored,
            "confirmed_observed_ticks": cursor.ticks,
            "chunks": cursor.decisions,
            "can_max_rise_m": max(0.0, peak - start_z),
            "verification_retries": int(cursor.verification_retry_used),
            "empty_set_recoveries": int(cursor.empty_recovery_used),
            "expert_result": None if result is None else asdict(result),
            "placement_verification": None if verdict is None else asdict(verdict),
        },
    )
    return terminal


def run_live(output: Path, expected_head: str) -> str:
    """Run one owner-scoped clear VALIDATION diagnostic from a detached checkout."""
    import subprocess
    import sys
    import time

    from vlarobot.execution.episode_runner import launch_webots_session
    from vlarobot.execution.worldmind_heuristic_qualification import (  # noqa: PLC2701
        _ROOT,
        _WORLD,
        _run_owned,
        _sha,
        _verified_head,
    )
    from vlarobot.execution.worldmind_idle_fingers import RUNTIME, RUNTIME_SHA

    head = _verified_head(expected_head)
    ref = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=_ROOT, check=False, capture_output=True
    )
    if ref.returncode != 1:
        raise ValueError("stateful diagnostic requires a detached campaign worktree")
    world = _ROOT / "worlds/factory.wbt"
    if _sha(world.read_bytes()) != _WORLD or _sha(RUNTIME.read_bytes()) != RUNTIME_SHA:
        raise ValueError("stateful diagnostic world or executable pin mismatch")

    output.mkdir(exist_ok=False)
    status = output / "controller-status.json"
    started = time.time()
    _write_event(
        output / "launch.json",
        {
            "head": head,
            "episode": "validation-clear-i00-s0",
            "scope": STATEFUL_DIAGNOSTIC_SCOPE,
            "scored": False,
            "started_at_epoch_s": started,
        },
    )
    with launch_webots_session(
        webots=str(RUNTIME),
        world=world,
        trace_dir=output,
        trace_prefix="stateful",
        rpc_shutdown_on_exit=False,
        runtime_attestation_required=True,
        extra_env={
            "VLAROBOT_DIAGNOSTIC_STATUS_PATH": str(status),
            "VLAROBOT_WORLD_SHA256": _WORLD,
            "VLAROBOT_FINGER_IDLE_ONLY": "0",
            "VLAROBOT_WEBOTS_MODE": "server",
            "WEBOTS_PYTHON": sys.executable,
            "VLAROBOT_URDF_REGISTRATION_FILE": "calibration/urdf-base-train-v1.yaml",
        },
    ) as session:
        terminal = _run_owned(
            session,
            status,
            started,
            output / "episode",
            head,
            "validation-clear-i00-s0",
            STATEFUL_DIAGNOSTIC_SCOPE,
        )
    if session.process.poll() != 0 or (output / "episode/shutdown-failure.json").exists():
        raise RuntimeError("owned stateful session shutdown failed")
    return terminal


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    run_live(args.output.resolve(), args.expected_head)
