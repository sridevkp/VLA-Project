"""WorldMind's stateful candidate-00 manipulation diagnostic.

The live expert owns waypoint/subphase control flow. The chunk boundary owns
candidate evidence, motor admission, simulator operations, and final verification.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

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
    ) -> None:
        self.observation = initial
        self.scenario, self.seed = scenario, seed
        self.horizon, self.max_ticks = horizon, max_ticks
        self.decide, self.advance, self.truth = decide, advance, truth
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

    def get_ground_truth(self) -> Any:
        return self.truth()

    def apply(self, action: Action) -> ApplyResult:
        return self._choose(action, continuation=False)

    def _choose(self, action: Action, *, continuation: bool) -> ApplyResult:
        if not self.initial_delivered or self.ticks >= self.max_ticks:
            raise RuntimeError("stateful episode control budget exhausted")
        self.armed = False
        result = self.decide(self, action, continuation)
        self.decisions += 1
        self.used = 0
        if result.accepted:
            self.target = action
            self.armed = True
        return result

    def step(self, count: int) -> Observation:
        if count != 1 or not self.armed or self.target is None:
            raise ValueError("stateful caller requires one tick under an accepted command")
        if self.ticks >= self.max_ticks:
            raise RuntimeError("stateful episode control budget exhausted")
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


def run_candidate_zero(
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
) -> str:
    """Execute the real WorldMind selection/receipt path with the live expert cursor."""
    scope = STATEFUL_DIAGNOSTIC_SCOPE
    active: dict[str, Any] = {}
    peak = float("-inf")
    start_z = session.client.get_ground_truth().object_poses[task.target].position.z
    limits = monitor.limits.joint_position
    settings = config.adapter.jitter

    def decide(cursor: StatefulChunkBackend, action: Action, continuation: bool) -> ApplyResult:
        if cursor.decisions >= config.bounds.max_decisions:
            raise RuntimeError("stateful episode decision budget exhausted")
        folder = directory / f"chunk-{cursor.decisions:04d}"
        folder.mkdir()
        state, _ = observe(cursor.observation, folder / "observed-state.msgpack")
        index = 0

        def draw(_task: Any, _state: Any, _memory: Any, seed: int) -> ActionChunk:
            nonlocal index
            chosen = action
            if index and not continuation and isinstance(action, JointTarget):
                rng = np.random.default_rng(seed)
                joints = {}
                for name, value in action.q.items():
                    lo, hi = limits[name]
                    for _ in range(settings.max_attempts):
                        delta = float(rng.normal(0.0, settings.joint_std_rad))
                        if abs(delta) <= settings.joint_limit_rad and lo <= value + delta <= hi:
                            joints[name] = value + delta
                            break
                    else:
                        raise ValueError("stateful waypoint jitter rejection budget exhausted")
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
        path = folder / "decision.msgpack"
        snapshot = _persist(
            path,
            _encode(
                {
                    "generation": asdict(generation),
                    "pool": [_row_data(row) for row in rows],
                    "selected": rows[0].candidate_id if selected == 0 else None,
                    "phase": cursor.phase,
                    "continuation": continuation,
                    "episode_tick": cursor.ticks,
                    "observation": cursor.observation,
                    "expiration": "earliest expert waypoint completion or horizon",
                },
                scope,
            ),
        )
        if selected != 0:
            return ApplyResult(False, "reference candidate-00 is ineligible")
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
            return motor_verdict
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
        return result

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

    cursor = StatefulChunkBackend(
        observation,
        entry.scene,
        entry.seed,
        horizon=config.pipeline.horizon_steps,
        max_ticks=config.bounds.max_steps,
        decide=decide,
        advance=advance,
        truth=session.client.get_ground_truth,
    )
    expert = ChunkExpert(scenario=entry.scene, trace_path=directory / "expert-phases.jsonl")
    result = expert.execute(cursor, entry.seed)
    verdict = None
    if result.home_reached:
        last, _ = observe(cursor.observation, directory / "final-state.msgpack")
        verdict = verifier.check_observed(task, last)
    success = bool(result.success and verdict is not None and verdict.success)
    terminal = "COMPLETE" if success else "STATEFUL_TASK_FAILED"
    _write_event(
        directory / "terminal.json",
        {
            "status": terminal,
            "scored": False,
            "confirmed_observed_ticks": cursor.ticks,
            "chunks": cursor.decisions,
            "can_max_rise_m": max(0.0, peak - start_z),
            "expert_result": asdict(result),
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
