"""Five frozen arm switches, deterministic candidate sampling and bounded recovery."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from vlarobot.core.types import ActionChunk, WorldState
from vlarobot.risk.gate import CandidateAssessment, RecoveryState, RiskGate
from vlarobot.world_model.rollout import Trajectory

ARMS = ("scripted_baseline", "vla_only", "vla_risk_veto", "vla_world_model", "full_system")


@dataclass(frozen=True)
class Selection:
    chunk: ActionChunk | None
    index: int | None
    rationale: str
    recovery_events: tuple[str, ...] = ()
    prediction_count: int = 0
    veto_count: int = 0
    score: float | None = None


def sampled_candidates(
    sample: Callable[[int], ActionChunk], *, seed: int, count: int
) -> tuple[tuple[ActionChunk, ...], tuple[int, ...]]:
    """K independent seeded policy draws; no unlogged perturbation of physical commands."""
    if count < 1 or seed < 0:
        raise ValueError("explicit nonnegative seed and positive candidate count required")
    seeds = tuple(map(int, np.random.default_rng(seed).integers(0, 2**31, size=count)))
    return tuple(sample(s) for s in seeds), seeds


class ComparisonPlanner:
    """Ablate prediction/heuristic veto while retaining downstream motor safety in every arm.

    J component meanings and validation-selected weights are injected, never guessed from
    the missing report PDF. No-veto means no predictive RiskGate, never no motor safety.
    """

    def __init__(
        self,
        arm: str,
        gate: RiskGate,
        *,
        predict: Callable[[WorldState, ActionChunk], Trajectory],
        assess: Callable[[WorldState, ActionChunk, Trajectory | None], CandidateAssessment],
    ) -> None:
        if arm not in ARMS:
            raise ValueError("unknown frozen comparison arm")
        self.arm, self.gate, self.predict, self.assess = arm, gate, predict, assess

    def select(self, state: WorldState, candidates: Sequence[ActionChunk]) -> Selection:
        if self.arm in ("scripted_baseline", "vla_only"):
            expected = "scripted" if self.arm == "scripted_baseline" else "vla"
            if any(c.source != expected for c in candidates):
                raise ValueError("baseline/VLA-only cannot silently substitute another proposer")
            return Selection(
                candidates[0] if candidates else None,
                0 if candidates else None,
                "first proposer chunk; motor safety remains authoritative",
            )
        assessments = []
        predicted = self.arm in ("vla_world_model", "full_system")
        for chunk in candidates:
            trajectory = self.predict(state, chunk) if predicted else None
            assessment = self.assess(state, chunk, trajectory)
            if trajectory is not None and self.arm == "full_system":
                assessment = CandidateAssessment(
                    (
                        *assessment.hard_checks,
                        ("prediction_complete", len(trajectory.states) == len(chunk.actions) + 1),
                        ("prediction_constraints", not trajectory.hard_reasons),
                    ),
                    assessment.components,
                )
            assessments.append(assessment)
        if self.arm == "vla_world_model":
            # Explicit no predictive veto ablation. Invalid J cannot be ranked as a real score.
            eligible = []
            for i, a in enumerate(assessments):
                if len(a.components) != 4 or not all(math.isfinite(v) for v in a.components):
                    continue
                score = sum(w * v for w, v in zip(self.gate.weights, a.components, strict=True))
                if math.isfinite(score):
                    eligible.append((i, score))
            if not eligible:
                return Selection(
                    None, None, "no finite prediction score", prediction_count=len(candidates)
                )
            chosen = (max if self.gate.maximize else min)(eligible, key=lambda p: p[1])
            return Selection(
                candidates[chosen[0]],
                chosen[0],
                f"prediction ranking without veto; J={chosen[1]:.9g}",
                prediction_count=len(candidates),
                score=chosen[1],
            )
        decision = self.gate.select(assessments)
        veto_count = sum(self.gate.select([a]).selected_index is None for a in assessments)
        index = decision.selected_index
        score = (
            None
            if index is None
            else sum(
                w * v for w, v in zip(self.gate.weights, assessments[index].components, strict=True)
            )
        )
        return Selection(
            None if index is None else candidates[index],
            index,
            decision.rationale,
            prediction_count=len(candidates) if predicted else 0,
            veto_count=veto_count,
            score=score,
        )

    def with_recovery(
        self,
        state: WorldState,
        primary: Callable[[WorldState], Sequence[ActionChunk]],
        *,
        reobserve: Callable[[], WorldState],
        alternate_family: Callable[[WorldState], Sequence[ActionChunk]],
    ) -> Selection:
        if self.arm in ("scripted_baseline", "vla_only"):
            return self.select(state, primary(state))
        recovery = RecoveryState()
        events, predictions, vetoes = [], 0, 0
        for stage in range(3):
            if stage == 1:
                state = reobserve()
            candidates = alternate_family(state) if stage == 2 else primary(state)
            result = self.select(state, candidates)
            predictions += result.prediction_count
            vetoes += result.veto_count
            if result.chunk is not None:
                return Selection(
                    result.chunk,
                    result.index,
                    result.rationale,
                    tuple(events),
                    predictions,
                    vetoes,
                    result.score,
                )
            events.append(recovery.no_safe())
        return Selection(
            None,
            None,
            "empty safe set after bounded recovery; abort",
            tuple(events),
            predictions,
            vetoes,
        )
