"""Protocol-compatible seeded candidate draws; predicted states are never fed back to VLA."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vlarobot.core.types import ActionChunk, TaskSpec, WorldState
from vlarobot.risk.comparison import sampled_candidates


class SeededCandidateGenerator:
    """Sample the installed proposer through an injected current-observation adapter."""

    def __init__(
        self,
        draw: Callable[[TaskSpec, WorldState, object, int], ActionChunk],
        *,
        seed: int,
        count: int,
        source: str,
    ) -> None:
        if seed < 0 or count < 1 or source not in ("vla", "scripted"):
            raise ValueError("explicit seed, candidate count and proposer source required")
        self.draw, self.seed, self.count, self.source = draw, seed, count, source
        self.decision_index = 0
        self.draw_log: list[dict[str, object]] = []

    def propose(self, task: TaskSpec, state: WorldState, memory: object) -> list[ActionChunk]:
        decision_seed = int(
            np.random.SeedSequence([self.seed, self.decision_index]).generate_state(1)[0]
        )
        chunks, seeds = sampled_candidates(
            lambda s: self.draw(task, state, memory, s), seed=decision_seed, count=self.count
        )
        if any(chunk.source != self.source for chunk in chunks):
            raise ValueError("proposer source mismatch; no hidden scripted/VLA replacement")
        self.draw_log.append(
            {
                "decision": self.decision_index,
                "seed": decision_seed,
                "draw_seeds": seeds,
                "source": self.source,
            }
        )
        self.decision_index += 1
        return list(chunks)
