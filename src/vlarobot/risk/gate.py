"""ADR-0029 RiskGate mechanics; component definitions/weights must come from validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vlarobot.core.types import Decision


@dataclass(frozen=True)
class CandidateAssessment:
    hard_checks: tuple[tuple[str, bool | None], ...]
    # Caller supplies report-defined Jg, Jl, Jr, Ju; no invented success probabilities.
    components: tuple[float | None, float | None, float | None, float | None]


class RiskGate:
    def __init__(
        self,
        *,
        weights: tuple[float, float, float, float],
        maximize: bool,
        required_constraints: tuple[str, ...],
        inactive_components: tuple[int, ...] = (),
    ) -> None:
        if (
            not required_constraints
            or len(set(required_constraints)) != len(required_constraints)
            or not isinstance(maximize, bool)
            or len(weights) != 4
        ):
            raise ValueError("explicit unique hard constraints and ranking direction required")
        if not all(math.isfinite(w) and w >= 0 for w in weights) or not any(weights):
            raise ValueError("finite explicit validation-selected weights required")
        if (
            len(set(inactive_components)) != len(inactive_components)
            or any(index not in range(4) for index in inactive_components)
            or not any(weight for i, weight in enumerate(weights) if i not in inactive_components)
        ):
            raise ValueError("explicit comparable active objective components required")
        self.inactive_components = frozenset(inactive_components)
        self.weights, self.maximize = weights, maximize
        self.required_constraints = required_constraints

    def select(self, candidates: list[CandidateAssessment]) -> Decision:
        safe = []
        rejected = []
        for index, candidate in enumerate(candidates):
            failed = [name for name, result in candidate.hard_checks if result is not True]
            names = [name for name, _ in candidate.hard_checks]
            failed.extend(
                "missing:" + name for name in self.required_constraints if name not in names
            )
            if len(set(names)) != len(names):
                failed.append("duplicated-hard-constraints")
            if not candidate.hard_checks:
                failed.append("missing-hard-constraints")
            if len(candidate.components) != 4 or not all(
                v is None if i in self.inactive_components else v is not None and math.isfinite(v)
                for i, v in enumerate(candidate.components)
            ):
                failed.append("nonfinite-utility-components")
            if failed:
                rejected.append(f"{index}:" + ",".join(failed))
                continue
            score = sum(
                w * v
                for i, (w, v) in enumerate(zip(self.weights, candidate.components, strict=True))
                if i not in self.inactive_components and v is not None
            )
            if not math.isfinite(score):
                rejected.append(f"{index}:nonfinite-utility")
                continue
            safe.append((index, score))
        if not safe:
            return Decision(
                None, "empty-safe-set; vetoes=" + (";".join(rejected) or "no candidates")
            )
        # Stable first-candidate tie breaking, independent of worker completion order.
        chosen = (
            max(safe, key=lambda pair: pair[1])
            if self.maximize
            else min(safe, key=lambda pair: pair[1])
        )
        return Decision(
            chosen[0],
            f"candidate={chosen[0]}; J={chosen[1]:.9g}; "
            f"components={candidates[chosen[0]].components}; weights={self.weights}; "
            f"vetoes={rejected}",
        )


@dataclass
class RecoveryState:
    no_safe_count: int = 0
    verification_retries: int = 0

    def no_safe(self) -> str:
        ladder = ("reobserve", "alternate_candidate_family", "abort")
        result = ladder[min(self.no_safe_count, len(ladder) - 1)]
        self.no_safe_count += 1
        return result

    def verify(self, success: bool) -> str:
        if success:
            return "complete"
        if self.verification_retries == 0:
            self.verification_retries = 1
            return "retry_once"
        return "abort"
