"""Reconcile the accepted scripted VALIDATION archive with the frozen trial keys."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.worldmind_config import FROZEN_BASE_SEED, FROZEN_EPISODE_COUNT, FROZEN_SPLIT

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

# The canonical physical baseline reviewed in VALIDATION_AND_PLANNING_REPORT.md.
BASELINE_ARTIFACT_SHA256 = "b1abca1c9ac90be542a37136d9ca4f3a34293622f6cea5b98bdb5b86dc092650"


@dataclass(frozen=True)
class ValidationCoverage:
    """Recorded outcomes and planned keys absent from the supplied evidence."""

    measured_episodes: tuple[str, ...]
    missing_episodes: tuple[str, ...]
    successes: int
    recorded_failure_states: tuple[tuple[str, int], ...]

    @property
    def complete(self) -> bool:
        return not self.missing_episodes


def reconcile_validation_outcomes(
    rows: Iterable[Mapping[str, Any]],
) -> ValidationCoverage:
    """Check trial keys and recorded terminal outcomes, without determining source eligibility."""
    plan = build_episode_plan(FROZEN_SPLIT, FROZEN_BASE_SEED, n=0, allow_test=False)
    if len(plan) != FROZEN_EPISODE_COUNT:
        raise ValueError("unexpected frozen VALIDATION plan length")
    expected = {
        item.index: (item.scene.name.rsplit("-s", 1)[0], item.seed, item.scene.name)
        for item in plan
    }
    measured: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        index = row.get("index")
        if type(index) is not int or index not in expected or index in measured:
            raise ValueError("duplicate or unexpected VALIDATION trial index")
        scene, seed, _ = expected[index]
        if row.get("scene") != scene or type(row.get("seed")) is not int or row["seed"] != seed:
            raise ValueError("trial scene or seed does not match frozen VALIDATION")
        success, failure_state = row.get("success"), row.get("failure_state")
        if type(success) is not bool or (
            failure_state is not None
            if success
            else not isinstance(failure_state, str) or not failure_state.strip()
        ):
            raise ValueError("record lacks a consistent terminal outcome")
        measured[index] = row

    failures = Counter(row["failure_state"] for row in measured.values() if not row["success"])
    return ValidationCoverage(
        tuple(episode for i, (_, _, episode) in expected.items() if i in measured),
        tuple(episode for i, (_, _, episode) in expected.items() if i not in measured),
        sum(row["success"] for row in measured.values()),
        tuple(sorted(failures.items())),
    )


def load_canonical_baseline_coverage(path: Path) -> ValidationCoverage:
    """Require the reviewed canonical baseline digest before attributing recorded results."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_ARTIFACT_SHA256:
        raise ValueError("not the reviewed canonical scripted baseline artifact")
    artifact = json.loads(raw)
    coverage = reconcile_validation_outcomes(artifact["records"])
    if not coverage.complete or artifact["episodes"] != FROZEN_EPISODE_COUNT:
        raise ValueError("canonical scripted baseline coverage is incomplete")
    return coverage
