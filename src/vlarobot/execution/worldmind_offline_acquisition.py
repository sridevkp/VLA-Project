"""Capture one ADR-0083 fixture decision without an evaluator or simulator entry point."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vlarobot.backends.fake import FakeBackend
from vlarobot.core.types import (
    Action,
    ActionChunk,
    ApplyResult,
    Observation,
    Pose3D,
    RobotState,
    WorldState,
)
from vlarobot.eval.report_contract import COMPONENTS
from vlarobot.eval.runner import build_episode_plan
from vlarobot.eval.worldmind_offline_evidence import (
    OfflineEvidence,
    evidence_digest,
    validate_offline_observed_state,
    validate_offline_records,
    write_evidence,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.worldmind_world_model_comparison import ARM, WorldModelComparisonFixture
from vlarobot.planning.execution import execute_first
from vlarobot.risk.comparison import ComparisonPlanner, Selection
from vlarobot.risk.gate import CandidateAssessment
from vlarobot.world_model.rollout import Trajectory

if TYPE_CHECKING:
    from vlarobot.execution.worldmind_offline_boundary import OfflineBoundary

_POLICY = "adr0083_fixture_eq38_motor_only_execute_first"


@dataclass(frozen=True)
class AcquisitionResult:
    """Verified offline bundle plus the original selection when planning returned one."""

    evidence: OfflineEvidence
    selection: Selection | None


@dataclass
class _ExecutionCapture:
    backend: FakeBackend
    monitor: SafetyMonitor
    motor: ApplyResult | None = None
    applied: str = "not_attempted"
    apply_reason: str | None = None
    stepped: str = "not_executed"
    observed: Observation | None = None
    boundary: OfflineBoundary | None = None
    records: dict[str, Any] | None = None
    recording_failed: bool = False

    def _record(self, event: str, **detail: Any) -> None:
        if self.boundary is not None:
            try:
                self.boundary.record(event, **detail)
            except Exception:
                self.recording_failed = True
                raise

    def validate(self, action: Action, state: RobotState) -> ApplyResult:
        verdict = self.monitor.validate(action, state)
        if not isinstance(verdict, ApplyResult) or type(verdict.accepted) is not bool:
            raise ValueError("invalid fixture motor verdict")
        self.motor = verdict
        return verdict

    def apply(self, action: Action) -> ApplyResult:
        if self.boundary is not None:
            try:
                self.boundary.prepare_apply(self.records, action, self.motor)
            except OSError:
                self.recording_failed = True
                raise
        self.applied = "unknown"
        try:
            result = (
                self.backend.apply(action)
                if self.boundary is None
                else self.boundary.submit(action)
            )
            if not isinstance(result, ApplyResult) or type(result.accepted) is not bool:
                raise ValueError("invalid fixture apply outcome")
        except Exception as exc:
            if self.boundary is not None and not self.boundary.submitted:
                self.applied = "not_attempted"
            self._record("apply_result", outcome=self.applied, reason=_reason(exc))
            raise
        self.applied = "accepted" if result.accepted else "rejected"
        self.apply_reason = result.reason
        self._record("apply_result", outcome=self.applied, reason=self.apply_reason)
        return result

    def step(self, n_steps: int) -> Observation:
        if self.boundary is not None:
            try:
                self.boundary.prepare_tick(n_steps)
            except OSError:
                self.recording_failed = True
                raise
        self.stepped = "unknown"
        try:
            if self.boundary is None:
                observation = self.backend.step(n_steps)
                if not isinstance(observation, Observation):
                    raise ValueError("fixture step returned no authoritative result")
                self.stepped = "confirmed"
            else:
                outcome = self.boundary.tick()
                self.stepped = outcome.status
                observation = outcome.observation
        except Exception as exc:
            if self.boundary is not None and not self.boundary.ticked:
                self.stepped = "not_executed"
            self._record("tick_result", outcome=self.stepped, reason=_reason(exc))
            raise
        self._record("tick_result", outcome=self.stepped)
        if self.stepped != "confirmed" or not isinstance(observation, Observation):
            raise ValueError("fixture tick or observation unavailable")
        self.observed = observation
        return observation


def _reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _candidate_rows(candidates: tuple[ActionChunk, ...], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index, (chunk, seed) in enumerate(zip(candidates, seeds, strict=True)):
        cid = f"candidate-{index:02d}"
        rows.append(
            {
                "id": cid,
                "index": index,
                "seed": seed,
                "draw_digest": evidence_digest({"id": cid, "seed": seed, "chunk": chunk}),
                "payload": {"chunk": deepcopy(chunk), "states": []},
                "rollout": "missing_prediction",
                "rollout_reason": "planner did not reach this draw",
                "checks": {},
                "components": None,
                "disposition": "unevaluable",
            }
        )
    return rows


def _plan(
    fixture: WorldModelComparisonFixture,
    state: WorldState,
    candidates: tuple[ActionChunk, ...],
    records: dict[str, Any],
) -> Selection | None:
    rows = records["candidates"]
    failures = records["failures"]
    original = fixture.planner
    index = -1

    def predict(current: WorldState, chunk: ActionChunk) -> Trajectory:
        nonlocal index
        index += 1
        row = rows[index]
        try:
            result = original.predict(current, chunk)
        except Exception as exc:
            row.update(rollout="failed_rollout", rollout_reason=_reason(exc))
            failures["failed_rollout"] = _reason(exc)
            raise
        row["prediction_hard_reasons"] = result.hard_reasons
        try:
            if result.dt_s != chunk.dt or len(result.states) > len(chunk.actions) + 1:
                raise ValueError("unaligned fixture prediction")
            initial = records["input"]["state"]
            if result.states and evidence_digest({"state": result.states[0]}) != evidence_digest(
                {"state": initial}
            ):
                raise ValueError("predicted initial state differs from the captured decision")
            for node in result.states:
                validate_offline_observed_state(node, records["input"]["expected_objects"])
                row["payload"]["states"].append(deepcopy(node))
        except Exception as exc:
            row.update(rollout="failed_rollout", rollout_reason=_reason(exc))
            failures["failed_rollout"] = _reason(exc)
            raise
        if len(result.states) == len(chunk.actions) + 1:
            row["rollout"] = "complete"
            row.pop("rollout_reason")
        else:
            row.update(rollout="failed_rollout", rollout_reason="prediction ended before H+1")
            failures["failed_rollout"] = row["rollout_reason"]
        return result

    def assess(
        current: WorldState, chunk: ActionChunk, trajectory: Trajectory | None
    ) -> CandidateAssessment:
        result = original.assess(current, chunk, trajectory)
        row = rows[index]
        # ADR-0083 supplies no predictive hard checks. Preserve the actual returned tuple.
        row["assessment_checks"] = result.hard_checks
        if len(result.components) == len(COMPONENTS) and all(map(math.isfinite, result.components)):
            values = dict(zip(COMPONENTS, result.components, strict=True))
            if records["bindings"]:
                row["components"] = values
            else:
                row["fixture_components"] = values
        else:
            row["component_failure"] = "no finite four-component assessment"
        return result

    selection = None
    try:
        selection = ComparisonPlanner(ARM, original.gate, predict=predict, assess=assess).select(
            state, candidates
        )
    except Exception as exc:
        failures.setdefault("interrupted_execution", _reason(exc))
    for row in rows:
        if row["rollout"] == "missing_prediction":
            failures.setdefault("missing_prediction", "the planner stopped before all rollouts")
        row["rollout_digest"] = evidence_digest(
            {
                "draw_digest": row["draw_digest"],
                "states": row["payload"]["states"],
            }
        )

    if selection is None:
        return None

    parameters = {"weights": tuple(original.gate.weights), "maximize": original.gate.maximize}
    scores = {}
    for row in rows:
        values = row.get("fixture_components", row["components"])
        if values is None:
            continue
        score = sum(
            w * values[name] for w, name in zip(parameters["weights"], COMPONENTS, strict=True)
        )
        if math.isfinite(score):
            scores[row["id"]] = score

    # Keep the planner's actual selection even when the acquisition schema prohibits execution.
    selected = None if selection.index is None else rows[selection.index]["id"]
    records["recovery"] = list(selection.recovery_events)
    records["planner_selection"] = {
        "id": selected,
        "score": selection.score,
        "reason": selection.rationale,
    }
    if selected is None and not failures:
        failures["interrupted_execution"] = selection.rationale
    if failures:
        return selection
    for row in rows:
        if row["id"] in scores:
            row["disposition"] = "selected" if row["id"] == selected else "unselected"
    records["selected"] = selected
    records["fixture_rank_parameters"] = parameters
    records["ranking"] = {
        "parameter_ref": evidence_digest(parameters),
        "maximize": parameters["maximize"],
        "scores": scores,
        "order": sorted(scores, key=scores.__getitem__, reverse=parameters["maximize"]),
    }
    return selection


def acquire_fixture_decision(
    path: Path,
    fixture: WorldModelComparisonFixture,
    backend: FakeBackend,
    observation: Observation,
    *,
    episode_id: str,
    seed: int,
    goal_base: Pose3D,
    source_hashes: Mapping[str, str],
    binding_refs: Mapping[str, str] | None = None,
    boundary: OfflineBoundary | None = None,
) -> AcquisitionResult:
    """Capture reset decision zero, an optional FakeBackend tick, and any valid failure evidence.

    Goal and binding references are explicit fixture declarations. Objective callables receive
    the captured state, chunk and trajectory unchanged. No calibration artifact is accepted.
    """
    if (
        not isinstance(backend, FakeBackend)
        or fixture.planner.arm != ARM
        or fixture.provenance.get("arm") != ARM
        or fixture.provenance.get("execution_scope") != "fixture_only"
        or fixture.provenance.get("calibration_status") != "fixture_inputs_only"
        or fixture.provenance.get("scientific_execution_approved") != "false"
    ):
        raise ValueError("acquisition requires the unapproved ADR-0083 FakeBackend fixture")
    plan = build_episode_plan("validation", 0, n=0, allow_test=False)
    episode_seeds = {item.scene.name: item.seed for item in plan}
    plan_payload = [
        {"index": item.index, "scene": asdict(item.scene), "seed": item.seed} for item in plan
    ]
    actual_hashes = {
        "config": hashlib.sha256(fixture.config.source_path.read_bytes()).hexdigest(),
        "definitions": hashlib.sha256(fixture.config.report.definitions.read_bytes()).hexdigest(),
        "plan": hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if any(source_hashes.get(name) != value for name, value in actual_hashes.items()):
        raise ValueError("fixture source provenance does not match config, definitions or plan")
    if type(seed) is not int or episode_seeds.get(episode_id) != seed:
        raise ValueError("fixture episode and seed must match frozen VALIDATION")
    if path.exists():
        raise FileExistsError(path)
    decision_seed = int(np.random.SeedSequence([seed, 0]).generate_state(1)[0])
    draws = np.random.default_rng(decision_seed).integers(0, 2**31, 32).tolist()
    outcome = {"episode": episode_id, "status": "in_progress", "classification": None}
    failures = {"incomplete_recording": "fixture capture is in progress"}
    records: dict[str, Any] = {
        "arm": ARM,
        "split": "validation",
        "proposer": "scripted_chunk_adapter",
        "episode": episode_id,
        "decision_index": 0,
        "decision_id": f"{episode_id}-d00",
        "decision_seed": decision_seed,
        "draw_seeds": draws,
        "source_hashes": dict(source_hashes),
        "provenance": dict(fixture.provenance),
        "policy": _POLICY,
        "episode_outcome": outcome,
        "outcome_ref": evidence_digest(outcome),
        "recovery": [],
        "timings": {},
        "failures": failures,
        "input": None,
        "bindings": dict(binding_refs or {}),
        "candidates": [],
        "selected": None,
        "ranking": None,
        "execution": [],
        "pairing": {"status": "not_recorded", "valid_steps": 0, "boundary": "before_execution"},
    }

    # Validate provenance, the episode key, and mandatory record metadata before any backend call.
    def finish(selection: Selection | None) -> AcquisitionResult:
        if failures:
            outcome.update(status="failed", classification=next(iter(failures)))
            records["outcome_ref"] = evidence_digest(outcome)
            if records["pairing"]["boundary"] == "before_execution":
                records["pairing"]["boundary"] = next(iter(failures))
        return AcquisitionResult(write_evidence(path, records), selection)

    validate_offline_records(records)
    if records["bindings"] and set(records["bindings"]) != set(COMPONENTS):
        raise ValueError("fixture binding references must cover all four components")

    try:
        task, state = fixture.prepare_fixture_input(backend, observation)
        records["input"] = {
            "state": deepcopy(state),
            "task": task,
            "goal": goal_base,
            "goal_provenance": "explicit fixture task-goal declaration",
            "expected_objects": (task.target,),
            "scene_complete": True,
            "time_s": observation.t,
            "frames": {"scene": "base", "goal": "base", "base_pose_world": "world"},
        }
        validate_offline_records(records)
        if boundary is not None:
            boundary.admit(task, state, observation)
    except Exception as exc:
        records["input"] = None
        failures.clear()
        failures["invalid_input"] = _reason(exc)
        return finish(None)
    failures.clear()

    try:
        generation, candidates = fixture.generate_fixture_pool(
            task, state, episode_id=episode_id, seed=seed, decision_index=0
        )
        if generation.draw_seed != decision_seed or list(generation.draw_seeds) != draws:
            raise ValueError("actual generator seeds disagree with the frozen schedule")
        records["candidates"] = _candidate_rows(candidates, draws)
    except Exception as exc:
        failures["failed_generation"] = _reason(exc)
        return finish(None)

    selection = _plan(fixture, state, candidates, records)
    validate_offline_records(records)
    if selection is not None and records["selected"] is not None and not failures:
        capture = _ExecutionCapture(backend, fixture.monitor, boundary=boundary, records=records)
        after = None
        try:
            execute_first(selection, observation, cast(Any, capture), cast(SafetyMonitor, capture))
            if capture.observed is not None:
                observed_task, observed_state = fixture.prepare_fixture_input(
                    backend, capture.observed
                )
                if boundary is not None:
                    boundary.admit(observed_task, observed_state, capture.observed)
                validate_offline_observed_state(
                    observed_state, records["input"]["expected_objects"]
                )
                if not math.isclose(capture.observed.t, observation.t + selection.chunk.dt):
                    raise ValueError("misaligned fixture observation time")
                after = observed_state
        except Exception as exc:
            failure = (
                "unknown_backend_outcome"
                if "unknown" in (capture.applied, capture.stepped)
                else "incomplete_recording"
                if capture.stepped == "confirmed" or capture.recording_failed
                else "interrupted_execution"
            )
            failures[failure] = _reason(exc)
        if capture.motor is not None:
            records["execution"] = [
                {
                    "index": 0,
                    "action": selection.chunk.actions[0],
                    "before": records["input"]["state"],
                    "motor": capture.motor,
                    "apply": capture.applied,
                    "apply_reason": capture.apply_reason,
                    "step": capture.stepped,
                    "after": after,
                    "observation_time_s": observation.t,
                    "result_time_s": None if after is None else capture.observed.t,
                }
            ]
        if after is not None:
            records["pairing"] = {
                "status": "truncated",
                "valid_steps": 1,
                "candidate_id": records["selected"],
                "boundary": "fixture_execute_first",
            }
        elif capture.applied == "rejected":
            records["pairing"]["boundary"] = "backend_rejected"
        elif capture.motor is not None and not capture.motor.accepted:
            records["pairing"]["boundary"] = "motor_veto"
        elif failures:
            records["pairing"]["boundary"] = next(iter(failures))
    return finish(selection)
