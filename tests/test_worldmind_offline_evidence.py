"""NON-SCIENTIFIC offline evidence fixtures. No simulator or calibration evaluator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from vlarobot.backends.worldmind_bindings import production_worldmind_bindings
from vlarobot.core.types import (
    ApplyResult,
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    TaskSpec,
    Vec3,
    WorldState,
)
from vlarobot.eval.decision_adapters import GenerationObservation
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config
from vlarobot.eval.worldmind_heuristic_reporting import heuristic_table51_population
from vlarobot.eval.worldmind_offline_evidence import (
    _decode,
    _encode,
    evidence_digest,
    read_evidence,
    write_evidence,
)
from vlarobot.execution.scripted_chunk_adapter import (
    JointJitter,
    ScriptedChunkAdapter,
    ScriptedWaypoint,
)
from vlarobot.execution.worldmind_heuristic_comparison import HeuristicCandidate, HeuristicDecision
from vlarobot.planning.candidates import SeededCandidateGenerator
from vlarobot.world_model.kinematic import VelocityLimitedKinematics
from vlarobot.world_model.rollout import AnalyticSurrogate

JOINT = "arm_7_joint"
EPISODE = "validation-clear-i00-s0"
DECISION = f"{EPISODE}-d00"


def _pose(x: float) -> Pose3D:
    return Pose3D(Vec3(x, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))


def _seal_rollout(row):
    row["rollout_digest"] = evidence_digest(
        {"draw_digest": row["draw_digest"], "states": row["payload"]["states"]}
    )


def _fixture(holding: bool | None = True):
    state = WorldState(
        SceneState([ObjectState("CAN", "can", _pose(0.5), Vec3(0.1, 0.1, 0.1), 1.0)]),
        RobotState(
            {JOINT: 0.0},
            {JOINT: 0.0},
            Pose2D(0.0, 0.0, 0.0),
            _pose(0.0),
            0.02,
            holding,
            Pose3D(Vec3(0.0, 0.0, 0.2), (0.6, 0.0, 0.0, 0.8)),
            {"raw": (b"fixture", -0.0)},
        ),
    )
    task = TaskSpec("Move the CAN", "CAN", "fixture goal")

    config = load_worldmind_diagnostic_config(
        Path(__file__).resolve().parents[1] / "configs/worldmind_diagnostic.approved.yaml"
    )
    bindings = production_worldmind_bindings()
    monitor = bindings.safety[config.safety.binding](config, bindings.inputs["safety"])
    solver = bindings.pose_solvers[config.adapter.pose_solver_binding]
    jitter = config.adapter.jitter
    adapter = ScriptedChunkAdapter(
        (
            ScriptedWaypoint("FIXTURE_A", 25, joints={JOINT: 0.02}),
            ScriptedWaypoint("FIXTURE_B", 25, joints={JOINT: 0.03}),
        ),
        solver=solver,
        monitor=monitor,
        horizon_steps=config.pipeline.horizon_steps,
        control_dt_s=config.pipeline.control_dt_s,
        jitter=JointJitter(jitter.joint_std_rad, jitter.joint_limit_rad, jitter.max_attempts),
        max_joint_step_rad=solver.execution_max_joint_step_rad,
        position_tolerance_m=config.adapter.ik_position_tolerance_m,
        orientation_tolerance_rad=config.adapter.ik_orientation_tolerance_rad,
    )
    generator = SeededCandidateGenerator(adapter.draw(), seed=0, count=32, source="scripted")
    pool = generator.propose(task, state, None)
    draw_log = generator.draw_log[0]
    predictor = AnalyticSurrogate(
        VelocityLimitedKinematics({JOINT: 1.0}, lambda joints: _pose(joints[JOINT])),
        inverse=lambda pose, robot: None,
        gripper_speed_m_s=0.03,
        gravity_m_s2=9.81,
        support_z_m={"CAN": 0.45},
        validate=lambda action, robot: monitor.validate(action, robot).accepted,
    )
    candidates = []
    for index, (seed, chunk) in enumerate(zip(draw_log["draw_seeds"], pool, strict=True)):
        candidate_id = f"candidate-{index:02d}"
        trajectory = predictor.predict_trajectory(state, chunk)
        row = {
            "id": candidate_id,
            "index": index,
            "seed": seed,
            "rollout": "complete",
            "draw_digest": evidence_digest({"id": candidate_id, "seed": seed, "chunk": chunk}),
            "payload": {"chunk": chunk, "states": list(trajectory.states)},
            "checks": {},
            "components": None,
            "disposition": "unselected" if index else "selected",
        }
        _seal_rollout(row)
        candidates.append(row)

    # Explicit synthetic observation, deliberately distinct from the predicted state.
    after = deepcopy(state)
    after.robot.joint_pos[JOINT] = 0.001
    outcome = {"episode": EPISODE, "status": "in_progress", "classification": None}
    return {
        "arm": "vla_world_model",
        "split": "validation",
        "proposer": "scripted_chunk_adapter",
        "episode": EPISODE,
        "decision_index": 0,
        "decision_id": DECISION,
        "decision_seed": draw_log["seed"],
        "draw_seeds": list(draw_log["draw_seeds"]),
        "source_hashes": {
            key: evidence_digest({"fixture_role": key})
            for key in ("source", "config", "plan", "definitions")
        },
        "policy": "fixture_only_explicit_candidate",
        "episode_outcome": outcome,
        "outcome_ref": evidence_digest(outcome),
        "recovery": [],
        "timings": {"planning_end_to_end_s": 0.0},
        "failures": {},
        "input": {
            "state": state,
            "task": task,
            "goal": _pose(0.8),
            "expected_objects": ("CAN",),
            "scene_complete": True,
            "time_s": 0.032,
            "frames": {"scene": "base", "goal": "base", "base_pose_world": "world"},
        },
        "bindings": {},
        "ranking": None,
        "candidates": candidates,
        "selected": "candidate-00",
        "execution": [
            {
                "index": 0,
                "action": pool[0].actions[0],
                "before": state,
                "motor": monitor.validate(pool[0].actions[0], state.robot),
                "apply": "accepted",
                "step": "confirmed",
                "after": after,
                "observation_time_s": 0.032,
                "result_time_s": 0.064,
            }
        ],
        "pairing": {
            "status": "truncated",
            "valid_steps": 1,
            "candidate_id": "candidate-00",
            "boundary": "fixture_one_tick",
        },
    }


@pytest.fixture(scope="module")
def records():
    return _fixture()


def test_exact_typed_round_trip_reproducible_digest(records, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    written = write_evidence(first, records)
    recovered = read_evidence(first, expected_digest=written.digest)
    assert recovered.records == records
    assert evidence_digest(recovered.records) == evidence_digest(records)
    assert written.digest == write_evidence(second, recovered.records).digest
    assert first.read_bytes() == second.read_bytes()
    independently_built = tmp_path / "independent"
    assert write_evidence(independently_built, _fixture()).digest == written.digest
    assert independently_built.read_bytes() == first.read_bytes()
    assert len(recovered.records["candidates"]) == 32
    assert all(len(row["payload"]["states"]) == 51 for row in recovered.records["candidates"])
    assert recovered.pool_complete
    assert not recovered.ranking_evidence_complete
    assert recovered.blockers == ("missing_objective_binding_or_components", "ranking_not_recorded")
    with pytest.raises(FileExistsError):
        write_evidence(first, records)


def test_unknown_holding_round_trips(tmp_path):
    records = _fixture(None)
    result = write_evidence(tmp_path / "unknown", records)
    assert result.records["input"]["state"].robot.gripper_holding is None
    assert result.pool_complete


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("quaternion", "base_pose_world"),
        ("pose", "base_pose_world"),
        ("scene", "scene objects"),
        ("duplicate", "ID, order or seed"),
        ("seed", "ID, order or seed"),
        ("payload_order", "ordered draw"),
        ("prediction_order", "ordered rollout"),
        ("partial", "status or horizon"),
        ("action", "applied command"),
        ("outcome", "outcome reference"),
    ],
)
def test_invalid_inputs_fail_closed(records, tmp_path, defect, reason):
    body = deepcopy(records)
    inp = body["input"]["state"]
    row = body["candidates"][0]
    if defect == "quaternion":
        inp.robot.base_pose_world = replace(inp.robot.base_pose_world, orientation=(0.0,) * 4)
    elif defect == "pose":
        inp.robot.base_pose_world = None
    elif defect == "scene":
        inp.scene.objects.clear()
    elif defect == "duplicate":
        body["candidates"][1]["id"] = row["id"]
    elif defect == "seed":
        row["seed"] += 1
    elif defect == "payload_order":
        row["payload"]["chunk"].actions.reverse()
    elif defect == "prediction_order":
        row["payload"]["states"].reverse()
    elif defect == "partial":
        row["payload"]["states"].pop()
        _seal_rollout(row)
    elif defect == "action":
        body["execution"][0]["action"] = JointTarget({JOINT: 0.0})
    else:
        body["episode_outcome"]["status"] = "failed"
    path = tmp_path / "invalid"
    with pytest.raises(ValueError, match=reason):
        write_evidence(path, body)
    assert not path.exists()


def test_partial_rollout_is_valid_failure_evidence(records, tmp_path):
    body = deepcopy(records)
    row = body["candidates"][-1]
    row["rollout"] = "failed_rollout"
    row["rollout_reason"] = "injected interruption at the final model step"
    row["disposition"] = "unevaluable"
    row["payload"]["states"].pop()
    _seal_rollout(row)
    body["failures"] = {"failed_rollout": "fixture interruption"}
    path = tmp_path / "partial"
    write_evidence(path, body)
    result = read_evidence(path)
    assert result.pool_complete
    assert not result.ranking_evidence_complete
    assert "failed_rollout" in result.blockers
    assert len(result.records["candidates"][-1]["payload"]["states"]) == 50


def test_failed_generation_and_interrupted_recording_are_distinct(records, tmp_path):
    body = deepcopy(records)
    body.update(
        failures={"failed_generation": "injected failure", "incomplete_recording": "missing tail"},
        candidates=[],
        selected=None,
        execution=[],
        pairing={"status": "not_recorded", "valid_steps": 0, "boundary": "generation"},
    )
    path = tmp_path / "failure"
    result = write_evidence(path, body)
    assert not result.pool_complete
    assert {"failed_generation", "incomplete_recording"} <= set(result.blockers)
    path.write_bytes(path.read_bytes()[:-7])
    with pytest.raises(ValueError):
        read_evidence(path)


def test_unknown_tick_does_not_claim_observation(records, tmp_path):
    body = deepcopy(records)
    body["execution"][0].update(step="unknown", after=None, result_time_s=None)
    body["failures"] = {"unknown_backend_outcome": "injected transport failure"}
    body["pairing"].update(status="not_recorded", valid_steps=0, boundary="unknown_tick")
    path = tmp_path / "unknown"
    result = write_evidence(path, body)
    assert "unknown_backend_outcome" in result.blockers
    body["execution"][0]["after"] = records["execution"][0]["after"]
    body["execution"][0]["result_time_s"] = 0.064
    with pytest.raises(ValueError, match="without confirmed step"):
        write_evidence(tmp_path / "fabricated", body)


def test_corrupted_inner_reference_rejected_even_with_rehashed_envelope(records, tmp_path):
    import hashlib

    path = tmp_path / "corrupt"
    write_evidence(path, records)
    envelope = _decode(path.read_bytes())
    body = _decode(envelope["body"])
    key = body["index"]["candidates"][0]["payload"]
    body["payloads"][key] += b"x"
    envelope["body"] = _encode(body)
    envelope["sha256"] = hashlib.sha256(envelope["body"]).hexdigest()
    path.write_bytes(_encode(envelope))
    with pytest.raises(ValueError, match="corrupted evidence reference"):
        read_evidence(path)


def test_heuristic_bridge_uses_frozen_population_and_excludes_motor_veto(records):
    rows = []
    for i, seed in enumerate(records["draw_seeds"]):
        finite = False if i == 0 else True
        first = None if i == 1 else ApplyResult(i != 2)
        label = "rejected" if i < 3 else "selected" if i == 3 else "eligible"
        rows.append(
            HeuristicCandidate(
                f"candidate-{i:02d}",
                i,
                seed,
                records["candidates"][i]["payload"]["chunk"],
                finite,
                first,
                label,
            )
        )
    generation = GenerationObservation(
        EPISODE, DECISION, "scripted", 0, records["decision_seed"], tuple(records["draw_seeds"]), 32
    )
    decision = HeuristicDecision(
        generation,
        tuple(rows),
        "candidate-03",
        "abort",
        "fixture",
        ApplyResult(False, "motor veto"),
    )
    result = heuristic_table51_population([decision])
    assert (result.hard_vetoed, result.evaluated, result.invalid, result.unevaluable) == (
        1,
        30,
        1,
        1,
    )
    assert result.rate == 1 / 30
    assert result.motor_vetoes == 1
    assert result.prediction_error.status == "not_applicable"
    assert result.prediction_error.value is None
    assert result.prediction_error.reason == "arm_has_no_prediction"
    assert result.prediction_error.valid_paired_steps == 0
    assert heuristic_table51_population([]).rate is None
    with pytest.raises(ValueError, match="duplicate"):
        heuristic_table51_population([decision, decision])


def test_partial_pool_cannot_reach_selection(records, tmp_path):
    body = deepcopy(records)
    body["candidates"].pop()
    body["failures"] = {"partial_pool": "injected generator interruption"}
    destination = tmp_path / "partial_pool"
    with pytest.raises(ValueError, match="selection from an incomplete pool"):
        write_evidence(destination, body)
    body["candidates"][0]["disposition"] = "unselected"
    body.update(
        selected=None,
        execution=[],
        pairing={"status": "not_recorded", "valid_steps": 0, "boundary": "partial_pool"},
    )
    assert not write_evidence(destination, body).pool_complete


def test_command_deviation_requires_non_comparable_pairing(records, tmp_path):
    body = deepcopy(records)
    body["execution"][0]["action"] = JointTarget({JOINT: 0.001})
    body["execution"][0]["deviation"] = "injected different command"
    body["failures"] = {"non_comparable_pairing": "command diverged"}
    body["pairing"].update(status="non_comparable", boundary="command_deviation", valid_steps=0)
    result = write_evidence(tmp_path / "deviation", body)
    assert "non_comparable_pairing" in result.blockers
    assert result.records["pairing"]["valid_steps"] == 0


def test_confirmed_step_preserved_without_observation(records, tmp_path):
    body = deepcopy(records)
    body["execution"][0].update(after=None, result_time_s=None)
    body["pairing"].update(status="not_recorded", valid_steps=0, boundary="observation_missing")
    path = tmp_path / "recording_failure"
    with pytest.raises(ValueError, match="requires an explicit failure"):
        write_evidence(path, body)
    body["failures"] = {"incomplete_recording": "fixture observation unavailable"}
    result = write_evidence(path, body)
    assert result.records["execution"][0]["step"] == "confirmed"
    assert result.records["execution"][0]["after"] is None
    assert result.records["pairing"]["valid_steps"] == 0
