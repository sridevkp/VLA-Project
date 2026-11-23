"""Deterministic, FakeBackend-only coverage of ADR-0084."""

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.worldmind_bindings import finite_chunk_check, production_worldmind_bindings
from vlarobot.core.types import ApplyResult, TaskSpec, VerifyResult
from vlarobot.eval.decision_adapters import CandidateGenerationProbe
from vlarobot.eval.report_contract import (
    TABLE51_NAMES,
    Table51MetricValue,
    table51_comparison_eligible,
    table51_nullable_values,
    table51_values,
)
from vlarobot.eval.worldmind_config import load_worldmind_diagnostic_config
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.scripted_chunk_adapter import ScriptedWaypoint
from vlarobot.execution.worldmind_heuristic_comparison import (
    HeuristicComparisonFixture,
    build_offline_heuristic_comparison,
)
from vlarobot.execution.worldmind_pipeline import WorldMindEpisodePolicy, _world_state  # noqa: PLC2701
from vlarobot.planning.verification import PlacementVerifier
from vlarobot.risk.comparison import ComparisonPlanner
from vlarobot.risk.gate import RiskGate

CONFIG = Path(__file__).resolve().parents[1] / "configs/worldmind_diagnostic.approved.yaml"
JOINT = "arm_7_joint"


@pytest.fixture
def config():
    return load_worldmind_diagnostic_config(CONFIG)


def _forbidden(*args, **kwargs):
    pytest.fail("prediction or ranking reached the prediction-free fixture")


class _Backend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.commands = []

    def apply(self, action):
        self.commands.append(action)
        return super().apply(action)


def _fixture(
    config,
    monkeypatch,
    *,
    rejected_rounds=(),
    prefilter=(),
    finite=True,
    unknown_prefilter=(),
    motor_veto=False,
    verification=(True,),
):
    monkeypatch.setattr(RiskGate, "select", _forbidden)
    monkeypatch.setattr(ComparisonPlanner, "select", _forbidden)
    base = production_worldmind_bindings()
    safety_factory = base.safety[config.safety.binding]
    seen = []
    count = 0
    active = False
    round_number = -1

    class Monitor(SafetyMonitor):
        def validate(self, action, state):
            nonlocal count
            if active:
                index = count
                count += 1
                seen.append(index)
                if index in unknown_prefilter:
                    return None
                if index in prefilter or (index == 32 and motor_veto):
                    return ApplyResult(False, "injected motor predicate rejection")
            return super().validate(action, state)

    def safety(config, inputs):
        return Monitor(safety_factory(config, inputs).limits)

    verdicts = iter(verification)

    class Verifier(PlacementVerifier):
        def check_observed(self, task, observed):
            return VerifyResult(next(verdicts), "fixture observed verdict")

        check = _forbidden

    def verifier(config, destination, rest, inputs):
        limits = config.verification
        return Verifier(
            destination_base=destination,
            rest_center_base=rest,
            up_axis_base=limits.up_axis_base,
            home_joints=limits.home_joints,
            distance_tolerance_m=limits.distance_tolerance_m,
            height_tolerance_m=limits.height_tolerance_m,
            tilt_limit_rad=limits.tilt_limit_rad,
            home_tolerance_rad=limits.home_tolerance_rad,
        )

    original_pool = HeuristicComparisonFixture._pool

    def pool(self, task, state, monitor, **kwargs):
        nonlocal active, count, round_number
        active = False
        round_number += 1
        generated = original_pool(self, task, state, monitor, **kwargs)
        active = True
        count = 0
        return generated

    monkeypatch.setattr(HeuristicComparisonFixture, "_pool", pool)

    def checked(state, chunk, trajectory):
        assert trajectory is None
        if round_number in rejected_rounds:
            return False
        if count in (0, 2):
            return finite
        return finite_chunk_check(state, chunk, None)

    waypoints = (ScriptedWaypoint("FIXTURE", 50, joints={JOINT: 0.02}),)
    bindings = replace(
        base,
        hard_checks={"finite_chunk_v1": checked},
        safety={config.safety.binding: safety},
        verification={config.verification.binding: verifier},
        surrogates={config.surrogate.binding: _forbidden},
        components={name: _forbidden for name in ("Jg", "Jl", "Jr", "Ju")},
        waypoint_factories={config.adapter.waypoints_binding: lambda *args: waypoints},
    )
    backend = _Backend()
    scenario = SimpleNamespace(name="adr0084-fixture", objects=(), can_xy_yaw_offset=None)
    # Use an ordinary FakeBackend observation, outside any frozen evaluation plan.
    observation = backend._observation()
    return bindings, backend, scenario, observation, waypoints, seen


@pytest.mark.parametrize("motor_veto", [False, True])
def test_complete_pool_accounting_and_independent_motor_veto(config, monkeypatch, motor_veto):
    bindings, backend, scenario, obs, waypoints, seen = _fixture(
        config, monkeypatch, prefilter=(1, 2), finite=False, motor_veto=motor_veto
    )
    task = TaskSpec(config.task.instruction, config.task.target, config.task.destination)
    state = _world_state(obs, backend.get_ground_truth(), config, target=task.target)
    production = SimpleNamespace(config=config, bindings=bindings)
    original = WorldMindEpisodePolicy._generator(production, waypoints, 0)
    expected = original.propose(task, state, None)

    result = build_offline_heuristic_comparison(config, bindings).execute_fixture_episode(
        backend, obs, episode_id=scenario.name, seed=0
    )
    (decision,) = result.decisions
    assert decision.generation.draw_seed == original.draw_log[0]["seed"]
    assert tuple(row.draw_seed for row in decision.candidates) == original.draw_log[0]["draw_seeds"]
    assert [row.draw_index for row in decision.candidates] == list(range(32))
    assert [row.candidate_id for row in decision.candidates] == [
        f"candidate-{i:02d}" for i in range(32)
    ]
    assert [row.chunk for row in decision.candidates] == expected
    assert [row.disposition for row in decision.candidates[:5]] == [
        "rejected",
        "rejected",
        "rejected",
        "selected",
        "eligible",
    ]
    assert decision.selected_candidate_id == "candidate-03"
    assert seen == list(range(33))
    assert result.evaluated_candidates == 32
    assert result.rejected_candidates == 3
    assert result.first_command_rejections == 2
    assert result.motor_vetoes == int(motor_veto)
    assert backend.commands == ([] if motor_veto else [expected[3].actions[0]])
    assert result.terminal_state == ("SAFETY_ABORT" if motor_veto else "COMPLETE")
    assert result.provenance["arm"] == "vla_risk_veto"
    assert result.provenance["scripted_adapter_binding"] == "scripted_chunk_adapter"
    assert result.provenance["prediction_mode"] == "none"
    assert result.provenance["execution_scope"] == "fixture_only"
    assert result.prediction_error == Table51MetricValue(
        "not_applicable", None, "arm_has_no_prediction", valid_paired_steps=0
    )
    assert not hasattr(decision, "prediction_pairing")


@pytest.mark.parametrize(
    ("empty", "verification", "transitions", "terminal", "ticks", "retries"),
    [
        ((0, 1), (), ("reobserve", "alternate_candidate_family:unavailable"), "NO_SAFE_SET", 1, 0),
        ((), (False, False), ("retry_once", "abort"), "VERIFICATION_ABORT", 2, 1),
        (
            (0, 2),
            (False,),
            ("reobserve", "retry_once", "alternate_candidate_family:unavailable"),
            "NO_SAFE_SET",
            2,
            1,
        ),
    ],
)
def test_recovery_allowances_are_separate_and_persist(
    config, monkeypatch, empty, verification, transitions, terminal, ticks, retries
):
    bindings, backend, scenario, obs, _, _ = _fixture(
        config, monkeypatch, rejected_rounds=empty, verification=verification
    )
    result = build_offline_heuristic_comparison(config, bindings).execute_fixture_episode(
        backend, obs, episode_id=scenario.name, seed=0
    )
    assert tuple(row.transition for row in result.decisions) == transitions
    assert result.terminal_state == terminal
    assert result.confirmed_ticks == ticks
    assert result.empty_sets == len(empty)
    assert result.verification_retries == retries


@pytest.mark.parametrize("bound", ["max_steps", "max_decisions"])
@pytest.mark.parametrize("empty", [False, True])
def test_recovery_respects_shared_budget(config, monkeypatch, bound, empty):
    config = replace(config, bounds=replace(config.bounds, **{bound: 1}))
    bindings, backend, scenario, obs, _, _ = _fixture(
        config, monkeypatch, rejected_rounds=(0,) if empty else (), verification=(False,)
    )
    result = build_offline_heuristic_comparison(config, bindings).execute_fixture_episode(
        backend, obs, episode_id=scenario.name, seed=0
    )
    assert result.terminal_state == "WORLDMIND_BOUND_EXHAUSTED"
    assert len(result.decisions) == result.confirmed_ticks == 1
    assert result.decisions[0].transition == ("reobserve" if empty else "retry_once")


def test_missing_dependency_and_partial_pool_fail_closed(config, monkeypatch):
    bindings, backend, scenario, obs, _, seen = _fixture(config, monkeypatch)
    with pytest.raises(ValueError, match="finite_chunk_v1"):
        build_offline_heuristic_comparison(config, replace(bindings, hard_checks={}))
    with pytest.raises(ValueError, match="frozen scripted"):
        build_offline_heuristic_comparison(
            replace(config, adapter=replace(config.adapter, alternate_waypoints_binding="extra")),
            bindings,
        )
    fixture = build_offline_heuristic_comparison(config, bindings)
    original = CandidateGenerationProbe.propose
    monkeypatch.setattr(
        CandidateGenerationProbe, "propose", lambda self, *args: original(self, *args)[:-1]
    )
    with pytest.raises(ValueError, match="inconsistent frozen"):
        fixture.execute_fixture_episode(backend, obs, episode_id=scenario.name, seed=0)
    assert not backend.commands and not seen


@pytest.mark.parametrize(
    "observed_defect", ["none", "missing", "duplicate", "bad_rotation", "home"]
)
def test_observed_verifier_matches_legacy_verdict(config, observed_defect):
    backend = FakeBackend()
    obs = backend._observation()
    state = _world_state(obs, backend.get_ground_truth(), config, target=config.task.target)
    (target,) = state.scene.objects
    if observed_defect == "missing":
        state.scene.objects = []
    elif observed_defect == "duplicate":
        state.scene.objects.append(target)
    elif observed_defect == "bad_rotation":
        state.scene.objects = [replace(target, pose=replace(target.pose, orientation=(0, 0, 0, 0)))]
    elif observed_defect == "home":
        state.robot.joint_pos.pop(JOINT)
    v = PlacementVerifier(
        destination_base=target.pose,
        rest_center_base=target.pose,
        up_axis_base=(0.0, 0.0, 1.0),
        home_joints={JOINT: 0.0},
        distance_tolerance_m=0.01,
        height_tolerance_m=0.01,
        tilt_limit_rad=0.1,
        home_tolerance_rad=0.02,
    )
    task = TaskSpec(config.task.instruction, config.task.target, config.task.destination)
    assert v.check_observed(task, state) == v.check(task, state, state)
    assert v.check_observed(task, state).success == (observed_defect == "none")


def test_unknown_check_verdicts_never_make_candidates_eligible(config, monkeypatch):
    bindings, backend, scenario, obs, _, _ = _fixture(
        config, monkeypatch, finite=None, unknown_prefilter=(1,)
    )
    result = build_offline_heuristic_comparison(config, bindings).execute_fixture_episode(
        backend, obs, episode_id=scenario.name, seed=0
    )
    (decision,) = result.decisions
    assert decision.selected_candidate_id == "candidate-03"
    assert decision.candidates[0].finite_chunk is None
    assert decision.candidates[1].first_command is None
    assert result.rejected_candidates == 3
    assert result.first_command_rejections == 1


def test_observed_verifier_preserves_inclusive_distance_height_home_and_strict_tilt(config):
    backend = FakeBackend()
    state = _world_state(
        backend._observation(), backend.get_ground_truth(), config, target=config.task.target
    )
    (target,) = state.scene.objects
    # Use exactly representable coordinates so equality tests are unambiguous.
    from vlarobot.core.types import Pose3D, Vec3

    origin = Pose3D(Vec3(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    state.scene.objects[0] = replace(target, pose=replace(origin, position=Vec3(0.0, 0.0, 0.125)))
    state.robot.joint_pos[JOINT] = 0.125
    common = dict(
        destination_base=origin,
        rest_center_base=origin,
        up_axis_base=(0.0, 0.0, 1.0),
        home_joints={JOINT: 0.0},
        distance_tolerance_m=0.125,
        height_tolerance_m=0.125,
        home_tolerance_rad=0.125,
    )
    task = TaskSpec(config.task.instruction, config.task.target, config.task.destination)
    assert PlacementVerifier(tilt_limit_rad=0.1, **common).check_observed(task, state).success

    state.scene.objects[0] = replace(
        state.scene.objects[0], pose=replace(state.scene.objects[0].pose, orientation=(1, 0, 0, 0))
    )
    assert (
        not PlacementVerifier(tilt_limit_rad=math.pi, **common).check_observed(task, state).success
    )


def test_nullable_table_preserves_denominators_and_requires_every_expected_arm(config):
    defs = config.report_definitions
    values = {
        name: Table51MetricValue("unavailable", None, "fixture_no_measurement")
        for name in TABLE51_NAMES
    }
    values["Prediction error"] = Table51MetricValue(
        "not_applicable", None, "arm_has_no_prediction", valid_paired_steps=0
    )
    values["Risk rejection rate"] = Table51MetricValue("measured", 0.0, denominator=32)
    table = table51_nullable_values(defs, values)
    assert table["schema_version"] == 1
    assert table["metrics"]["Prediction error"]["value"] is None
    assert table["metrics"]["Prediction error"]["valid_paired_steps"] == 0
    assert table["metrics"]["Risk rejection rate"]["value"] == 0
    assert table["metrics"]["Risk rejection rate"]["denominator"] == 32
    arms = ("vla_risk_veto", "vla_world_model")
    both = dict.fromkeys(arms, values)
    assert not table51_comparison_eligible("Prediction error", arms, both)
    assert table51_comparison_eligible("Risk rejection rate", arms, both)
    assert not table51_comparison_eligible("Risk rejection rate", arms, {arms[0]: values})
    with pytest.raises((TypeError, ValueError)):
        table51_values(defs, {name: item.value for name, item in values.items()})


@pytest.mark.parametrize(
    ("fault", "terminal", "expected"),
    [
        (
            None,
            "one_tick_observed",
            ["pin", "inspect", "reset", "gt", "apply", "tick", "gt", "shutdown", "close"],
        ),
        ("negative", None, ["pin", "inspect", "close"]),
        ("bad_observation", None, ["pin", "inspect", "reset", "close"]),
        ("empty", "empty_eligible_set", ["pin", "inspect", "reset", "gt", "shutdown", "close"]),
        ("veto", "motor_veto", ["pin", "inspect", "reset", "gt", "shutdown", "close"]),
        ("storage", None, ["pin", "inspect", "reset", "gt", "close"]),
        ("corrupt_intent", None, ["pin", "inspect", "reset", "gt", "close"]),
        (
            "reject",
            "backend_rejected",
            ["pin", "inspect", "reset", "gt", "apply", "shutdown", "close"],
        ),
        ("unknown_apply", None, ["pin", "inspect", "reset", "gt", "apply", "close"]),
        ("receipt_storage", None, ["pin", "inspect", "reset", "gt", "apply", "close"]),
        (
            "unknown_tick",
            "tick_unknown",
            ["pin", "inspect", "reset", "gt", "apply", "tick", "close"],
        ),
        (
            "missing_observation",
            "tick_unobserved",
            ["pin", "inspect", "reset", "gt", "apply", "tick", "close"],
        ),
        (
            "episode_complete",
            "COMPLETE",
            ["pin", "inspect", "reset", "gt"]
            + ["apply", "tick", "gt"] * 50
            + ["shutdown", "close"],
        ),
        (
            "episode_retry",
            "VERIFICATION_ABORT",
            ["pin", "inspect", "reset", "gt"]
            + ["apply", "tick", "gt"] * 100
            + ["shutdown", "close"],
        ),
        (
            "episode_empty",
            "NO_SAFE_SET",
            ["pin", "inspect", "reset", "gt", "tick", "gt", "shutdown", "close"],
        ),
        (
            "episode_interleaved",
            "NO_SAFE_SET",
            ["pin", "inspect", "reset", "gt", "tick", "gt"]
            + ["apply", "tick", "gt"] * 50
            + ["shutdown", "close"],
        ),
        (
            "episode_partial_budget",
            "WORLDMIND_BOUND_EXHAUSTED",
            ["pin", "inspect", "reset", "gt", "apply", "tick", "gt", "shutdown", "close"],
        ),
        (
            "episode_mid_veto",
            "motor_veto",
            ["pin", "inspect", "reset", "gt", "apply", "tick", "gt", "shutdown", "close"],
        ),
        (
            "episode_mid_unknown",
            "tick_unknown",
            ["pin", "inspect", "reset", "gt", "apply", "tick", "gt", "apply", "tick", "close"],
        ),
        ("episode_corrupt_reobserve", None, ["pin", "inspect", "reset", "gt", "close"]),
        (
            "episode_unknown",
            "tick_unknown",
            ["pin", "inspect", "reset", "gt", "tick", "close"],
        ),
        (
            "episode_budget",
            "WORLDMIND_BOUND_EXHAUSTED",
            ["pin", "inspect", "reset", "gt", "tick", "gt", "shutdown", "close"],
        ),
    ],
)
def test_owned_guarded_arm3(config, monkeypatch, tmp_path, fault, terminal, expected):
    import json

    from vlarobot.core.wire import decode_message
    from vlarobot.execution import worldmind_heuristic_qualification as qualification

    episode = fault is not None and fault.startswith("episode_")
    if fault in {"episode_budget", "episode_partial_budget"}:
        config = replace(config, bounds=replace(config.bounds, max_steps=1))
    bindings, backend, *_ = _fixture(
        config,
        monkeypatch,
        motor_veto=fault == "veto",
        prefilter=(33,) if fault == "episode_mid_veto" else (),
        rejected_rounds={
            "empty": (0,),
            "episode_empty": (0, 1),
            "episode_interleaved": (0, 2),
            "episode_unknown": (0,),
            "episode_corrupt_reobserve": (0,),
            "episode_budget": (0,),
        }.get(fault, ()),
        verification=(False, False)
        if fault in {"episode_retry", "episode_interleaved"}
        else (True,),
    )
    calls = []
    owned = {"pid": 42, "start_time": 1.0, "nonce": "fixture"}
    output = tmp_path / "qualification"

    class Client:
        def require_owned_diagnostic_session(self, *, session_id, world_sha256):
            assert session_id == owned and world_sha256 == qualification._WORLD
            calls.append("pin")

        def inspect_fingers(self):
            calls.append("inspect")
            rows = []
            for index, name in enumerate(qualification._FINGERS):
                captured = (-7.142594888992681e-07, 7.171095938981087e-07)
                reading = captured[index] if fault == "negative" else 0.0
                sensor = name.replace("_finger_joint", "_sensor_finger_joint")
                rows.append(
                    {
                        "expected_motor": name,
                        "resolved_motor": name,
                        "resolved_sensor": sensor,
                        "cached_sensor": sensor,
                        "mapping": "matches",
                        "raw_coordinate_m": reading,
                        "cached_raw_coordinate_m": reading,
                        "cached_pair_count": 1,
                        "motor_limits_m": [0.0, 0.045],
                        "sampling_period_ms": 32,
                    }
                )
            return dict(
                scope="read_only_finger_admission",
                session_id=owned,
                reviewed_limits_m=[0.0, 0.045],
                physics_step_count=0,
                simulator_time_s=0.096,
                readings=rows,
            )

        def reset(self, scenario, seed):
            calls.append("reset")
            assert (scenario.name, seed) == ("validation-clear-i00-s0", 0)
            result = backend.reset(scenario, seed)
            if fault != "bad_observation":
                result.robot.diagnostics["finger_joint_sensor_rad"] = dict.fromkeys(
                    qualification._FINGERS, 0.0
                )
            return result

        def get_ground_truth(self):
            calls.append("gt")
            return backend.get_ground_truth()

        def apply(self, action):
            calls.append("apply")
            d = sorted(output.glob("decision-*"))[-1] if episode else output
            decision = decode_message((d / "decision.msgpack").read_bytes())["payload"]
            selected = [row for row in decision["pool"] if row["disposition"] == "selected"]
            assert len(decision["pool"]) == 32 and len(selected) == 1
            action_dir = sorted(d.glob("action-*"))[-1] if episode else d
            intent = json.loads((action_dir / "apply-intent.json").read_bytes())["payload"]
            assert selected[0]["chunk"].actions[intent["action_index"]] == action
            if episode:
                verdict = decode_message((action_dir / "motor-verdict.msgpack").read_bytes())
                assert verdict["payload"]["motor_verdict"].accepted
            else:
                assert decision["motor_verdict"].accepted
            if fault == "unknown_apply":
                raise TimeoutError("lost apply reply")
            if fault == "reject":
                return ApplyResult(False)
            return backend.apply(action)

        def diagnostic_step(self):
            calls.append("tick")
            d = sorted(output.glob("decision-*"))[-1] if episode else output
            action_dirs = sorted(d.glob("action-*"))
            d = action_dirs[-1] if action_dirs else d
            intent = json.loads((d / "tick-intent.json").read_bytes())["payload"]
            if "reobserve_decision_sha256" in intent:
                assert not (d / "apply-intent.json").exists()
            else:
                assert (d / "apply-result.json").is_file()
            if fault in {"unknown_tick", "episode_unknown"} or (
                fault == "episode_mid_unknown" and d.name == "action-01"
            ):
                return {"confirmed_ticks": None}
            if fault == "missing_observation":
                return {"confirmed_ticks": 1, "error": "fixture observation failed"}
            obs = backend.step(1)
            obs.robot.diagnostics["finger_joint_sensor_rad"] = dict.fromkeys(
                qualification._FINGERS, 0.0
            )
            return {"confirmed_ticks": 1, "observation": obs}

        def shutdown_diagnostic_session(self, *, session_id):
            assert session_id == owned
            calls.append("shutdown")
            return {"status": "stopped", "session_id": owned}

        def close(self):
            calls.append("close")

    session = SimpleNamespace(
        client=Client(),
        endpoint="tcp://127.0.0.1:5555",
        process=object(),
        runtime_attestation={
            "reported_version": "R2025a",
            "executable_sha256": qualification._RUNTIME,
        },
    )
    monkeypatch.setattr(qualification, "_verified_head", lambda expected: expected)
    monkeypatch.setattr(qualification, "production_worldmind_bindings", lambda: bindings)
    monkeypatch.setattr(qualification, "read_owned_controller_session", lambda *a, **kw: owned)
    monkeypatch.setattr(qualification, "load_worldmind_diagnostic_config", lambda path: config)
    writer = qualification._write_event

    def injected_write(path, payload):
        if path.name == "apply-intent.json" and fault == "storage":
            raise OSError("fixture storage fault")
        if path.name == "apply-result.json" and fault == "receipt_storage":
            raise OSError("fixture receipt storage fault")
        digest = writer(path, payload)
        if path.name == "apply-intent.json" and fault == "corrupt_intent":
            path.write_bytes(b"{}")
        if path.name == "tick-intent.json" and fault == "episode_corrupt_reobserve":
            path.with_name("decision.msgpack").write_bytes(b"corrupt decision")
        return digest

    monkeypatch.setattr(qualification, "_write_event", injected_write)
    args = dict(
        status_path=tmp_path / "status.json",
        launch_started_at_s=1.0,
        directory=output,
        expected_head="approved-fixture",
    )
    run = qualification.run_heuristic_qualification
    if episode:
        run = qualification.run_heuristic_episode
        args["episode_id"] = "validation-clear-i00-s0"
    if terminal is None:
        with pytest.raises((ValueError, OSError, KeyError)):
            run(session, **args)
    else:
        assert run(session, **args) == terminal
    assert calls == expected
    if fault in {"episode_partial_budget", "episode_mid_veto", "episode_mid_unknown"}:
        assert not (output / "decision-0000" / "verification.json").exists()
    if fault == "missing_observation":
        receipt = json.loads((output / "tick-result.json").read_bytes())["payload"]
        assert receipt["confirmed_ticks"] == 1 and not receipt["observation_returned"]
    with pytest.raises(FileExistsError):
        run(session, **args)
    assert calls == expected
