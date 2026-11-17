"""Offline A-D2 tests: no simulator, model, solver or actuator is reachable."""

from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from vlarobot.diagnostics.ad2.admission import Bound, Protocol, State, compare
from vlarobot.diagnostics.ad2.evidence import EvidenceError, Targets, digest, load_targets
from vlarobot.diagnostics.ad2.fixtures import (
    FixtureProcess,
    FixtureSession,
    FixtureTransport,
    SnapshotBuffer,
    run_fixtures,
    snapshot,
)
from vlarobot.diagnostics.ad2.geometry import geometry_inventory


@pytest.fixture
def state():
    return State(
        (("scene", "train-clear-i00-s100"), ("gripper_hold", "unchanged")),
        (("arm_rad", 0.0), ("object_v_m_s", 0.0)),
        (("base_rotation", (0.0, 0.0, 0.0, 1.0)),),
        (("arm_rad", "rad"), ("object_v_m_s", "m/s"), ("base_rotation", "rad")),
    )


@pytest.fixture
def protocol():
    # Synthetic numbers deliberately do not match the exploratory physical proposals.
    return Protocol(
        "fixture-v1",
        "fixture:explicit-criteria",
        "fixture-observed-v1",
        (
            Bound("arm_rad", 0.1, 0.1, "rad"),
            Bound("object_v_m_s", 0.2, 0.2, "m/s"),
            Bound("base_rotation", 0.01, 0.01, "rad"),
        ),
        (),
        1e-12,
        1e-12,
    )


@pytest.fixture
def targets():
    q = (0.0,) * 7
    return Targets(
        "fixture-only",
        "fixture-seal",
        tuple(f"arm_{i}_joint" for i in range(1, 8)),
        q,
        q,
        (0.01,) * 7,
        "train-clear-i00-s100",
        100,
    )


@pytest.fixture
def sessions(state):
    return tuple(
        FixtureSession(
            case,
            FixtureProcess(i + 1, f"nonce-{i}", 6000 + i),
            FixtureTransport(True),
            SnapshotBuffer(state, 80, 2.56, 3, 77),
            SnapshotBuffer(state, 81, 2.592, 3, 77),
        )
        for i, case in enumerate("ABC")
    )


def test_explicit_criteria_and_reference_pairwise(state, protocol):
    assert compare(state, state, (), protocol).accepted
    before = replace(state, scalars=(("arm_rad", -0.09), ("object_v_m_s", 0.0)))
    now = replace(state, scalars=(("arm_rad", 0.09), ("object_v_m_s", 0.0)))
    result = compare(state, now, (before,), protocol)
    assert not result.accepted
    assert any("pairwise" in x for x in result.issues)
    assert compare(state, now, (), protocol).accepted


@pytest.mark.parametrize(
    "change",
    [
        {"bounds": ()},
        {"approval_id": ""},
        {"version": ""},
        {"observability_version": ""},
        {"quaternion_norm_tolerance": float("nan")},
        {"timestamp_tolerance_s": -1.0},
        {"bounds": (Bound("arm_rad", 1, 1, "wrong-unit"),)},
    ],
)
def test_missing_protocol_or_criteria_fail_closed(state, protocol, change):
    assert not compare(state, state, (), replace(protocol, **change)).accepted


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_live_missing_nonfinite_not_zero(state, protocol, value):
    changed = replace(state, scalars=(("arm_rad", value), ("object_v_m_s", 0.0)))
    assert not compare(state, changed, (), protocol).accepted


def test_omission_requires_named_version_and_live_observation(state, protocol):
    ref = replace(state, scalars=(("arm_rad", 0.0), ("object_v_m_s", None)))
    assert not compare(ref, state, (), protocol).accepted
    limited = replace(
        protocol,
        observability_version="fixture-partial-v1",
        historical_omissions=(("object_v_m_s", "not historically recorded"),),
    )
    assert compare(ref, state, (), limited).accepted
    assert not compare(ref, ref, (), limited).accepted
    assert not compare(state, state, (), limited).accepted
    assert not compare(ref, state, (ref,), limited).accepted


def test_quaternion_sign_invariance_and_bad_norm(state, protocol):
    negative = replace(state, orientations=(("base_rotation", (0.0, 0.0, 0.0, -1.0)),))
    assert compare(state, negative, (), protocol).accepted
    bad = replace(state, orientations=(("base_rotation", (0.0, 0.0, 0.0, 0.0)),))
    assert not compare(state, bad, (), protocol).accepted
    rotated = replace(state, orientations=(("base_rotation", (0.0, 0.0, 1.0, 0.0)),))
    assert not compare(state, rotated, (), protocol).accepted


def test_configuration_schema_and_units_are_exact(state, protocol):
    for changed in [
        replace(state, configuration=(("scene", "other"),)),
        replace(state, units=(("arm_rad", "m"),)),
        replace(state, scalars=(("arm_rad", 0.0),)),
    ]:
        assert not compare(state, changed, (), protocol).accepted


def test_snapshot_is_deeply_immutable_and_has_no_callbacks(state):
    buffer = SnapshotBuffer(state, 5, 0.16, 3, 2)
    assert snapshot(buffer) is buffer
    with pytest.raises(FrozenInstanceError):
        buffer.tick = 6
    with pytest.raises(TypeError):
        snapshot(object())
    with pytest.raises(TypeError):
        replace(state, scalars=[("arm_rad", 0.0)])
    with pytest.raises(TypeError):
        replace(state, orientations=(("base_rotation", [0, 0, 0, 1]),))
    with pytest.raises(ValueError):
        snapshot(replace(buffer, tick=-1))


def test_three_ordered_isolated_fixture_ticks(tmp_path, targets, state, protocol, sessions):
    result = run_fixtures(targets, state, protocol, sessions, tmp_path / "new")
    assert result.completed == ("A", "B", "C")
    assert len([e for e in result.events if e["event"] == "fixture-step-dispatched"]) == 3
    assert all(not s.process.alive for s in sessions)
    preparations = [e for e in result.events if e["event"] == "preparation"]
    assert all(e["startup_ticks"] == 3 and e["reset_ticks"] == 77 for e in preparations)
    cleanups = [e for e in result.events if e["event"] == "fixture-cleanup"]
    assert len(cleanups) == 3 and all(e["physics_progression"] == 0 for e in cleanups)


@pytest.mark.parametrize("where,steps", [("apply", 0), ("step", 1)])
def test_ambiguous_ack_is_not_retried(tmp_path, targets, state, protocol, sessions, where, steps):
    a = replace(sessions[0], transport=FixtureTransport(True, ambiguous_at=where))
    out = tmp_path / "out"
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), out)
    assert result.status == "fixture-stopped" and not result.completed
    assert len([e for e in result.events if e["event"] == "fixture-apply-dispatched"]) == 1
    assert len([e for e in result.events if e["event"] == "fixture-step-dispatched"]) == steps
    assert len([e for e in result.events if e["event"] == "fixture-start"]) == 1
    assert "TimeoutError" in (out / "fixture-events.jsonl").read_text()
    assert (out / "fixture-result.json").is_file()


def test_rejection_never_steps_or_replaces(tmp_path, targets, state, protocol, sessions):
    a = replace(sessions[0], transport=FixtureTransport(False))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert not any(e["event"] == "fixture-step-dispatched" for e in result.events)
    assert not sessions[1].process.alive
    assert any(e.get("reason") == "SAFETY rejection" for e in result.events)


@pytest.mark.parametrize(
    "attribute,value", [("tick", 80), ("tick", 82), ("time_s", 2.624), ("reset_ticks", 78)]
)
def test_tick_and_time_mismatch_stop(
    tmp_path, targets, state, protocol, sessions, attribute, value
):
    a = replace(sessions[0], after=replace(sessions[0].after, **{attribute: value}))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert result.status == "fixture-stopped" and not result.completed


def test_reset_mismatch_before_any_dispatch(tmp_path, targets, state, protocol, sessions):
    bad = replace(state, configuration=(("scene", "wrong"),))
    a = replace(sessions[0], before=replace(sessions[0].before, state=bad))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert not any(e["event"].endswith("dispatched") for e in result.events)


def test_conflict_no_fourth_or_live_adapter(tmp_path, targets, state, protocol, sessions):
    for values in [sessions[::-1], (*sessions, sessions[0]), (object(), *sessions[1:])]:
        with pytest.raises(ValueError):
            run_fixtures(targets, state, protocol, values, tmp_path / "uncreated")
    assert not (tmp_path / "uncreated").exists()
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        run_fixtures(targets, state, protocol, sessions, out)
    with pytest.raises(ValueError):
        run_fixtures(
            targets,
            state,
            replace(protocol, approval_id="not-live-approved"),
            sessions,
            tmp_path / "not-live",
        )


def test_identity_reuse_preserves_partial_and_cleans(tmp_path, targets, state, protocol, sessions):
    sessions[1].process.nonce = sessions[0].process.nonce
    result = run_fixtures(targets, state, protocol, sessions, tmp_path / "out")
    assert result.completed == ("A",)
    assert len([e for e in result.events if e["event"] == "fixture-start"]) == 1
    assert not any(s.process.alive for s in sessions)


def test_unowned_conflicting_process_not_cleaned(tmp_path, targets, state, protocol, sessions):
    sessions[0].process.alive = True
    result = run_fixtures(targets, state, protocol, sessions, tmp_path / "out")
    assert result.status == "fixture-stopped" and sessions[0].process.alive
    assert result.events[-1]["owned"] is False


def test_infrastructure_failure_preserved(tmp_path, targets, state, protocol, sessions):
    a = replace(sessions[0], transport=FixtureTransport(True, infrastructure_error="fake startup"))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert result.status == "fixture-stopped"
    assert not any(e["event"].endswith("dispatched") for e in result.events)
    assert all(not s.process.alive for s in sessions)


def sealed_fixture(tmp_path):
    """Build tiny synthetic inputs; no real model/dataset dependency in unit tests."""
    q = {f"arm_{i}_joint": 0.0 for i in range(1, 8)}
    cases = {"A": {"joints": q}, "B": {"joints": q}, "C": {"joints": {name: 0.01 for name in q}}}
    objects = {
        "result": {
            "identity": "A-D1-fixture",
            "cases": cases,
            "actual_numerical_ik_calls": 2,
            "original_gate": "FAILED_0_of_24",
        },
        "provenance": {"identity": "A-D1-fixture"},
        "native": {"row": {"episode_index": 0, "frame_index": 0}},
        "meta": {"scenario": {"name": "train-clear-i00-s100"}, "reset_seed": 100},
        "B": cases["B"],
        "C": cases["C"],
    }
    files = {}
    for key, value in objects.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        files[key] = {
            "root": "fixture",
            "path": path.name,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    spec = {
        "schema_version": 1,
        "identity": "A-D1-fixture",
        "scene": "train-clear-i00-s100",
        "seed": 100,
        "frame": [0, 0],
        "files": files,
        "targets": {k: list(v["joints"].values()) for k, v in cases.items()},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec, {"fixture": tmp_path}


def test_pinned_loader_immutable_and_c_exact(tmp_path):
    manifest, _, roots = sealed_fixture(tmp_path)
    targets = load_targets(manifest, digest(manifest), roots)
    assert targets.a == targets.b == (0.0,) * 7
    assert targets.c == (0.01,) * 7
    with pytest.raises(FrozenInstanceError):
        targets.c = targets.a
    with pytest.raises(EvidenceError):
        targets.case("D")


@pytest.mark.parametrize("part", ["manifest", "C", "provenance", "native", "meta"])
def test_sealed_source_tampering_rejected(tmp_path, part):
    manifest, spec, roots = sealed_fixture(tmp_path)
    trust = digest(manifest)
    path = manifest if part == "manifest" else tmp_path / spec["files"][part]["path"]
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(EvidenceError):
        load_targets(manifest, trust, roots)


@pytest.mark.parametrize("change", ["C_value", "A_B", "frame", "scene", "missing_root"])
def test_semantic_checks_independent_of_resealed_file(tmp_path, change):
    manifest, spec, roots = sealed_fixture(tmp_path)
    if change in ("C_value", "A_B"):
        part = "C" if change == "C_value" else "B"
        path = tmp_path / spec["files"][part]["path"]
        obj = json.loads(path.read_text())
        obj["joints"]["arm_1_joint"] = 0.02
        path.write_text(json.dumps(obj), encoding="utf-8")
        spec["files"][part].update(sha256=digest(path), bytes=path.stat().st_size)
    elif change == "frame":
        spec["frame"] = [0, 1]
    elif change == "scene":
        spec["scene"] = "not-authorized"
    else:
        roots = {}
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(EvidenceError):
        load_targets(manifest, digest(manifest), roots)


@pytest.mark.parametrize(
    "relative",
    ["../result.json", "/result.json", "C:/result.json", "x\\result.json", "./result.json"],
)
def test_path_escape_rejected(tmp_path, relative):
    manifest, spec, roots = sealed_fixture(tmp_path)
    spec["files"]["result"]["path"] = relative
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(EvidenceError):
        load_targets(manifest, digest(manifest), roots)


def test_real_safety_monitor_rejects_synthetic_joint_command_without_backend():
    from vlarobot.core.types import JointTarget, Pose2D, Pose3D, RobotState, Vec3
    from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor

    monitor = SafetyMonitor(
        SafetyLimits(
            joint_position={"arm_1_joint": (0.07, 2.68)}, joint_velocity={"arm_1_joint": 1.95}
        )
    )
    state = RobotState(
        joint_pos={"arm_1_joint": 0.1},
        joint_vel={"arm_1_joint": 0.0},
        base_pose=Pose2D(0, 0, 0),
        ee_pose=Pose3D(Vec3(0.5, 0, 1), (0, 0, 0, 1)),
        gripper_opening=0.0,
    )
    for q in [{"arm_1_joint": -1.0}, {"arm_1_joint": float("nan")}, {"unknown": 0.0}]:
        assert not monitor.validate(JointTarget(q=q), state).accepted


def test_geometry_inventory_finds_nested_table_and_never_certifies(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="fake"><link name="arm"><collision><geometry>'
        '<box size="1 1 1"/></geometry></collision></link>'
        '<link name="wrist"/><joint name="j" type="fixed">'
        '<parent link="arm"/><child link="wrist"/></joint></robot>'
    )
    world = tmp_path / "world.wbt"
    world.write_text('DEF CAN Can {}\nTable {}\nRobot { controller "moving" }')
    table = tmp_path / "Table.proto"
    table.write_text(
        'Solid { children [ SolidBox { name "main frame" } SolidBox { name "foot 1" } ] }'
    )
    result = geometry_inventory(urdf, world, {"Table": table})
    assert not result["collision_safety_certified"]
    assert result["overall_coverage"] == "partial"
    assert len(result["sources"]["assets"]["Table"]["nested_solidbox_lines"]) == 2
    assert result["world_controller_literals"] == ["moving"]
    assert not result["adjacency_not_approved_exclusions"][0]["exclusion_approved"]
    assert "Tiago" in result["missing_assets"]
    assert result["required_scene_geometry"]["swept_volumes"].startswith("unsupported")


def test_infrastructure_import_surface_has_no_live_or_solver_dependency():
    root = Path(__file__).parents[1] / "src/vlarobot/diagnostics/ad2"
    forbidden = {
        "controller",
        "torch",
        "transformers",
        "lerobot",
        "ikpy",
        "subprocess",
        "zmq",
        "socket",
        "vlarobot.backends",
        "vlarobot.vla",
    }
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n == f or n.startswith(f + ".") for n in names for f in forbidden)


@pytest.mark.parametrize(
    "override",
    [
        {"reply_nonce_override": "other-session"},
        {"reply_sequence_override": 2},
        {"ambiguous_at": "unrecognized"},
    ],
)
def test_bad_receipt_identity_stops_without_retry(
    tmp_path, targets, state, protocol, sessions, override
):
    a = replace(sessions[0], transport=replace(sessions[0].transport, **override))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert result.status == "fixture-stopped"
    assert not any(e["event"] == "fixture-step-dispatched" for e in result.events)
    assert len([e for e in result.events if e["event"] == "fixture-start"]) == 1


def test_after_nonfinite_state_is_a_failure(tmp_path, targets, state, protocol, sessions):
    bad = replace(state, scalars=(("arm_rad", float("nan")), ("object_v_m_s", 0.0)))
    a = replace(sessions[0], after=replace(sessions[0].after, state=bad))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert result.status == "fixture-stopped" and not result.completed


def test_summary_write_failure_retains_acknowledged_ledger(
    tmp_path, targets, state, protocol, sessions, monkeypatch
):
    original = Path.open

    def fail_summary(path, *args, **kwargs):
        if path.name == "fixture-result.json":
            raise OSError("fake storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_summary)
    with pytest.raises(OSError, match="fake storage failure"):
        run_fixtures(targets, state, protocol, sessions, tmp_path / "out")
    assert all(not s.process.alive for s in sessions)
    ledger = (tmp_path / "out/fixture-events.jsonl").read_text()
    assert ledger.count('"event": "fixture-after"') == 3
    assert ledger.count('"event": "fixture-cleanup"') == 3


def test_duplicate_json_and_symlink_components_rejected_without_creating_links(
    tmp_path, monkeypatch
):
    from vlarobot.diagnostics.ad2.evidence import confined, read_json

    p = tmp_path / "duplicate.json"
    p.write_text('{"case":1,"case":2}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="Duplicate"):
        read_json(p)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "alias" or original(p))
    with pytest.raises(EvidenceError, match="Symlink"):
        confined(tmp_path, "alias/file.json")


def test_units_and_numeric_bounds_individually_validated(state, protocol):
    wrong = replace(protocol.bounds[0], unit="m")
    assert not compare(
        state, state, (), replace(protocol, bounds=(wrong, *protocol.bounds[1:]))
    ).accepted
    for value in [-1.0, float("nan"), float("inf")]:
        bad = replace(protocol.bounds[0], pairwise=value)
        assert not compare(
            state, state, (), replace(protocol, bounds=(bad, *protocol.bounds[1:]))
        ).accepted


def test_exact_a_b_identity_checked_even_when_both_case_copies_agree(tmp_path):
    manifest, spec, roots = sealed_fixture(tmp_path)
    for key in ["result", "B"]:
        path = tmp_path / spec["files"][key]["path"]
        obj = json.loads(path.read_text())
        target = obj["cases"]["B"]["joints"] if key == "result" else obj["joints"]
        target["arm_1_joint"] = 0.02
        path.write_text(json.dumps(obj), encoding="utf-8")
        spec["files"][key].update(sha256=digest(path), bytes=path.stat().st_size)
    spec["targets"]["B"][0] = 0.02
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(EvidenceError, match="identical"):
        load_targets(manifest, digest(manifest), roots)


def test_original_provenance_cannot_omit_inputs(tmp_path):
    manifest, spec, roots = sealed_fixture(tmp_path)
    path = tmp_path / spec["files"]["provenance"]["path"]
    obj = json.loads(path.read_text())
    obj["inputs_sha256"] = {"original/input": "0" * 64}
    path.write_text(json.dumps(obj), encoding="utf-8")
    spec["files"]["provenance"].update(sha256=digest(path), bytes=path.stat().st_size)
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(EvidenceError, match="provenance coverage"):
        load_targets(manifest, digest(manifest), roots)


def test_unaccounted_preparation_tick_stops_before_dispatch(
    tmp_path, targets, state, protocol, sessions
):
    a = replace(sessions[0], before=replace(sessions[0].before, startup_ticks=2))
    result = run_fixtures(targets, state, protocol, (a, *sessions[1:]), tmp_path / "out")
    assert result.status == "fixture-stopped"
    assert not any(e["event"].endswith("dispatched") for e in result.events)


def test_protocol_cannot_hold_mutable_criteria(protocol):
    with pytest.raises(TypeError):
        replace(protocol, bounds=list(protocol.bounds))
    with pytest.raises(TypeError):
        replace(protocol, historical_omissions=(("object_v_m_s", 123),))


def test_verified_json_checks_the_bytes_it_actually_parses(tmp_path):
    from vlarobot.diagnostics.ad2.evidence import read_json

    path = tmp_path / "input.json"
    path.write_text('{"a":1}', encoding="utf-8")
    old = digest(path)
    path.write_text('{"a":2}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="changed after verification"):
        read_json(path, old)
