"""Scope tests for the narrowed Packet-B collision proposal (offline, read-only).

The module pins three things that matter for scientific honesty:

* the manifest is an exact partition and any weakening of it fails closed;
* each removed gripper/arm pair is justified by a *measured* campaign maximum, never by an
  assumed pose;
* the narrowed checker still rejects real violations (table, unintended object contact, joint
  limits, malformed input), and the only way to observe its accept path is an explicitly
  labelled control clone that removes pairs the strict checker reports as violated.

No test launches Webots, moves the robot or reads TRAIN/TEST.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from vlarobot.backends.worldmind_bindings import (
    ARM_SWEEP_CONFIG_PATH,
    PacketBManifest,
    load_arm_sweep_config,
    load_packet_b_manifest,
    packet_b_narrowed_sweep_check,
    production_worldmind_bindings,
    runtime_safety_limits,
)
from vlarobot.core.types import (
    JointTarget,
    ObjectState,
    Pose2D,
    Pose3D,
    RobotState,
    Vec3,
)
from vlarobot.eval.worldmind_packet_b import (
    PACKET_B_EVIDENCE_PIN_PATH,
    PacketBCampaignReport,
    evaluate_campaign_gripper_mounts,
    evaluate_campaign_pair_compatibility,
    recorded_gripper_mount_clearance,
    recorded_world_state,
    verify_campaign_evidence,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.world_model.arm_geometry import (
    _object_box,
    primitive_clearance,
    primitive_lower_bound,
    static_box_primitive,
)
from vlarobot.world_model.arm_sweep import SweepGeometryError, urdf_collision_boxes
from vlarobot.world_model.scene_geometry import (
    SceneCollisionCatalog,
    load_scene_collision_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/geometry/packet-b-exclusions-v1.yaml"
SCENE_CATALOG = ROOT / "configs/geometry/scene-collision-v1.yaml"
CAMPAIGN = Path("/home/tj330/programs/VLA-Project-analytic/runs/validation-evidence-v2-canonical")
EVIDENCE_PIN = PACKET_B_EVIDENCE_PIN_PATH

_TABLE_TOP_M = 0.6588858201145767
_TABLE_AABB_TOP_M = 0.7221534676512351


def _campaign_available() -> bool:
    return (CAMPAIGN / "campaign").is_dir()


needs_campaign = pytest.mark.skipif(
    not _campaign_available(), reason="recorded expert campaign is not present"
)


@lru_cache(maxsize=1)
def _manifest() -> PacketBManifest:
    return load_packet_b_manifest()


@lru_cache(maxsize=1)
def _catalog() -> SceneCollisionCatalog:
    return load_scene_collision_catalog(SCENE_CATALOG)


@lru_cache(maxsize=1)
def _campaign_report() -> PacketBCampaignReport:
    return evaluate_campaign_pair_compatibility(_manifest(), evidence_dir=CAMPAIGN)


@lru_cache(maxsize=1)
def _checker():
    catalog = _catalog()
    return packet_b_narrowed_sweep_check(
        _manifest(),
        scene=catalog.geometry,
        expected_base_pose_world=catalog.expected_base_pose_world,
        required_object_ids=frozenset({"CAN"}),
        contact_object_ids=frozenset({"CAN"}),
        contact_permitted=None,
    )


def _worst_recorded_state() -> tuple[object, str, int]:
    report = _campaign_report()
    worst = min(report.pairs, key=lambda item: item.worst_slack_m)
    path = CAMPAIGN / "campaign" / f"{worst.worst_episode}.steps.npz"
    episode = np.load(path)
    state = recorded_world_state(episode, worst.worst_step, catalog=_catalog())
    return state, worst.worst_episode, worst.worst_step


def _violating_pairs(checker, state) -> set[frozenset[str]]:
    """Return every checked pair whose exact separation is below the required allowance."""
    required = checker.manifest.required_separation_m
    capsules = checker.inner.geometry(state)
    boxes = checker.inner.box_geometry(state)
    excluded = checker.inner.excluded_self_pairs
    violations: set[frozenset[str]] = set()
    for index, first in enumerate(capsules):
        for second in capsules[index + 1 :]:
            pair = frozenset((first.name, second.name))
            if pair in excluded:
                continue
            if primitive_clearance(first, second) < required:
                violations.add(pair)
    for capsule in capsules:
        for box in boxes:
            pair = frozenset((capsule.name, box.name))
            if pair in excluded:
                continue
            if primitive_clearance(capsule, box) < required:
                violations.add(pair)
    return violations


def _control_clone(checker, state, *, scene=None):
    """Build a labelled control clone that additionally ignores pairs this state violates."""
    catalog = _catalog()
    clone = packet_b_narrowed_sweep_check(
        _manifest(),
        scene=scene or catalog.geometry,
        expected_base_pose_world=catalog.expected_base_pose_world,
        required_object_ids=frozenset({"CAN"}),
        contact_object_ids=frozenset({"CAN"}),
        contact_permitted=None,
    )
    violating = _violating_pairs(checker, state)
    assert violating, "control state must actually violate the strict pair set"
    clone.inner.excluded_self_pairs = frozenset(checker.inner.excluded_self_pairs | violating)
    return clone, violating


# --------------------------------------------------------------------------------------- manifest


def test_manifest_pins_a_complete_partition_with_measured_exclusions():
    manifest = _manifest()
    report = manifest.report()
    assert report["status"] == "diagnostic_proposal"
    assert report["classification"] == "not_a_motion_certificate"
    assert manifest.counts == {
        "capsules": 19,
        "pairs": 171,
        "same_rigid_body": 15,
        "directly_jointed": 39,
        "urdf_volumes_intersect": 8,
        "retained": 109,
        "retained_compatible_with_all_steps": 101,
        "retained_rejected_at_some_step": 8,
        "gripper_mount": 16,
    }
    assert manifest.required_separation_m == pytest.approx(0.039807151, abs=1e-9)
    excluded = {tuple(sorted(pair)) for pair in manifest.manifest.excluded_pairs()}
    retained = {tuple(sorted(pair)) for pair in manifest.manifest.retained_pairs}
    assert len(excluded) == 62
    assert len(retained) == 109
    assert not excluded & retained
    assert len(excluded) + len(retained) == 19 * 18 // 2
    assert len(manifest.gripper_mount_pairs) == 16
    assert all(
        item.max_campaign_clearance_m < manifest.required_separation_m
        for item in manifest.gripper_mount_pairs
    )
    coverage = report["coverage_statement"]
    assert "42 URDF collision bodies" in coverage
    assert "does not cover" in coverage


def _mutate(tmp_path: Path, text: str) -> Path:
    urdf = (ROOT / "src/vlarobot/backends/webots/tiago.urdf").resolve()
    text = text.replace(
        "source_urdf: ../../src/vlarobot/backends/webots/tiago.urdf",
        f"source_urdf: {urdf.as_posix()}",
        1,
    )
    target = tmp_path / "packet-b-exclusions-v1.yaml"
    target.write_text(text, encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "mutator",
    [
        pytest.param(
            lambda text, manifest: text + "\nunexpected_key: 1\n",
            id="unknown-key",
        ),
        pytest.param(
            lambda text, manifest: text.replace(
                "status: diagnostic_proposal", "status: certificate"
            ),
            id="status-escalation",
        ),
        pytest.param(
            lambda text, manifest: text.replace(
                f"source_urdf_sha256: {manifest.urdf_sha256}",
                f"source_urdf_sha256: {manifest.urdf_sha256[:-1]}0",
            ),
            id="urdf-hash",
        ),
        pytest.param(
            lambda text, manifest: text.replace("motion_m: 0.014706151", "motion_m: 0.0147062"),
            id="motion-mismatch",
        ),
        pytest.param(
            lambda text, manifest: text.replace("reason: directly_jointed", "reason: jointed", 1),
            id="bogus-reason",
        ),
        pytest.param(
            lambda text, manifest: text.replace("retained: 109", "retained: 108", 1),
            id="retained-count",
        ),
        pytest.param(
            lambda text, manifest: re.sub(
                r"max_campaign_clearance_m: [0-9.]+",
                "max_campaign_clearance_m: 0.5",
                text,
                count=1,
            ),
            id="satisfiable-gripper-mount",
        ),
        pytest.param(
            lambda text, manifest: text.replace("gripper_mount: 16", "gripper_mount: 15", 1),
            id="gripper-mount-count",
        ),
        pytest.param(
            lambda text, manifest: re.sub(
                r"max_campaign_clearance_step: episode-\d+",
                "max_campaign_clearance_step: episode-99",
                text,
                count=1,
            ),
            id="witness-outside-campaign",
        ),
        pytest.param(
            lambda text, manifest: re.sub(
                r"max_campaign_clearance_step: episode-\d+:\d+",
                "max_campaign_clearance_step: later",
                text,
                count=1,
            ),
            id="malformed-witness",
        ),
    ],
)
def test_tampered_manifest_fails_closed(tmp_path, mutator):
    text = MANIFEST.read_text(encoding="utf-8")
    tampered = mutator(text, _manifest())
    assert tampered != text
    with pytest.raises(SweepGeometryError):
        load_packet_b_manifest(_mutate(tmp_path, tampered))


def test_dropping_one_retained_pair_breaks_the_partition(tmp_path):
    text = MANIFEST.read_text(encoding="utf-8")
    marker = "- pair:\n  - TIAGo front arm:0\n  - arm_6_link:0\n"
    assert marker in text
    stripped = text.replace(marker, "", 1)
    stripped = stripped.replace("retained: 109", "retained: 108", 1)
    stripped = stripped.replace(
        "retained_compatible_with_all_steps: 101",
        "retained_compatible_with_all_steps: 100",
        1,
    )
    with pytest.raises(SweepGeometryError):
        load_packet_b_manifest(_mutate(tmp_path, stripped))


def test_duplicate_gripper_mount_row_fails_closed(tmp_path):
    text = MANIFEST.read_text(encoding="utf-8")
    head, rows = text.split("gripper_mount_pairs:\n", 1)
    first_row = rows.split("\n- first:", 1)[0]
    tampered = head + "gripper_mount_pairs:\n" + rows.rstrip("\n") + "\n" + first_row + "\n"
    with pytest.raises(SweepGeometryError):
        load_packet_b_manifest(_mutate(tmp_path, tampered))


def test_dropping_a_pinned_mount_row_keeps_that_pair_vetoed(tmp_path):
    """Removing a measurement can only skip an exemption, so the pair stays fail-closed."""
    text = MANIFEST.read_text(encoding="utf-8")
    head, rows = text.split("gripper_mount_pairs:\n", 1)
    entries = rows.split("\n- first:")
    dropped_row = "\n- first:" + entries[1]
    tampered = (
        head
        + "gripper_mount_pairs:\n"
        + entries[0]
        + "".join("\n- first:" + item for item in entries[2:])
    )
    tampered = tampered.replace("gripper_mount: 16", "gripper_mount: 15", 1)
    manifest = load_packet_b_manifest(_mutate(tmp_path, tampered))
    match = re.search(r"- first: (front:\d+)\n  second: (\S+)", dropped_row)
    assert match is not None
    pair = frozenset((match.group(1), match.group(2)))
    assert pair not in {item.pair for item in manifest.gripper_mount_pairs}
    catalog = _catalog()
    narrowed = packet_b_narrowed_sweep_check(
        manifest,
        scene=catalog.geometry,
        expected_base_pose_world=catalog.expected_base_pose_world,
        required_object_ids=frozenset({"CAN"}),
        contact_object_ids=frozenset({"CAN"}),
        contact_permitted=None,
    )
    assert pair not in narrowed.inner.excluded_self_pairs


# ------------------------------------------------------------------------------- measured evidence


@needs_campaign
def test_every_pinned_gripper_mount_pair_is_measured_unsatisfiable():
    manifest = _manifest()
    results = evaluate_campaign_gripper_mounts(manifest, evidence_dir=CAMPAIGN)
    pinned = {
        (item.first, item.second): item.max_campaign_clearance_m
        for item in manifest.gripper_mount_pairs
    }
    assert len(results) == 16
    assert {result.pair for result in results} == set(pinned)
    for result in results:
        assert result.max_clearance_m == pytest.approx(pinned[result.pair], abs=1e-6)
        assert result.unsatisfiable_in_campaign is True


@needs_campaign
def test_pinned_witness_reproduces_the_measured_maximum():
    manifest = _manifest()
    for item in manifest.gripper_mount_pairs:
        episode, step = item.max_campaign_clearance_step.split(":")
        value = recorded_gripper_mount_clearance(
            manifest, item.first, item.second, episode, int(step)
        )
        assert value == pytest.approx(item.max_campaign_clearance_m, abs=1e-6)
        assert value < manifest.required_separation_m


@needs_campaign
def test_recorded_campaign_matches_the_committed_sha256_pin():
    report = verify_campaign_evidence(CAMPAIGN)
    assert report["status"] == "verified"
    assert report["files"] == 18
    assert report["steps"] == 13766


@needs_campaign
def test_tampered_evidence_pin_fails_closed(tmp_path):
    pin_path = EVIDENCE_PIN
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["files"][0]["sha256"] = "0" * 64
    tampered = tmp_path / "pin.json"
    tampered.write_text(json.dumps(pin), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_campaign_evidence(CAMPAIGN, pin_path=tampered)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["steps"] = int(pin["steps"]) + 1
    tampered.write_text(json.dumps(pin), encoding="utf-8")
    with pytest.raises(ValueError, match="totals"):
        verify_campaign_evidence(CAMPAIGN, pin_path=tampered)


@needs_campaign
def test_campaign_regression_reproduces_101_compatible_and_8_rejected():
    report = _campaign_report()
    assert report.episodes == 18
    assert report.steps == 13766
    assert len(report.pairs) == 109
    assert len(report.compatible_pairs) == 101
    assert len(report.rejected_pairs) == 8
    slacks = {row.pair: row.worst_slack_m for row in report.pairs}
    assert slacks[("TIAGo front arm_3:2", "arm_3_link:0")] == pytest.approx(-0.031549, abs=1e-6)
    assert slacks[("arm_6_link:0", "wrist_ft_tool_link:0")] == pytest.approx(-0.030749, abs=1e-6)
    assert slacks[("TIAGo front arm:2", "arm_2_link:1")] == pytest.approx(-0.021381, abs=1e-6)
    worst = min(report.pairs, key=lambda item: item.worst_slack_m)
    assert worst.worst_episode.startswith("episode-")
    assert worst.worst_step >= 0


# ------------------------------------------------------------------------------- narrowed checker


def test_structural_exclusions_keep_every_unproven_pair():
    manifest = _manifest()
    checker = _checker()
    reasons = Counter(item.reason for item in checker.structural_exclusions)
    assert reasons == {
        "same_rigid_body": 16,
        "directly_jointed": 35,
        "parallel_linkage": 9,
        "gripper_mount": 16,
    }
    pinned = {item.pair for item in manifest.gripper_mount_pairs}
    mount_pairs = {
        item.pair for item in checker.structural_exclusions if item.reason == "gripper_mount"
    }
    assert mount_pairs == pinned
    boxes = urdf_collision_boxes(
        manifest.urdf_path,
        ("front",),
        max_point_speed_m_s=manifest.speed_bound_m_s,
        clearance_by_link={"front": manifest.margin_m},
    )["front"]
    assert {box.name for box in boxes} == {f"front:{index}" for index in range(5)}
    # Only the measured-unsatisfiable wrist pairs are removed; the rest stay vetoed.
    for box in boxes:
        assert frozenset((box.name, "arm_5_link:0")) not in checker.inner.excluded_self_pairs
        if box.name == "front:0":
            assert frozenset((box.name, "arm_5_link:1")) in checker.inner.excluded_self_pairs
        else:
            assert frozenset((box.name, "arm_5_link:1")) not in checker.inner.excluded_self_pairs
    assert frozenset(("front:0", "wrist_ft_tool_link:0")) in checker.inner.excluded_self_pairs


def test_oriented_table_exposes_exact_top_and_conservative_aabb():
    catalog = _catalog()
    table = next(box for box in catalog.geometry.boxes if box.name == "table")
    assert table.exact is True
    assert table.axes_base is not None and table.half_extents is not None
    exact_top = table.center_base[2] + table.half_extents[2]
    aabb_top = table.center_base[2] + table.half_base[2]
    assert exact_top == pytest.approx(_TABLE_TOP_M, abs=1e-9)
    assert aabb_top == pytest.approx(_TABLE_AABB_TOP_M, abs=1e-9)
    assert aabb_top - exact_top > 0.06
    report = catalog.report()
    printed = next(row for row in report["static_boxes"] if row["name"] == "table")
    assert printed["axes_base"] is not None
    assert printed["half_extents"] is not None


@needs_campaign
def test_measured_table_clearance_witnesses_are_pinned():
    """Pin the two table margins cited by ADR-0061 and section 9 of the readiness doc.

    Over all 18 recorded episodes / 13,766 steps the tightest exact gripper-box/table
    clearance is 0.053529841 m at ``episode-11:253`` and the tightest AABB *lower bound*
    on any capsule/table pair is 0.148097 m at ``episode-06:306``. Both stay above the
    required 0.039807 m, so the recorded expert campaign is not table-blocked.
    """
    manifest = _manifest()
    catalog = _catalog()
    checker = _checker()
    required = manifest.required_separation_m
    table = static_box_primitive(next(box for box in catalog.geometry.boxes if box.name == "table"))

    episode = np.load(CAMPAIGN / "campaign" / "episode-11.steps.npz")
    state = recorded_world_state(episode, 253, catalog=catalog)
    boxes = checker.inner.box_geometry(state)
    tightest = min((primitive_clearance(box, table), box.name) for box in boxes)
    assert tightest[0] == pytest.approx(0.05352984062466254, abs=1e-9)
    assert tightest[1] == "gripper_right_finger_link:0"
    assert tightest[0] > required

    episode = np.load(CAMPAIGN / "campaign" / "episode-06.steps.npz")
    state = recorded_world_state(episode, 306, catalog=catalog)
    bounds = {
        capsule.name: primitive_lower_bound(capsule, table)
        for capsule in checker.inner.geometry(state)
    }
    name, bound = min(bounds.items(), key=lambda item: item[1])
    assert name == "TIAGo front arm:2"
    assert bound == pytest.approx(0.1480970663250895, abs=1e-9)
    assert bound > required


@needs_campaign
def test_strict_checker_rejects_a_recorded_expert_configuration():
    state, episode, step = _worst_recorded_state()
    checker = _checker()
    assert checker.evaluate(state, state) is False
    assert _violating_pairs(checker, state)
    assert episode.startswith("episode-") and step >= 0


@needs_campaign
def test_relaxed_control_clone_is_the_only_observed_accept_path():
    state, _, _ = _worst_recorded_state()
    checker = _checker()
    clone, violating = _control_clone(checker, state)
    assert checker.evaluate(state, state) is False
    assert clone.evaluate(state, state) is True
    assert len(violating) >= 1
    # The clone is a control, never a shipped binding: the manifest itself is unchanged.
    assert _manifest().counts["retained_rejected_at_some_step"] == 8


@needs_campaign
def test_gripper_below_the_permitted_table_clearance_is_rejected():
    """A non-penetrating fingertip closer than `required` to the table must still be vetoed."""
    catalog = _catalog()
    checker = _checker()
    state, _, _ = _worst_recorded_state()
    clone, _ = _control_clone(checker, state)
    assert clone.evaluate(state, state) is True

    geometry = catalog.geometry
    table = next(box for box in geometry.boxes if box.name == "table")
    table_primitive = static_box_primitive(table)
    boxes = clone.inner.box_geometry(state)
    name, gap = min(
        ((box.name, primitive_clearance(box, table_primitive)) for box in boxes),
        key=lambda item: item[1],
    )
    assert name.startswith("gripper_")
    required = checker.manifest.required_separation_m
    shift = gap - 0.02
    assert shift > 0.0
    raised = dataclasses.replace(
        table,
        center_base=(
            table.center_base[0],
            table.center_base[1],
            table.center_base[2] + shift,
        ),
    )
    raised_scene = dataclasses.replace(
        geometry,
        boxes=tuple(raised if box.name == "table" else box for box in geometry.boxes),
    )
    raised_table = static_box_primitive(raised)
    tightest = min(primitive_clearance(box, raised_table) for box in boxes)
    assert 0.0 < tightest < required
    # The raised table must not also break an arm capsule, otherwise the reject is ambiguous.
    assert all(
        primitive_clearance(capsule, raised_table) >= required
        for capsule in clone.inner.geometry(state)
    )
    raised_clone, _ = _control_clone(checker, state, scene=raised_scene)
    assert raised_clone.evaluate(state, state) is False


@needs_campaign
def test_arm_capsule_intersecting_the_table_is_rejected():
    state, _, _ = _worst_recorded_state()
    checker = _checker()
    clone, _ = _control_clone(checker, state)
    assert clone.evaluate(state, state) is True

    catalog = _catalog()
    geometry = catalog.geometry
    table = next(box for box in geometry.boxes if box.name == "table")
    exact_top = table.center_base[2] + table.half_extents[2]
    lowest = min(
        min(capsule.start[2], capsule.end[2]) - capsule.radius_m
        for capsule in clone.inner.geometry(state)
    )
    shift = (lowest + 0.01) - exact_top
    assert shift > 0.0
    raised_table = dataclasses.replace(
        table,
        center_base=(
            table.center_base[0],
            table.center_base[1],
            table.center_base[2] + shift,
        ),
    )
    raised_scene = dataclasses.replace(
        geometry,
        boxes=tuple(raised_table if box.name == "table" else box for box in geometry.boxes),
    )
    raised_clone, _ = _control_clone(checker, state, scene=raised_scene)
    assert raised_clone.evaluate(state, state) is False


@needs_campaign
def test_unintended_can_contact_is_rejected_unless_the_phase_permits_it():
    state, _, _ = _worst_recorded_state()
    checker = _checker()
    boxes = {box.name: box for box in checker.inner.box_geometry(state)}
    finger = boxes["gripper_right_finger_link:0"]
    can = next(item for item in _catalog().geometry.carried_objects if item.name == "CAN")
    can_dims = (
        float(can.radius_m),
        float(can.radius_m),
        float(can.height_m) / 2.0,
    )
    rotation = np.column_stack(finger.axes)
    w, x, y, z = _matrix_quaternion(rotation)
    axes = np.asarray(finger.axes, dtype=float)
    required = checker.manifest.required_separation_m
    others = [
        primitive
        for primitive in (*checker.inner.geometry(state), *boxes.values())
        if not (primitive.name.startswith("gripper_") and "finger" in primitive.name)
    ]
    best: tuple[float, np.ndarray, object] | None = None
    for axis in range(3):
        for sign in (1.0, -1.0):
            direction = sign * axes[axis]
            face = np.asarray(finger.center) + direction * float(finger.half_extents[axis])
            center = face + direction * (can_dims[axis] + 0.001)
            candidate = _object_box(
                ObjectState(
                    object_id="CAN",
                    label="CAN",
                    pose=Pose3D(Vec3(*map(float, center)), (w, x, y, z)),
                    dimensions=Vec3(
                        2.0 * float(can.radius_m),
                        2.0 * float(can.radius_m),
                        float(can.height_m),
                    ),
                    confidence=1.0,
                )
            )
            clearance = primitive_clearance(finger, candidate)
            if not 0.0 < clearance < required:
                continue
            margin = min(primitive_clearance(item, candidate) for item in others)
            if margin < required:
                continue
            if best is None or margin > best[0]:
                best = (margin, center, candidate)
    assert best is not None, "no CAN placement leaves every other primitive clear"
    _, center, can_box = best
    contact_state = dataclasses.replace(
        state,
        scene=dataclasses.replace(
            state.scene,
            objects=[
                ObjectState(
                    object_id="CAN",
                    label="CAN",
                    pose=Pose3D(Vec3(*map(float, center)), (w, x, y, z)),
                    dimensions=Vec3(
                        2.0 * float(can.radius_m),
                        2.0 * float(can.radius_m),
                        float(can.height_m),
                    ),
                    confidence=1.0,
                )
            ],
        ),
    )
    gap = primitive_clearance(finger, can_box)
    assert 0.0 < gap < required
    # Every non-finger primitive must stay clear, otherwise this would test another failure.
    for primitive in (*checker.inner.geometry(state), *boxes.values()):
        if primitive.name.startswith("gripper_") and "finger" in primitive.name:
            continue
        assert primitive_clearance(primitive, can_box) >= required

    catalog = _catalog()
    strict = packet_b_narrowed_sweep_check(
        _manifest(),
        scene=catalog.geometry,
        expected_base_pose_world=catalog.expected_base_pose_world,
        required_object_ids=frozenset({"CAN"}),
        contact_object_ids=frozenset({"CAN"}),
        contact_permitted=None,
    )
    strict.inner.excluded_self_pairs = frozenset(
        checker.inner.excluded_self_pairs | _violating_pairs(checker, state)
    )
    assert strict.evaluate(contact_state, contact_state) is False

    for predicate, expected in ((lambda _: True, True), (lambda _: False, False)):
        permitted = packet_b_narrowed_sweep_check(
            _manifest(),
            scene=catalog.geometry,
            expected_base_pose_world=catalog.expected_base_pose_world,
            required_object_ids=frozenset({"CAN"}),
            contact_object_ids=frozenset({"CAN"}),
            contact_permitted=predicate,
        )
        permitted.inner.excluded_self_pairs = strict.inner.excluded_self_pairs
        assert permitted.evaluate(contact_state, contact_state) is expected


def test_safety_monitor_still_rejects_joint_limit_and_velocity_violations():
    limits = runtime_safety_limits()
    monitor = SafetyMonitor(limits)
    joint = "arm_1_joint"
    lower, upper = limits.joint_position[joint]
    current = 0.5 * (lower + upper)
    state = RobotState(
        joint_pos={joint: current},
        joint_vel={},
        base_pose=Pose2D(0.0, 0.0, 0.0),
        ee_pose=Pose3D(Vec3(0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)),
        gripper_opening=0.02,
    )
    too_far = monitor.validate(JointTarget(q={joint: upper + 1.0}), state)
    assert too_far.accepted is False
    max_velocity = limits.joint_velocity[joint]
    step = max_velocity * limits.command_horizon * 2.0
    too_fast = monitor.validate(JointTarget(q={joint: current + step}), state)
    assert too_fast.accepted is False
    within = monitor.validate(
        JointTarget(q={joint: current + 0.5 * max_velocity * limits.command_horizon}), state
    )
    assert within.accepted is True


@needs_campaign
def test_checker_returns_none_for_malformed_or_missing_input():
    state, _, _ = _worst_recorded_state()
    checker = _checker()
    assert checker.evaluate(None, state) is None
    assert checker.evaluate(state, "not a state") is None
    broken = dataclasses.replace(
        state,
        robot=dataclasses.replace(state.robot, gripper_opening=math.nan),
    )
    assert checker.evaluate(broken, broken) is None


def test_shipped_contract_stays_blocked_and_production_bindings_stay_empty():
    config = load_arm_sweep_config(ARM_SWEEP_CONFIG_PATH)
    assert config.status == "blocked"
    assert config.certificate is None
    bindings = production_worldmind_bindings()
    assert dict(bindings.components) == {}
    assert dict(bindings.calibration_evaluators) == {}
    package = __import__("vlarobot.backends.worldmind_bindings", fromlist=["x"])
    assert package.PACKET_B_MANIFEST_PATH == MANIFEST.resolve()


def _matrix_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    norm = float(np.linalg.norm(values))
    return tuple(float(value) / norm for value in values)
