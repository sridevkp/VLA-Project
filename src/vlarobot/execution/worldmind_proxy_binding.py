"""Bind the November 24 proxy scope to pinned scene and surrogate inputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from vlarobot.backends.worldmind_bindings import GRASP_Z, PLACE_CLEAR
from vlarobot.core.types import (
    ActionChunk,
    GroundTruth,
    ObjectState,
    Pose3D,
    SceneState,
    Vec3,
    WorldState,
)
from vlarobot.eval.worldmind_objectives import MissingObjectiveInput
from vlarobot.eval.worldmind_proxy_policy import (
    assess_proxy,
    observe_proxy_clearance,
    selections_by_policy,
)
from vlarobot.execution.safety_monitor import SafetyMonitor
from vlarobot.execution.worldmind_pipeline import _inverse_pose  # noqa: PLC2701
from vlarobot.risk.gate import CandidateAssessment
from vlarobot.world_model.rollout import AnalyticSurrogate, compose

_MODEL_PINS = {
    "TiagoGripper.proto": "2519a5091aee31a682df489a568c30f77bed170be00054dd71ef3b621862eec9",
    "Hammer.proto": "9672241772c97d74a21244ddaec36ad5f01677fc708d469b8938d8786698b736",
    "Apple.proto": "86535a2243d9ee303e57292c134d4fdf4e24f1ee9fd34d536c4da208781e3c2e",
}
_TABLE_SHA = "9258b376f80309ed074130c61e784b9e0c18c71d48b980102d364dd0e60d66e7"
# HAM: union bound of four boxes and the rotated handle cylinder in pinned Hammer.proto.
# APP: 0.05 m sphere with the default scale of 1 in the factory world.
_OBSTACLE_BOUNDS = {
    "HAM": (Vec3(0.0055, 0, 0), Vec3(0.267, 0.099, 0.027)),
    "APP": (Vec3(0, 0, 0), Vec3(0.1, 0.1, 0.1)),
}
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
PLACEMENT_STATE_INDEX = 43


def verified_table_world(root: Path) -> tuple[ObjectState, ...]:
    """Load only the five world-frame tables matched against live runtime geometry."""
    for name, digest in _MODEL_PINS.items():
        model = root / "assets/webots/protos/r2025a" / name
        if hashlib.sha256(model.read_bytes()).hexdigest() != digest:
            raise ValueError(f"proxy model provenance mismatch: {name}")

    source = root / "runs/worldmind/geometry-inspection-v5-2026-11-19/geometry.json"
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _TABLE_SHA:
        raise ValueError("verified table evidence hash mismatch")
    evidence = json.loads(payload)["table_target_reconciliation"]
    if evidence["runtime_grounded"] is not True or evidence["blockers"]:
        raise ValueError("table evidence is not runtime-grounded")
    boxes = evidence["expected_source_geometry"]["boxes"]
    if {box["name"] for box in boxes} != {"main frame", *(f"foot {i}" for i in range(1, 5))}:
        raise ValueError("verified table geometry is incomplete")
    if any(box["world_rotation_axis_angle"] != [0.0, 0.0, 1.0, 0.0] for box in boxes):
        raise ValueError("expected verified unrotated table boxes")
    return tuple(
        ObjectState(
            f"TABLE:{box['name']}",
            "table",
            Pose3D(Vec3(*box["world_translation_m"]), _IDENTITY),
            Vec3(*box["size_m"]),
            1.0,
        )
        for box in boxes
    )


def evaluate_proxy_pool(
    state: WorldState,
    truth: GroundTruth,
    pool: Sequence[ActionChunk],
    monitor: SafetyMonitor,
    predictor: AnalyticSurrogate,
    tables_world: Sequence[ObjectState],
    config: Any,
) -> tuple[tuple[CandidateAssessment, ...], dict[str, tuple[int | None, ...]]]:
    """Evaluate modelled CAN/HAM/APP and table geometry without a full-arm claim."""
    base = state.robot.base_pose_world
    if base is None or set(truth.object_poses) != {"CAN", "HAM", "APP"}:
        raise MissingObjectiveInput("proxy scene requires exact CAN/HAM/APP and base pose")
    world_to_base = _inverse_pose(base)
    obstacles = [
        ObjectState(
            name,
            name,
            compose(compose(world_to_base, truth.object_poses[name]), Pose3D(center, _IDENTITY)),
            size,
            1.0,
        )
        for name, (center, size) in _OBSTACLE_BOUNDS.items()
    ]
    complete = replace(state, scene=SceneState([*state.scene.objects, *obstacles]))
    tables = [replace(table, pose=compose(world_to_base, table.pose)) for table in tables_world]
    destination = compose(
        world_to_base, Pose3D(Vec3(*config.verification.destination_world), _IDENTITY)
    ).position
    tool_goal = Vec3(destination.x, destination.y, destination.z + GRASP_Z + PLACE_CLEAR)
    results = []
    for chunk in pool:
        prediction = predictor.predict_trajectory(state, chunk)
        try:
            limits = monitor.limits.joint_position
            positions = [item.robot.joint_pos for item in prediction.states]
            joint_pass = bool(limits) and all(
                name in joints and lo < joints[name] < hi
                for joints in positions
                for name, (lo, hi) in limits.items()
            )
            evidence = observe_proxy_clearance(
                complete,
                chunk,
                prediction,
                table_boxes=tables,
                expected_object_ids=frozenset({"CAN", "HAM", "APP"}),
                joint_limits_pass=joint_pass,
                # Pool generation completed the existing waypoint IK/residual gates.
                ik_pass=True,
            )
            assessment = assess_proxy(
                complete,
                chunk,
                prediction,
                evidence,
                placement_state_index=PLACEMENT_STATE_INDEX,
                placement_tool_goal_base=tool_goal,
                placement_tolerance_m=config.verification.distance_tolerance_m,
            )
        except MissingObjectiveInput:
            # An incomplete analytic rollout cannot be ranked or treated as clear.
            if len(prediction.states) == len(chunk.actions) + 1:
                raise
            assessment = CandidateAssessment(
                (("prediction_complete", False),), (0.0, 0.0, 0.0, 0.0)
            )
        results.append(assessment)
    if len(results) != 32 or any(not math.isfinite(v) for row in results for v in row.components):
        raise MissingObjectiveInput("proxy ranking requires 32 reconciled assessments")
    return tuple(results), selections_by_policy((results,), include_optional=True)
