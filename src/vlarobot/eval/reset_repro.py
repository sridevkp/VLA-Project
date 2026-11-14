"""Reset reproducibility evidence: fresh session vs reused session (ADR-0029 discipline).

Compares a fixed seeded scenario executed (a) in a fresh Webots session and (b) after a
different scenario in a reused session. Evidence is MEASURED only when run against Webots.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.core.types import GroundTruth, Observation, ScenarioSpec
from vlarobot.eval.scenarios import generate_scenes
from vlarobot.execution.episode_runner import DEFAULT_WORLD, launch_webots_session


@dataclass(frozen=True)
class ResetSnapshot:
    scenario_name: str
    seed: int
    t: float
    joint_pos: dict[str, float]
    joint_vel: dict[str, float]
    gripper_opening: float
    base_pose: tuple[float, float, float]
    base_pose_world_pos: tuple[float, float, float]
    base_pose_world_quat: tuple[float, float, float, float]
    ee_pos: tuple[float, float, float]
    ee_quat: tuple[float, float, float, float]
    object_poses: dict[
        str, dict[str, tuple[float, float, float] | tuple[float, float, float, float]]
    ]


def snapshot_from(
    obs: Observation, gt: GroundTruth | None, scenario: ScenarioSpec, seed: int
) -> ResetSnapshot:
    robot = obs.robot
    objects: dict[
        str, dict[str, tuple[float, float, float] | tuple[float, float, float, float]]
    ] = {}
    poses = gt.object_poses if gt is not None else {}
    for name, pose in sorted(poses.items()):
        objects[name] = {
            "pos": tuple(float(v) for v in pose.position.as_array()),
            "quat": tuple(float(v) for v in pose.orientation),
        }
    return ResetSnapshot(
        scenario_name=scenario.name,
        seed=int(seed),
        t=float(obs.t),
        joint_pos={k: float(v) for k, v in sorted(robot.joint_pos.items())},
        joint_vel={k: float(v) for k, v in sorted(robot.joint_vel.items())},
        gripper_opening=float(robot.gripper_opening),
        base_pose=(float(robot.base_pose.x), float(robot.base_pose.y), float(robot.base_pose.yaw)),
        base_pose_world_pos=tuple(float(v) for v in robot.base_pose_world.position.as_array()),
        base_pose_world_quat=tuple(float(v) for v in robot.base_pose_world.orientation),
        ee_pos=tuple(float(v) for v in robot.ee_pose.position.as_array()),
        ee_quat=tuple(float(v) for v in robot.ee_pose.orientation),
        object_poses=objects,
    )


@dataclass(frozen=True)
class ComparisonResult:
    ok: bool
    max_abs_diff: float
    diffs: dict[str, float]
    tolerances: dict[str, float]
    notes: list[str]


def _quat_angle_diff(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> float:
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    a, b = a / na, b / nb
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.acos(dot))


def compare_snapshots(a: ResetSnapshot, b: ResetSnapshot) -> ComparisonResult:
    pos_tol = 5e-4  # m
    joint_tol = 5e-4  # rad
    quat_tol = 1e-3  # rad (angle)
    vel_tol = 1e-2
    grip_tol = 1e-4
    diffs: dict[str, float] = {}
    notes: list[str] = []

    def rec(name: str, value: float) -> None:
        diffs[name] = max(diffs.get(name, 0.0), float(value))

    for k in sorted(set(a.joint_pos) | set(b.joint_pos)):
        rec("joint_pos", abs(a.joint_pos.get(k, 0.0) - b.joint_pos.get(k, 0.0)))
    for k in sorted(set(a.joint_vel) | set(b.joint_vel)):
        rec("joint_vel", abs(a.joint_vel.get(k, 0.0) - b.joint_vel.get(k, 0.0)))
    rec("gripper", abs(a.gripper_opening - b.gripper_opening))
    rec("base_pose", max(abs(x - y) for x, y in zip(a.base_pose, b.base_pose, strict=True)))
    rec(
        "base_pose_world_pos",
        max(abs(x - y) for x, y in zip(a.base_pose_world_pos, b.base_pose_world_pos, strict=True)),
    )
    rec("base_pose_world_quat", _quat_angle_diff(a.base_pose_world_quat, b.base_pose_world_quat))
    rec("ee_pos", max(abs(x - y) for x, y in zip(a.ee_pos, b.ee_pos, strict=True)))
    rec("ee_quat", _quat_angle_diff(a.ee_quat, b.ee_quat))

    keys = sorted(set(a.object_poses) | set(b.object_poses))
    if set(a.object_poses) != set(b.object_poses):
        notes.append(f"object set differs: {sorted(a.object_poses)} vs {sorted(b.object_poses)}")
        rec("object_set", 1.0)
    for name in keys:
        pa = a.object_poses.get(name)
        pb = b.object_poses.get(name)
        if pa is None or pb is None:
            continue
        rec(f"obj.{name}.pos", max(abs(x - y) for x, y in zip(pa["pos"], pb["pos"], strict=True)))  # type: ignore[arg-type]
        rec(f"obj.{name}.quat", _quat_angle_diff(pa["quat"], pb["quat"]))  # type: ignore[arg-type]

    tolerances = {
        "joint_pos": joint_tol,
        "joint_vel": vel_tol,
        "gripper": grip_tol,
        "base_pose": pos_tol,
        "base_pose_world_pos": pos_tol,
        "base_pose_world_quat": quat_tol,
        "ee_pos": pos_tol,
        "ee_quat": quat_tol,
        "object_set": 0.0,
    }
    # per-object tolerances same as pose/quat
    for k in list(diffs):
        if k.startswith("obj.") and k.endswith(".pos"):
            tolerances[k] = pos_tol
        elif k.startswith("obj.") and k.endswith(".quat"):
            tolerances[k] = quat_tol

    ok = True
    for k, v in diffs.items():
        tol = tolerances.get(k, pos_tol)
        if not (math.isfinite(v) and v <= tol + 1e-12):
            ok = False
            notes.append(f"{k} diff={v} > tol={tol}")
    max_abs = max(diffs.values()) if diffs else 0.0
    return ComparisonResult(
        ok=ok, max_abs_diff=max_abs, diffs=diffs, tolerances=tolerances, notes=notes
    )


def capture_fresh(client: Any, target: ScenarioSpec, seed: int) -> ResetSnapshot:
    obs = client.reset(target, seed)
    gt = client.get_ground_truth() if hasattr(client, "get_ground_truth") else None
    return snapshot_from(obs, gt, target, seed)


def capture_after_other(
    client: Any, target: ScenarioSpec, other: ScenarioSpec, seed: int
) -> ResetSnapshot:
    client.reset(other, seed)
    for _ in range(30):
        client.step(1)
    obs = client.reset(target, seed)
    gt = client.get_ground_truth() if hasattr(client, "get_ground_truth") else None
    return snapshot_from(obs, gt, target, seed)


def run_reset_repro(
    *,
    webots: str | None,
    world: Path | None,
    out_dir: Path,
    split: str = "train",
    seed: int = 0,
    headless: bool = False,
    startup_timeout_s: float = 180.0,
) -> dict[str, Any]:
    """MEASURED (Webots): compare fresh-session target reset vs reused-session-after-other reset."""
    from vlarobot.eval.metrics import write_json

    scenes = generate_scenes(split, seed)
    target = scenes[0]
    other = scenes[1] if len(scenes) > 1 else scenarios_fallback(target)
    world_path = world or DEFAULT_WORLD
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) fresh session
    with launch_webots_session(
        webots=webots,
        world=world_path,
        trace_dir=out_dir / "fresh",
        trace_prefix="reset-repro-fresh",
        headless=headless,
        startup_timeout_s=startup_timeout_s,
    ) as session:
        fresh = capture_fresh(session.client, target, seed)

    # (b) reused session: perturb with other, then reset to target
    with launch_webots_session(
        webots=webots,
        world=world_path,
        trace_dir=out_dir / "reused",
        trace_prefix="reset-repro-reused",
        headless=headless,
        startup_timeout_s=startup_timeout_s,
    ) as session:
        reused = capture_after_other(session.client, target, other, seed)

    comp = compare_snapshots(fresh, reused)
    report = {
        "measurement": "measured",
        "backend": "webots",
        "split": split,
        "seed": seed,
        "target_scenario": target.name,
        "other_scenario": other.name,
        "fresh": asdict(fresh),
        "reused": asdict(reused),
        "comparison": {
            "ok": comp.ok,
            "max_abs_diff": comp.max_abs_diff,
            "diffs": comp.diffs,
            "tolerances": comp.tolerances,
            "notes": comp.notes,
        },
        "tolerance_policy": {
            "position_m": 5e-4,
            "joint_rad": 5e-4,
            "quaternion_angle_rad": 1e-3,
            "joint_vel": 1e-2,
            "gripper_m": 1e-4,
            "note": "differences <= tolerance treated as reset-equivalent for this check",
        },
    }
    write_json(report, out_dir / "reset_repro.json")
    return report


def scenarios_fallback(target: ScenarioSpec) -> ScenarioSpec:
    return ScenarioSpec(name=f"{target.name}-alt", can_xy_yaw_offset=(0.03, -0.03, 0.1))
