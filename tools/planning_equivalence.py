"""Identical 32x50 inputs: frozen-reference versus optimized nominal planning and vetoes."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import sys
import types
from pathlib import Path
from time import perf_counter

import numpy as np


def reference(name: str, path: Path, replacements: dict[str, str]) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    source = path.read_text(encoding="utf-8")
    for a, b in replacements.items():
        source = source.replace(a, b)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-valid", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve earlier performance evidence")
    args.out.mkdir(parents=True)
    isolated = Path(__file__).resolve().parents[1]
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = "calibration/urdf-base-train-v1.yaml"
    from vlarobot.execution.vla_campaign import make_vla_adapter
    from vlarobot.vla.settings import load_vla_settings

    import vlarobot
    from vlarobot.backends.webots.backend import _rotation_to_quaternion
    from vlarobot.backends.webots.config import ARM_JOINTS, SETTINGS
    from vlarobot.backends.webots.kinematics import Kinematics

    vlarobot.__path__.insert(0, str(isolated / "src/vlarobot"))
    from vlarobot.core.types import (
        ActionChunk,
        EEDelta,
        Pose2D,
        Pose3D,
        RobotState,
        SceneState,
        Vec3,
        WorldState,
    )
    from vlarobot.world_model.kinematic import MemoizedForward, VelocityLimitedKinematics
    from vlarobot.world_model.linearized_ik import LinearizedIK
    from vlarobot.world_model.prepared_fk import PreparedForward
    from vlarobot.world_model.rollout import AnalyticSurrogate

    kin = Kinematics()
    settings = load_vla_settings()
    adapter = make_vla_adapter(kin.registration_sha256)
    monitor = adapter.safety_monitor

    def forward(joints):
        q = np.zeros(len(kin.chain.links))
        for n, v in joints.items():
            q[kin.idx[n]] = v
        matrix = kin.forward(q)
        return Pose3D(Vec3(*map(float, matrix[:3, 3])), _rotation_to_quaternion(matrix[:3, :3]))

    raw = isolated / "runs/validation-evidence-v1/campaign/episode-00.steps.npz"
    with np.load(raw, allow_pickle=False) as a:
        index = len(a["t"]) // 2
        joints = dict(
            zip(a["joint_names"].tolist(), map(float, a["joint_pos"][index]), strict=True)
        )
    state = WorldState(SceneState(), RobotState(joints, {}, Pose2D(0, 0, 0), forward(joints), 0.03))
    folder = isolated / "tests/fixtures/planning_reference"
    rk = reference("reference_kinematic", folder / "kinematic.py", {})
    rr = reference(
        "reference_rollout",
        folder / "rollout.py",
        {"from vlarobot.world_model.kinematic import": "from reference_kinematic import"},
    )
    ri = reference(
        "reference_ik",
        folder / "linearized_ik.py",
        {"from vlarobot.world_model.rollout import": "from reference_rollout import"},
    )

    prepared = PreparedForward(
        [(link.name, link.get_link_frame_matrix) for link in kin.chain.links],
        active_joints=frozenset(ARM_JOINTS),
        reference_joints=joints,
        to_pose=lambda matrix: Pose3D(
            Vec3(*map(float, matrix[:3, 3])), _rotation_to_quaternion(matrix[:3, :3])
        ),
        geometry_id=kin.registration_sha256,
    )
    fk_rng = np.random.default_rng(917)
    max_fk_difference = 0.0
    for _ in range(100):
        probe = dict(joints)
        for name in ARM_JOINTS:
            probe[name] = float(fk_rng.uniform(*kin.bounds(name)))
        a, b = forward(probe), prepared(probe)
        max_fk_difference = max(
            max_fk_difference,
            float(
                np.max(
                    np.abs(
                        np.r_[a.position.as_array(), a.orientation]
                        - np.r_[b.position.as_array(), b.orientation]
                    )
                )
            ),
        )
    if max_fk_difference != 0:
        raise AssertionError("prepared FK is not exactly equivalent")

    def model(old):
        fk = forward if old else MemoizedForward(prepared, geometry_id=kin.registration_sha256)
        solver = (ri.LinearizedIK if old else LinearizedIK)(
            fk,
            bounds={n: kin.bounds(n) for n in ARM_JOINTS},
            max_joint_step_rad=SETTINGS.max_joint_step_rad,
            position_tolerance_m=settings.ik_position_tolerance_m,
            orientation_tolerance_rad=settings.ik_orientation_tolerance_rad,
            difference_step_rad=1e-5,
            damping=1e-6,
        )

        def prepare(robot):
            if not old:
                fk.clear()
            solver.reset(robot)

        tracking = (rk.VelocityLimitedKinematics if old else VelocityLimitedKinematics)(
            {n: 0.5 for n in ARM_JOINTS}, fk
        )
        return (rr.AnalyticSurrogate if old else AnalyticSurrogate)(
            tracking,
            inverse=solver.solve,
            prepare_inverse=prepare,
            gripper_speed_m_s=1.0,
            gravity_m_s2=9.81,
            support_z_m={},
            validate=lambda action, robot: monitor.validate(action, robot).accepted,
        )

    before, after = model(True), model(False)
    rng = np.random.default_rng(20261115)
    chunks = [
        ActionChunk(
            [
                EEDelta(
                    Vec3(*map(float, rng.normal(0, 0.0005, 3))),
                    Vec3(*map(float, rng.normal(0, 0.0001, 3))),
                    0.03,
                )
                for _ in range(50)
            ],
            0.032,
            "vla",
        )
        for _ in range(32)
    ]
    if not args.all_valid:
        chunks[-1].actions[0] = EEDelta(Vec3(2.0, 0, 0), Vec3(0, 0, 0), 0.03)
    inputs = np.array(
        [[(*a.dpos.as_array(), *a.drot.as_array(), a.gripper) for a in c.actions] for c in chunks]
    )
    np.savez(
        args.out / "inputs.npz",
        state_joint_names=list(joints),
        state_joint_values=list(joints.values()),
        actions=inputs,
    )

    def run(m):
        return [m.predict_trajectory(state, c) for c in chunks]

    old, new = run(before), run(after)
    maximum = 0.0
    for a, b in zip(old, new, strict=True):
        if a.hard_reasons != b.hard_reasons or len(a.states) != len(b.states):
            raise AssertionError("veto/horizon divergence")
        for x, y in zip(a.states, b.states, strict=True):
            aa = [
                *x.robot.joint_pos.values(),
                *x.robot.joint_vel.values(),
                *x.robot.ee_pose.position.as_array(),
                *x.robot.ee_pose.orientation,
                x.robot.gripper_opening,
            ]
            bb = [
                *y.robot.joint_pos.values(),
                *y.robot.joint_vel.values(),
                *y.robot.ee_pose.position.as_array(),
                *y.robot.ee_pose.orientation,
                y.robot.gripper_opening,
            ]
            maximum = max(maximum, float(np.max(np.abs(np.asarray(aa) - bb))))
    if maximum > 1e-10:
        raise AssertionError("numerical equivalence failed")
    times = {"before": [], "after": []}
    for repeat in range(5):
        for label, m in (
            [("before", before), ("after", after)]
            if repeat % 2 == 0
            else [("after", after), ("before", before)]
        ):
            start = perf_counter()
            run(m)
            times[label].append(perf_counter() - start)
    cold_after = model(False)
    cold_started = perf_counter()
    cold_trajectories = run(cold_after)
    cold_after_s = perf_counter() - cold_started
    if sum(len(t.states) - 1 for t in cold_trajectories) != sum(len(t.states) - 1 for t in new):
        raise AssertionError("cold-cache candidate horizon differs")
    components = {}
    for label, m in [("before", before), ("after", after)]:
        profile = cProfile.Profile()
        profile.runcall(run, m)
        profile.dump_stats(str(args.out / (label + ".pstats")))
        stats = pstats.Stats(profile).stats
        components[label] = {
            name: sum(v[3] for k, v in stats.items() if predicate(k))
            for name, predicate in {
                "FK_inclusive_s": lambda k: (
                    k[2] == "forward_kinematics"
                    or (k[2] == "__call__" and k[0].endswith("prepared_fk.py"))
                ),
                "Jacobian_inclusive_s": lambda k: k[2] == "_refresh",
                "IK_inclusive_s": lambda k: k[2] == "solve",
                "state_deepcopy_inclusive_s": lambda k: k[2] == "deepcopy",
                "world_copy_inclusive_s": lambda k: k[2] == "copy_world",
                "robot_copy_inclusive_s": lambda k: k[2] == "copy_robot",
            }.items()
        }
    report = {
        "status": "MEASURED identical inputs/reference versus optimized nominal planning",
        "candidate_count": 32,
        "horizon": 50,
        "completed_control_steps_per_batch": sum(len(t.states) - 1 for t in new),
        "all_valid_fixture_requested": args.all_valid,
        "seed": 20261115,
        "initialization": "actual VALIDATION episode-00 middle observation",
        "max_absolute_state_difference": maximum,
        "all_vetoes_identical": True,
        "max_prepared_fk_difference_100_joint_configurations": max_fk_difference,
        "inputs_sha256": hashlib.sha256((args.out / "inputs.npz").read_bytes()).hexdigest(),
        "cold_after_full_batch_s": cold_after_s,
        "timings_s": {
            label: {
                "samples": values,
                "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for label, values in times.items()
        },
        "components_profiled_inclusive_not_additive": components,
        "uncertified_full_arm_sweep_remains_vetoed": True,
        "includes_model_inference": False,
        "safety_and_trust_thresholds_unchanged": True,
        "test_accessed": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*folder.glob("*.py"), *(isolated / "src/vlarobot/world_model").glob("*.py")]
        },
    }
    (args.out / "equivalence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
