"""Read-only measured TRAIN replay using the immutable accepted main-worktree FK adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--attachment-only", action="store_true")
    args = parser.parse_args()
    isolated = Path(__file__).resolve().parents[1]
    project = args.project.resolve()
    if args.out.exists():
        raise FileExistsError("preserve measured replay evidence")
    sys.path.insert(0, str(project / "src"))
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = "calibration/urdf-base-train-v1.yaml"
    from vlarobot.backends.webots.pose_solver import BoundedPoseSolver
    from vlarobot.vla.settings import load_vla_settings

    import vlarobot
    import vlarobot.execution
    from vlarobot.backends.webots.backend import _rotation_to_quaternion
    from vlarobot.backends.webots.config import ARM_JOINTS, SETTINGS
    from vlarobot.backends.webots.kinematics import Kinematics
    from vlarobot.core.types import (
        ActionChunk,
        EEDelta,
        JointTarget,
        Pose2D,
        Pose3D,
        RobotState,
        SceneState,
        Vec3,
        WorldState,
    )

    # Composition only: frozen registered FK comes from main, new pure dynamics from this branch.
    vlarobot.__path__.insert(0, str(isolated / "src/vlarobot"))
    vlarobot.execution.__path__.insert(0, str(isolated / "src/vlarobot/execution"))
    from vlarobot.execution.replay_errors import recorded_chunk_errors
    from vlarobot.world_model.kinematic import VelocityLimitedKinematics

    kin = Kinematics()
    collection = project / "runs/demos/train-s100-registered-v1"
    provenance = json.loads((collection / "provenance.json").read_text(encoding="utf-8"))
    assert kin.registration_sha256 == provenance["registration_sha256"]

    def forward(joints):
        q = np.zeros(len(kin.chain.links))
        for name, value in joints.items():
            q[kin.idx[name]] = value
        matrix = kin.forward(q)
        return Pose3D(Vec3(*map(float, matrix[:3, 3])), _rotation_to_quaternion(matrix[:3, :3]))

    if args.attachment_only:
        from vlarobot.execution.attachment_validation import conditional_attachment_errors

        report = conditional_attachment_errors(
            collection,
            forward,
            _rotation_to_quaternion,
            joint_names=ARM_JOINTS,
            speed_rad_s=0.5,
            horizons=(1, 10, 50),
        )
        report["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                isolated / "src/vlarobot/execution/attachment_validation.py",
                isolated / "src/vlarobot/world_model/rollout.py",
                project / "configs/calibration/urdf-base-train-v1.yaml",
                collection / "accepted.json",
            )
        }
        args.out.mkdir(parents=True)
        (args.out / "conditional-attachment-errors.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return
    started = perf_counter()
    if args.replay_report:
        report = json.loads(args.replay_report.read_text(encoding="utf-8"))
        for name, digest in report["source_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError("cached replay source changed")
        report["cached_replay_report"] = str(args.replay_report.resolve())
    else:
        report = recorded_chunk_errors(
            collection,
            forward,
            joint_names=ARM_JOINTS,
            speed_rad_s=0.5,
            horizons=(1, 10, 50),
            stride=50,
        )
        report["replay_wall_s"] = perf_counter() - started
    first = json.loads((collection / "accepted.json").read_text(encoding="utf-8"))["episodes"][0]
    with np.load(Path(first["raw_dir"]) / "episode-00.steps.npz") as a:
        i = len(a["t"]) // 2
        joints = dict(zip(a["joint_names"].tolist(), map(float, a["joint_pos"][i]), strict=True))
    state = RobotState(joints, {}, Pose2D(0, 0, 0), forward(joints), 0.03)
    tracking = VelocityLimitedKinematics({name: 0.5 for name in ARM_JOINTS}, forward)
    times = []
    for _ in range(7):
        started = perf_counter()
        for _ in range(32):
            current = state
            for _ in range(50):
                current = tracking.step(
                    current, JointTarget({n: joints[n] + 0.01 for n in ARM_JOINTS}), 0.032
                )
        times.append(perf_counter() - started)
    solver = BoundedPoseSolver(kin, max_joint_step_rad=SETTINGS.max_joint_step_rad)
    ik_times = []
    for _ in range(12):
        target = Pose3D(
            Vec3(
                state.ee_pose.position.x + 0.001, state.ee_pose.position.y, state.ee_pose.position.z
            ),
            state.ee_pose.orientation,
        )
        started = perf_counter()
        solution = solver.solve(target, state)
        ik_times.append(perf_counter() - started)
    report["latency"] = {
        "registered_fk_32_chunks_x50_steps_s": {
            "p50": float(np.median(times[1:])),
            "p95": float(np.percentile(times[1:], 95)),
        },
        "registered_full_pose_ik_s": {
            "p50": float(np.median(ik_times[2:])),
            "p95": float(np.percentile(ik_times[2:], 95)),
        },
        "serial_1600_ik_p50_projection_s": float(np.median(ik_times[2:]) * 1600),
        "ik_last_position_error_m": solution.position_error_m,
        "excludes_model_inference_full_arm_sweep_and_state_copy": True,
        "not_full_pipeline_acceptance": True,
    }
    from vlarobot.world_model.linearized_ik import LinearizedIK
    from vlarobot.world_model.rollout import AnalyticSurrogate

    settings = load_vla_settings()
    linear = LinearizedIK(
        forward,
        bounds={n: kin.bounds(n) for n in ARM_JOINTS},
        max_joint_step_rad=SETTINGS.max_joint_step_rad,
        position_tolerance_m=settings.ik_position_tolerance_m,
        orientation_tolerance_rad=settings.ik_orientation_tolerance_rad,
        difference_step_rad=1e-5,
        damping=1e-6,
    )
    # No full-arm geometry certificate is supplied: hard_reasons must retain uncertainty.
    model = AnalyticSurrogate(
        tracking,
        inverse=linear.solve,
        prepare_inverse=linear.reset,
        gripper_speed_m_s=1.0,
        gravity_m_s2=9.81,
        support_z_m={},
        validate=lambda a, s: True,
    )
    chunk = ActionChunk([EEDelta(Vec3(0.0005, 0, 0), Vec3(0, 0, 0), 0.03)] * 50, 0.032, "vla")
    times = []
    incomplete = 0
    for _ in range(4):
        started = perf_counter()
        for _ in range(32):
            trajectory = model.predict_trajectory(WorldState(SceneState(), state), chunk)
            incomplete += len(trajectory.states) != 51
        times.append(perf_counter() - started)
    report["latency"]["linearized_32_chunks_x50_states_s"] = {
        "p50": float(np.median(times[1:])),
        "p95": float(np.percentile(times[1:], 95)),
        "incomplete_trajectories": incomplete,
        "batches": 4,
        "includes_state_copy_ik_and_fk": True,
        "excludes_inference_full_arm_geometry": True,
        "fixture": "repeated reachable 0.5mm base-x deltas; fixed measured TRAIN joint seed",
        "numerics": {"central_difference_rad": 1e-5, "damping": 1e-6},
        "not_execution_ik_or_task_success": True,
    }
    report["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            isolated / "src/vlarobot/execution/replay_errors.py",
            isolated / "src/vlarobot/world_model/kinematic.py",
            project / "src/vlarobot/backends/webots/kinematics.py",
            project / "configs/calibration/urdf-base-train-v1.yaml",
        ]
    }
    args.out.mkdir(parents=True)
    (args.out / "registered-replay.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("sources", "source_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
