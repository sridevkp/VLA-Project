"""Small predeclared independent VALIDATION chunks, fresh physical sessions, no training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-validation", type=Path)
    parser.add_argument("--webots", required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    isolated = Path(__file__).resolve().parents[1]
    if args.out.exists():
        raise FileExistsError("preserve diagnostic version")
    args.out.mkdir(parents=True)
    code_sha_at_launch = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan = [
        {
            "kind": kind,
            "sample_seed": 4100 + i,
            "prefix": prefix,
            "scene_index": 0,
            "scene_seed": 0,
            "steps": 50,
        }
        for i, (kind, prefix) in enumerate(
            [
                ("free_joint", None),
                ("closure", "DESCEND"),
                ("carried_joint", "LIFT"),
                ("release", "LIFT"),
            ]
        )
    ]
    (args.out / "plan.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "cases": plan,
                "predeclared": True,
                "sampling": "Gaussian joint-target draw, std 0.015 rad; no retries/tuning",
                "not_demonstrations": True,
                "test_accessed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.wait_validation:
        while True:
            try:
                status = json.loads(args.wait_validation.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(15)
                continue
            if status["status"] == "completed":
                break
            if status["status"] == "failed-preserving-evidence":
                raise RuntimeError("validation source campaign failed")
            time.sleep(15)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code_sha_at_launch:
        raise RuntimeError("driver changed while waiting; preserve this version and restart")
    os.environ["VLAROBOT_URDF_REGISTRATION_FILE"] = "calibration/urdf-base-train-v1.yaml"
    sys.path.insert(0, str(project / "src"))
    import vlarobot
    from vlarobot.backends.webots.backend import _rotation_to_quaternion
    from vlarobot.backends.webots.config import ARM_JOINTS, FINGER_SPEED
    from vlarobot.backends.webots.kinematics import Kinematics
    from vlarobot.core.types import (
        ActionChunk,
        GripperCmd,
        JointTarget,
        ObjectState,
        Pose3D,
        SceneState,
        Vec3,
        WorldState,
    )
    from vlarobot.eval.recorder import EpisodeRecorder, RecordingBackend
    from vlarobot.eval.scenarios import generate_scenes
    from vlarobot.execution.episode_runner import launch_webots_session
    from vlarobot.execution.scripted_can import ScriptedCanPolicy
    from vlarobot.execution.vla_campaign import make_vla_adapter

    vlarobot.__path__.insert(0, str(isolated / "src/vlarobot"))
    from vlarobot.world_model.evaluation import prediction_errors
    from vlarobot.world_model.kinematic import VelocityLimitedKinematics
    from vlarobot.world_model.rollout import AnalyticSurrogate, Attachment, compose, rotate
    from vlarobot.world_model.state_copy import copy_robot

    geometry = json.loads(
        (isolated / "runs/live-geometry-v3/live-bounding-objects.json").read_text(encoding="utf-8")
    )
    can = next(o for o in geometry["objects"] if o["def"] == "CAN")["shapes"][0]["parameters"]
    radius, height = can["radius"], can["height"]
    pad = next(o for o in geometry["objects"] if o["name"] == "gripper_left_finger_link")["shapes"][
        0
    ]["parameters"]["size"]
    release_aperture = radius + pad[0] / 2
    scene = generate_scenes("validation", 0)[0]
    kin = Kinematics()
    monitor = make_vla_adapter(kin.registration_sha256).safety_monitor

    def forward(joints):
        q = np.zeros(len(kin.chain.links))
        for name, value in joints.items():
            q[kin.idx[name]] = value
        m = kin.forward(q)
        return Pose3D(Vec3(*map(float, m[:3, 3])), _rotation_to_quaternion(m[:3, :3]))

    def inverse_pose(pose):
        q = np.asarray(pose.orientation)
        inverse = np.r_[-q[:3], q[3]]
        return Pose3D(
            Vec3(*map(float, rotate(inverse, -pose.position.as_array()))),
            tuple(map(float, inverse)),
        )

    completed = []

    def publish():
        (args.out / "progress.json").write_text(
            json.dumps(
                {
                    "completed": len(completed),
                    "planned": 4,
                    "cases": completed,
                    "test_accessed": False,
                    "not_demonstrations": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for case in plan:
        folder = args.out / case["kind"]
        folder.mkdir()
        recorder = EpisodeRecorder(folder, record_frames=True, frame_hz=1.0)
        recorder.open_episode(
            0,
            0,
            "independent_candidate_diagnostic",
            "PRIVILEGED DIAGNOSTIC oracle initialization; future GT metrics only",
        )
        with launch_webots_session(
            webots=args.webots,
            world=project / "worlds/factory.wbt",
            trace_dir=folder,
            trace_prefix="candidate",
            headless=False,
        ) as session:
            backend = RecordingBackend(session.client, recorder)
            if case["prefix"]:
                policy = ScriptedCanPolicy(
                    scenario=scene, stop_after=case["prefix"], trace_path=folder / "prefix.jsonl"
                )
                prefix = policy.execute(backend, 0)
                if prefix.failure_state:
                    recorder.record_outcome(
                        {
                            "success": False,
                            "failure_state": prefix.failure_state,
                            "failure_reason": prefix.failure_reason,
                            "diagnostic_completed": False,
                            "not_a_demonstration": True,
                        }
                    )
                    recorder.close_episode()
                    completed.append({**case, "prefix_failed": True})
                    publish()
                    continue
                first = json.loads(
                    (folder / "prefix.jsonl").read_text(encoding="utf-8").splitlines()[0]
                )
                rest_world = Vec3(*map(float, first["can_world_xyz_m"]))
                observation = backend.step(1)
            else:
                observation = backend.reset(scene, 0)
                rest_world = backend.get_ground_truth().object_poses["CAN"].position
            actual = backend.get_ground_truth().object_poses["CAN"]
            base = observation.robot.base_pose_world
            if base is None:
                raise RuntimeError("full initial base transform unavailable")
            base_inverse = inverse_pose(base)
            initial_can = compose(base_inverse, actual)
            rest = compose(base_inverse, Pose3D(rest_world, actual.orientation))
            robot = copy_robot(observation.robot)
            held = case["prefix"] == "LIFT"
            robot.gripper_holding = True if held else None
            initial = WorldState(
                SceneState(
                    [
                        ObjectState(
                            "CAN", "can", initial_can, Vec3(2 * radius, 2 * radius, height), 1.0
                        )
                    ]
                ),
                robot,
            )
            rng = np.random.default_rng(case["sample_seed"])
            if case["kind"] in ("free_joint", "carried_joint"):
                target = {n: float(robot.joint_pos[n] + rng.normal(0, 0.015)) for n in ARM_JOINTS}
                action = JointTarget(target)
            else:
                action = GripperCmd(0.0 if case["kind"] == "closure" else 0.045)
            chunk = ActionChunk([action] * 50, 0.032, "diagnostic")
            attachment = (
                Attachment(
                    "CAN", compose(inverse_pose(robot.ee_pose), initial_can), release_aperture
                )
                if held
                else None
            )
            model = AnalyticSurrogate(
                VelocityLimitedKinematics({n: 0.5 for n in ARM_JOINTS}, forward),
                inverse=lambda pose, robot: None,
                gripper_speed_m_s=FINGER_SPEED,
                gravity_m_s2=9.81,
                support_z_m={"CAN": rest.position.z - height / 2},
                validate=lambda a, s: monitor.validate(a, s).accepted,
            )
            predicted = model.predict_trajectory(initial, chunk, attachment=attachment)
            observed = [initial]
            veto = None
            for index, command in enumerate(chunk.actions):
                verdict = monitor.validate(command, observation.robot)
                if not verdict.accepted:
                    veto = verdict.reason
                    break
                if isinstance(command, JointTarget):
                    pose = forward({**observation.robot.joint_pos, **command.q})
                    from vlarobot.core.types import EEPoseTarget

                    cartesian = monitor.validate(
                        EEPoseTarget(pose, None, monitor.limits.command_horizon), observation.robot
                    )
                    if not cartesian.accepted:
                        veto = cartesian.reason
                        break
                applied = backend.apply(command)
                if not applied.accepted:
                    veto = applied.reason
                    break
                observation = backend.step(1)
                truth = backend.get_ground_truth().object_poses["CAN"]
                actual_can = compose(base_inverse, truth)
                observed.append(
                    WorldState(
                        SceneState(
                            [
                                ObjectState(
                                    "CAN",
                                    "can",
                                    actual_can,
                                    Vec3(2 * radius, 2 * radius, height),
                                    1.0,
                                )
                            ]
                        ),
                        copy_robot(observation.robot),
                    )
                )
                if index + 1 < len(predicted.states):
                    p = predicted.states[index + 1]
                    recorder.record_predicted(
                        observation.t,
                        {
                            "joint_pos": p.robot.joint_pos,
                            "CAN_base_xyz": p.scene.objects[0].pose.position.as_array().tolist(),
                            "gripper_m": p.robot.gripper_opening,
                            "hard_reasons": predicted.hard_reasons,
                        },
                    )
            errors = (
                prediction_errors(predicted, observed)
                if len(predicted.states) == len(observed) and len(observed) > 1
                else None
            )
            row = {
                **case,
                "observed_steps": len(observed) - 1,
                "errors": errors,
                "execution_veto": veto,
                "prediction_hard_reasons": predicted.hard_reasons,
                "not_a_demonstration": True,
                "release_half_aperture_geometry_m": release_aperture,
                "conditional_on_observed_lift": held,
                "joint_quantity": "7 arm joints plus torso feedback",
                "gravity_model": "base-z gravity, fixed support; contact/tilt uncalibrated",
                "initial_can_rise_world_m": actual.position.z - rest_world.z,
            }
            recorder.record_outcome(
                {
                    "success": False,
                    "diagnostic_completed": len(observed) == 51,
                    "not_a_demonstration": True,
                    "failure_state": "DIAGNOSTIC_ONLY",
                    "failure_reason": veto,
                    **row,
                }
            )
            recorder.close_episode()
            completed.append(row)
            (folder / "candidate-errors.json").write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8"
            )
        (args.out / "progress.json").write_text(
            json.dumps(
                {
                    "completed": len(completed),
                    "planned": 4,
                    "cases": completed,
                    "test_accessed": False,
                    "training_artifacts_modified": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (args.out / "source-sha256.json").write_text(
        json.dumps(
            {
                "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "geometry_probe": hashlib.sha256(
                    (isolated / "runs/live-geometry-v3/live-bounding-objects.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
