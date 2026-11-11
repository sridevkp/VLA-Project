"""Static finger/pad GT verification at safe arm poses (campaign item 4).

Measures finger_midpoint_world and arm_7 endPoint GT at the reset pose and at three
small, safe arm-joint perturbations (torso fixed, tool kept above the table, never
approaching the can). Reports the arm_7->midpoint vector expressed in the arm_7 frame
(FK orientation from measured joints) for each pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import ARM_JOINTS, SETTINGS
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.core.types import JointTarget, ScenarioSpec
from vlarobot.execution.episode_runner import (
    _popen,
    _stop_client,
    _terminate_process,
    _write_pid_file,
    build_webots_command,
    resolve_webots,
    wait_for_simulator,
    webots_environment,
)

# Small, safe joint perturbations (rad): [arm_1, arm_2, arm_4] deltas only.
POSE_DELTAS = [
    {"arm_1_joint": +0.08},  # ~4.6 deg yaw sweep
    {"arm_1_joint": -0.08},  # ~-4.6 deg
    {"arm_2_joint": +0.05, "arm_4_joint": -0.05},  # small shoulder/elbow compensation
]
MIN_TOOL_HEIGHT_M = 0.80  # above the 0.75 m table top with margin
MAX_CONTACT_NODES = 0


def _reserve_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _measure(client: ZmqBackendClient, kin: Kinematics, obs) -> dict[str, object]:
    diag = obs.robot.diagnostics
    midpoint = diag.get("finger_midpoint_world_xyz_m")
    arm7_gt = diag.get("arm_7_joint_ground_truth_world_xyz_m")
    joints = {name: float(obs.robot.joint_pos.get(name, float("nan"))) for name in ARM_JOINTS}
    q = np.zeros(len(kin.chain.links), dtype=float)
    for name in ARM_JOINTS:
        if name in obs.robot.joint_pos and name in kin.idx:
            q[kin.idx[name]] = float(obs.robot.joint_pos[name])
    # arm_7 frame from FK on measured joints (urdf chain, M1 model).
    try:
        transforms = kin.chain.forward_kinematics(list(q), full_kinematics=True)
        arm7_fk = np.asarray(transforms[kin.idx["arm_7_joint"]], dtype=float)
    except Exception as error:  # pragma: no cover - diagnostic only
        arm7_fk = None
        fk_error = str(error)
    else:
        fk_error = None
    result: dict[str, object] = {
        "joints_rad": joints,
        "finger_midpoint_world_xyz_m": midpoint,
        "arm_7_gt_world_xyz_m": arm7_gt,
        "arm_7_fk_world_xyz_m": (None if arm7_fk is None else arm7_fk[:3, 3].tolist()),
        "contacts": diag.get("contact_nodes", []),
        "sim_time_s": float(obs.t),
        "fk_error": fk_error,
    }
    if midpoint is not None and arm7_gt is not None:
        mid = np.asarray(midpoint, dtype=float)
        arm7 = np.asarray(arm7_gt, dtype=float)
        d_world = mid - arm7
        result["arm7_to_midpoint_world_xyz_m"] = d_world.tolist()
        result["arm7_to_midpoint_distance_m"] = float(np.linalg.norm(d_world))
        if arm7_fk is not None:
            try:
                transform = np.asarray(arm7_fk, dtype=float)
                rotation = transform[:3, :3]
                d_arm7 = rotation.T @ d_world
                result["arm7_to_midpoint_arm7frame_xyz_m"] = d_arm7.tolist()
            except Exception as error:  # pragma: no cover - diagnostic only
                result["arm7_frame_error"] = str(error)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webots", required=True)
    ap.add_argument(
        "--world", default=str(Path(__file__).resolve().parents[1] / "worlds/factory.wbt")
    )
    ap.add_argument("--out", default="runs/campaign-2026-11-11/phase2/finger_gt_poses.json")
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    webots = resolve_webots(args.webots)
    world = Path(args.world).resolve()
    port = _reserve_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    log_path = out.with_suffix(".log")
    pid_path = out.with_suffix(".pid")

    with log_path.open("w", encoding="utf-8") as logf:
        env = webots_environment(webots)
        env["VLAROBOT_ZMQ_PORT"] = str(port)
        env["VLAROBOT_CONTROLLER_STDERR_PATH"] = str(out.with_suffix(".controller.log"))
        proc = _popen(build_webots_command(webots, world, headless=False), env, logf)
        _write_pid_file(pid_path, proc.pid)
        client: ZmqBackendClient | None = None
        try:
            client = wait_for_simulator(endpoint, proc, args.startup_timeout)
            kin = Kinematics()
            obs = client.reset(ScenarioSpec(), seed=0)
            poses: list[dict[str, object]] = []
            poses.append({"name": "reset", **_measure(client, kin, obs)})

            for index, deltas in enumerate(POSE_DELTAS, start=1):
                # Safety pre-check: tool must be above the table and no contacts.
                diag = obs.robot.diagnostics
                midpoint = diag.get("finger_midpoint_world_xyz_m")
                if midpoint is None or float(midpoint[2]) < MIN_TOOL_HEIGHT_M:
                    raise RuntimeError(f"pose {index}: unsafe start height {midpoint}")
                if diag.get("contact_nodes"):
                    raise RuntimeError(f"pose {index}: unexpected contacts at start")
                targets: dict[str, float] = {}
                for name, delta in deltas.items():
                    base_value = float(obs.robot.joint_pos.get(name, 0.0))
                    targets[name] = base_value + delta
                client.apply(JointTarget(targets))
                for _ in range(60):
                    obs = client.step(1)
                diag = obs.robot.diagnostics
                midpoint = diag.get("finger_midpoint_world_xyz_m")
                if midpoint is None or float(midpoint[2]) < MIN_TOOL_HEIGHT_M:
                    raise RuntimeError(
                        f"pose {index}: tool below safe height after move: {midpoint}"
                    )
                contacts = diag.get("contact_nodes", [])
                if contacts:
                    raise RuntimeError(f"pose {index}: contacts after move: {contacts}")
                poses.append(
                    {
                        "name": f"pose{index}",
                        "joint_deltas_rad": deltas,
                        "table_top_height_m": SETTINGS.table_top_height_m,
                        **_measure(client, kin, obs),
                    }
                )

            # Constant-vector check across poses.
            vectors = [
                np.asarray(pose["arm7_to_midpoint_arm7frame_xyz_m"], dtype=float)
                for pose in poses
                if isinstance(pose, dict) and pose.get("arm7_to_midpoint_arm7frame_xyz_m")
            ]
            summary: dict[str, object] = {}
            if len(vectors) >= 2:
                stacked = np.vstack(vectors)
                spread = stacked.max(axis=0) - stacked.min(axis=0)
                summary = {
                    "pose_count": len(vectors),
                    "per_axis_spread_m": spread.tolist(),
                    "max_spread_m": float(spread.max()),
                    "constant_within_5mm": bool(spread.max() < 0.005),
                }
            payload = {
                "status": "ok",
                "poses": poses,
                "summary": summary,
                "proto_pad_center_arm7_m": [0.0, 0.1901, 0.0],
                "proto_pad_bottom_arm7_m": [0.0, 0.235, 0.0],
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({"out": str(out), "summary": summary}, indent=2))
            return 0
        finally:
            if client is not None:
                _stop_client(client)
            _terminate_process(proc, 30.0, pid_path=pid_path)


if __name__ == "__main__":
    raise SystemExit(main())
