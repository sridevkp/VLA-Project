"""Live pad-position verification probe (campaign Phase 2a).

Safe, above-can descent in 5 mm steps with contact-API detection.
Reaches PREGRASP via ScriptedCanPolicy (stop_after=PREGRASP), then probes downward.
Launches its own Webots process and terminates it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import ARM_JOINTS, CAN_DEF
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.webots.sim_kinematics import (
    PROTO_PAD_BOTTOM_ARM7_M,
    PROTO_PAD_CENTER_ARM7_M,
)
from vlarobot.backends.zmq.client import ZmqBackendClient
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
from vlarobot.execution.scripted_can import ScriptedCanPolicy


def _reserve_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pad_mid_world(obs) -> list[float] | None:
    mid = obs.robot.diagnostics.get("finger_midpoint_world_xyz_m")
    if isinstance(mid, list) and len(mid) == 3:
        return [float(x) for x in mid]
    return None


def _contacts(obs) -> list:
    c = obs.robot.diagnostics.get("contact_nodes", [])
    return c if isinstance(c, list) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webots", required=True)
    ap.add_argument(
        "--world",
        default=str(Path(__file__).resolve().parents[1] / "worlds/factory.wbt"),
    )
    ap.add_argument("--out", default="runs/campaign-2026-11-11/phase2/pad_probe.json")
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
        cmd = build_webots_command(webots, world, headless=False)
        proc = _popen(cmd, env, logf)
        _write_pid_file(pid_path, proc.pid)
        client: ZmqBackendClient | None = None
        try:
            client = wait_for_simulator(endpoint, proc, args.startup_timeout)
            kin = Kinematics()

            # Reach PREGRASP using scripted expert semantics.
            expert = ScriptedCanPolicy(kinematics=kin, grasp_servo="none", stop_after="PREGRASP")
            result_exec = expert.execute(client, 0)
            obs = result_exec.observation
            if obs is None or result_exec.final_state != "PREGRASP":
                raise RuntimeError(
                    "expert did not reach PREGRASP: "
                    f"state={result_exec.final_state} reason={result_exec.failure_reason}"
                )

            targets = expert._base_frame_targets(client, obs)
            pregrasp = np.array(targets["pregrasp"], dtype=float)
            grasp = np.array(targets["grasp"], dtype=float)

            q = np.zeros(len(kin.chain.links), dtype=float)
            for name in ARM_JOINTS:
                if name in obs.robot.joint_pos and name in kin.idx:
                    q[kin.idx[name]] = float(obs.robot.joint_pos[name])

            # Safe descent from achieved pregrasp toward grasp in 5mm target steps.
            total = float(pregrasp[2] - grasp[2])
            n_steps = max(1, int(np.ceil(abs(total) / 0.005)))
            samples = []
            first_contact = None
            aborted_early = False
            current = q
            mover = ScriptedCanPolicy(kinematics=kin, grasp_servo="none")
            for i in range(1, n_steps + 1):
                alpha = min(1.0, (i * 0.005) / abs(total)) if abs(total) > 1e-9 else 1.0
                target = pregrasp + alpha * (grasp - pregrasp)
                moved = mover._move_to(client, current, target, "DESCEND")
                if moved is None:
                    samples.append(
                        {
                            "i": i,
                            "alpha": alpha,
                            "error": mover._last_failure_reason or "move_failed",
                            "target_base_xyz": target.tolist(),
                            "sim_time_s": float(obs.t),
                        }
                    )
                    break
                current, obs = moved
                pad_mid = _pad_mid_world(obs)
                gt = client.get_ground_truth()
                can_world = np.array(
                    [
                        gt.object_poses[CAN_DEF].position.x,
                        gt.object_poses[CAN_DEF].position.y,
                        gt.object_poses[CAN_DEF].position.z,
                    ],
                    dtype=float,
                )
                sample = {
                    "i": i,
                    "alpha": alpha,
                    "target_base_xyz": target.tolist(),
                    "achieved_ee_base_xyz": obs.robot.ee_pose.position.as_array().tolist(),
                    "pad_mid_world_xyz": pad_mid,
                    "can_world_xyz": can_world.tolist(),
                    "contacts": _contacts(obs),
                    "sim_time_s": float(obs.t),
                }
                samples.append(sample)
                if sample["contacts"]:
                    first_contact = sample
                    if alpha < 0.7:
                        aborted_early = True
                    break

            prediction = {
                "proto_pad_center_arm7_m": PROTO_PAD_CENTER_ARM7_M.tolist(),
                "proto_pad_bottom_arm7_m": PROTO_PAD_BOTTOM_ARM7_M.tolist(),
                "note": (
                    "Compare measured pad_mid_world_xyz[2] at first contact vs can_world_xyz[2] "
                    "and expected pad/can geometry to derive MEASURED pad-bottom/offset."
                ),
            }
            result = {
                "status": "ok",
                "endpoint": endpoint,
                "seed": 0,
                "aborted_on_contact": first_contact is not None,
                "aborted_early_contact": aborted_early,
                "first_contact": first_contact,
                "pregrasp_target_base": pregrasp.tolist(),
                "grasp_target_base": grasp.tolist(),
                "prediction": prediction,
                "samples": samples,
            }
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "out": str(out),
                        "samples": len(samples),
                        "first_contact": first_contact is not None,
                        "aborted_early_contact": aborted_early,
                    },
                    indent=2,
                )
            )
            return 0
        finally:
            if client is not None:
                _stop_client(client)
            _terminate_process(proc, 30.0, pid_path=pid_path)


if __name__ == "__main__":
    raise SystemExit(main())
