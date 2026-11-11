"""Contact-height pad probe (campaign addition 2; descent stop height source).

Per repetition, in a fresh world and its own Webots session:
1. The scripted policy drives PREGRASP with the gt_pad servo correction ON, so the hover
   above the can is aligned using the MEASURED pad reference (arm_7 Solid + measured
   offset, ADR-0022). It stops there (``stop_after="PREGRASP"``).
2. The probe then descends from the achieved hover in 5 mm commanded steps at fixed XY,
   checking ``contact_nodes`` after every step. At the first finger contact with the can
   lid it stops (never squeezes the can) and records:
   - the Solid-based pad reference height at contact,
   - the can centre z from ground truth and the lid z (can centre z + CAN_HALF_HEIGHT_M),
   - the resulting offset from the reference to the pad bottom (reference z - lid z),
   - the XY error at contact, the contact nodes/points, and the descended distance.
3. Repeats ``--repetitions`` times from fresh worlds and reports mean/std. If std > 2 mm
   the summary says so and flags that a single value must not be used.

Safety: torso fixed at reset height, tool always above the table floor guard, descent
aborts on contact or on the floor guard, no threshold or servo behaviour changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import (
    CAN_DEF,
    CAN_HALF_HEIGHT_M,
    PAD_OFFSET_ARM7_SIMFRAME_M,
    TABLE_TOP_HEIGHT_M,
)
from vlarobot.backends.webots.kinematics import Kinematics
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

STEP_DOWN_M = 0.005  # 5 mm commanded steps (addition 2 spec)
MAX_DESCENT_M = 0.30
# Never descend the tool reference below table top + this margin.
SAFETY_FLOOR_PADREF_Z_M = TABLE_TOP_HEIGHT_M + 0.05
PAD_REF_KEY = "pad_midpoint_arm7_solid_measured_world_xyz_m"


def _reserve_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pad_ref_z(observation) -> float | None:
    raw = observation.robot.diagnostics.get(PAD_REF_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    arr = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return float(arr[2])


def _contacts(observation) -> list[dict[str, object]]:
    raw = observation.robot.diagnostics.get("contact_nodes", [])
    return raw if isinstance(raw, list) else []


def _run_repetition(
    webots: str,
    world: Path,
    index: int,
    out_dir: Path,
    startup_timeout: float,
) -> dict[str, object]:
    suffix = f"rep{index:02d}"
    trace_path = out_dir / f"{suffix}.jsonl"
    log_path = out_dir / f"{suffix}.log"
    pid_path = out_dir / f"{suffix}.pid"
    controller_log = out_dir / f"{suffix}.controller.log"
    port = _reserve_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    record: dict[str, object] = {"repetition": index, "endpoint": endpoint}
    with log_path.open("w", encoding="utf-8") as logf:
        env = webots_environment(resolve_webots(webots))
        env["VLAROBOT_ZMQ_PORT"] = str(port)
        env["VLAROBOT_CONTROLLER_STDERR_PATH"] = str(controller_log)
        proc = _popen(
            build_webots_command(resolve_webots(webots), world, headless=False), env, logf
        )
        _write_pid_file(pid_path, proc.pid)
        client: ZmqBackendClient | None = None
        try:
            client = wait_for_simulator(endpoint, proc, startup_timeout)
            kin = Kinematics()
            policy = ScriptedCanPolicy(
                kinematics=kin,
                trace_path=trace_path,
                grasp_servo="gt_pad",
                stop_after="PREGRASP",
            )
            result = policy.execute(client, seed=index)
            if result.failure_state is not None or result.final_state != "PREGRASP":
                record["status"] = "pregrasp_failed"
                record["failure_state"] = result.failure_state
                record["failure_reason"] = result.failure_reason
                return record
            current_q = policy._commanded_vector()
            observation = client.step(1)
            pad_ref_z = _pad_ref_z(observation)
            if pad_ref_z is None:
                record["status"] = "pad_reference_unavailable"
                return record
            ee_base = observation.robot.ee_pose.position.as_array()
            target = np.array([ee_base[0], ee_base[1], float(ee_base[2])], dtype=float)
            descended = 0.0
            contact_info: list[dict[str, object]] | None = None
            while descended < MAX_DESCENT_M:
                target = target - np.array([0.0, 0.0, STEP_DOWN_M])
                moved = policy._move_to(client, current_q, target, "DESCEND")
                if moved is None:
                    record["status"] = "descent_move_failed"
                    record["failure_reason"] = policy._last_failure_reason
                    record["descended_m"] = descended
                    return record
                current_q, observation = moved
                descended += STEP_DOWN_M
                contacts = _contacts(observation)
                if contacts:
                    contact_info = contacts
                    break
                pad_ref_now = _pad_ref_z(observation)
                if pad_ref_now is None:
                    record["status"] = "pad_reference_lost"
                    record["descended_m"] = descended
                    return record
                if pad_ref_now < SAFETY_FLOOR_PADREF_Z_M:
                    record["status"] = "safety_floor"
                    record["descended_m"] = descended
                    return record
            can_pose = client.get_ground_truth().object_poses[CAN_DEF]
            can_xyz = can_pose.position.as_array()
            pad_ref_final = _pad_ref_z(observation)
            if pad_ref_final is None:
                record["status"] = "pad_reference_lost_at_end"
                record["descended_m"] = descended
                return record
            record["status"] = "contact" if contact_info is not None else "no_contact"
            record["descended_m"] = descended
            record["pad_ref_z_at_contact_m"] = pad_ref_final
            record["can_centre_z_m"] = float(can_xyz[2])
            record["can_lid_z_m"] = float(can_xyz[2] + CAN_HALF_HEIGHT_M)
            record["offset_reference_to_pad_bottom_m"] = pad_ref_final - record["can_lid_z_m"]
            delta = can_xyz - np.asarray(observation.robot.diagnostics[PAD_REF_KEY], dtype=float)
            record["e_xy_at_contact_m"] = float(np.linalg.norm(delta[:2]))
            record["contact_nodes"] = contact_info if contact_info is not None else []
            return record
        finally:
            if client is not None:
                _stop_client(client)
            _terminate_process(proc, 30.0, pid_path=pid_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webots", required=True)
    parser.add_argument(
        "--world", default=str(Path(__file__).resolve().parents[1] / "worlds/factory.wbt")
    )
    parser.add_argument(
        "--out", default="runs/campaign-2026-11-11/phase3_contact_probe/pad_contact_probe.json"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args()

    out = Path(args.out)
    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    world = Path(args.world).resolve()
    results = [
        _run_repetition(args.webots, world, index, out_dir, args.startup_timeout)
        for index in range(1, args.repetitions + 1)
    ]
    offsets = [
        row["offset_reference_to_pad_bottom_m"]
        for row in results
        if row.get("status") == "contact"
        and isinstance(row.get("offset_reference_to_pad_bottom_m"), (int, float))
    ]
    summary: dict[str, object] = {
        "contact_repetitions": len(offsets),
        "requested_repetitions": args.repetitions,
    }
    if len(offsets) >= 2:
        mean = float(np.mean(np.asarray(offsets, dtype=float)))
        std = float(np.std(np.asarray(offsets, dtype=float), ddof=1))
        summary.update(
            {
                "offset_mean_m": mean,
                "offset_std_m": std,
                "offset_std_above_2mm": bool(std > 0.002),
                "use_single_value": bool(std <= 0.002),
            }
        )
    elif len(offsets) == 1:
        summary["offset_only_m"] = offsets[0]
        summary["use_single_value"] = False
        summary["note"] = "only one contact repetition; std not computable"
    payload = {
        "constants": {
            "can_half_height_m": CAN_HALF_HEIGHT_M,
            "pad_offset_arm7_simframe_m": list(PAD_OFFSET_ARM7_SIMFRAME_M),
            "step_down_m": STEP_DOWN_M,
            "safety_floor_padref_z_m": SAFETY_FLOOR_PADREF_Z_M,
        },
        "results": results,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
