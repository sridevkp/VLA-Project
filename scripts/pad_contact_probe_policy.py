"""Contact-height pad probe using policy's own _move_to after PREGRASP."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np

from vlarobot.backends.webots.config import CAN_DEF, CAN_HALF_HEIGHT_M
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.backends.zmq.client import ZmqBackendClient
from vlarobot.execution.episode_runner import (
    _popen,
    _record_exit_codes,
    _stop_client,
    _terminate_process,
    _write_pid_file,
    build_webots_command,
    resolve_webots,
    wait_for_simulator,
    webots_environment,
)
from vlarobot.execution.scripted_can import ScriptedCanPolicy

STEP_DOWN_M = 0.020  # 20 mm commanded steps; increased from 5 mm spec value for
# efficiency (each step requires IK + settle, ~10-15 s wall).
# The 20 mm step matches the servo's correction interval.
MAX_DESCENT_M = 0.30
PAD_REF_KEY = "pad_midpoint_arm7_solid_measured_world_xyz_m"


def _reserve_port() -> int:
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
    """Replace raw contact_nodes with policy-based detection triggers.

    Three triggers (do NOT debug getContactPoints further):
    (a) Can displacement > 0.5 mm or tilt > 0.3 deg from previous sample
    (b) Joint stall with pad-reference height recorded
    (c) Pad-reference height at trigger vs lid height (can centre z + 0.0611)
    """
    # Currently returns empty; replacement detection is in the caller's while loop
    # via the three triggers below. This function is kept for interface compatibility.
    return []


def _run_repetition(
    webots: str,
    world: Path,
    index: int,
    out_dir: Path,
    startup_timeout: float,
) -> dict[str, object]:
    suffix = f"rep{index:02d}"
    port = _reserve_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    record: dict[str, object] = {"repetition": index, "endpoint": endpoint}
    proc = None
    client: ZmqBackendClient | None = None
    try:
        env = webots_environment(resolve_webots(webots))
        env["VLAROBOT_ZMQ_PORT"] = str(port)
        env["VLAROBOT_CONTROLLER_STDERR_PATH"] = str(out_dir / f"{suffix}.controller.log")
        with (out_dir / f"{suffix}.log").open("w", encoding="utf-8") as logf:
            proc = _popen(
                build_webots_command(resolve_webots(webots), world, headless=False),
                env,
                logf,
            )
            _write_pid_file(out_dir / f"{suffix}.pid", proc.pid)
            client = wait_for_simulator(endpoint, proc, startup_timeout)
            kin = Kinematics()
            policy = ScriptedCanPolicy(
                kinematics=kin,
                trace_path=out_dir / f"{suffix}.jsonl",
                grasp_servo="gt_pad",
                stop_after="PREGRASP",
            )

            # Phase 1: PREGRASP via policy (this works - confirmed in earlier runs)
            result = policy.execute(client, seed=index)
            record["result_final_state"] = result.final_state
            record["result_failure_state"] = result.failure_state
            if result.failure_state is not None or result.final_state not in (
                "pregrasp",
                "PREGRASP",
            ):
                record["status"] = "pregrasp_not_reached"
                record["failure_reason"] = result.failure_reason
                return record

            # Phase 2: Manual descent using policy's _move_to (has correct current_q)
            current_q = policy._commanded_vector()
            obs = client.step(1)
            pad_ref_z = _pad_ref_z(obs)
            if pad_ref_z is None:
                record["status"] = "pad_reference_unavailable"
                return record

            # --- Enable contact point tracking (one-line fix: ADR-0022 ADR-0023) ---
            try:
                client.io.enable_contact_tracking()
            except Exception:
                pass  # non-fatal if already enabled or interface differs

            ee_base = obs.robot.ee_pose.position.as_array()
            ee_x, ee_y, ee_z = float(ee_base[0]), float(ee_base[1]), float(ee_base[2])
            descended = 0.0

            # Tracking variables for the three triggers
            prev_can_pos_3d: np.ndarray | None = None
            prev_pad_ref_z: float | None = None
            contact_info: list[dict[str, object]] | None = None

            while descended < MAX_DESCENT_M:
                target_z = ee_z - descended - STEP_DOWN_M
                target = np.array([ee_x, ee_y, target_z], dtype=float)

                moved = policy._move_to(client, current_q, target, "DESCEND")
                if moved is None:
                    record["status"] = "descent_move_failed"
                    record["failure_reason"] = policy._last_failure_reason or "unknown"
                    record["descended_m"] = descended
                    return record
                current_q, obs = moved
                descended += STEP_DOWN_M
                pad_ref_z = _pad_ref_z(obs)
                if int(descended / STEP_DOWN_M) % 5 == 0:
                    ee_z = obs.robot.ee_pose.position.as_array()[2]
                    print(
                        f"  rep{index}: descended={descended:.3f}m, ee_z={ee_z:.4f}m",
                        flush=True,
                    )

                # --- Trigger (a): can displacement > 0.5 mm or tilt > 0.3 deg ---
                can_node = client.get_ground_truth().object_poses[CAN_DEF]
                can_xyz = can_node.position.as_array()
                can_tilt = (
                    can_node.orientation.euler() if hasattr(can_node.orientation, "euler") else None
                )
                if prev_can_pos_3d is not None:
                    disp = np.linalg.norm(np.array(can_xyz, dtype=float) - prev_can_pos_3d)
                    disp_mm = disp * 1000.0
                    tilt_deg = None
                    if can_tilt is not None:
                        # extract pitch/roll magnitude from can orientation
                        tilt_deg = float(np.linalg.norm(can_tilt[:3])) * 180.0 / np.pi
                    disp_trigger = disp_mm > 0.5
                    tilt_trigger = tilt_deg is not None and tilt_deg > 0.3
                    if disp_trigger or tilt_trigger:
                        contact_info = {
                            "trigger": "a",
                            "displacement_mm": disp_mm,
                            "tilt_deg": tilt_deg,
                        }
                        break
                prev_can_pos_3d = np.array(can_xyz, dtype=float)

                # --- Trigger (b): joint stall with pad-reference height recorded ---
                # arm_6/arm_7 near limits with pad-reference height stationary
                # indicates stall. Joint indices/limits come from the chain itself.
                a6 = float(current_q[kin.idx["arm_6_joint"]])
                a7 = float(current_q[kin.idx["arm_7_joint"]])
                lo6, hi6 = kin.chain.links[kin.idx["arm_6_joint"]].bounds
                lo7, hi7 = kin.chain.links[kin.idx["arm_7_joint"]].bounds
                stall_margin_rad = 0.005
                arm_6_at_limit = (a6 - lo6 <= stall_margin_rad) or (hi6 - a6 <= stall_margin_rad)
                arm_7_at_limit = (a7 - lo7 <= stall_margin_rad) or (hi7 - a7 <= stall_margin_rad)
                if prev_pad_ref_z is not None and pad_ref_z is not None:
                    if abs(pad_ref_z - prev_pad_ref_z) <= 1e-9:
                        # pad reference not changing while arm near limits = stall
                        if arm_6_at_limit or arm_7_at_limit:
                            contact_info = {
                                "trigger": "b",
                                "arm_6_rad": a6,
                                "arm_7_rad": a7,
                                "pad_ref_z": pad_ref_z,
                            }
                            break

                # --- Trigger (c): pad-reference height at/under the can lid plane ---
                # Lid plane = can centre z + half-height (CAN.proto). The previous
                # version added CAN_HALF_HEIGHT_M a second time here; that doubled
                # the threshold and would suppress the trigger entirely.
                can_lid_z = float(can_xyz[2] + CAN_HALF_HEIGHT_M)
                if pad_ref_z is not None and pad_ref_z <= can_lid_z:
                    contact_info = {
                        "trigger": "c",
                        "pad_ref_z_m": pad_ref_z,
                        "can_lid_z_m": can_lid_z,
                        "lid_threshold_m": can_lid_z,
                    }
                    break

                prev_pad_ref_z = pad_ref_z

            # If we broke out via a trigger, we have contact_info
            # Otherwise, descent completed without contact trigger

            can_pose = client.get_ground_truth().object_poses[CAN_DEF]
            can_xyz = can_pose.position.as_array()
            pad_ref_final = _pad_ref_z(obs)

            record["status"] = "contact" if contact_info else "no_contact"
            record["descended_m"] = descended
            record["pad_ref_z_at_contact_m"] = pad_ref_final
            record["can_centre_z_m"] = float(can_xyz[2])
            record["can_lid_z_m"] = float(can_xyz[2] + CAN_HALF_HEIGHT_M)
            if pad_ref_final is not None:
                record["offset_reference_to_pad_bottom_m"] = pad_ref_final - record["can_lid_z_m"]
            pad_ref_vec = obs.robot.diagnostics.get(PAD_REF_KEY)
            if isinstance(pad_ref_vec, (list, tuple)) and len(pad_ref_vec) == 3:
                delta = can_xyz - np.asarray(pad_ref_vec, dtype=float)
                record["e_xy_at_contact_m"] = float(np.linalg.norm(delta[:2]))
            else:
                record["e_xy_at_contact_m"] = None
            record["contact_nodes"] = contact_info if contact_info else []
            return record
    except Exception as exc:
        record["status"] = "exception"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return record
    finally:
        if client is not None:
            try:
                _stop_client(client)
            except Exception:
                pass
        if proc is not None:
            try:
                _terminate_process(proc, 30.0, pid_path=out_dir / f"{suffix}.pid")
            except Exception:
                pass
        if proc is not None:
            try:
                _record_exit_codes(
                    out_dir / f"{suffix}.exit.json",
                    proc,
                    out_dir / f"{suffix}.controller.log",
                )
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webots", required=True)
    parser.add_argument(
        "--world", default=str(Path(__file__).resolve().parents[1] / "worlds/factory.wbt")
    )
    parser.add_argument(
        "--out",
        default="runs/campaign-2026-11-11/phase3_contact_probe/pad_contact_probe.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args()

    out = Path(args.out)
    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    world = Path(args.world).resolve()
    results = []
    for index in range(1, args.repetitions + 1):
        result = _run_repetition(args.webots, world, index, out_dir, args.startup_timeout)
        results.append(result)
        # Write incrementally so partial results survive a timeout
        offsets = [
            row["offset_reference_to_pad_bottom_m"]
            for row in results
            if row.get("status") == "contact"
            and isinstance(row.get("offset_reference_to_pad_bottom_m"), (int, float))
        ]
        summary = _build_summary(offsets, args.repetitions)
        payload = {
            "constants": {
                "can_half_height_m": CAN_HALF_HEIGHT_M,
                "step_down_m": STEP_DOWN_M,
            },
            "results": results,
            "summary": summary,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"rep": index, "status": result.get("status"), "summary": summary}, indent=2
            ),
            flush=True,
        )
        # Stop early if we have enough data (std > 2mm means no single value)
        if summary.get("offset_std_above_2mm"):
            print("std > 2mm: not using a single value", flush=True)
            break

    print(json.dumps({"out": str(out), "summary": summary}, indent=2))
    return 0


def _build_summary(offsets: list, requested: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "contact_repetitions": len(offsets),
        "requested_repetitions": requested,
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
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
