"""Per-state report for a scripted CAN trace (no Webots import).

Prints, for every state exit, the CAN world pose, displacement, lift, the live-base
fingertip-to-CAN xy error, base tilt, and the realign/CLOSE-guard events. Also prints the
four final success criteria from the ``DONE`` boundary.

Usage:
    uv run python scripts/analyze_can_run.py runs/m1-07-realign-on-01.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _rotation(quaternion: Any) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _tip_to_can_xy_m(record: dict[str, Any]) -> float:
    rotation = _rotation(record["base_world_orientation_xyzw"])
    tip_world = rotation @ np.asarray(record["achieved_ee_base_xyz_m"], dtype=float) + np.asarray(
        record["base_world_xyz_m"], dtype=float
    )
    can_world = np.asarray(record["can_world_xyz_m"], dtype=float)
    return float(np.linalg.norm((tip_world - can_world)[:2]))


def _mm(value: Any, width: int = 10) -> str:
    """Field renderer for a metres value; ``none`` when the trace carries no number."""
    if not isinstance(value, (int, float)):
        return "none".rjust(width)
    return f"{float(value) * 1000.0:{width}.2f}"


def _print_gt_section(rows: list[dict[str, Any]]) -> None:
    """Print ``gt_tip_telemetry`` measurements and any ``TOOL_OFFSET`` calibration warning."""
    gt_rows = [record for record in rows if record.get("event") == "gt_tip_telemetry"]
    print()
    if not gt_rows:
        print("gt_tip_telemetry: none in this trace (expected on the live Webots backend only).")
        return
    header = (
        f"{'state':14s} {'phase':>6s} {'sim_time_s':>10s} {'fk_gt_mm':>9s} "
        f"{'gt_mid_can_mm':>13s} {'fk_can_mm':>10s} "
        f"{'clear_L_mm':>10s} {'clear_R_mm':>10s} {'tool_off_err':>12s}"
    )
    print("gt_tip_telemetry (real finger midpoint from Supervisor endPoint Solids")
    print("vs the FK tool tip; TOOL_OFFSET tolerance is 5 mm, report-only):")
    print(header)
    print("-" * len(header))
    for record in gt_rows:
        print(
            f"{record.get('state'):14s} {record.get('phase'):>6s} "
            f"{float(record.get('sim_time_s', 0.0)):10.3f} "
            f"{_mm(record.get('fk_gt_offset_m'), 9)} "
            f"{_mm(record.get('gt_to_can_xy_m'), 13)} "
            f"{_mm(record.get('fk_to_can_xy_m'), 10)} "
            f"{_mm(record.get('clearance_left_m'), 10)} "
            f"{_mm(record.get('clearance_right_m'), 10)} "
            f"{str(record.get('tool_offset_error')):>12s}"
        )
        proposal = record.get("tool_offset_calibration_proposal")
        if isinstance(proposal, dict):
            print(
                f"    TOOL_OFFSET calibration error: {proposal.get('proposal_note')} "
                f"configured={proposal.get('configured_finger_tip_offset_m')} "
                f"proposed={proposal.get('proposed_finger_tip_offset_m')} "
                f"(report only, not applied)"
            )


def main(path: Path) -> int:
    rows = _load(path)
    exits = [
        record
        for record in rows
        if record.get("event") == "state_boundary" and record.get("phase") == "exit"
    ]
    header = (
        f"{'state':14s} {'can_world_xyz_m':30s} {'disp_mm':>8s} {'lift_mm':>8s} "
        f"{'tip_can_mm':>10s} {'gt_mid_can_mm':>13s} {'fk_gt_mm':>9s} "
        f"{'clear_L_mm':>10s} {'clear_R_mm':>10s} "
        f"{'roll_deg':>9s} {'pitch_deg':>9s} {'realign':>7s}"
    )
    print(f"trace: {path}")
    print(header)
    print("-" * len(header))
    for record in exits:
        can = np.asarray(record["can_world_xyz_m"], dtype=float)
        displacement = np.asarray(record["can_displacement_world_xyz_m"], dtype=float)
        print(
            f"{record['state']:14s} {np.round(can, 4).tolist()!s:30s} "
            f"{np.linalg.norm(displacement) * 1000.0:8.2f} "
            f"{displacement[2] * 1000.0:8.2f} "
            f"{_tip_to_can_xy_m(record) * 1000.0:10.2f} "
            f"{_mm(record.get('gt_to_can_xy_m'), 13)} "
            f"{_mm(record.get('fk_gt_offset_m'), 9)} "
            f"{_mm(record.get('clearance_left_m'), 10)} "
            f"{_mm(record.get('clearance_right_m'), 10)} "
            f"{float(record.get('base_tilt_roll_deg', 0.0)):9.3f} "
            f"{float(record.get('base_tilt_pitch_deg', 0.0)):9.3f} "
            f"{str(record.get('pregrasp_realign')):>7s}"
        )

    _print_gt_section(rows)

    print()
    for event in (
        "pregrasp_realign",
        "pregrasp_realigned",
        "lift_target_refreshed",
        "close_alignment_check",
        "close_realign",
        "close_misaligned",
    ):
        for record in rows:
            if record.get("event") == event:
                print(event, json.dumps(record, sort_keys=True))

    done = next(
        (
            record
            for record in rows
            if record.get("event") == "state_boundary" and record.get("state") == "DONE"
        ),
        None,
    )
    if done is not None:
        keys = (
            "pregrasp_realign",
            "distance_m",
            "height_error_m",
            "tilt_deg",
            "home_reached",
            "place_distance_ok",
            "can_height_ok",
            "can_tilt_ok",
            "success",
        )
        print()
        print("DONE criteria:", json.dumps({key: done.get(key) for key in keys}, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/analyze_can_run.py <trace.jsonl>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
