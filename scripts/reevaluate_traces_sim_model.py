"""Webots-free re-evaluation of recorded traces with the sim-derived FK chain (ADR-0020).

This script is the offline counterpart of ``validate-kinematics``. It cannot
capture live poses: it only re-scores trace rows that already carry the
per-pose ``kinematics_telemetry`` record produced by a server run with
``export_sim_kinematics: true`` (the validation sweep stores that payload in its
JSON report; ordinary brain traces do not embed the Solid-pose telemetry). When
the simulator kinematics snapshot is absent, or when no provided input carries
the required telemetry, the script reports that it is **blocked on a live
export** and exits without claiming a numeric result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlarobot.backends.webots.config import SIM_KINEMATICS_PATH  # noqa: E402
from vlarobot.execution.kinematics_validation import (  # noqa: E402
    JointSample,
    evaluate_sweep,
    format_report,
)


def _samples_from_payload(payload: object) -> list[tuple[JointSample, str]]:
    """Extract validation samples from a validate-kinematics JSON report, if present."""
    if not isinstance(payload, dict):
        return []
    raw_samples = payload.get("samples")
    samples: list[tuple[JointSample, str]] = []
    if isinstance(raw_samples, list):
        for row in raw_samples:
            if not isinstance(row, dict):
                continue
            joints = row.get("joints") or row.get("joint_positions_rad")
            telemetry = row.get("telemetry") or row.get("kinematics_telemetry")
            group = row.get("group")
            if not isinstance(group, str):
                group = "spread"
            if isinstance(joints, dict) and isinstance(telemetry, dict):
                samples.append(
                    (
                        JointSample(
                            joints={str(k): float(v) for k, v in joints.items()},
                            telemetry=telemetry,
                        ),
                        group,
                    )
                )
    return samples


def _load_samples(paths: list[Path]) -> list[tuple[JointSample, str]]:
    samples: list[tuple[JointSample, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            payload = json.loads(text)
            samples.extend(_samples_from_payload(payload))
            continue
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.extend(_samples_from_payload(row))
            joints = row.get("joint_positions_rad")
            telemetry = row.get("kinematics_telemetry")
            group = row.get("group") if isinstance(row.get("group"), str) else "spread"
            if isinstance(joints, dict) and isinstance(telemetry, dict):
                samples.append(
                    (
                        JointSample(
                            joints={str(k): float(v) for k, v in joints.items()},
                            telemetry=telemetry,
                        ),
                        group,
                    )
                )
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="validate-kinematics JSON reports or JSONL traces that embed kinematics_telemetry.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=SIM_KINEMATICS_PATH,
        help=(
            "Simulator kinematics snapshot JSON "
            "(default: configs/robot.yaml sim_kinematics_filename)."
        ),
    )
    args = parser.parse_args(argv)

    snapshot = args.snapshot.expanduser()
    if not snapshot.exists():
        print(
            "blocked on live export: sim kinematics snapshot not found at "
            f"{snapshot}. Run `uv run python -m vlarobot.cli export-sim-kinematics` "
            "against a live Webots session (export_sim_kinematics: true), then re-run "
            "this script. No numeric re-evaluation was performed."
        )
        return 2

    samples = _load_samples([path.expanduser() for path in args.inputs])
    if not samples:
        print(
            "blocked on live export: the sim snapshot is present but none of the provided "
            "inputs contain per-pose kinematics_telemetry samples. Run "
            "`uv run python -m vlarobot.cli validate-kinematics --report <path.json>` "
            "against a live Webots session (or record traces that embed kinematics_telemetry), "
            "then pass that report to this script. Existing PREGRASP/DESCEND traces do not carry "
            "the Solid-pose telemetry required for a like-for-like re-evaluation, so no numeric "
            "result is claimed."
        )
        return 2

    report = evaluate_sweep(samples, tool_verified=False)
    print(f"reevaluation snapshot={snapshot} samples={len(samples)} inputs={len(args.inputs)}")
    for line in format_report(report):
        print(line)
    return 0 if report.accepted_model is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
