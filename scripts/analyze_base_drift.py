"""Report base drift per state from recorded JSONL traces (Webots-free)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlarobot.execution.trace_compare import base_drift_by_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="JSONL trace files to analyze.")
    args = parser.parse_args()

    for path in args.traces:
        drift = base_drift_by_state(path)
        print(path.name)
        if not drift:
            print("  no state_boundary records with base_world_xyz_m")
            continue
        for state, summary in drift.items():
            print(
                f"  {state:12s} max={summary.max_distance_m * 1000:8.2f} mm "
                f"yaw={summary.max_yaw_deg:7.2f} deg "
                f"tilt={summary.max_tilt_deg:7.2f} deg "
                f"end={summary.end_distance_m * 1000:8.2f} mm "
                f"t={summary.sim_time_s:8.3f} s "
                f"avg_rate={summary.rate_m_per_s * 1000:6.2f} mm/s "
                f"max_rate={summary.max_rate_m_per_s * 1000:6.2f} mm/s"
            )


if __name__ == "__main__":
    main()
