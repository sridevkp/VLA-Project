"""Reproducible post-hoc metrics for V2 TRAIN closed-loop diagnostics.

Ground truth is read only for this measurement; it is never supplied to the
policy. The calculation is shared by the baseline endpoint-decoder run and any
recovered checkpoint so their reported fields are directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def summarize(run: Path) -> list[dict[str, Any]]:
    """Derive one comparable metric row per recorded episode in a V2 diagnostic run."""
    import numpy as np

    rows: list[dict[str, Any]] = []
    for meta_path in sorted(run.glob("episode-*.meta.json")):
        stem = meta_path.name.split(".")[0]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        outcome = meta["outcome"]
        steps = np.load(run / f"{stem}.steps.npz")
        motion = _rows(run / f"{stem}.latency.motion.jsonl")
        times = np.asarray(steps["t"], dtype=np.float64)
        motion_times = np.asarray([entry["t"] for entry in motion], dtype=np.float64)
        if times.shape != motion_times.shape or not np.array_equal(times, motion_times):
            raise ValueError(f"{stem}: motion log is not aligned with recorded steps")
        names = [str(name) for name in steps["gt_names"]]
        if "CAN" not in names:
            raise ValueError(f"{stem}: CAN ground truth is unavailable for post-hoc metrics")
        can = np.asarray(steps["gt_xyz"], dtype=np.float64)[:, names.index("CAN")]
        tcp = np.asarray([entry["tcp_world_m"] for entry in motion], dtype=np.float64)
        if len(times) == 0 or tcp.shape != (len(times), 3) or can.shape != (len(times), 3):
            raise ValueError(f"{stem}: incompatible tool/CAN trace shapes")
        gripper = np.asarray(steps["gripper"], dtype=np.float64)
        if gripper.shape != (len(times),):
            raise ValueError(f"{stem}: incompatible gripper trace shape")

        events: list[dict[str, Any]] = []
        for entry in _rows(run / f"{stem}.actions.jsonl"):
            if not entry.get("accepted") or "opening" not in entry.get("action", {}):
                continue
            target = entry["action"]["opening"]
            if events and events[-1]["target"] == target:
                continue
            index = int(np.argmin(np.abs(times - float(entry["t"]))))
            events.append(
                {
                    "t": float(entry["t"]),
                    "target": float(target),
                    "alignment_m": float(np.linalg.norm(tcp[index] - can[index])),
                    "observed_opening_m": float(gripper[index]),
                }
            )

        latency = _rows(run / f"{stem}.latency.jsonl")
        raw = np.concatenate(
            [np.asarray(entry["actions"], dtype=np.float64)[:, -1] for entry in latency]
        )
        rows.append(
            {
                "trial": stem,
                "scenario": meta["scenario"],
                "terminal": outcome.get("failure_state"),
                "reason": outcome.get("failure_reason"),
                "queries": outcome.get("model_chunks"),
                "actions": outcome.get("executed_actions"),
                "fallbacks": outcome.get("ik_fallback_solutions"),
                "closest_m": float(np.min(np.linalg.norm(tcp - can, axis=1))),
                "rise_m": float(np.max(can[:, 2]) - can[0, 2]),
                "can_xy_m": float(np.max(np.linalg.norm(can[:, :2] - can[0, :2], axis=1))),
                "gripper_events": events,
                "holding_true": sum(entry.get("observed_holding") is True for entry in motion),
                "projected": sum(int(entry["projected_count"]) for entry in latency),
                "raw_outside": int(np.count_nonzero((raw < 0) | (raw > 0.045))),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "verification": {
                    "success": bool(outcome.get("success")),
                    "home": bool(outcome.get("home_reached")),
                    "placement_m": float(outcome.get("final_target_distance", float("nan"))),
                },
            }
        )
    return rows
