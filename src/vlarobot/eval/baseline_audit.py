"""Read-only physical and provenance audit of a completed repaired frozen baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.eval.demo_audit import DemoAuditSpec
from vlarobot.eval.runner import _completed_recording, build_episode_plan


def audit_completed_baseline(out: Path, spec: DemoAuditSpec) -> dict[str, Any]:
    """Require all frozen recordings, then measure CAN rise independently of HOME."""
    status = json.loads((out / "campaign-status.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    plan = build_episode_plan("train", 0)
    if status["status"] != "completed" or summary["n_episodes"] != len(plan):
        raise ValueError("completed repaired 24-episode campaign required")
    episodes: list[dict[str, Any]] = []
    failure_categories: Counter[str] = Counter()
    hashes = {}
    for entry in plan:
        if _completed_recording(out, entry, "baseline") is None:
            raise ValueError(f"missing completed frozen physical episode {entry.index}")
        prefix = f"episode-{entry.index:02d}"
        meta = json.loads((out / f"{prefix}.meta.json").read_text(encoding="utf-8"))
        if meta["scenario"] != json.loads(json.dumps(asdict(entry.scene))):
            raise ValueError(f"episode {entry.index} geometry differs from the frozen plan")
        outcome = meta["outcome"]
        with np.load(out / f"{prefix}.steps.npz", allow_pickle=False) as samples:
            can = samples["gt_xyz"][:, samples["gt_names"].tolist().index("CAN")]
            quat = samples["gt_quat"][:, samples["gt_names"].tolist().index("CAN")]
            t = samples["t"]
            rise = can[:, 2] - can[0, 2]
            elevated = rise >= spec.lift_min_m
            carried = can[elevated, :2]
            # This is trajectory evidence, not a replacement success rule or contact test.
            travel = (
                float(np.linalg.norm(carried - carried[0], axis=1).max()) if len(carried) else 0.0
            )
            physical = {
                "max_can_rise_m": float(rise.max()),
                "samples_with_can_rise_ge_40mm": int(elevated.sum()),
                "elevated_horizontal_travel_m": travel,
                "final_height_change_m": float(rise[-1]),
                "recorded_sim_duration_s": float(t[-1] - t[0]),
                "samples": len(t),
                "measured_final_target_distance_m": float(
                    np.linalg.norm(can[-1] - spec.place_world_m)
                ),
                "measured_final_tilt_deg": float(
                    np.degrees(
                        np.arccos(np.clip(1 - 2 * (quat[-1, 0] ** 2 + quat[-1, 1] ** 2), -1, 1))
                    )
                ),
            }
            physical["independent_placement_ok"] = (
                physical["measured_final_target_distance_m"] <= spec.place_tolerance_m
                and abs(physical["final_height_change_m"]) <= spec.height_tolerance_m
                and physical["measured_final_tilt_deg"] < spec.tilt_limit_deg
            )
        if not outcome["success"]:
            category = outcome.get("failure_state") or "completed-but-unsuccessful-placement"
            failure_categories[category] += 1
        episodes.append(
            {
                "index": entry.index,
                "scene": entry.scene.name,
                "seed": entry.seed,
                "outcome": outcome,
                **physical,
            }
        )
        for suffix in ["meta.json", "steps.npz", "actions.jsonl", "contacts.jsonl"]:
            name = f"{prefix}.{suffix}"
            hashes[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    successes = sum(e["outcome"]["success"] for e in episodes)
    if successes != summary["success"]["successes"]:
        raise ValueError("aggregate success does not match retained outcomes")
    return {
        "campaign": str(out.resolve()),
        "summary": summary,
        "runtime": status,
        "provenance": provenance,
        "failure_categories": dict(failure_categories),
        "episodes": episodes,
        "recording_sha256": hashes,
        "physical_successes_with_rise_ge_40mm": sum(
            e["outcome"]["success"] and e["max_can_rise_m"] >= spec.lift_min_m for e in episodes
        ),
        "successful_outcome_placement_disagreements": [
            e["index"]
            for e in episodes
            if e["outcome"]["success"] and not e["independent_placement_ok"]
        ],
        "pad_contact_limit": (
            "baseline NPZ retains CAN/EE trajectories; individual pads are measured in the "
            "instrumented comparison and new demonstration physical streams, not this campaign"
        ),
        "literal_gate_threshold": min(0.4, 0.6 * successes / len(plan)),
        "vla_gate_result": "UNMEASURED: no VLA-only evaluation",
        "zero_baseline_owner_decision_required": successes == 0,
        "fresh_v1_not_merged": True,
        "full_scene_geometry_matches_frozen_plan": True,
    }
