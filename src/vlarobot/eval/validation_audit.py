"""Independent physical VALIDATION evidence; never convert partial runs into final rates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vlarobot.eval.metrics import summarize_success
from vlarobot.eval.runner import build_episode_plan


def audit_validation(
    campaign: Path,
    *,
    place_world_m: tuple[float, float, float],
    distance_tolerance_m: float,
    height_tolerance_m: float,
    tilt_limit_deg: float,
    home_tolerance_rad: float,
    arm_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compare actual CAN/joint feedback to unchanged frozen thresholds for every pair."""
    plan = build_episode_plan("validation", 0)
    rows = []
    hashes = {}
    scenes = defaultdict(list)
    families = defaultdict(list)
    for entry in plan:
        stem = f"episode-{entry.index:02d}"
        meta = campaign / (stem + ".meta.json")
        npz = campaign / (stem + ".steps.npz")
        if not meta.is_file() or not npz.is_file():
            raise ValueError("full 18-pair VALIDATION campaign not complete")
        data = json.loads(meta.read_text(encoding="utf-8"))
        outcome = data["outcome"]
        if (
            data["scenario"]["metadata"]["split"] != "validation"
            or data["reset_seed"] != entry.seed
            or data["scenario"]["name"] != entry.scene.name
        ):
            raise ValueError("recording differs from frozen validation pair")
        if data["pipeline"] != "baseline":
            raise ValueError("validation expert provenance mismatch")
        with np.load(npz, allow_pickle=False) as a:
            can = a["gt_names"].tolist().index("CAN")
            names = a["joint_names"].tolist()
            xyz = a["gt_xyz"][:, can]
            quaternion = a["gt_quat"][-1, can]
            if not np.isfinite(np.r_[xyz.reshape(-1), quaternion]).all() or not np.isclose(
                quaternion @ quaternion, 1
            ):
                raise ValueError("invalid physical CAN evidence")
            distance = float(np.linalg.norm(xyz[-1] - place_world_m))
            height = float(xyz[-1, 2] - xyz[0, 2])
            tilt = math.degrees(
                math.acos(float(np.clip(1 - 2 * (quaternion[0] ** 2 + quaternion[1] ** 2), -1, 1)))
            )
            home = max(
                abs(float(a["joint_pos"][-1, names.index(n)] - a["joint_pos"][0, names.index(n)]))
                for n in arm_names
            )
            checks = {
                "measured_home_return": home <= home_tolerance_rad,
                "placement_distance": distance <= distance_tolerance_m,
                "placement_height": abs(height) <= height_tolerance_m,
                "upright": tilt < tilt_limit_deg,
            }
            independent = all(checks.values()) and outcome.get("home_reached") is True
            if outcome.get("success") is not independent:
                raise ValueError(
                    f"outcome/independent physical evidence disagreement at {entry.index}"
                )
            family = entry.scene.metadata["family"]
            scene = entry.scene.name.rsplit("-s", 1)[0]
            row = {
                "index": entry.index,
                "scene": scene,
                "family": family,
                "seed": entry.seed,
                "success": independent,
                "checks": checks,
                "max_can_rise_m": float(xyz[:, 2].max() - xyz[0, 2]),
                "final_can_world_xyz_m": xyz[-1].tolist(),
                "final_target_distance_m": distance,
                "final_height_error_m": height,
                "final_tilt_deg": tilt,
                "home_max_joint_error_rad": home,
                "recorded_steps": len(a["t"]),
                "sim_duration_s": float(a["t"][-1] - a["t"][0]),
                "failure_state": outcome.get("failure_state"),
                "failure_reason": outcome.get("failure_reason"),
            }
        rows.append(row)
        scenes[scene].append(independent)
        families[family].append(independent)
        for p in (meta, npz):
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    def stats(flags):
        from dataclasses import asdict

        return asdict(summarize_success(flags))

    return {
        "status": "MEASURED completed physical VALIDATION expert campaign",
        "episodes": 18,
        "overall": stats([r["success"] for r in rows]),
        "per_scene": {s: stats(v) for s, v in scenes.items()},
        "per_family": {s: stats(v) for s, v in families.items()},
        "records": rows,
        "failure_states": dict(Counter(r["failure_state"] for r in rows if not r["success"])),
        "files_sha256": hashes,
        "test_accessed": False,
        "training_data_admitted": False,
        "reproducibility_limits": [
            "one fresh session per pair; no exact-pair repeated trials",
            "worker scheduling and runtime provenance must be read from campaign sidecars",
            "seeds vary geometry; repeated scene indices require clustering",
            "Wilson intervals are descriptive; no seed-only determinism or TEST claim",
        ],
    }
