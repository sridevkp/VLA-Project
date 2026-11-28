"""Extract TRAIN demonstration decision boundaries and observed skill outcomes."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vlarobot.execution.skills import Skill

_PHASE_SKILL = {
    "INIT": Skill.OPEN,
    "PREGRASP": Skill.APPROACH,
    "DESCEND": Skill.DESCEND,
    "CLOSE": Skill.CLOSE,
    "LIFT": Skill.LIFT,
    "TRANSIT": Skill.TRANSFER,
    "PLACE_DESCEND": Skill.LOWER,
    "RELEASE": Skill.OPEN,
    "RETREAT": Skill.RETREAT,
}


def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sim_time_s": row["sim_time_s"],
        "joint_position_rad": row["measured_joint_positions_rad"],
        "tool_base_m": row["achieved_ee_base_xyz_m"],
        "gripper_opening_m": row["gripper_opening_m"],
        "gripper_holding": row["grasp_detected"],
        "oracle_can_base_m": row["can_base_xyz_m"],
        "oracle_can_world_m": row["can_world_xyz_m"],
        "robot_base_world_m": row["base_world_xyz_m"],
        "robot_base_world_orientation_xyzw": row["base_world_orientation_xyzw"],
    }


def demonstration_decisions(manifest: Path) -> Iterator[dict[str, Any]]:
    """Read only physically admitted TRAIN demonstrations from one explicit manifest."""
    records = json.loads(manifest.read_text(encoding="utf-8"))
    for demo in records["episodes"]:
        scene = demo["scene"]
        metadata = scene["metadata"]
        if metadata["split"] != "train":
            raise ValueError("V4 demonstration extraction requires TRAIN")
        if demo["status"] != "accepted" or demo["outcome"]["success"] is not True:
            raise ValueError("V4 demonstration extraction requires an accepted episode")
        raw = Path(demo["raw_dir"])
        audit = json.loads((raw / "audit.json").read_text(encoding="utf-8"))
        if audit["accepted"] is not True or not all(audit["checks"].values()):
            raise ValueError(f"missing independent physical admission for {raw}")
        pending: dict[str, dict[str, Any]] = {}
        with (raw / "policy.jsonl").open(encoding="utf-8") as trace:
            for line in trace:
                row = json.loads(line)
                phase = row.get("state")
                if row.get("event") != "state_boundary" or phase not in _PHASE_SKILL:
                    continue
                if row["phase"] == "enter":
                    pending[phase] = row
                if row["phase"] != "exit" or phase not in pending:
                    continue
                before = pending.pop(phase)
                skill = _PHASE_SKILL[phase]
                target = (
                    None if skill in (Skill.OPEN, Skill.CLOSE) else before["commanded_base_xyz_m"]
                )
                if skill not in (Skill.OPEN, Skill.CLOSE) and target is None:
                    raise ValueError(f"missing Cartesian target for {phase} in {raw}")
                yield {
                    "scene_group": f"train-{metadata['family']}-i{metadata['index']:02d}",
                    "scene": scene["name"],
                    "seed": demo["seed"],
                    "attempt": demo["attempt"],
                    "skill": skill.value,
                    "target_base_m": target,
                    "observed_before": _snapshot(before),
                    "offline_observed_after": _snapshot(row),
                    "offline_failed": row["failed"],
                    "offline_can_rise_m": row["can_world_xyz_m"][2] - before["can_world_xyz_m"][2],
                    "label_source": "demonstration",
                    "measurement_source": "simulator-observed",
                }
