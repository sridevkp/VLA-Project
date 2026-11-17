"""Audit completed frozen VALIDATION recordings with independent CAN/joint evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve previous report version")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vlarobot.eval.validation_audit import audit_validation

    config = yaml.safe_load((args.project / "configs/robot.yaml").read_text(encoding="utf-8"))
    report = audit_validation(
        args.campaign,
        place_world_m=tuple(config["place_world"]),
        distance_tolerance_m=config["place_tolerance_m"],
        height_tolerance_m=config["can_place_z_tolerance_m"],
        tilt_limit_deg=config["can_tilt_limit_deg"],
        home_tolerance_rad=config["home_settle_tolerance_rad"],
        arm_names=tuple(f"arm_{i}_joint" for i in range(1, 8)),
    )
    args.out.mkdir(parents=True)
    (args.out / "physical-validation-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: report[k] for k in ("overall", "per_scene", "per_family", "failure_states")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
