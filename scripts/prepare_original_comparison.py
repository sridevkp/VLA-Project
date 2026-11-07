"""Prepare an original-controller worktree for an M1-06a comparison run."""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise ValueError(f"Expected fixture anchor was not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def prepare(worktree: Path) -> None:
    controller = worktree / "controllers" / "tiago_controller"
    config_path = controller / "config.py"
    pick_place_path = controller / "pick_place.py"
    kinematics_path = controller / "kinematics.py"
    motion_path = controller / "motion.py"
    world_path = worktree / "worlds" / "factory.wbt"

    if "VLAROBOT_COMPARISON_FIXTURE" in pick_place_path.read_text(encoding="utf-8"):
        print(f"Comparison fixture is already prepared in {worktree}")
        return
    _replace_once(
        config_path,
        "PLACE_BASE = [-1.07, -3.9, 0.8]  # target can centre in robot-base coordinates",
        "PLACE_WORLD = [-1.0, -4.3, 0.821]  # VLAROBOT_COMPARISON_FIXTURE",
    )
    _replace_once(config_path, "STOP_AFTER = None", 'STOP_AFTER = "LIFT"')
    _replace_once(
        kinematics_path,
        "self.rng = np.random.default_rng(1)",
        "self.rng = np.random.default_rng(0)  # VLAROBOT_COMPARISON_FIXTURE",
    )
    _replace_once(
        pick_place_path,
        "import numpy as np",
        "import json\nimport os\nimport numpy as np  # VLAROBOT_COMPARISON_FIXTURE",
    )
    pick_text = pick_place_path.read_text(encoding="utf-8")
    pick_text = pick_text.replace("PLACE_BASE,", "PLACE_WORLD,", 1)
    pick_text = pick_text.replace(
        "    PLACE_WORLD, GRASP_Z, APPROACH_H, PLACE_CLEAR, ALIGN_TOL, SKIP_ALIGN_CHECK,",
        "    PLACE_WORLD, ARM_JOINTS, GRASP_Z, APPROACH_H, PLACE_CLEAR, "
        "ALIGN_TOL, SKIP_ALIGN_CHECK,",
        1,
    )
    pick_text = pick_text.replace(
        """                self.place_tip = np.asarray(PLACE_BASE, dtype=float) + np.array(
                    [0, 0, GRASP_Z + PLACE_CLEAR]
                )""",
        """                self.place_base = self.frames.world_to_base(PLACE_WORLD)
                self.place_tip = self.place_base + np.array(
                    [0, 0, GRASP_Z + PLACE_CLEAR]
                )""",
        1,
    )
    pick_text = pick_text.replace(
        """                    self.place_tip = np.asarray(PLACE_BASE, dtype=float) + np.array(
                        [0, 0, PLACE_CLEAR]
                    ) + held""",
        """                    self.place_tip = self.place_base + np.array(
                        [0, 0, PLACE_CLEAR]
                    ) + held""",
        1,
    )
    if "VLAROBOT_COMPARISON_FIXTURE" not in pick_text:
        raise ValueError(f"Could not prepare target frame in {pick_place_path}")
    if "PLACE_BASE" in pick_text:
        raise ValueError(f"Legacy PLACE_BASE references remain in {pick_place_path}")
    pick_text = pick_text.replace(
        """            finished = self.state""",
        """            finished = self.state
            self.motion.trace_state = finished
            self._trace_boundary(finished, "enter")""",
        1,
    )
    pick_text = pick_text.replace(
        """            if STOP_AFTER and finished == STOP_AFTER:""",
        """            self._trace_boundary(finished, "exit")
            if STOP_AFTER and finished == STOP_AFTER:""",
        1,
    )
    boundary_method = """
    def _trace_boundary(self, state, phase):
        path = os.environ.get("VLAROBOT_TRACE_LOG")
        if not path:
            return
        record = {
            "event": "state_boundary",
            "phase": phase,
            "state": state,
            "commanded_base_xyz_m": None,
            "achieved_ee_base_xyz_m": self.frames.tool_position().tolist(),
            "joint_positions_rad": {
                name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
            },
            "gripper_opening_m": None,
            "grasp_detected": None,
        }
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\\n")

"""
    pick_text = pick_text.replace(
        "    def run(self):",
        "    # VLAROBOT_COMPARISON_FIXTURE: comparison-only state trace.\n"
        + boundary_method
        + "    def run(self):",
        1,
    )
    pick_place_path.write_text(pick_text, encoding="utf-8")

    _replace_once(
        motion_path,
        "import numpy as np",
        "import json\nimport os\nimport numpy as np  # VLAROBOT_COMPARISON_FIXTURE",
    )
    motion_text = motion_path.read_text(encoding="utf-8")
    motion_text = motion_text.replace(
        "        self.io, self.frames, self.ik, self.vis = robot_io, frames, "
        "kinematics, visualization",
        "        self.io, self.frames, self.ik, self.vis = robot_io, frames, "
        "kinematics, visualization\n"
        '        self.trace_state = "INIT"',
        1,
    )
    motion_text = motion_text.replace(
        "                return True",
        "                self._trace_waypoint(q)\n                return True",
        1,
    )
    logger = """
    def _trace_waypoint(self, q):
        path = os.environ.get("VLAROBOT_TRACE_LOG")
        if not path:
            return
        joint_target = {name: float(q[self.ik.idx[name]]) for name in ARM_JOINTS}
        record = {
            "event": "state_boundary",
            "phase": "waypoint",
            "state": self.trace_state,
            "commanded_base_xyz_m": self.ik.forward(q)[:3, 3].tolist(),
            "achieved_ee_base_xyz_m": self.frames.tool_position().tolist(),
            "joint_positions_rad": {
                name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
            },
            "joint_target": joint_target,
            "gripper_opening_m": None,
            "grasp_detected": None,
        }
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\\n")

"""
    motion_text = motion_text.replace(
        "    def go_to(self, target, tol=0.01, ik_tol=IK_TOL):",
        "    # VLAROBOT_COMPARISON_FIXTURE: comparison-only waypoint trace.\n"
        + logger
        + "    def go_to(self, target, tol=0.01, ik_tol=IK_TOL):",
        1,
    )
    if "VLAROBOT_COMPARISON_FIXTURE" not in motion_text:
        raise ValueError(f"Could not instrument motion waypoints in {motion_path}")
    motion_path.write_text(motion_text, encoding="utf-8")

    _replace_once(
        world_path,
        'controller "tiago_controller"',
        'controller "<extern>"  # VLAROBOT_COMPARISON_FIXTURE',
    )
    print(f"Prepared comparison-only original fixture in {worktree}")
    print("Target: PLACE_WORLD converted with Frames.world_to_base; STOP_AFTER=LIFT; IK seed=0.")
    print("The world uses controller <extern>; set VLAROBOT_TRACE_LOG to write JSONL.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path, help="Path to the original-controller worktree.")
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    if not (worktree / ".git").exists() and not (worktree / ".git").is_file():
        raise SystemExit(f"Not a Git worktree: {worktree}")
    prepare(worktree)


if __name__ == "__main__":
    main()
