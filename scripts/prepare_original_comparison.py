"""Prepare a clean original-controller archive/worktree for M1-06a trace capture."""

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
    gripper_path = controller / "gripper.py"
    world_path = worktree / "worlds" / "factory.wbt"

    pick_source = pick_place_path.read_text(encoding="utf-8")
    if "VLAROBOT_COMPARISON_FIXTURE_V2" in pick_source:
        print(f"Comparison fixture v2 is already prepared in {worktree}")
        return
    if "VLAROBOT_COMPARISON_FIXTURE" in pick_source:
        raise ValueError(
            "Fixture v1 is already applied. Prepare fixture v2 from a clean archive/worktree "
            "to preserve existing source and traces."
        )

    _replace_once(
        config_path,
        "PLACE_BASE = [-1.07, -3.9, 0.8]  # target can centre in robot-base coordinates",
        "PLACE_WORLD = [-1.0, -4.3, 0.821]  # VLAROBOT_COMPARISON_FIXTURE_V2",
    )
    _replace_once(config_path, "STOP_AFTER = None", 'STOP_AFTER = "LIFT"')
    _replace_once(
        kinematics_path,
        "self.rng = np.random.default_rng(1)",
        "self.rng = np.random.default_rng(0)  # VLAROBOT_COMPARISON_FIXTURE_V2",
    )
    _replace_once(
        pick_place_path,
        "import numpy as np",
        "import json\nimport os\nimport numpy as np  # VLAROBOT_COMPARISON_FIXTURE_V2",
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
    if "VLAROBOT_COMPARISON_FIXTURE_V2" not in pick_text or "PLACE_BASE" in pick_text:
        raise ValueError(f"Could not prepare the target frame in {pick_place_path}")
    pick_text = pick_text.replace(
        "        self.motion, self.gripper, self.vis = motion, gripper, visualization",
        "        self.motion, self.gripper, self.vis = motion, gripper, visualization\n"
        "        self.motion.trace_task = self",
        1,
    )
    pick_text = pick_text.replace(
        "            finished = self.state",
        "            finished = self.state\n"
        "            self.motion.trace_state = finished\n"
        '            self._trace_boundary(finished, "enter")',
        1,
    )
    pick_text = pick_text.replace(
        "            if STOP_AFTER and finished == STOP_AFTER:",
        '            self._trace_boundary(finished, "exit")\n'
        '            if finished == "INIT":\n'
        '                self._trace_cartesian_waypoint("descend_grasp", self.grasp_tip)\n'
        '                self._trace_cartesian_waypoint("lift", self.pregrasp)\n'
        "            if STOP_AFTER and finished == STOP_AFTER:",
        1,
    )

    trace_methods = """
    def _trace_measurements(self):
        can_world = np.asarray(self.io.can_node.getPosition(), dtype=float)
        if getattr(self, "can_initial_world_xyz", None) is None:
            self.can_initial_world_xyz = can_world.copy()
        base_rotation = np.asarray(self.io.me.getOrientation(), dtype=float).reshape(3, 3)
        can_rotation = np.asarray(self.io.can_node.getOrientation(), dtype=float).reshape(3, 3)
        joint_limits = {
            name: [float(value) for value in self.ik.bounds(name)] for name in ARM_JOINTS
        }
        joint_positions = {
            name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
        }
        return {
            "can_world_xyz_m": can_world.tolist(),
            "can_base_xyz_m": self.frames.world_to_base(can_world).tolist(),
            "can_displacement_world_xyz_m": (can_world - self.can_initial_world_xyz).tolist(),
            "lift_height_m": float(can_world[2] - self.can_initial_world_xyz[2]),
            "can_world_orientation_matrix": can_rotation.reshape(-1).tolist(),
            "can_base_orientation_matrix": (base_rotation.T @ can_rotation).reshape(-1).tolist(),
            "base_world_xyz_m": [float(value) for value in self.io.me.getPosition()],
            "base_world_orientation_matrix": base_rotation.reshape(-1).tolist(),
            "tool_base_xyz_m": self.frames.tool_position().tolist(),
            "joint_limits_rad": joint_limits,
            "measured_joint_positions_rad": joint_positions,
            "commanded_joint_positions_rad": joint_positions,
            "commanded_joint_target_rad": getattr(
                self.motion, "last_joint_target", None
            ) or joint_positions,
            "gripper_opening_m": float(self.gripper.opening()),
            "grasp_detected": getattr(self.gripper, "last_grasp_detected", None),
            "close_stalled": getattr(self.gripper, "last_close_stalled", None),
            "close_stall_counter": getattr(self.gripper, "last_close_stall_counter", None),
            "close_opening_m": getattr(self.gripper, "last_close_opening", None),
            "ik_error_m": self.motion.last_ik_error,
            "ik_tilt_deg": self.motion.last_ik_tilt,
        }

    def _trace_boundary(self, state, phase):
        path = os.environ.get("VLAROBOT_TRACE_LOG")
        if not path:
            return
        commanded_target = {
            "PREGRASP": getattr(self, "pregrasp", None),
            "DESCEND": getattr(self, "grasp_tip", None),
            "CLOSE": getattr(self, "grasp_tip", None),
            "LIFT": getattr(self, "pregrasp", None),
            "TRANSIT": getattr(self, "place_above", None),
            "PLACE_DESCEND": getattr(self, "place_tip", None),
            "RELEASE": getattr(self, "place_tip", None),
            "RETREAT": getattr(self, "place_above", None),
            "HOME": None if self.home_q is None else self.ik.forward(self.home_q)[:3, 3],
        }.get(state)
        joint_positions = {
            name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
        }
        record = {
            "event": "state_boundary",
            "phase": phase,
            "state": state,
            "commanded_base_xyz_m": None,
            "measured_joint_positions_rad": joint_positions,
            "commanded_joint_target_rad": getattr(
                self.motion, "last_joint_target", None
            ) or joint_positions,
            "commanded_joint_positions_rad": joint_positions,
            "commanded_ee_base_xyz_m": (
                None if commanded_target is None else commanded_target.tolist()
            ),
            "achieved_ee_base_xyz_m": self.frames.tool_position().tolist(),
            "joint_positions_rad": joint_positions,
        }
        record.update(self._trace_measurements())
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\\n")

    def _trace_cartesian_waypoint(self, name, point):
        path = os.environ.get("VLAROBOT_TRACE_LOG")
        if not path:
            return
        record = {
            "event": "waypoint",
            "state": "INIT",
            "waypoint": name,
            "base_xyz_m": [float(value) for value in point],
        }
        record.update(self._trace_measurements())
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\\n")

"""
    pick_text = pick_text.replace(
        "    def run(self):",
        "    # VLAROBOT_COMPARISON_FIXTURE_V2: diagnostics only.\n"
        + trace_methods
        + "    def run(self):",
        1,
    )
    pick_place_path.write_text(pick_text, encoding="utf-8")

    _replace_once(
        motion_path,
        "import numpy as np",
        "import json\nimport os\nimport numpy as np  # VLAROBOT_COMPARISON_FIXTURE_V2",
    )
    motion_text = motion_path.read_text(encoding="utf-8")
    motion_text = motion_text.replace(
        "        self.io, self.frames, self.ik, self.vis = robot_io, frames, "
        "kinematics, visualization",
        "        self.io, self.frames, self.ik, self.vis = robot_io, frames, "
        "kinematics, visualization\n"
        '        self.trace_state = "INIT"\n'
        "        self.trace_task = None\n"
        "        self.last_joint_target = {}\n"
        "        self.last_ik_error = None\n"
        "        self.last_ik_tilt = None",
        1,
    )
    motion_text = motion_text.replace(
        "                return True",
        "                self._trace_waypoint(q)\n                return True",
        1,
    )
    motion_logger = """
    def _trace_waypoint(self, q):
        path = os.environ.get("VLAROBOT_TRACE_LOG")
        if not path:
            return
        joint_target = {name: float(q[self.ik.idx[name]]) for name in ARM_JOINTS}
        self.last_joint_target = joint_target.copy()
        record = {
            "event": "state_boundary",
            "phase": "waypoint",
            "state": self.trace_state,
            "commanded_base_xyz_m": self.ik.forward(q)[:3, 3].tolist(),
            "commanded_ee_base_xyz_m": self.ik.forward(q)[:3, 3].tolist(),
            "achieved_ee_base_xyz_m": self.frames.tool_position().tolist(),
            "joint_positions_rad": {
                name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
            },
            "measured_joint_positions_rad": {
                name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
            },
            "commanded_joint_positions_rad": {
                name: float(self.io.sensors[name].getValue()) for name in ARM_JOINTS
            },
            "joint_target": joint_target,
            "commanded_joint_target_rad": joint_target,
            "ik_error_m": self.last_ik_error,
            "ik_tilt_deg": self.last_ik_tilt,
        }
        if self.trace_task is not None:
            record.update(self.trace_task._trace_measurements())
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\\n")

"""
    motion_text = motion_text.replace(
        "    def go_to(self, target, tol=0.01, ik_tol=IK_TOL):",
        "    # VLAROBOT_COMPARISON_FIXTURE_V2: diagnostics only.\n"
        + motion_logger
        + "    def go_to(self, target, tol=0.01, ik_tol=IK_TOL):",
        1,
    )
    motion_text = motion_text.replace(
        "        q, error, tilt = self.ik.solve(target, self.frames.current_q(), restarts=True)",
        "        q, error, tilt = self.ik.solve(target, self.frames.current_q(), restarts=True)\n"
        "        self.last_ik_error, self.last_ik_tilt = error, tilt",
        1,
    )
    motion_text = motion_text.replace(
        "            q, error, tilt = self.ik.solve(point, q)",
        "            q, error, tilt = self.ik.solve(point, q)\n"
        "            self.last_ik_error, self.last_ik_tilt = error, tilt",
    )
    motion_path.write_text(motion_text, encoding="utf-8")

    _instrument_gripper(gripper_path)
    _replace_once(
        world_path,
        'controller "tiago_controller"',
        'controller "<extern>"  # VLAROBOT_COMPARISON_FIXTURE_V2',
    )
    print(f"Prepared comparison-only original fixture v2 in {worktree}")
    print("Target frame, stop-after-LIFT, and deterministic IK seed are fixed for comparison.")
    print("Instrumentation records robot/object/gripper/IK measurements; behavior is unchanged.")


def _instrument_gripper(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "        self.fingers = robot_io.fingers",
        "        self.fingers = robot_io.fingers\n"
        "        self.last_close_stalled = None\n"
        "        self.last_close_stall_counter = 0\n"
        "        self.last_close_opening = None\n"
        "        self.last_grasp_detected = None",
        1,
    )
    text = text.replace(
        "        previous, still = self.opening(), 0",
        "        previous, still = self.opening(), 0\n        self.last_close_stall_counter = 0",
        1,
    )
    text = text.replace(
        "            still = still + 1 if abs(current - previous) < 2e-5 else 0\n"
        "            previous = current",
        "            still = still + 1 if abs(current - previous) < 2e-5 else 0\n"
        "            self.last_close_stall_counter = still\n"
        "            previous = current",
        1,
    )
    text = text.replace(
        '        print(f"  fingers stalled at opening={opening:.4f} m")\n'
        "        return opening > MIN_GRASP_OPEN, opening",
        '        print(f"  fingers stalled at opening={opening:.4f} m")\n'
        "        self.last_close_stalled = still >= 15\n"
        "        self.last_close_opening = opening\n"
        "        self.last_grasp_detected = opening > MIN_GRASP_OPEN\n"
        "        return self.last_grasp_detected, opening",
        1,
    )
    if "last_close_stall_counter" not in text:
        raise ValueError(f"Could not instrument gripper diagnostics in {path}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path, help="Path to an original-controller tree.")
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    if not (worktree / ".git").exists() and not (worktree / ".git").is_file():
        raise SystemExit(f"Not a Git worktree: {worktree}")
    prepare(worktree)


if __name__ == "__main__":
    main()
