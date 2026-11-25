"""Joint-space moves and straight-line Cartesian moves."""

import numpy as np

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    IK_TOL,
    IK_TOL_GRASP,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
)
from vlarobot.backends.webots.kinematics import IKTrustRegionError


class Motion:
    def __init__(self, robot_io, frames, kinematics, visualization):
        self.io, self.frames, self.ik, self.vis = robot_io, frames, kinematics, visualization
        self.last_ik_error = None
        self.last_ik_tilt = None
        self.last_ik_failure_reason = None
        self.last_waypoint_index = None
        self.last_waypoint_count = None
        self.last_target = None

    def move_to_q(self, q, tol=0.01, max_steps=800):
        start = self.frames.current_q()
        for waypoint in self.ik.interpolate_joint_path(start, q, MAX_JOINT_STEP_RAD):
            for name in ARM_JOINTS:
                self.io.motors[name].setPosition(float(waypoint[self.ik.idx[name]]))
            for _ in range(max_steps):
                if self.vis.step() == -1:
                    return False
                errors = {
                    name: abs(self.io.sensors[name].getValue() - waypoint[self.ik.idx[name]])
                    for name in ARM_JOINTS
                }
                if max(errors.values()) < tol:
                    break
            else:
                worst = max(errors, key=lambda name: errors[name])
                print(
                    f"  [!] move timed out (possibly collision). Worst joint: {worst}, "
                    f"error {errors[worst]:.3f} rad"
                )
                return False
        return True

    def go_to(self, target, tol=0.01, ik_tol=IK_TOL):
        self.vis.set_target(target)
        q, error, tilt = self.ik.solve(
            target,
            self.frames.current_q(),
            restarts=True,
            position_tolerance=ik_tol,
        )
        self.last_ik_error, self.last_ik_tilt = error, tilt
        self.last_waypoint_index, self.last_waypoint_count = 1, 1
        self.last_target = np.asarray(target, dtype=float).tolist()
        print(f"  IK -> {np.round(target, 3)} err={error:.4f} m tilt={tilt:.1f} deg")
        if error > ik_tol or tilt > MAX_TILT_DEG:
            print("  [!] IK rejected (unreachable or tilted)")
            return False
        return self.move_to_q(q, tol=tol)

    def move_linear(self, target, step=0.03, ik_tol=IK_TOL_GRASP):
        start = self.frames.tool_position()
        target = np.asarray(target, dtype=float)
        self.vis.set_target(target)
        count = max(2, int(np.linalg.norm(target - start) / step))
        q = self.frames.current_q()
        self.last_target = target.tolist()
        self.last_waypoint_index, self.last_waypoint_count = 0, count
        for index, alpha in enumerate(np.linspace(0, 1, count + 1)[1:], 1):
            point = (1 - alpha) * start + alpha * target
            try:
                q, error, tilt = self.ik.solve(
                    point,
                    q,
                    restarts=True,
                    position_tolerance=ik_tol,
                    max_joint_step_rad=MAX_JOINT_STEP_RAD,
                )
            except IKTrustRegionError as error:
                self.last_ik_failure_reason = str(error)
                self.last_waypoint_index = index
                print(f"  [!] linear move failed at waypoint {index}/{count}: {error}")
                return False
            self.last_ik_failure_reason = None
            self.last_ik_error, self.last_ik_tilt = error, tilt
            self.last_waypoint_index = index
            if error > ik_tol or tilt > MAX_TILT_DEG:
                print(
                    f"  [!] linear move failed at waypoint {index}/{count}: "
                    f"err={error:.4f} tilt={tilt:.1f}"
                )
                return False
            final = index == count
            if not self.move_to_q(q, tol=0.01 if final else 0.03, max_steps=600):
                return False
        return True
