"""Joint-space moves and straight-line Cartesian moves."""
import numpy as np
from config import ARM_JOINTS, IK_TOL, IK_TOL_GRASP, MAX_TILT_DEG

class Motion:
    def __init__(self, robot_io, frames, kinematics, visualization):
        self.io, self.frames, self.ik, self.vis = robot_io, frames, kinematics, visualization

    def move_to_q(self, q, tol=0.01, max_steps=800):
        for name in ARM_JOINTS:
            self.io.motors[name].setPosition(float(q[self.ik.idx[name]]))
        for _ in range(max_steps):
            if self.vis.step() == -1:
                return False
            errors = {
                name: abs(self.io.sensors[name].getValue() - q[self.ik.idx[name]])
                for name in ARM_JOINTS
            }
            if max(errors.values()) < tol:
                return True
        worst = max(ARM_JOINTS, key=lambda n: abs(
            self.io.sensors[n].getValue() - q[self.ik.idx[n]]
        ))
        error = abs(self.io.sensors[worst].getValue() - q[self.ik.idx[worst]])
        print(f"  [!] move timed out (possibly collision). Worst joint: {worst}, error {error:.3f} rad")
        return False

    def go_to(self, target, tol=0.01, ik_tol=IK_TOL):
        self.vis.set_target(target)
        q, error, tilt = self.ik.solve(target, self.frames.current_q(), restarts=True)
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
        for index, alpha in enumerate(np.linspace(0, 1, count + 1)[1:], 1):
            point = (1 - alpha) * start + alpha * target
            q, error, tilt = self.ik.solve(point, q)
            if error > ik_tol or tilt > MAX_TILT_DEG:
                print(f"  [!] linear move failed at waypoint {index}/{count}: "
                      f"err={error:.4f} tilt={tilt:.1f}")
                return False
            final = index == count
            if not self.move_to_q(q, tol=0.01 if final else 0.03, max_steps=600):
                return False
        return True
