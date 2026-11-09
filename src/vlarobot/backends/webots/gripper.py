"""Gripper open/close commands and a basic stall-based grasp heuristic."""

from vlarobot.backends.webots.config import FINGER_SPEED, GRIP_FORCE, MIN_GRASP_OPEN


class Gripper:
    def __init__(self, robot_io, visualization):
        self.io, self.vis = robot_io, visualization
        self.fingers = robot_io.fingers
        self.open_pos, self.closed_pos = robot_io.open_pos, robot_io.closed_pos
        self.last_close_stalled = None
        self.last_close_opening = None

    def values(self):
        return self.io.current_finger_values()

    def opening(self):
        if not self.fingers:
            return 0.0
        return sum(abs(s.getValue() - self.closed_pos) for _, s in self.fingers) / len(self.fingers)

    def open(self, timeout_s=4.0, cycle=False):
        if not self.fingers:
            print("[!] no gripper motors found")
            return False
        if cycle:
            for motor, _ in self.fingers:
                motor.setAvailableForce(motor.getMaxForce())
                motor.setVelocity(FINGER_SPEED)
                motor.setPosition(self.closed_pos)
            for _ in range(int(timeout_s * 1000 / self.io.ts)):
                if self.vis.step() == -1:
                    return False
                if all(abs(s.getValue() - self.closed_pos) < 0.001 for _, s in self.fingers):
                    break
        for motor, _ in self.fingers:
            motor.setAvailableForce(motor.getMaxForce())
            motor.setVelocity(FINGER_SPEED)
            motor.setPosition(self.open_pos)
        for _ in range(int(timeout_s * 1000 / self.io.ts)):
            if self.vis.step() == -1:
                return False
            if all(abs(s.getValue() - self.open_pos) < 0.001 for _, s in self.fingers):
                break
        values = self.values()
        ok = all(abs(v - self.open_pos) < 0.002 for v in values)
        print(f"  fingers {values} (target {self.open_pos}) -> {'OPEN' if ok else 'NOT OPEN'}")
        return ok

    def close(self):
        self.last_close_stalled = False
        if not self.fingers:
            self.last_close_opening = 0.0
            return False, 0.0
        for motor, _ in self.fingers:
            if GRIP_FORCE is not None:
                max_force = (
                    float(motor.getMaxForce())
                    if hasattr(motor, "getMaxForce")
                    else float(GRIP_FORCE)
                )
                motor.setAvailableForce(min(float(GRIP_FORCE), max_force))
            motor.setPosition(self.closed_pos)
        previous, still, stalled = self.opening(), 0, False
        for i in range(300):
            if self.vis.step() == -1:
                return False, 0.0
            current = self.opening()
            still = still + 1 if abs(current - previous) < 2e-5 else 0
            previous = current
            if i > 10 and still >= 15:
                stalled = True
                break
        opening = self.opening()
        self.last_close_stalled = stalled
        self.last_close_opening = opening
        print(f"  fingers close ended opening={opening:.4f} m stalled={stalled}")
        return opening > MIN_GRASP_OPEN, opening
