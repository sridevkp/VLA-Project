"""Gripper open/close commands and a basic stall-based grasp heuristic."""
from config import FINGER_SPEED, GRIP_FORCE, MIN_GRASP_OPEN

class Gripper:
    def __init__(self, robot_io, visualization):
        self.io, self.vis = robot_io, visualization
        self.fingers = robot_io.fingers
        self.open_pos, self.closed_pos = robot_io.open_pos, robot_io.closed_pos

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
        if not self.fingers:
            return False, 0.0
        for motor, _ in self.fingers:
            if GRIP_FORCE is not None:
                motor.setAvailableForce(GRIP_FORCE)
            motor.setPosition(self.closed_pos)
        previous, still = self.opening(), 0
        for i in range(300):
            if self.vis.step() == -1:
                return False, 0.0
            current = self.opening()
            still = still + 1 if abs(current - previous) < 2e-5 else 0
            previous = current
            if i > 10 and still >= 15:
                break
        opening = self.opening()
        print(f"  fingers stalled at opening={opening:.4f} m")
        return opening > MIN_GRASP_OPEN, opening
