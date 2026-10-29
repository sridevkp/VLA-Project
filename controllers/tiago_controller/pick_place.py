"""High-level pick-and-place state machine."""
import numpy as np
from config import (
    PLACE_BASE, GRASP_Z, APPROACH_H, PLACE_CLEAR, ALIGN_TOL, SKIP_ALIGN_CHECK,
    LIFT_MIN, MAX_RETRIES, STOP_AFTER, IK_TOL_GRASP, TORSO_HEIGHT,
    HEAD_PAN, HEAD_TILT, HEAD_JOINTS, TOOL_OFFSET, GRIPPER_TEST, CALIBRATE_TOOL,
)

class PickPlace:
    def __init__(self, robot_io, frames, kinematics, motion, gripper, visualization):
        self.io, self.frames, self.ik = robot_io, frames, kinematics
        self.motion, self.gripper, self.vis = motion, gripper, visualization
        self.state = "INIT"
        self.retries = 0
        self.gripper_primed = False
        self.home_q = None
        self.can_z0_world = None

    def initialize_posture(self):
        if TORSO_HEIGHT is not None:
            self.io.motors["torso_lift_joint"].setPosition(TORSO_HEIGHT)
            for _ in range(1000):
                if self.vis.step() == -1:
                    return False
                if abs(self.io.sensors["torso_lift_joint"].getValue() - TORSO_HEIGHT) < 0.002:
                    break
            print("Torso at", round(self.io.sensors["torso_lift_joint"].getValue(), 3))

        for name, target in zip(HEAD_JOINTS, [HEAD_PAN, HEAD_TILT]):
            motor = self.io.robot.getDevice(name)
            if motor is None:
                print(f"[warn] head motor {name} not found")
                continue
            sensor = motor.getPositionSensor()
            if sensor is not None:
                sensor.enable(self.io.ts)
            lo, hi = motor.getMinPosition(), motor.getMaxPosition()
            if hi > lo:
                target = min(max(target, lo), hi)
            motor.setVelocity(0.5)
            motor.setPosition(target)
        self.vis.wait(int(2000 / self.io.ts))
        print("Finger sensors before any finger command:", self.gripper.values())

        if GRIPPER_TEST:
            for label, target in [
                ("CLOSE", self.gripper.closed_pos), ("OPEN", self.gripper.open_pos),
                ("CLOSE", self.gripper.closed_pos), ("OPEN", self.gripper.open_pos),
            ]:
                for motor, _ in self.gripper.fingers:
                    motor.setPosition(target)
                self.vis.wait(int(3000 / self.io.ts))
                print(f"  [gripper test] {label}: sensors {self.gripper.values()} <- check 3D view")
            print("Gripper test finished; set GRIPPER_TEST=False to run pick-and-place.")
            while self.vis.step() != -1:
                pass
            return False

        if CALIBRATE_TOOL:
            from calibration import calibrate_tool_offset
            self.gripper.open()
            calibrate_tool_offset(self.io, self.frames, self.ik)

        self.home_q = self.frames.current_q()
        return True

    def run(self):
        print("Starting pick-and-place")
        while self.vis.step() != -1 and self.state not in ("DONE", "FAILED"):
            finished = self.state

            if self.state == "INIT":
                can_base = self.frames.can_base_position()
                self.grasp_tip = can_base + np.array([0, 0, GRASP_Z])
                self.pregrasp = self.grasp_tip + np.array([0, 0, APPROACH_H])
                self.place_tip = np.asarray(PLACE_BASE, dtype=float) + np.array(
                    [0, 0, GRASP_Z + PLACE_CLEAR]
                )
                self.place_above = self.place_tip + np.array([0, 0, APPROACH_H])
                self.can_z0_world = self.io.can_node.getPosition()[2]
                print("Can (base frame):", np.round(can_base, 3))
                if not self.gripper.open(cycle=not self.gripper_primed):
                    print("[!] gripper failed to open; aborting")
                    self.state = "FAILED"
                    continue
                self.gripper_primed = True
                self.state = "PREGRASP"

            elif self.state == "PREGRASP":
                print("[PREGRASP]")
                self.state = "DESCEND" if self.motion.go_to(self.pregrasp) else "FAILED"

            elif self.state == "DESCEND":
                print("[DESCEND]")
                print("  finger sensors:", self.gripper.values())
                before = np.array(self.io.can_node.getPosition())
                ok = self.motion.move_linear(self.grasp_tip, ik_tol=IK_TOL_GRASP)
                print("  can moved by", np.round(np.array(self.io.can_node.getPosition()) - before, 4))
                print("  finger sensors:", self.gripper.values())
                self.state = "CLOSE" if ok else "FAILED"

            elif self.state == "CLOSE":
                print("[CLOSE]")
                transform = self.ik.forward(self.frames.current_q())
                error_base = self.frames.can_base_position() - transform[:3, 3]
                error_base[2] = 0.0
                error_local = transform[:3, :3].T @ error_base
                print("  suggested TOOL_OFFSET (only if can is visibly between fingers):",
                      np.round(np.array(TOOL_OFFSET) + error_local, 4).tolist())
                xy_error = np.linalg.norm(
                    (self.frames.can_base_position() - self.frames.tool_position())[:2]
                )
                print(f"  x,y misalignment can vs tool: {xy_error * 1000:.1f} mm")
                if xy_error > ALIGN_TOL and not SKIP_ALIGN_CHECK:
                    print("[!] misaligned beyond ALIGN_TOL; retrying")
                    self.state = "RETRY"
                else:
                    if xy_error > ALIGN_TOL:
                        print("[skip] alignment check disabled; closing anyway")
                    self.vis.wait(10)
                    grasped, _ = self.gripper.close()
                    self.vis.wait(20)
                    self.state = "LIFT" if grasped else "RETRY"

            elif self.state == "LIFT":
                print("[LIFT]")
                ok = self.motion.move_linear(self.pregrasp)
                self.vis.wait(20)
                rose = self.io.can_node.getPosition()[2] - self.can_z0_world
                distance = np.linalg.norm(
                    self.frames.can_base_position() - self.frames.tool_position()
                )
                print(f"  can rose {rose:.3f} m, can-to-tool {distance:.3f} m")
                if ok and rose > LIFT_MIN and distance < 0.10:
                    held = self.frames.tool_position() - self.frames.can_base_position()
                    self.place_tip = np.asarray(PLACE_BASE, dtype=float) + np.array(
                        [0, 0, PLACE_CLEAR]
                    ) + held
                    self.place_above = self.place_tip + np.array([0, 0, APPROACH_H])
                    print("  held offset tool-can:", np.round(held, 4))
                    self.state = "TRANSIT"
                else:
                    self.state = "RETRY"

            elif self.state == "TRANSIT":
                print("[TRANSIT]")
                self.state = "PLACE_DESCEND" if self.motion.move_linear(self.place_above) else "FAILED"

            elif self.state == "PLACE_DESCEND":
                print("[PLACE_DESCEND]")
                self.state = "RELEASE" if self.motion.move_linear(self.place_tip) else "FAILED"

            elif self.state == "RELEASE":
                print("[RELEASE]")
                self.gripper.open()
                self.vis.wait(30)
                self.state = "RETREAT"

            elif self.state == "RETREAT":
                print("[RETREAT]")
                self.state = "HOME" if self.motion.move_linear(self.place_above) else "FAILED"

            elif self.state == "HOME":
                print("[HOME]")
                self.vis.set_target(None)
                self.motion.move_to_q(self.home_q, tol=0.02)
                self.state = "DONE"

            elif self.state == "RETRY":
                self.retries += 1
                print(f"[RETRY] attempt {self.retries}/{MAX_RETRIES}")
                if self.retries > MAX_RETRIES:
                    self.state = "FAILED"
                else:
                    self.gripper.open()
                    self.motion.move_linear(
                        self.frames.tool_position() + np.array([0, 0, APPROACH_H])
                    )
                    self.state = "INIT"

            if STOP_AFTER and finished == STOP_AFTER:
                print(f"Stopped after {finished}; next state would be {self.state}.")
                break

        print("Final state:", self.state)
        while self.vis.step() != -1:
            pass
