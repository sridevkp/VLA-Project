"""Debug markers and optional UI bridge ticking."""
import numpy as np

class Visualization:
    def __init__(self, robot_io, frames, period_ms=250, quality=60):
        self.io, self.frames = robot_io, frames
        self.current_target = None
        self.target_translation = self._marker("T_TARGET", "1 0 0")
        self.tool_translation = self._marker("T_TOOL", "0 1 0")
        self.bridge = None
        try:
            from ui_bridge import UIBridge
            rgb = self.io.robot.getDevice("Astra rgb")
            depth = self.io.robot.getDevice("Astra depth")
            if rgb is not None:
                rgb.enable(self.io.ts)
            if depth is not None:
                depth.enable(self.io.ts)
            if rgb is not None:
                self.bridge = UIBridge(self.io.robot, rgb, period_ms=period_ms, quality=quality)
        except Exception as exc:
            print(f"[warn] UI bridge initialization skipped: {exc}")

    def _marker(self, name, rgb):
        robot = self.io.robot
        node = robot.getFromDef(name)
        if node is None:
            robot.getRoot().getField("children").importMFNodeFromString(
                -1,
                f"DEF {name} Solid {{ children [ Shape {{ appearance PBRAppearance "
                f"{{ baseColor {rgb} roughness 1 metalness 0 }} "
                f"geometry Sphere {{ radius 0.015 }} }} ] }}"
            )
            node = robot.getFromDef(name)
        return node.getField("translation")

    def set_target(self, target):
        self.current_target = None if target is None else np.asarray(target, dtype=float)
        self.update()

    def update(self):
        if self.current_target is not None:
            self.target_translation.setSFVec3f(
                self.frames.base_to_world(self.current_target).tolist()
            )
        self.tool_translation.setSFVec3f(
            self.frames.base_to_world(self.frames.tool_position()).tolist()
        )

    def step(self):
        result = self.io.robot.step(self.io.ts)
        if result != -1:
            if self.bridge is not None:
                self.bridge.tick()
            self.update()
        return result

    def wait(self, count):
        for _ in range(count):
            if self.step() == -1:
                return False
        return True
