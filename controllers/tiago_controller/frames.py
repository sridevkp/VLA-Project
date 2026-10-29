"""Coordinate transforms and current joint/tool state."""
import numpy as np

class Frames:
    def __init__(self, robot_io, kinematics):
        self.io, self.ik = robot_io, kinematics

    def base_to_world(self, point):
        R = np.array(self.io.me.getOrientation()).reshape(3, 3)
        return R @ np.asarray(point, dtype=float) + np.array(self.io.me.getPosition())

    def world_to_base(self, point):
        R = np.array(self.io.me.getOrientation()).reshape(3, 3)
        return R.T @ (np.asarray(point, dtype=float) - np.array(self.io.me.getPosition()))

    def current_q(self):
        return [
            self.io.sensors[link.name].getValue() if link.name in self.io.sensors else 0.0
            for link in self.ik.chain.links
        ]

    def tool_position(self):
        return self.ik.forward(self.current_q())[:3, 3]

    def can_base_position(self):
        return self.world_to_base(self.io.can_node.getPosition())
