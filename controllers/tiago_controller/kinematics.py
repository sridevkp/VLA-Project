"""IKPy chain, bounded inverse kinematics, and tool-frame scoring."""
import warnings
import numpy as np
from ikpy.chain import Chain
from config import ARM_JOINTS, TOOL_OFFSET, URDF_PATH

warnings.filterwarnings("ignore", message=".*is of type 'fixed'.*")

class Kinematics:
    def __init__(self):
        if not URDF_PATH.exists():
            raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
        base_elements = [
            "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
            "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm",
        ]
        self.chain = Chain.from_urdf_file(
            str(URDF_PATH), base_elements=base_elements, last_link_vector=TOOL_OFFSET
        )
        for i, link in enumerate(self.chain.links):
            self.chain.active_links_mask[i] = link.name in ARM_JOINTS
        self.idx = {link.name: i for i, link in enumerate(self.chain.links)}
        self.rng = np.random.default_rng(1)
        for name in ARM_JOINTS:
            if name not in self.idx:
                raise RuntimeError(f"Arm joint {name} missing from IKPy chain.")
        print("IKPy chain:", [link.name for link in self.chain.links])

    def forward(self, q):
        return self.chain.forward_kinematics(q)

    def bounds(self, name):
        lo, hi = self.chain.links[self.idx[name]].bounds
        lo = -np.pi if lo is None or not np.isfinite(lo) else lo
        hi = np.pi if hi is None or not np.isfinite(hi) else hi
        return lo, hi

    def clip(self, q):
        q = np.array(q, dtype=float)
        for name in ARM_JOINTS:
            lo, hi = self.bounds(name)
            q[self.idx[name]] = np.clip(q[self.idx[name]], lo, hi)
        return q

    def score(self, target, q):
        T = self.forward(q)
        error = float(np.linalg.norm(T[:3, 3] - target))
        tool_dir = T[:3, :3] @ np.array([0., 0., -1.])
        tilt = float(np.degrees(np.arccos(np.clip(-tool_dir[2], -1, 1))))
        return error, tilt

    def solve(self, target, seed, restarts=False, n_random=20):
        seed = self.clip(seed)
        seeds = [seed]
        if restarts:
            seeds.append(self.clip(self.chain.inverse_kinematics(target, initial_position=seed)))
            for _ in range(n_random):
                candidate = seed.copy()
                for name in ARM_JOINTS:
                    lo, hi = self.bounds(name)
                    candidate[self.idx[name]] = self.rng.uniform(lo, hi)
                seeds.append(candidate)
        best = None
        for candidate in seeds:
            q = self.chain.inverse_kinematics(
                target, target_orientation=[0, 0, 1],
                orientation_mode="Z", initial_position=candidate
            )
            error, tilt = self.score(target, q)
            cost = error + 0.002 * tilt
            if best is None or cost < best[0]:
                best = (cost, q, error, tilt)
            if error < 0.005 and tilt < 5:
                break
        return best[1], best[2], best[3]
