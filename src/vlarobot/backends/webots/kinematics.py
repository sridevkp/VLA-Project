"""IKPy chain, bounded inverse kinematics, and tool-frame scoring."""

import warnings
from copy import copy
from typing import Literal, TypeAlias

import numpy as np
from ikpy.chain import Chain
from ikpy.link import Link

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    IK_TOL_GRASP,
    KINEMATIC_MODEL,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
    SIM_KINEMATICS_PATH,
    TOOL_OFFSET,
    TORSO_JOINT,
    URDF_PATH,
)
from vlarobot.backends.webots.sim_kinematics import (
    SimKinematics,
    build_chain,
    load_snapshot,
    validate_torso,
)

warnings.filterwarnings("ignore", message=".*is of type 'fixed'.*")

IKResult: TypeAlias = tuple[np.ndarray, float, float]


class IKTrustRegionError(RuntimeError):
    """Raised when no tolerance-satisfying IK solution exists in the joint trust region."""


class Kinematics:
    def __init__(self, model_override: Literal["urdf", "sim"] | None = None):
        self.model = model_override if model_override is not None else KINEMATIC_MODEL
        self._sim: SimKinematics | None = None
        if self.model == "sim":
            if not SIM_KINEMATICS_PATH.exists():
                raise FileNotFoundError(
                    f"Sim kinematics snapshot not found: {SIM_KINEMATICS_PATH}; run "
                    "uv run python -m vlarobot.cli export-sim-kinematics first (ADR-0020)."
                )
            self._sim = load_snapshot(SIM_KINEMATICS_PATH)
            self.chain = build_chain(self._sim)
        else:
            if not URDF_PATH.exists():
                raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
            base_elements = [
                "base_link",
                "base_link_Torso_joint",
                "Torso",
                "torso_lift_joint",
                "torso_lift_link",
                "torso_lift_link_TIAGo front arm_joint",
                "TIAGo front arm",
            ]
            self.chain = Chain.from_urdf_file(
                str(URDF_PATH), base_elements=base_elements, last_link_vector=TOOL_OFFSET
            )
        # Same active-link semantics for both models: only the arm joints are
        # IK variables (the torso is baked/constant in this M1 chain).
        for i, link in enumerate(self.chain.links):
            self.chain.active_links_mask[i] = link.name in ARM_JOINTS
        self.idx = {link.name: i for i, link in enumerate(self.chain.links)}
        self.rng = np.random.default_rng(1)
        for name in ARM_JOINTS:
            if name not in self.idx:
                raise RuntimeError(f"Arm joint {name} missing from IKPy chain.")

    def set_seed(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer.")
        self.rng = np.random.default_rng(seed)
        print("IKPy chain:", [link.name for link in self.chain.links])

    def forward(self, q):
        if self._sim is not None:
            # The sim model bakes the torso at its export posture; FK with a
            # drifted torso value would silently return a wrong pose, so fail
            # fast (tolerance mirrors the reset settle tolerance).
            validate_torso(self._sim, float(q[self.idx[TORSO_JOINT]]))
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

    def interpolate_joint_path(
        self,
        start: np.ndarray,
        target: np.ndarray,
        max_step_rad: float = MAX_JOINT_STEP_RAD,
        min_steps: int = 1,
    ) -> list[np.ndarray]:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        if start.shape != target.shape or start.shape != (len(self.chain.links),):
            raise ValueError("Joint path endpoints must match the configured IK chain shape.")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
            raise ValueError("Joint path endpoints must contain only finite values.")
        if max_step_rad <= 0.0 or min_steps < 1:
            raise ValueError("max_step_rad and min_steps must be positive.")
        max_delta = max(
            (abs(float(target[self.idx[name]] - start[self.idx[name]])) for name in ARM_JOINTS),
            default=0.0,
        )
        step_count = max(min_steps, int(np.ceil(max_delta / max_step_rad)))
        return [
            (1.0 - alpha) * start + alpha * target
            for alpha in np.linspace(0.0, 1.0, step_count + 1)[1:]
        ]

    def score(self, target, q):
        T = self.forward(q)
        error = float(np.linalg.norm(T[:3, 3] - target))
        tool_dir = T[:3, :3] @ np.array([0.0, 0.0, -1.0])
        tilt = float(np.degrees(np.arccos(np.clip(-tool_dir[2], -1, 1))))
        return error, tilt

    def solve(
        self,
        target,
        seed,
        restarts=False,
        n_random=20,
        position_tolerance=IK_TOL_GRASP,
        max_joint_step_rad=None,
    ):
        if position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive.")
        if max_joint_step_rad is not None and max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive when provided.")
        seed = self.clip(seed)
        solve_chain = (
            self._trust_region_chain(seed, max_joint_step_rad)
            if max_joint_step_rad is not None
            else self.chain
        )
        seeds = [seed]
        if restarts:
            seeds.append(self.clip(solve_chain.inverse_kinematics(target, initial_position=seed)))
            for _ in range(n_random):
                candidate = seed.copy()
                for name in ARM_JOINTS:
                    lo, hi = (
                        self._trust_bounds(name, seed, max_joint_step_rad)
                        if max_joint_step_rad is not None
                        else self.bounds(name)
                    )
                    candidate[self.idx[name]] = self.rng.uniform(lo, hi)
                seeds.append(candidate)
        candidates: list[IKResult] = []
        for candidate in seeds:
            q = solve_chain.inverse_kinematics(
                target,
                target_orientation=[0, 0, 1],
                orientation_mode="Z",
                initial_position=candidate,
            )
            error, tilt = self.score(target, q)
            candidates.append((q, error, tilt))
            if error <= position_tolerance and tilt <= MAX_TILT_DEG:
                return q, error, tilt
        best = self._select_solution(candidates, seed, position_tolerance)
        if max_joint_step_rad is not None:
            raise IKTrustRegionError(
                f"no solution within {max_joint_step_rad:.6f} rad trust region met "
                f"position tolerance {position_tolerance:.6f} m; "
                f"best error={best[1]:.6f} m, tilt={best[2]:.3f} deg"
            )
        return best

    def _select_solution(
        self,
        candidates: list[tuple[np.ndarray, float, float]],
        seed: np.ndarray,
        position_tolerance: float,
    ) -> tuple[np.ndarray, float, float]:
        if not candidates:
            raise RuntimeError("Inverse kinematics did not produce a candidate solution.")

        def distance(q: np.ndarray) -> float:
            return float(
                np.linalg.norm([q[self.idx[name]] - seed[self.idx[name]] for name in ARM_JOINTS])
            )

        feasible = [
            candidate
            for candidate in candidates
            if candidate[1] <= position_tolerance and candidate[2] <= MAX_TILT_DEG
        ]
        if feasible:
            return min(feasible, key=lambda candidate: (distance(candidate[0]), candidate[1]))
        return min(
            candidates,
            key=lambda candidate: (
                candidate[1] + 0.002 * candidate[2],
                distance(candidate[0]),
            ),
        )

    def _trust_region_chain(self, seed: np.ndarray, step: float) -> Chain:
        links: list[Link] = []
        for link in self.chain.links:
            copied_link = copy(link)
            if link.name in ARM_JOINTS:
                copied_link.bounds = self._trust_bounds(link.name, seed, step)
            links.append(copied_link)
        return Chain(
            links=links,
            active_links_mask=self.chain.active_links_mask.copy(),
            name=self.chain.name,
            jax_precompile=False,
        )

    def _trust_bounds(self, name: str, seed: np.ndarray, step: float | None) -> tuple[float, float]:
        lower, upper = self.bounds(name)
        if step is not None:
            current = float(seed[self.idx[name]])
            lower = max(lower, current - step)
            upper = min(upper, current + step)
        if lower >= upper:
            raise IKTrustRegionError(
                f"joint {name!r} has no nonzero trust-region interval around "
                f"{seed[self.idx[name]]:.6f} rad"
            )
        return lower, upper
