"""Exact-order serial FK with fixed prefix/matrices prepared once for a stationary torso."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from vlarobot.core.types import Pose3D


class PreparedForward:
    """Retain original left-to-right dot order; never approximate the active joint transforms."""

    def __init__(
        self,
        links: Sequence[tuple[str, Callable[[float], np.ndarray]]],
        *,
        active_joints: frozenset[str],
        reference_joints: Mapping[str, float],
        to_pose: Callable[[np.ndarray], Pose3D],
        geometry_id: str,
    ) -> None:
        if not links or not active_joints or not geometry_id:
            raise ValueError("complete chain and geometry fingerprint required")
        names = [n for n, _ in links]
        if len(set(names)) != len(names) or not active_joints.issubset(names):
            raise ValueError("unique chain links and complete active-joint coverage required")
        if not set(reference_joints).issubset(names):
            raise ValueError("unknown reference joint")
        self.reference = dict(reference_joints)
        self.active = active_joints
        self.geometry_id = geometry_id
        self.to_pose = to_pose
        self.prefix = np.eye(4)
        self.operations: list[
            tuple[str, Callable[[float], np.ndarray] | None, np.ndarray | None]
        ] = []
        first = False
        for name, transform in links:
            if name in active_joints:
                first = True
                self.operations.append((name, transform, None))
            else:
                matrix = transform(float(self.reference.get(name, 0.0)))
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError("invalid fixed-link transform")
                if first:
                    self.operations.append((name, None, matrix.copy()))
                else:
                    self.prefix = np.dot(self.prefix, matrix)

    def __call__(self, joints: dict[str, float]) -> Pose3D:
        if set(joints) != set(self.reference):
            raise ValueError("prepared FK requires the same complete joint set")
        if any(joints[n] != v for n, v in self.reference.items() if n not in self.active):
            raise ValueError("fixed joint changed; rebuild prepared FK before prediction")
        matrix = self.prefix.copy()
        for name, transform, fixed in self.operations:
            if transform is not None:
                matrix = np.dot(matrix, transform(joints[name]))
            else:
                assert fixed is not None
                matrix = np.dot(matrix, fixed)
        return self.to_pose(matrix)
