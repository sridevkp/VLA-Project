
from pathlib import Path

import numpy as np
from ikpy.chain import Chain


# Use the URDF you downloaded from PAL Robotics.
URDF_PATH = Path(__file__).with_name("only_arm.urdf")

# Limits copied from your actual Webots TIAGo, in radians.
WEBOTS_LIMITS = [
    (0.070, 2.680),    # arm_1_joint
    (-1.500, 1.020),   # arm_2_joint
    (-3.460, 1.500),   # arm_3_joint
    (-0.320, 2.290),   # arm_4_joint
    (-2.070, 2.070),   # arm_5_joint
    (-1.390, 1.390),   # arm_6_joint
    (-2.070, 2.070),   # arm_7_joint
]

ARM_NAMES = [f"arm_{i}_joint" for i in range(1, 8)]


class TIAGoIK:
    def __init__(self):
        self.chain = Chain.from_urdf_file(
            str(URDF_PATH),
            active_links_mask=[
                False, True, True, True, True,
                True, True, True, False
            ]
        )

        if len(self.chain.links) != 9:
            raise RuntimeError(
                f"Expected 9 chain links, got {len(self.chain.links)}"
            )

        # Chain indices:
        # 0 = base, 1..7 = arm joints, 8 = tool link.
        # Only the seven actual arm joints are controllable.
        # self.chain.active_links_mask = [
        #     False, True, True, True, True,
        #     True, True, True, False
        # ]

        # Constrain the solver to the Webots joint limits.
        for i, limits in enumerate(WEBOTS_LIMITS, start=1):
            self.chain.links[i].bounds = limits

        # Seven arm joints, in radians.
        # The leading and trailing zeros represent non-actuated links.
        self.home = np.array([
            0.0,
            0.070, 0.260, -3.160, 1.270,
            1.320, 0.000, 1.410,
            0.0,
        ], dtype=float)

    def forward_position(self, joint_angles=None):
        """Return tool XYZ in the URDF chain's root frame, in metres."""
        angles = self.home.copy() if joint_angles is None else np.asarray(
            [0.0, *joint_angles, 0.0], dtype=float
        )

        transform = self.chain.forward_kinematics(angles)
        return transform[:3, 3].copy()

    def solve_position(self, xyz, initial_angles=None):
        """
        Solve for seven arm angles for a target XYZ position.
        xyz must be relative to the URDF chain root, in metres.
        """
        target = np.asarray(xyz, dtype=float)

        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("xyz must contain three finite numbers.")

        initial = self.home.copy()
        if initial_angles is not None:
            initial[1:8] = np.asarray(initial_angles, dtype=float)

        result = self.chain.inverse_kinematics(
            target_position=target,
            initial_position=initial,
        )

        angles = np.asarray(result[1:8], dtype=float)

        if not np.all(np.isfinite(angles)):
            raise RuntimeError("IK returned non-finite joint angles.")

        for name, angle, (low, high) in zip(
            ARM_NAMES, angles, WEBOTS_LIMITS
        ):
            if angle < low - 1e-5 or angle > high + 1e-5:
                raise RuntimeError(
                    f"IK result violates Webots limits: "
                    f"{name}={angle:.4f}, valid [{low}, {high}]"
                )

        achieved = self.forward_position(angles)
        error = float(np.linalg.norm(achieved - target))

        if error > 0.005:
            raise RuntimeError(
                f"Target not reached accurately. Position error: "
                f"{error * 1000:.1f} mm"
            )

        return angles, achieved, error


if __name__ == "__main__":
    ik = TIAGoIK()

    # First validate forward kinematics at the known starting pose.
    current_xyz = ik.forward_position()
    print("Tool XYZ in URDF root frame:", current_xyz)

    # Solve for the same point; this should be a reachable target.
    angles, achieved, error = ik.solve_position(current_xyz)

    print("\nIK solution:")
    for name, angle in zip(ARM_NAMES, angles):
        print(f"{name:15s}: {angle: .5f} rad")

    print("\nAchieved XYZ:", achieved)
    print(f"Position error: {error * 1000:.3f} mm")
