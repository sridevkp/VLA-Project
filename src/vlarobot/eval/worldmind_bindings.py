"""Pure binding identities for the production WorldMind composition.

This module intentionally contains no simulator, backend, Webots, ROS or MuJoCo import. The
concrete factories are registered in ``vlarobot.backends.worldmind_bindings`` so that all
geometry and safety construction stays behind the backend boundary. Keeping identities here
lets configuration validation and tests refer to stable names without importing a simulator.

No value in this module supplies a component formula, unit, weight, ranking direction or risk
threshold. The scientific component registry remains empty until owner-reviewed values exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

ANALYTIC_SURROGATE_BINDING = "analytic_surrogate_corrected_v1"
RUNTIME_SAFETY_BINDING = "runtime_safety_monitor_v1"
PLACEMENT_VERIFIER_BINDING = "placement_verifier_v1"
CORRECTED_POSE_SOLVER_BINDING = "corrected_bounded_pose_solver_v1"
CORRECTED_WAYPOINTS_BINDING = "corrected_scripted_waypoints_v1"
FINITE_CHUNK_CHECK = "finite_chunk_v1"
# Deliberately NOT in WORLD_MIND_BINDING_IDS: no owner-reviewed full-arm certificate exists.
# This identity exists only so an opt-in, unregistered probe can be named in diagnostics.
UNVERIFIED_ARM_SWEEP_BINDING = "unverified_urdf_arm_sweep_v1"

WORLD_MIND_BINDING_IDS: Mapping[str, str] = MappingProxyType(
    {
        "surrogate": ANALYTIC_SURROGATE_BINDING,
        "safety": RUNTIME_SAFETY_BINDING,
        "verification": PLACEMENT_VERIFIER_BINDING,
        "pose_solver": CORRECTED_POSE_SOLVER_BINDING,
        "waypoints": CORRECTED_WAYPOINTS_BINDING,
        "hard_check": FINITE_CHUNK_CHECK,
    }
)

# No approved calibration evaluator exists. The key is declared so a ready config fails closed
# rather than silently scoring with an implicit objective.
CALIBRATION_EVALUATOR_BINDING = "registered_calibration_evaluator_v1"
