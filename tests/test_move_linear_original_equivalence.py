"""Strengthened OFF-path equivalence test vs the original 1b2e4da ``move_linear``.

The earlier "OFF byte-matches the original arithmetic" claim was asserted only by key
absence (``closed_loop_waypoints`` does not exist at ``1b2e4da``). This test replaces the
weak claim with two graded reference implementations of
``1b2e4da:controllers/tiago_controller/motion.py`` ``Motion.move_linear``, both driven
from the identical inputs the OFF policy sees (``closed_loop_waypoints=False``,
``pregrasp_realign=False``):

- Mode 'bounded' (full 1e-9 parity): the reference keeps the original's waypoint
  arithmetic (3 cm spacing, alpha schedule, ``IK_TOL_GRASP``/``MAX_TILT_DEG`` rejection,
  0.03/0.01 rad settling, 600-step budget) and calls IK with the shipped bounded argument
  set (``restarts=True``, ``max_joint_step_rad=MAX_JOINT_STEP_RAD``), i.e. the continuation
  parameters of the current controller lineage. Cartesian waypoint sequences,
  per-waypoint IK seeds, AND the accepted joint solutions must all match within 1e-9.
- Mode 'literal-1b2e4da' (documented deviations): the reference calls IK with the
  historical unbounded defaults (``restarts=False``, unbounded chain). The Cartesian
  waypoint sequences still match within 1e-9, but the unbounded ikpy solve can wander
  up to ~0.35 rad away from the warm seed while remaining at the same tip (~0.001 mm);
  from waypoint 2 onward both continuations legitimately diverge. Each solution is
  asserted against the common acceptance criterion (tip within ``IK_TOL_GRASP``) rather
  than bitwise equality, and the policy solution is additionally asserted to stay within
  its own joint trust region.
"""

import json
from dataclasses import dataclass

import numpy as np
import pytest
from test_scripted_can_policy import RateLimitedCanBackend

from vlarobot.backends.webots.config import (
    ARM_JOINTS,
    IK_TOL_GRASP,
    LINEAR_FINAL_SETTLE_TOLERANCE_RAD,
    LINEAR_MOVE_TIMEOUT_STEPS,
    LINEAR_SETTLE_TOLERANCE_RAD,
    LINEAR_WAYPOINT_SPACING_M,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
    TORSO_HEIGHT,
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import Kinematics
from vlarobot.execution.scripted_can import ScriptedCanPolicy

TOL = 1e-9


def _tip_deviation(reference_q: np.ndarray, policy_q: np.ndarray, kin: Kinematics) -> float:
    """Base-frame tip distance between the reference and policy IK solutions."""
    ref_tip = kin.forward(reference_q)[:3, 3]
    pol_tip = kin.forward(policy_q)[:3, 3]
    return float(np.linalg.norm(ref_tip - pol_tip))


class _RecordingKinematics(Kinematics):
    """Kinematics wrapper that records every ``solve`` call (args and accepted result)."""

    def __init__(self) -> None:
        super().__init__()
        self.set_seed(0)
        self.solve_records: list[dict[str, object]] = []
        original = self.solve

        def recording_solve(
            target,
            seed,
            restarts=False,
            n_random=20,
            position_tolerance=IK_TOL_GRASP,
            max_joint_step_rad=None,
        ):
            result = original(
                target,
                seed,
                restarts=restarts,
                n_random=n_random,
                position_tolerance=position_tolerance,
                max_joint_step_rad=max_joint_step_rad,
            )
            self.solve_records.append(
                {
                    "target": np.asarray(target, dtype=float).copy(),
                    "seed": np.asarray(seed, dtype=float).copy(),
                    "result": np.asarray(result[0], dtype=float).copy(),
                    "restarts": restarts,
                    "max_joint_step_rad": max_joint_step_rad,
                }
            )
            return result

        self.solve = recording_solve  # type: ignore[method-assign]


def _full_vector(kin: Kinematics, q: dict[str, float]) -> np.ndarray:
    """Chain-length joint vector from a name->value mapping, torso included."""
    vector = np.zeros(len(kin.chain.links), dtype=float)
    vector[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    for name, value in q.items():
        if name in kin.idx:
            vector[kin.idx[name]] = float(value)
    return vector


@dataclass
class SegmentRecord:
    """Inputs and solve-call range of one policy ``_move_linear`` invocation."""

    state: str
    start: np.ndarray
    target: np.ndarray
    seed0: np.ndarray
    begin: int
    end: int


def _wrap_move_linear(policy: ScriptedCanPolicy) -> dict[str, SegmentRecord]:
    """Record ``(start, target, seed0, solve range)`` per ``_move_linear`` call."""
    linear: dict[str, SegmentRecord] = {}
    kin = policy.kinematics
    if isinstance(kin, _RecordingKinematics):
        records = kin.solve_records
    else:  # pragma: no cover - defensive
        raise TypeError("expected a _RecordingKinematics")
    original_move_linear = policy._move_linear

    def wrapped(backend, observation, target, state, target_provider=None):
        begin = len(records)
        seed0 = policy._commanded_vector()  # entry-time commanded vector (the seam seed)
        result = original_move_linear(
            backend, observation, target, state, target_provider=target_provider
        )
        linear[state] = SegmentRecord(
            state=state,
            start=observation.robot.ee_pose.position.as_array().copy(),
            target=np.asarray(target, dtype=float).copy(),
            seed0=seed0,
            begin=begin,
            end=len(records),
        )
        return result

    policy._move_linear = wrapped  # type: ignore[method-assign]
    return linear


def _original_settle(
    backend: RateLimitedCanBackend,
    joint_target: dict[str, float],
    settle_tolerance: float,
) -> bool:
    """Settle loop of the original ``Motion.move_to_q`` on the fake backend.

    The original controller drives the motors directly (``setPosition``) and has no
    velocity safety gate, so the reference bypasses ``RateLimitedCanBackend.apply``'s
    safety monitor and writes ``motor_targets`` the way ``move_to_q`` would have. Mirrors
    ``_command_and_wait`` for the no-stall case: step until every joint is within
    ``settle_tolerance`` or the 600-step budget expires.
    """
    backend.motor_targets.update(joint_target)
    for _ in range(LINEAR_MOVE_TIMEOUT_STEPS):
        observation = backend.step(1)
        measured = observation.robot.joint_pos
        errors = (abs(joint_target[name] - measured[name]) for name in ARM_JOINTS)
        if max(errors, default=0.0) < settle_tolerance:
            return True
    return False


def drive_original_move_linear(
    kin: Kinematics,
    backend: RateLimitedCanBackend,
    start: np.ndarray,
    target: np.ndarray,
    seed0: np.ndarray,
    bounded: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reference implementation of the 1b2e4da ``Motion.move_linear`` arithmetic.

    Faithful to the original: waypoint count from the 3 cm spacing, the linear alpha
    schedule, one IK call per waypoint from the previous accepted solution, the
    ``IK_TOL_GRASP``/``MAX_TILT_DEG`` rejection test, and per-waypoint settling mirrored
    from ``move_to_q`` (0.03 rad mid-path, 0.01 rad final, 600-step budget).

    ``bounded`` selects the IK continuation parameters of one of the two graded modes:

    - ``bounded=True`` -- the shipped bounded argument set the current controller
      lineage runs its Cartesian segments with (``restarts=True``,
      ``max_joint_step_rad=MAX_JOINT_STEP_RAD``).
    - ``bounded=False`` -- the literal 1b2e4da call (``self.ik.solve(point, q)`` =
      ``restarts=False``, unbounded chain, factory ``n_random=20``).

    Returns one ``(point, seed_used, accepted_q)`` triple per waypoint.
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    count = max(2, int(float(np.linalg.norm(target - start)) / LINEAR_WAYPOINT_SPACING_M))
    q = np.asarray(seed0, dtype=float).copy()
    waypoint_records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, alpha in enumerate(np.linspace(0.0, 1.0, count + 1)[1:], start=1):
        point = (1.0 - alpha) * start + alpha * target
        seed = q.copy()
        q, error, tilt = kin.solve(
            point,
            q,
            restarts=bounded,
            n_random=20,
            position_tolerance=IK_TOL_GRASP,
            max_joint_step_rad=MAX_JOINT_STEP_RAD if bounded else None,
        )
        if error > IK_TOL_GRASP or tilt > MAX_TILT_DEG:
            pytest.fail(
                f"reference move_linear rejected at waypoint {index}/{count}: "
                f"err={error:.6f} tilt={tilt:.3f}"
            )
        final = index == count
        settled = _original_settle(
            backend,
            {name: float(q[kin.idx[name]]) for name in ARM_JOINTS},
            LINEAR_FINAL_SETTLE_TOLERANCE_RAD if final else LINEAR_SETTLE_TOLERANCE_RAD,
        )
        if not settled:
            pytest.fail(f"reference move_linear did not settle at waypoint {index}/{count}")
        waypoint_records.append((point, seed, q))
    return waypoint_records


def _measured_at_enter(rows: list[dict], state: str) -> dict[str, float]:
    boundary = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == state
        and row.get("phase") == "enter"
    )
    return boundary["measured_joint_positions_rad"]


def _replay_measured(
    backend: RateLimitedCanBackend,
    kin: Kinematics,
    measured: dict[str, float],
) -> None:
    """Put a fresh reference backend at a recorded measured joint state."""
    for name in ARM_JOINTS:
        value = float(measured[name])
        backend._robot_state.joint_pos[name] = value
        backend.motor_targets[name] = value
    backend._sync_observation()


def _run_off_policy(tmp_path):
    """Full policy run with all OFF flags; returns (kin, backend, policy, trace rows, linear)."""
    kin = _RecordingKinematics()
    backend = RateLimitedCanBackend(kin)
    policy = ScriptedCanPolicy(
        kin,
        trace_path=tmp_path / "off-run.jsonl",
        closed_loop_waypoints=False,
        pregrasp_realign=False,
        stop_after="LIFT",
    )
    linear = _wrap_move_linear(policy)
    result = policy.execute(backend, seed=0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "off-run.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert result.final_state == "lift"
    assert backend.rejections == 0
    assert set(linear) == {"DESCEND", "LIFT"}
    return kin, backend, policy, rows, linear


def _record_arrays(record: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``(target, seed, accepted result)`` arrays of one recorded solve call."""
    return (
        np.asarray(record["target"], dtype=float),
        np.asarray(record["seed"], dtype=float),
        np.asarray(record["result"], dtype=float),
    )


def _collect_policy_segments(
    kin: _RecordingKinematics, linear: dict[str, SegmentRecord]
) -> dict[str, list[dict[str, object]]]:
    """Assert the OFF policy's solver argument set and return its solve records per state."""
    segments: dict[str, list[dict[str, object]]] = {}
    for state in ("DESCEND", "LIFT"):
        solves = kin.solve_records[linear[state].begin : linear[state].end]
        assert solves, f"no IK calls recorded for {state}"
        for record in solves:
            # The OFF path documents its solver arg set: bounded trust region + restarts.
            # This is the deliberate divergence from the literal 1b2e4da call site; the
            # tests below quantify its effect instead of hiding it.
            assert record["restarts"] is True
            assert record["max_joint_step_rad"] == pytest.approx(MAX_JOINT_STEP_RAD)
        segments[state] = solves
    return segments


def _segment_inputs(
    linear: dict[str, SegmentRecord], state: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The policy's ``(start, target, seed0)`` inputs for one recorded Cartesian state."""
    segment = linear[state]
    return segment.start, segment.target, segment.seed0


def test_off_path_bounded_reference_matches_waypoints_seeds_solutions_1e9(tmp_path) -> None:
    """Bounded mode: DESCEND/LIFT points, seeds AND solutions all match within 1e-9."""
    kin, _backend, _policy, rows, linear = _run_off_policy(tmp_path)
    ref_kin = _RecordingKinematics()
    ref_backend = RateLimitedCanBackend(ref_kin)

    for state in ("DESCEND", "LIFT"):
        solves = _collect_policy_segments(kin, linear)[state]
        _replay_measured(ref_backend, ref_kin, _measured_at_enter(rows, state))
        start, target, seed0 = _segment_inputs(linear, state)
        reference = drive_original_move_linear(
            ref_kin, ref_backend, start=start, target=target, seed0=seed0
        )
        assert len(reference) == len(solves)
        for wp_index, (ref_point, ref_seed, ref_q) in enumerate(reference, start=1):
            pol_target, pol_seed, pol_result = _record_arrays(solves[wp_index - 1])
            assert np.allclose(ref_point, pol_target, atol=TOL, rtol=0.0), (
                f"{state} waypoint {wp_index}: Cartesian waypoint differs"
            )
            assert np.allclose(ref_seed, pol_seed, atol=TOL, rtol=0.0), (
                f"{state} waypoint {wp_index}: IK seed differs"
            )
            assert np.allclose(ref_q, pol_result, atol=TOL, rtol=0.0), (
                f"{state} waypoint {wp_index}: accepted IK solution differs"
            )


def test_off_path_literal_original_matches_waypoints_and_tip_criterion(tmp_path) -> None:
    """Literal 1b2e4da mode: waypoints 1e-9, solutions within the shared IK tolerance."""
    kin, _backend, _policy, rows, linear = _run_off_policy(tmp_path)
    ref_kin = _RecordingKinematics()
    ref_backend = RateLimitedCanBackend(ref_kin)

    for state in ("DESCEND", "LIFT"):
        solves = _collect_policy_segments(kin, linear)[state]
        _replay_measured(ref_backend, ref_kin, _measured_at_enter(rows, state))
        start, target, seed0 = _segment_inputs(linear, state)
        reference = drive_original_move_linear(
            ref_kin, ref_backend, start=start, target=target, seed0=seed0, bounded=False
        )
        assert len(reference) == len(solves)
        for wp_index, (ref_point, ref_seed, ref_q) in enumerate(reference, start=1):
            pol_target, pol_seed, pol_result = _record_arrays(solves[wp_index - 1])
            assert np.allclose(ref_point, pol_target, atol=TOL, rtol=0.0), (
                f"{state} waypoint {wp_index}: Cartesian waypoint differs"
            )
            if wp_index == 1:
                # The first seed is the shared seam seed, so it must agree exactly.
                assert np.allclose(ref_seed, pol_seed, atol=TOL, rtol=0.0), (
                    f"{state} waypoint 1: IK seed differs"
                )
            deviation = _tip_deviation(ref_q, pol_result, kin)
            assert deviation <= IK_TOL_GRASP, (
                f"{state} waypoint {wp_index}: reference and policy solutions disagree at "
                f"tip distance {deviation * 1e3:.4f} mm (beyond the shared IK acceptance "
                f"criterion {IK_TOL_GRASP * 1e3:.1f} mm)"
            )
            # The bounded solver must never command further than its own trust region.
            policy_deviation = max(
                abs(float(pol_result[kin.idx[name]]) - float(pol_seed[kin.idx[name]]))
                for name in ARM_JOINTS
            )
            assert policy_deviation <= MAX_JOINT_STEP_RAD + 1e-12, (
                f"{state} waypoint {wp_index}: policy commanded {policy_deviation:.3f} rad "
                f"away from its seed, beyond the trust region {MAX_JOINT_STEP_RAD}"
            )


def test_off_path_seed0_is_within_final_settle_tolerance_of_measurement(tmp_path) -> None:
    """Seam deviation: policy seeds from commanded, original from measured (< 0.01 rad)."""
    kin, _backend, _policy, rows, linear = _run_off_policy(tmp_path)
    for state in ("DESCEND", "LIFT"):
        seed0 = linear[state].seed0
        measured = _measured_at_enter(rows, state)
        residuals = [
            abs(float(seed0[kin.idx[name]]) - float(measured[name])) for name in ARM_JOINTS
        ]
        assert max(residuals) < LINEAR_FINAL_SETTLE_TOLERANCE_RAD + 1e-12, (
            f"{state} seam: max commanded-vs-measured residual {max(residuals):.6f} rad "
            f"exceeded the final settle tolerance {LINEAR_FINAL_SETTLE_TOLERANCE_RAD}"
        )


def test_reference_move_linear_solver_arguments_per_mode() -> None:
    """Each mode must call ``solve`` with exactly its documented argument set."""
    kin = Kinematics()
    backend = RateLimitedCanBackend(kin)

    # Stub solve: identity result (accepts any point and needs no settling) so the drive
    # is solvability-independent and this test purely checks argument forwarding.
    received: list[dict[str, object]] = []

    def stub_solve(
        target,
        seed,
        restarts=False,
        n_random=20,
        position_tolerance=IK_TOL_GRASP,
        max_joint_step_rad=None,
    ):
        received.append({"restarts": restarts, "max_joint_step_rad": max_joint_step_rad})
        return np.asarray(seed, dtype=float).copy(), 0.0, 0.0

    kin.solve = stub_solve  # type: ignore[method-assign]

    start = np.array([0.65, -0.23, 0.90])
    target = np.array([0.65, -0.23, 0.84])
    seed0 = np.zeros(len(kin.chain.links), dtype=float)
    seed0[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT

    for bounded, expected in (
        (False, {"restarts": False, "max_joint_step_rad": None}),
        (True, {"restarts": True, "max_joint_step_rad": MAX_JOINT_STEP_RAD}),
    ):
        received.clear()
        drive_original_move_linear(kin, backend, start, target, seed0, bounded=bounded)
        assert len(received) >= 2
        for call in received:
            assert call == expected, f"bounded={bounded}: bad solve args {call}"
