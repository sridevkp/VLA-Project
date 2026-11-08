import json
import math
from dataclasses import replace

import numpy as np
import pytest

from vlarobot.backends.fake import FakeBackend
from vlarobot.backends.webots.config import (
    ALIGN_TOL,
    APPROACH_H,
    ARM_JOINTS,
    CLOSE_ALIGN_TOL,
    FINGER_SPEED,
    GRASP_Z,
    GRIPPER_OPEN,
    IK_TOL,
    IK_TOL_GRASP,
    IK_TOL_TRANSIT,
    MAX_JOINT_STEP_RAD,
    MAX_TILT_DEG,
    PLACE_CLEAR,
    PLACE_WORLD,
    TORSO_HEIGHT,
    TORSO_JOINT,
)
from vlarobot.backends.webots.kinematics import IKTrustRegionError, Kinematics
from vlarobot.core.types import ApplyResult, GripperCmd, GroundTruth, JointTarget, Pose3D, Vec3
from vlarobot.execution.safety_monitor import SafetyLimits, SafetyMonitor
from vlarobot.execution.scripted_can import (
    CanSuccessCriteria,
    ScriptedCanPolicy,
    _ik_tolerance_for_state,
    _world_to_base,
)


class FakeCanBackend(FakeBackend):
    def __init__(self, *args, **kwargs):
        self.kinematics = Kinematics()
        super().__init__(*args, **kwargs)

    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        self._robot_state.base_pose_world = Pose3D(
            position=Vec3(-0.53683, -3.70906, 0.0885438),
            orientation=(
                0.01121522834116144,
                -0.007522733160055142,
                -0.717650835270671,
                0.6962721492163582,
            ),
        )
        return self._sync_observation()

    def apply(self, action):
        result = super().apply(action)
        self._sync_observation()
        return result

    def step(self, n_steps=1):
        super().step(n_steps)
        return self._sync_observation()

    def _sync_observation(self):
        q = np.zeros(len(self.kinematics.chain.links), dtype=float)
        for name, value in self._robot_state.joint_pos.items():
            q[self.kinematics.idx[name]] = value
        q[self.kinematics.idx[TORSO_JOINT]] = TORSO_HEIGHT
        transform = self.kinematics.forward(q)
        self._robot_state.ee_pose = Pose3D(
            Vec3(*map(float, transform[:3, 3])),
            (0.0, 0.0, 0.0, 1.0),
        )
        return self._observation()

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            object_poses={
                "CAN": Pose3D(
                    Vec3(-0.799879, -4.38998, 0.810949),
                    (0.0, 0.0, 0.0, 1.0),
                )
            }
        )


class RateLimitedCanBackend(FakeCanBackend):
    def __init__(self, kinematics, *, stuck=False):
        limits = SafetyLimits(
            joint_position={name: kinematics.bounds(name) for name in ARM_JOINTS},
            joint_velocity={name: 1.76 for name in ARM_JOINTS},
            command_horizon=1.0,
        )
        super().__init__(SafetyMonitor(limits), step_duration=0.032)
        self.kinematics = kinematics
        self.stuck = stuck
        self.motor_targets = dict(self._robot_state.joint_pos)
        self.rejections = 0

    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        for name in ARM_JOINTS:
            lo, hi = self.kinematics.bounds(name)
            self._robot_state.joint_pos[name] = min(max(0.0, lo + 0.001), hi)
        self.motor_targets = dict(self._robot_state.joint_pos)
        return self._sync_observation()

    def apply(self, action):
        if isinstance(action, JointTarget):
            result = self.safety_monitor.validate(action, self._robot_state)
            if not result.accepted:
                self.rejections += 1
                return result
            self.motor_targets.update(action.q)
            return ApplyResult(True)
        return super().apply(action)

    def step(self, n_steps=1):
        for _ in range(n_steps):
            if not self.stuck:
                max_delta = 0.5 * self.step_duration
                for name, target in self.motor_targets.items():
                    current = self._robot_state.joint_pos[name]
                    delta = max(-max_delta, min(max_delta, target - current))
                    self._robot_state.joint_pos[name] = current + delta
            super().step(1)
        return self._sync_observation()


class BaseDriftCanBackend(RateLimitedCanBackend):
    def __init__(
        self,
        kinematics,
        *,
        drift_m: float = 0.006,
        yaw_deg: float = 0.1,
        rate_m_per_s: float | None = None,
        contact_nodes: list[dict[str, object]] | None = None,
    ):
        super().__init__(kinematics)
        self.drift_m = drift_m
        self.yaw_deg = yaw_deg
        self.rate_m_per_s = rate_m_per_s
        self.contact_nodes = contact_nodes or []

    def step(self, n_steps=1):
        observation = super().step(n_steps)
        drift_m = self.drift_m if self.rate_m_per_s is None else self.rate_m_per_s * observation.t
        diagnostics: dict[str, object] = {
            "base_drift_distance_m": drift_m,
            "base_drift_yaw_deg": self.yaw_deg,
            "contact_nodes": self.contact_nodes,
        }
        observation.robot.diagnostics = diagnostics
        self._robot_state.diagnostics = diagnostics
        return observation


class FingerprintCanBackend(RateLimitedCanBackend):
    """Injects a deterministic reset fingerprint, like the Webots backend does."""

    def reset(self, scenario, seed):
        observation = super().reset(scenario, seed)
        fingerprint = {
            "motors": {
                name: {
                    "target_position_rad": float(value),
                    "velocity_rad_s": 0.0,
                    "control_mode": "position",
                }
                for name, value in sorted(observation.robot.joint_pos.items())
            },
            "joint_sensors_rad": {
                name: float(value) for name, value in sorted(observation.robot.joint_pos.items())
            },
            "base": {
                "world_xyz_m": observation.robot.base_pose_world.position.as_array().tolist(),
                "world_orientation_matrix": [1.0] * 9,
            },
            "can": {"world_xyz_m": [1.0, 2.0, 3.0], "world_orientation_matrix": [1.0] * 9},
            "torso_sensor_after_rad": TORSO_HEIGHT,
            "torso_commanded_rad": TORSO_HEIGHT,
        }
        observation.robot.diagnostics["reset_fingerprint"] = fingerprint
        observation.robot.diagnostics["reset_torso_before_rad"] = 0.0
        observation.robot.diagnostics["reset_torso_after_rad"] = TORSO_HEIGHT
        self._robot_state.diagnostics = observation.robot.diagnostics
        return observation


class ClosedGripperCanBackend(RateLimitedCanBackend):
    """Start with closed fingers and ramp the opening at the configured finger speed."""

    def __init__(self, kinematics):
        super().__init__(kinematics)
        self.gripper_target = 0.0
        self.gripper_commands: list[float] = []

    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        self._robot_state.gripper_opening = 0.0
        self.gripper_target = 0.0
        self.gripper_commands.clear()
        return self._sync_observation()

    def apply(self, action):
        if isinstance(action, GripperCmd):
            result = self.safety_monitor.validate(action, self._robot_state)
            if not result.accepted:
                return result
            self.gripper_commands.append(action.opening)
            self.gripper_target = action.opening
            return ApplyResult(True)
        return super().apply(action)

    def step(self, n_steps=1):
        for _ in range(n_steps):
            current = self._robot_state.gripper_opening
            delta = self.gripper_target - current
            limit = FINGER_SPEED * self.step_duration
            self._robot_state.gripper_opening = (
                self.gripper_target
                if abs(delta) <= limit
                else current + math.copysign(limit, delta)
            )
            super().step(1)
        return self._sync_observation()


class StuckGripperCanBackend(ClosedGripperCanBackend):
    """Accept gripper commands without ever moving the fingers."""

    def apply(self, action):
        if isinstance(action, GripperCmd):
            return ApplyResult(True)
        return super().apply(action)


def test_policy_opens_gripper_at_init_before_descend(tmp_path) -> None:
    backend = ClosedGripperCanBackend(Kinematics())
    trace = tmp_path / "gripper-open.jsonl"
    result = ScriptedCanPolicy(backend.kinematics, trace_path=trace, stop_after="DESCEND").execute(
        backend, seed=0
    )

    assert backend.gripper_commands[0] == pytest.approx(GRIPPER_OPEN)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    init_exit = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "INIT"
        and row.get("phase") == "exit"
    )
    descend_enter = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "DESCEND"
        and row.get("phase") == "enter"
    )
    assert init_exit["gripper_opening_m"] == pytest.approx(GRIPPER_OPEN, abs=0.002)
    assert descend_enter["gripper_opening_m"] == pytest.approx(GRIPPER_OPEN, abs=0.002)
    assert result.final_state == "grasp"


def test_policy_fails_init_when_gripper_cannot_open(tmp_path) -> None:
    backend = StuckGripperCanBackend(Kinematics())
    trace = tmp_path / "gripper-stuck.jsonl"
    result = ScriptedCanPolicy(backend.kinematics, trace_path=trace).execute(backend, seed=0)

    assert result.success is False
    assert result.failure_state == "INIT"
    assert result.failure_reason == "gripper failed to open"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("event") == "gripper_timeout" for row in rows)


def _quaternion_multiply(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quaternion_from_rpy_deg(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class TiltingCanBackend(RateLimitedCanBackend):
    """Rock the base by a fixed roll/pitch mid-PREGRASP, like the real robot does."""

    def __init__(
        self,
        kinematics,
        *,
        tilt_at_s: float = 2.5,
        roll_deg: float = -1.3,
        pitch_deg: float = 1.5,
    ):
        super().__init__(kinematics)
        self.tilt_at_s = tilt_at_s
        self.roll_deg = roll_deg
        self.pitch_deg = pitch_deg
        self.tilted = False

    def reset(self, scenario, seed):
        observation = super().reset(scenario, seed)
        self.tilted = False
        return observation

    def step(self, n_steps=1):
        observation = super().step(n_steps)
        if not self.tilted and observation.t >= self.tilt_at_s:
            self.tilted = True
            base = self._robot_state.base_pose_world
            assert base is not None
            self._robot_state.base_pose_world = Pose3D(
                base.position,
                _quaternion_multiply(
                    base.orientation,
                    _quaternion_from_rpy_deg(self.roll_deg, self.pitch_deg, 0.0),
                ),
            )
            observation = self._sync_observation()
        return observation


def _rotation_from_quaternion(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def _tip_world(observation) -> np.ndarray:
    base = observation.robot.base_pose_world
    assert base is not None
    rotation = _rotation_from_quaternion(base.orientation)
    return rotation @ observation.robot.ee_pose.position.as_array() + base.position.as_array()


class NudgedCanBackend(TiltingCanBackend):
    """Rock the base, then nudge the CAN once when the fingertip reaches grasp height."""

    def __init__(self, kinematics, *, nudge_m: float = 0.012):
        super().__init__(kinematics)
        self.nudge_m = nudge_m
        self.can_world = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
        self.nudged = False

    def reset(self, scenario, seed):
        observation = super().reset(scenario, seed)
        self.can_world = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
        self.nudged = False
        return observation

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            object_poses={"CAN": Pose3D(Vec3(*map(float, self.can_world)), (0.0, 0.0, 0.0, 1.0))}
        )

    def step(self, n_steps=1):
        observation = super().step(n_steps)
        tip = _tip_world(observation)
        if not self.nudged and tip[2] < self.can_world[2] + 0.04:
            self.nudged = True
            self.can_world[0] += self.nudge_m
        return observation


class DraggedCanBackend(TiltingCanBackend):
    """Rock the base, then let the CAN stick to the fingertip at a fixed lateral offset."""

    def __init__(self, kinematics, *, drag_offset_m: float = 0.012):
        super().__init__(kinematics)
        self.drag_offset_m = drag_offset_m
        self.can_world = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
        self.contacted = False

    def reset(self, scenario, seed):
        observation = super().reset(scenario, seed)
        self.can_world = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
        self.contacted = False
        return observation

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            object_poses={"CAN": Pose3D(Vec3(*map(float, self.can_world)), (0.0, 0.0, 0.0, 1.0))}
        )

    def step(self, n_steps=1):
        observation = super().step(n_steps)
        tip = _tip_world(observation)
        if tip[2] < self.can_world[2] + 0.04:
            self.contacted = True
        if self.contacted:
            self.can_world[0] = tip[0] + self.drag_offset_m
            self.can_world[1] = tip[1]
        return observation


def _tip_to_can_xy_m(record: dict[str, object]) -> float:
    quaternion = record["base_world_orientation_xyzw"]
    assert isinstance(quaternion, (list, tuple)) and len(quaternion) == 4
    x, y, z, w = (float(value) for value in quaternion)
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )
    tip_world = rotation @ np.asarray(record["achieved_ee_base_xyz_m"], dtype=float) + np.asarray(
        record["base_world_xyz_m"], dtype=float
    )
    can_world = np.asarray(record["can_world_xyz_m"], dtype=float)
    return float(np.linalg.norm((tip_world - can_world)[:2]))


def test_pregrasp_realign_corrects_tilt_induced_offset(tmp_path) -> None:
    kin = Kinematics()
    backend = TiltingCanBackend(kin)
    trace = tmp_path / "pregrasp-realign.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace, stop_after="DESCEND").execute(backend, seed=0)

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    realign = next(row for row in rows if row.get("event") == "pregrasp_realign")
    assert result.max_base_tilt_deg == pytest.approx(1.5, abs=0.05)
    assert realign["xy_error_m"] > ALIGN_TOL
    realigned = next(row for row in rows if row.get("event") == "pregrasp_realigned")
    assert realigned["xy_error_after_m"] < ALIGN_TOL
    assert any(row.get("event") == "lift_target_refreshed" for row in rows)
    descend_enter = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "DESCEND"
        and row.get("phase") == "enter"
    )
    assert _tip_to_can_xy_m(descend_enter) < ALIGN_TOL


def test_pregrasp_realign_is_skipped_without_base_tilt(tmp_path) -> None:
    kin = Kinematics()
    backend = RateLimitedCanBackend(kin)
    trace = tmp_path / "pregrasp-no-tilt.jsonl"

    ScriptedCanPolicy(kin, trace_path=trace, stop_after="DESCEND").execute(backend, seed=0)

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    realign = next(row for row in rows if row.get("event") == "pregrasp_realign")
    assert realign["xy_error_m"] <= ALIGN_TOL
    assert not any(row.get("event") == "pregrasp_realigned" for row in rows)


def test_pregrasp_realign_flag_is_recorded_and_can_be_disabled(tmp_path) -> None:
    kin = Kinematics()
    backend = TiltingCanBackend(kin)
    enabled_trace = tmp_path / "realign-on.jsonl"
    enabled = ScriptedCanPolicy(
        kin,
        trace_path=enabled_trace,
        pregrasp_realign=True,
        stop_after="DESCEND",
    ).execute(backend, seed=0)

    disabled_trace = tmp_path / "realign-off.jsonl"
    disabled = ScriptedCanPolicy(
        kin,
        trace_path=disabled_trace,
        pregrasp_realign=False,
        stop_after="DESCEND",
    ).execute(backend, seed=0)

    assert enabled.pregrasp_realign is True
    assert disabled.pregrasp_realign is False
    enabled_rows = [
        json.loads(line) for line in enabled_trace.read_text(encoding="utf-8").splitlines()
    ]
    disabled_rows = [
        json.loads(line) for line in disabled_trace.read_text(encoding="utf-8").splitlines()
    ]
    enabled_init = next(
        row
        for row in enabled_rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "INIT"
        and row.get("phase") == "exit"
    )
    disabled_init = next(
        row
        for row in disabled_rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "INIT"
        and row.get("phase") == "exit"
    )
    assert enabled_init["pregrasp_realign"] is True
    assert disabled_init["pregrasp_realign"] is False
    assert any(row.get("event") == "pregrasp_realign" for row in enabled_rows)
    assert any(row.get("event") == "pregrasp_realigned" for row in enabled_rows)
    assert not any(row.get("event") == "pregrasp_realign" for row in disabled_rows)
    assert not any(row.get("event") == "pregrasp_realigned" for row in disabled_rows)
    assert not any(row.get("event") == "lift_target_refreshed" for row in disabled_rows)


def test_close_guard_redescends_once_when_can_is_nudged(tmp_path) -> None:
    kin = Kinematics()
    backend = NudgedCanBackend(kin)
    trace = tmp_path / "close-guard-success.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace, pregrasp_realign=True).execute(
        backend, seed=0
    )

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    check = next(row for row in rows if row.get("event") == "close_alignment_check")
    realigned = next(row for row in rows if row.get("event") == "close_realign")
    assert check["xy_error_m"] > CLOSE_ALIGN_TOL
    assert realigned["xy_error_after_m"] <= CLOSE_ALIGN_TOL
    assert not any(row.get("event") == "close_misaligned" for row in rows)
    assert result.failure_state != "CLOSE"
    assert result.final_state == "home"


def test_close_guard_fails_when_can_stays_dragged(tmp_path) -> None:
    kin = Kinematics()
    backend = DraggedCanBackend(kin)
    trace = tmp_path / "close-guard-failure.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace, pregrasp_realign=True).execute(
        backend, seed=0
    )

    assert result.success is False
    assert result.failure_state == "CLOSE"
    assert result.failure_reason == "misaligned at close"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    realigned = next(row for row in rows if row.get("event") == "close_realign")
    assert realigned["xy_error_after_m"] > CLOSE_ALIGN_TOL
    misaligned = next(row for row in rows if row.get("event") == "close_misaligned")
    assert misaligned["reason"] == "misaligned at close"
    assert misaligned["xy_error_m"] > CLOSE_ALIGN_TOL


def test_success_criteria_are_reported_separately(tmp_path) -> None:
    backend = FakeCanBackend()
    trace = tmp_path / "criteria.jsonl"
    result = ScriptedCanPolicy(backend.kinematics, trace_path=trace).execute(backend, seed=0)

    assert result.home_reached is True
    assert result.place_distance_ok is False
    assert result.can_height_ok is True
    assert result.can_tilt_ok is True
    assert result.can_height_error_m == pytest.approx(0.0, abs=1e-9)
    assert result.can_tilt_deg == pytest.approx(0.0, abs=1e-9)
    assert result.final_can_world_xyz_m == pytest.approx((-0.799879, -4.38998, 0.810949))
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    done = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "DONE"
        and row.get("phase") == "exit"
    )
    for key in (
        "distance_m",
        "can_world_xyz_m",
        "rest_height_m",
        "height_error_m",
        "tilt_deg",
        "home_reached",
        "place_distance_ok",
        "can_height_ok",
        "can_tilt_ok",
        "success",
    ):
        assert key in done
    assert done["home_reached"] is True
    assert done["place_distance_ok"] is False
    assert done["can_height_ok"] is True
    assert done["can_tilt_ok"] is True
    assert done["success"] is False


def test_can_success_criteria_require_every_criterion() -> None:
    base = CanSuccessCriteria(
        distance_m=0.0,
        can_world_xyz_m=(0.0, 0.0, 0.0),
        rest_height_m=0.0,
        height_error_m=0.0,
        tilt_deg=0.0,
        home_reached=True,
        place_distance_ok=True,
        can_height_ok=True,
        can_tilt_ok=True,
    )
    assert base.success is True
    assert replace(base, home_reached=False).success is False
    assert replace(base, place_distance_ok=False).success is False
    assert replace(base, can_height_ok=False).success is False
    assert replace(base, can_tilt_ok=False).success is False


def test_scripted_can_policy_allows_transit_ik_with_specific_tolerance(tmp_path) -> None:
    backend = FakeCanBackend()
    trace_path = tmp_path / "trace.jsonl"
    policy = ScriptedCanPolicy(kinematics=backend.kinematics, trace_path=trace_path)

    result = policy.execute(backend, seed=0)

    assert result.seed == 0
    assert result.rejections == 0
    assert result.final_state == "home"
    assert result.success is False
    assert result.failure_state is None
    expected_distance = np.linalg.norm(
        np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
        - np.asarray(PLACE_WORLD, dtype=float)
    )
    assert result.final_target_distance == pytest.approx(expected_distance, abs=1e-12)
    assert result.failure_reason is None
    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert not any(
        record.get("event") == "ik_rejection" and record.get("state") == "TRANSIT"
        for record in records
    )


def test_watched_pregrasp_stop_leaves_later_states_unexecuted(tmp_path) -> None:
    backend = RateLimitedCanBackend(Kinematics())
    trace = tmp_path / "pregrasp-only.jsonl"
    result = ScriptedCanPolicy(backend.kinematics, trace_path=trace, stop_after="PREGRASP").execute(
        backend, seed=0
    )

    assert result.final_state == "pregrasp"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    states = [row["state"] for row in rows if row.get("event") == "state_boundary"]
    assert "PREGRASP" in states
    assert "DESCEND" not in states


class ArmLinkDiagnosticsBackend(RateLimitedCanBackend):
    def step(self, n_steps=1):
        observation = super().step(n_steps)
        diagnostics: dict[str, object] = {
            "arm_link_world_xyz_m": {
                "elbow": [-1.1, -4.5, 0.95],
                "wrist": [-1.2, -4.4, 0.90],
                "tool": [-1.25, -4.35, 0.86],
            },
            "arm_7_joint_fk_ground_truth_mismatch": False,
        }
        observation.robot.diagnostics = diagnostics
        self._robot_state.diagnostics = diagnostics
        return observation


def test_trace_records_supervisor_arm_links_and_table_clearance(tmp_path) -> None:
    backend = ArmLinkDiagnosticsBackend(Kinematics())
    trace = tmp_path / "supervisor-links.jsonl"
    ScriptedCanPolicy(backend.kinematics, trace_path=trace, stop_after="PREGRASP").execute(
        backend, seed=0
    )

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    pregrasp = [
        row
        for row in rows
        if row.get("event") == "state_boundary" and row.get("state") == "PREGRASP"
    ]
    assert pregrasp
    record = pregrasp[-1]
    assert record["supervisor_arm_link_world_xyz_m"]["tool"] == [-1.25, -4.35, 0.86]
    # The tool link is the lowest inside the footprint: 0.86 m - 0.75 m table top.
    assert record["supervisor_min_table_clearance_m"] == pytest.approx(0.11)
    assert record["arm_7_joint_fk_ground_truth_mismatch"] is False
    assert record["table_top_height_m"] == 0.75
    assert record["table_clearance_required_m"] == 0.05


def test_full_ik_vector_uses_configured_torso_height() -> None:
    policy = ScriptedCanPolicy()

    vector = policy._full_joint_vector({})

    assert vector[policy.kinematics.idx[TORSO_JOINT]] == TORSO_HEIGHT


def test_kinematics_seed_reproduces_random_restart_sequence() -> None:
    kin = Kinematics()
    kin.set_seed(17)
    first = kin.rng.uniform(-1.0, 1.0, size=8)
    kin.set_seed(17)
    second = kin.rng.uniform(-1.0, 1.0, size=8)

    np.testing.assert_array_equal(first, second)


def test_fixture_seed_zero_matches_original_restart_stream_and_torso() -> None:
    kin = Kinematics()
    kin.set_seed(0)
    actual = np.asarray(
        [[kin.rng.uniform(*kin.bounds(name)) for name in ARM_JOINTS] for _ in range(20)]
    )
    rng = np.random.default_rng(0)
    expected = np.asarray(
        [[rng.uniform(*kin.bounds(name)) for name in ARM_JOINTS] for _ in range(20)]
    )
    np.testing.assert_array_equal(actual, expected)
    seed = ScriptedCanPolicy(kin)._full_joint_vector({name: 0.0 for name in ARM_JOINTS})
    assert seed[kin.idx[TORSO_JOINT]] == TORSO_HEIGHT


def test_unpaced_rate_limited_joint_waypoints_eventually_hit_velocity_rejection() -> None:
    kin = Kinematics()
    backend = RateLimitedCanBackend(kin)
    observation = backend.reset(None, 0)
    start = np.zeros(len(kin.chain.links), dtype=float)
    for name, value in observation.robot.joint_pos.items():
        start[kin.idx[name]] = value
    start[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    goal = start.copy()
    goal[kin.idx["arm_7_joint"]] = 2.0
    steps = kin.interpolate_joint_path(start, goal, MAX_JOINT_STEP_RAD, min_steps=2)

    rejected = False
    for waypoint in steps:
        target = {name: float(waypoint[kin.idx[name]]) for name in ARM_JOINTS}
        result = backend.apply(JointTarget(target))
        if not result.accepted:
            rejected = True
            break
        observation = backend.step(1)

    assert rejected
    assert backend.rejections == 1
    assert "violates velocity limits" in (result.reason or "")
    assert observation.t > 0.0


def test_paced_policy_settles_prefix_and_seeds_descend_from_pregrasp(monkeypatch, tmp_path) -> None:
    import vlarobot.execution.scripted_can as scripted_can

    monkeypatch.setattr(scripted_can, "STOP_AFTER", "LIFT")
    kin = Kinematics()
    kin.set_seed(0)
    original_solve = kin.solve
    solve_calls = []

    def recording_solve(
        target: np.ndarray,
        seed: np.ndarray,
        restarts: bool = False,
        n_random: int = 20,
        position_tolerance: float = IK_TOL_GRASP,
        max_joint_step_rad: float | None = None,
    ):
        result = original_solve(
            target,
            seed,
            restarts=restarts,
            n_random=n_random,
            position_tolerance=position_tolerance,
            max_joint_step_rad=max_joint_step_rad,
        )
        solve_calls.append((np.asarray(seed).copy(), np.asarray(result[0]).copy()))
        return result

    kin.solve = recording_solve
    backend = RateLimitedCanBackend(kin)
    trace_path = tmp_path / "paced-prefix.jsonl"
    policy = ScriptedCanPolicy(kin, trace_path=trace_path)

    result = policy.execute(backend, seed=0)

    assert result.final_state == "lift"
    assert backend.rejections == 0
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    entered = [
        record["state"]
        for record in records
        if record.get("event") == "state_boundary" and record.get("phase") == "enter"
    ]
    assert entered[:5] == ["INIT", "PREGRASP", "DESCEND", "CLOSE", "LIFT"]
    assert not any(record.get("event") == "safety_rejection" for record in records)
    for state, tolerance in (("PREGRASP", 0.01), ("DESCEND", 0.01), ("LIFT", 0.01)):
        boundary = next(
            record
            for record in records
            if record.get("event") == "state_boundary"
            and record.get("state") == state
            and record.get("phase") == "exit"
        )
        measured = boundary["measured_joint_positions_rad"]
        commanded = boundary["commanded_joint_target_rad"]
        assert max(abs(measured[name] - commanded[name]) for name in ARM_JOINTS) < tolerance
    assert len(solve_calls) > 2
    initial_seed, pregrasp_solution = solve_calls[0]
    descend_seed = solve_calls[1][0]
    for name in ARM_JOINTS:
        lo, hi = kin.bounds(name)
        assert initial_seed[kin.idx[name]] == pytest.approx(min(max(0.0, lo + 0.001), hi))
    assert initial_seed[kin.idx[TORSO_JOINT]] == TORSO_HEIGHT
    np.testing.assert_allclose(descend_seed, pregrasp_solution, atol=0.0, rtol=0.0)
    assert policy.state_sim_durations["PREGRASP"] == pytest.approx(
        policy.state_settle_durations["PREGRASP"]
    )


def test_joint_stall_watchdog_fails_the_active_state(monkeypatch) -> None:
    import vlarobot.execution.scripted_can as scripted_can

    monkeypatch.setattr(scripted_can, "JOINT_STALL_STEPS", 4)
    kin = Kinematics()
    backend = RateLimitedCanBackend(kin, stuck=True)
    result = ScriptedCanPolicy(kin).execute(backend, seed=0)

    assert result.final_state == "FAILED"
    assert result.failure_state == "PREGRASP"
    assert result.failure_reason == "joint stall"
    assert backend.rejections == 0


def test_base_invariant_fails_above_the_hard_limit_with_contact_context(tmp_path) -> None:
    kin = Kinematics()
    backend = BaseDriftCanBackend(
        kin,
        drift_m=0.016,
        yaw_deg=0.1,
        contact_nodes=[{"node_name": "TABLE"}],
    )
    trace_path = tmp_path / "base-drift.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace_path).execute(backend, seed=0)

    assert result.failure_state == "PREGRASP"
    assert result.failure_reason == "base moved"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event = next(row for row in records if row.get("event") == "base_moved")
    assert event["contact_nodes"] == [{"node_name": "TABLE"}]
    assert event["position_exceeded"] is True
    assert event["rate_exceeded"] is False


def test_base_drift_between_tiers_warns_once_per_state_without_failing(tmp_path) -> None:
    kin = Kinematics()
    backend = BaseDriftCanBackend(kin, drift_m=0.006, yaw_deg=0.1)
    trace_path = tmp_path / "base-drift-warning.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace_path, stop_after="PREGRASP").execute(
        backend, seed=0
    )

    assert result.final_state == "pregrasp"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    warnings = [
        row
        for row in records
        if row.get("event") == "base_drift_warning" and row.get("state") == "PREGRASP"
    ]
    assert len(warnings) == 1
    assert not any(row.get("event") == "base_moved" for row in records)
    assert warnings[0]["warn_position_m"] == pytest.approx(0.005)
    assert warnings[0]["fail_position_m"] == pytest.approx(0.015)


def test_base_drift_below_the_warning_tier_is_silent(tmp_path) -> None:
    kin = Kinematics()
    backend = BaseDriftCanBackend(kin, drift_m=0.003, yaw_deg=0.05)
    trace_path = tmp_path / "base-drift-silent.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace_path, stop_after="PREGRASP").execute(
        backend, seed=0
    )

    assert result.final_state == "pregrasp"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert not any(row.get("event") == "base_drift_warning" for row in records)
    assert not any(row.get("event") == "base_moved" for row in records)


def test_base_drift_rate_limit_fails_a_fast_shove(tmp_path) -> None:
    kin = Kinematics()
    backend = BaseDriftCanBackend(kin, rate_m_per_s=0.025, yaw_deg=0.0)
    trace_path = tmp_path / "base-drift-rate.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace_path).execute(backend, seed=0)

    assert result.failure_state == "PREGRASP"
    assert result.failure_reason == "base moved"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event = next(row for row in records if row.get("event") == "base_moved")
    assert event["rate_exceeded"] is True
    assert event["position_exceeded"] is False
    assert event["base_drift_distance_m"] < 0.015


def test_repeat_episodes_have_identical_reset_fingerprints_and_torso(tmp_path) -> None:
    backend = FingerprintCanBackend(Kinematics())
    policy = ScriptedCanPolicy(
        backend.kinematics,
        trace_path=tmp_path / "repeat.jsonl",
        stop_after="PREGRASP",
    )

    first = policy.execute(backend, seed=0)
    second = policy.execute(backend, seed=0)

    assert first.reset_fingerprint is not None
    assert first.reset_fingerprint == second.reset_fingerprint
    assert first.torso_start_rad == pytest.approx(TORSO_HEIGHT)
    assert first.torso_end_rad == pytest.approx(TORSO_HEIGHT)
    assert second.torso_start_rad == pytest.approx(TORSO_HEIGHT)
    assert second.torso_end_rad == pytest.approx(TORSO_HEIGHT)
    assert first.max_base_drift_m == pytest.approx(0.0)


def test_trace_records_reset_fingerprint_and_controller_reset_state(tmp_path) -> None:
    backend = FingerprintCanBackend(Kinematics())
    trace = tmp_path / "fingerprint.jsonl"
    ScriptedCanPolicy(backend.kinematics, trace_path=trace, stop_after="PREGRASP").execute(
        backend, seed=0
    )

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    init_enter = next(
        row
        for row in rows
        if row.get("event") == "state_boundary"
        and row.get("state") == "INIT"
        and row.get("phase") == "enter"
    )
    assert init_enter["reset_fingerprint"] is not None
    assert init_enter["reset_fingerprint"]["torso_commanded_rad"] == pytest.approx(TORSO_HEIGHT)
    assert init_enter["controller_reset_state"]["ik_seed"] == 0
    assert init_enter["controller_reset_state"]["torso_command_rad"] == pytest.approx(TORSO_HEIGHT)
    assert init_enter["base_drift_rate_m_per_s"] == pytest.approx(0.0)


def test_opt_in_timings_record_ik_transport_and_trace_components(tmp_path) -> None:
    backend = RateLimitedCanBackend(Kinematics())
    policy = ScriptedCanPolicy(
        backend.kinematics,
        trace_path=tmp_path / "timings.jsonl",
        stop_after="PREGRASP",
        timing_enabled=True,
    )

    policy.execute(backend, seed=0)
    summary = policy.timing_summary()

    assert summary["enabled"] is True
    totals = summary["totals"]
    assert totals["backend_reset_s"] >= 0.0
    assert totals["ik_solve_s"] > 0.0
    assert totals["backend_apply_s"] > 0.0
    assert totals["backend_step_s"] > 0.0
    assert totals["trace_write_s"] > 0.0
    assert "PREGRASP" in summary["by_state"]


def test_timings_are_disabled_by_default(tmp_path) -> None:
    backend = RateLimitedCanBackend(Kinematics())
    policy = ScriptedCanPolicy(
        backend.kinematics,
        trace_path=tmp_path / "no-timings.jsonl",
        stop_after="PREGRASP",
    )

    policy.execute(backend, seed=0)

    assert policy.timing_summary()["enabled"] is False
    assert policy.state_timings == {}


def test_failed_boundary_uses_last_observation_not_reset_observation(tmp_path, monkeypatch) -> None:
    import vlarobot.execution.scripted_can as scripted_can

    monkeypatch.setattr(scripted_can, "JOINT_STALL_STEPS", 4)
    kin = Kinematics()
    backend = RateLimitedCanBackend(kin, stuck=True)
    trace_path = tmp_path / "failed-last-observation.jsonl"

    result = ScriptedCanPolicy(kin, trace_path=trace_path).execute(backend, seed=0)

    assert result.failure_reason == "joint stall"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    exit_record = next(
        record
        for record in records
        if record.get("event") == "state_boundary"
        and record.get("state") == "PREGRASP"
        and record.get("phase") == "exit"
    )
    assert exit_record["sim_time_s"] > 0.0
    assert exit_record["sim_time_s"] == pytest.approx(backend._time)
    assert exit_record["measured_joint_positions_rad"] == backend._robot_state.joint_pos
    assert exit_record["final_target_distance_m"] == pytest.approx(result.final_target_distance)
    base_pose_world = backend._robot_state.base_pose_world
    assert base_pose_world is not None
    assert exit_record["base_world_xyz_m"] == pytest.approx(
        base_pose_world.position.as_array().tolist()
    )


def test_kinematics_prefers_nearest_feasible_joint_solution() -> None:
    kin = Kinematics()
    seed = np.zeros(len(kin.chain.links), dtype=float)
    near = seed.copy()
    near[kin.idx["arm_1_joint"]] = 0.1
    far = seed.copy()
    far[kin.idx["arm_1_joint"]] = 0.4

    selected, error, _ = kin._select_solution(
        [(far, 0.0001, 0.0), (near, 0.004, 0.0)],
        seed,
        position_tolerance=IK_TOL_GRASP,
    )

    np.testing.assert_array_equal(selected, near)
    assert error == pytest.approx(0.004)


def test_joint_interpolation_respects_configured_maximum_step() -> None:
    kin = Kinematics()
    start = np.zeros(len(kin.chain.links), dtype=float)
    target = start.copy()
    target[kin.idx["arm_7_joint"]] = 2.0

    waypoints = kin.interpolate_joint_path(start, target, MAX_JOINT_STEP_RAD)

    assert len(waypoints) == 14
    previous = start
    for waypoint in waypoints:
        assert max(
            abs(waypoint[kin.idx[name]] - previous[kin.idx[name]]) for name in ARM_JOINTS
        ) <= (MAX_JOINT_STEP_RAD + 1e-12)
        previous = waypoint
    np.testing.assert_array_equal(waypoints[-1], target)


def test_safe_via_is_selected_by_fk_and_both_joint_segments_clear_table() -> None:
    kin = Kinematics()
    policy = ScriptedCanPolicy(kin)
    policy._seed = 0
    start = np.zeros(len(kin.chain.links), dtype=float)
    goal = start.copy()
    start[kin.idx[TORSO_JOINT]] = goal[kin.idx[TORSO_JOINT]] = 0.35
    for name, value in {
        "arm_1_joint": 0.7123,
        "arm_2_joint": 0.7360,
        "arm_3_joint": -0.7142,
        "arm_4_joint": 1.0956,
        "arm_5_joint": 0.9456,
        "arm_6_joint": -1.2348,
        "arm_7_joint": 0.5786,
    }.items():
        goal[kin.idx[name]] = value
    base = Pose3D(
        Vec3(-0.53679259, -3.70923266, 0.08531599),
        (0.01175238, -0.00664384, -0.71763008, 0.69629363),
    )

    via = policy._find_safe_via(start, goal, base)

    assert via is not None
    assert not np.allclose(via, start)
    assert policy._joint_path_table_clear(start, via, base)
    assert policy._joint_path_table_clear(via, goal, base)


def test_ik_tolerance_is_relaxed_only_for_non_grasp_states() -> None:
    assert _ik_tolerance_for_state("PREGRASP") == IK_TOL
    assert _ik_tolerance_for_state("TRANSIT") == IK_TOL_TRANSIT == 0.015
    assert _ik_tolerance_for_state("RETREAT") == IK_TOL_TRANSIT
    assert _ik_tolerance_for_state("HOME") == IK_TOL_TRANSIT
    assert _ik_tolerance_for_state("DESCEND") == IK_TOL_GRASP == 0.005
    assert _ik_tolerance_for_state("PLACE_DESCEND") == IK_TOL_GRASP
    assert _ik_tolerance_for_state("LIFT") == IK_TOL_GRASP


def test_world_to_base_matches_frames_transform() -> None:
    half_angle = np.sqrt(0.5)
    base_pose_world = Pose3D(
        position=Vec3(1.0, 2.0, 0.5),
        orientation=(0.0, 0.0, half_angle, half_angle),
    )

    point_base = _world_to_base(np.asarray((1.0, 3.0, 0.5)), base_pose_world)

    np.testing.assert_allclose(point_base, (1.0, 0.0, 0.0), atol=1e-12)


def test_approved_world_target_and_linear_path_ik_diagnostics() -> None:
    kin = Kinematics()
    base_pose_world = Pose3D(
        position=Vec3(-0.53683, -3.70906, 0.0885438),
        orientation=(
            0.01121522834116144,
            -0.007522733160055142,
            -0.717650835270671,
            0.6962721492163582,
        ),
    )
    can_world = np.asarray((-0.799879, -4.38998, 0.810949), dtype=float)
    can_base = _world_to_base(can_world, base_pose_world)
    place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)

    q = np.zeros(len(kin.chain.links), dtype=float)
    q[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    pregrasp = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    grasp = can_base + np.asarray((0.0, 0.0, GRASP_Z))
    place_descend = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR))
    place_above = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR + APPROACH_H))

    for point in (pregrasp, grasp, place_descend):
        _, error, tilt = kin.solve(point, q, restarts=False)
        assert error <= IK_TOL_GRASP
        assert tilt <= 15.0

    _, transit_error, transit_tilt = kin.solve(place_above, q, restarts=False)
    assert transit_error == pytest.approx(0.010474806, abs=1e-6)
    assert transit_error > IK_TOL_GRASP
    assert transit_error <= IK_TOL
    assert transit_error <= IK_TOL_TRANSIT
    assert transit_tilt <= MAX_TILT_DEG

    q, _, _ = kin.solve(pregrasp, q, restarts=True)
    start = kin.forward(q)[:3, 3]
    count = max(2, int(np.linalg.norm(place_above - start) / 0.03))
    first_rejected_waypoint = None
    for index, alpha in enumerate(np.linspace(0.0, 1.0, count + 1)[1:], 1):
        point = (1.0 - alpha) * start + alpha * place_above
        q, error, tilt = kin.solve(point, q)
        if error > IK_TOL_GRASP or tilt > MAX_TILT_DEG:
            first_rejected_waypoint = (index, count, error, tilt)
            break
    assert first_rejected_waypoint is not None
    assert first_rejected_waypoint[:2] == (7, 7)
    assert first_rejected_waypoint[2] == pytest.approx(0.010474, abs=1e-5)


def test_trust_region_keeps_scripted_cartesian_waypoints_continuous() -> None:
    kin = Kinematics()
    kin.set_seed(0)
    base_pose_world = Pose3D(
        position=Vec3(-0.53683, -3.70906, 0.0885438),
        orientation=(
            0.01121522834116144,
            -0.007522733160055142,
            -0.717650835270671,
            0.6962721492163582,
        ),
    )
    can_base = _world_to_base(
        np.asarray((-0.799879, -4.38998, 0.810949), dtype=float),
        base_pose_world,
    )
    place_base = _world_to_base(np.asarray(PLACE_WORLD, dtype=float), base_pose_world)
    pregrasp = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    grasp = can_base + np.asarray((0.0, 0.0, GRASP_Z))
    place_tip = place_base + np.asarray((0.0, 0.0, GRASP_Z + PLACE_CLEAR))
    place_above = place_tip + np.asarray((0.0, 0.0, APPROACH_H))

    q = np.zeros(len(kin.chain.links), dtype=float)
    q[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    q, _, _ = kin.solve(pregrasp, q, restarts=True, position_tolerance=IK_TOL)

    for target, tolerance in (
        (grasp, IK_TOL_GRASP),
        (pregrasp, IK_TOL_GRASP),
        (place_above, IK_TOL_TRANSIT),
        (place_tip, IK_TOL_GRASP),
        (place_above, IK_TOL_TRANSIT),
    ):
        start = kin.forward(q)[:3, 3].copy()
        count = max(2, int(np.linalg.norm(target - start) / 0.03))
        for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
            point = (1.0 - alpha) * start + alpha * target
            previous = q.copy()
            q, error, tilt = kin.solve(
                point,
                q,
                restarts=True,
                position_tolerance=tolerance,
                max_joint_step_rad=MAX_JOINT_STEP_RAD,
            )
            step = max(abs(q[kin.idx[name]] - previous[kin.idx[name]]) for name in ARM_JOINTS)
            assert step <= MAX_JOINT_STEP_RAD + 1e-9
            assert error <= tolerance
            assert tilt <= MAX_TILT_DEG


def test_trust_region_rejects_reachable_target_that_requires_branch_jump() -> None:
    kin = Kinematics()
    kin.set_seed(0)
    base_pose_world = Pose3D(
        position=Vec3(-0.53683, -3.70906, 0.0885438),
        orientation=(
            0.01121522834116144,
            -0.007522733160055142,
            -0.717650835270671,
            0.6962721492163582,
        ),
    )
    can_base = _world_to_base(
        np.asarray((-0.799879, -4.38998, 0.810949), dtype=float),
        base_pose_world,
    )
    target = can_base + np.asarray((0.0, 0.0, GRASP_Z + APPROACH_H))
    seed = np.zeros(len(kin.chain.links), dtype=float)
    seed[kin.idx[TORSO_JOINT]] = TORSO_HEIGHT
    goal, _, _ = kin.solve(target, seed, restarts=True, position_tolerance=IK_TOL_GRASP)
    branch_seed = goal.copy()
    branch_seed[kin.idx["arm_1_joint"]] -= 0.5

    unbounded, error, _ = kin.solve(
        target,
        branch_seed,
        restarts=True,
        position_tolerance=IK_TOL_GRASP,
    )
    assert error <= IK_TOL_GRASP
    assert (
        max(abs(unbounded[kin.idx[name]] - branch_seed[kin.idx[name]]) for name in ARM_JOINTS)
        > MAX_JOINT_STEP_RAD
    )

    with pytest.raises(IKTrustRegionError, match="no solution within 0.150000 rad trust region"):
        kin.solve(
            target,
            branch_seed,
            restarts=True,
            position_tolerance=IK_TOL_GRASP,
            max_joint_step_rad=MAX_JOINT_STEP_RAD,
        )
