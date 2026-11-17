"""Tests for the repeat-check reset-fingerprint comparison."""

import pytest

from vlarobot.cli import _fingerprint_comparison


def _fingerprint() -> dict[str, object]:
    return {
        "motors": {
            "arm_1_joint": {
                "target_position_rad": 0.1,
                "velocity_rad_s": 0.5,
                "control_mode": "position",
            }
        },
        "joint_sensors_rad": {"arm_1_joint": 0.1},
        "base": {
            "world_xyz_m": [0.0, 0.0, 0.0],
            "world_orientation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "velocity_world_xyz_m_s": [0.0, 0.0, 0.0],
            "angular_velocity_world_xyz_rad_s": [0.0, 0.0, 0.0],
        },
        "can": {"world_xyz_m": [1.0, 2.0, 3.0]},
    }


def test_identical_fingerprints_pass():
    comparison = _fingerprint_comparison(_fingerprint(), _fingerprint(), 1e-3)

    assert comparison.passed
    assert comparison.commanded == []
    assert comparison.measured == []
    assert comparison.transient == []


def test_sub_tolerance_measurement_noise_passes_but_is_reported():
    first = _fingerprint()
    second = _fingerprint()
    sensors = second["joint_sensors_rad"]
    assert isinstance(sensors, dict)
    sensors["arm_1_joint"] = 0.1 + 1.5e-4
    base = second["base"]
    assert isinstance(base, dict)
    base["world_xyz_m"] = [1.3e-4, 0.0, 0.0]

    comparison = _fingerprint_comparison(first, second, 1e-3)

    assert comparison.passed
    assert comparison.measured == []
    assert comparison.max_measured_delta == pytest.approx(1.5e-4)


def test_measured_change_above_tolerance_fails():
    first = _fingerprint()
    second = _fingerprint()
    sensors = second["joint_sensors_rad"]
    assert isinstance(sensors, dict)
    sensors["arm_1_joint"] = 0.11

    comparison = _fingerprint_comparison(first, second, 1e-3)

    assert not comparison.passed
    assert comparison.measured
    assert "joint_sensors_rad.arm_1_joint" in comparison.measured[0]


def test_commanded_motor_change_fails_even_with_loose_tolerance():
    first = _fingerprint()
    second = _fingerprint()
    motors = second["motors"]
    assert isinstance(motors, dict)
    arm = motors["arm_1_joint"]
    assert isinstance(arm, dict)
    arm["target_position_rad"] = 0.2

    comparison = _fingerprint_comparison(first, second, 10.0)

    assert not comparison.passed
    assert comparison.commanded
    assert "motors.arm_1_joint.target_position_rad" in comparison.commanded[0]


def test_base_velocity_is_transient_and_does_not_fail():
    first = _fingerprint()
    second = _fingerprint()
    base = second["base"]
    assert isinstance(base, dict)
    base["velocity_world_xyz_m_s"] = [3.4e-3, 0.0, 0.0]

    comparison = _fingerprint_comparison(first, second, 1e-9)

    assert comparison.passed
    assert comparison.transient
    assert comparison.max_transient_delta == pytest.approx(3.4e-3)
