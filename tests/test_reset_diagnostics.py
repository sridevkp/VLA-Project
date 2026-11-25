import numpy as np

from vlarobot.cli import _compare_reset_probes, _image_similarity
from vlarobot.execution.scripted_can import ResetProbeResult, ResetProbeWaypoint


def _probe(shift: float = 0.0) -> ResetProbeResult:
    return ResetProbeResult(
        seed=0,
        reset_duration_s=0.1,
        probe_duration_s=0.2,
        state_sequence=("INIT", "PREGRASP"),
        commanded_pregrasp=(0.5 + shift, 0.1, 0.8),
        waypoints=(
            ResetProbeWaypoint(
                joint_targets=(("arm_1_joint", 0.2 + shift),),
                achieved_position=(0.5 + shift, 0.1, 0.8),
                achieved_orientation=(0.0, 0.0, 0.0, 1.0),
                gripper_opening=0.04,
            ),
        ),
        achieved_position=(0.5 + shift, 0.1, 0.8),
        achieved_orientation=(0.0, 0.0, 0.0, 1.0),
        gripper_opening=0.04,
        reset_joint_positions=(("arm_1_joint", 0.1 + shift),),
        reset_base_position=(0.0, 0.0, 0.0),
        reset_base_orientation=(0.0, 0.0, 0.0, 1.0),
        reset_ee_position=(0.5 + shift, 0.1, 0.8),
        reset_ee_orientation=(0.0, 0.0, 0.0, 1.0),
        reset_can_position=(1.0, 2.0, 3.0),
        reset_can_orientation=(0.0, 0.0, 0.0, 1.0),
        rgb_at_reset=np.zeros((16, 16, 3), dtype=np.uint8),
        rgb_warmup_samples=((1, np.zeros((16, 16, 3), dtype=np.uint8)),),
        rgb_after_warmup=np.zeros((16, 16, 3), dtype=np.uint8),
    )


def test_reset_behavior_comparison_uses_one_millimeter_and_milliradian_tolerance() -> None:
    assert _compare_reset_probes(_probe(), _probe(0.0005))["result"] == "PASS"
    assert _compare_reset_probes(_probe(), _probe(0.002))["result"] == "FAIL"


def test_reset_image_metrics_return_perfect_similarity_for_identical_rgb() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    assert _image_similarity(image, image) == (0.0, 1.0, 0)


def test_reset_image_metrics_detect_changed_pixels() -> None:
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    second = first.copy()
    second[4, 4] = 100

    mean_absolute_difference, ssim, maximum = _image_similarity(first, second)

    assert mean_absolute_difference > 0.0
    assert ssim < 1.0
    assert maximum == 100
