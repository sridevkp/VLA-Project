import pytest

from vlarobot.eval.worldmind_endpoint_reference import slider_displacement_m

ROTATION = (0, -1, 0, 1, 0, 0, 0, 0, 1)


@pytest.mark.parametrize("axis_x", [-1, 1])
@pytest.mark.parametrize("displacement", [-1e-6, 0, 0.045])
def test_reference_preserves_signed_slide_in_rotated_translated_parent(axis_x, displacement):
    assert slider_displacement_m(
        (1, 2, 3),
        ROTATION,
        (1.01, 2.02 + axis_x * displacement, 3.03),
        zero_marker_parent_m=(0.02, -0.01, 0.03),
        slider_axis_parent=(axis_x, 0, 0),
    ) == pytest.approx(displacement, abs=1e-12)


def test_reference_rejects_reflection_as_rotation():
    with pytest.raises(ValueError, match="proper rotation"):
        slider_displacement_m(
            (0, 0, 0),
            (1, 0, 0, 0, 1, 0, 0, 0, -1),
            (0, 0, 0),
            zero_marker_parent_m=(0, 0, 0),
            slider_axis_parent=(1, 0, 0),
        )
