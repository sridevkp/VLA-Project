import numpy as np
import pytest

from vlarobot.backends.webots.images import camera_array_to_rgb


def test_non_square_webots_array_preserves_pixel_coordinates_and_rgb_channels():
    # R2025a getImageArray consumes consecutive pixels but groups width then height.
    # Six distinct colours expose the old transpose, unlike all-black camera fakes.
    pixels = [
        [[3, 2, 1], [6, 5, 4]],
        [[9, 8, 7], [12, 11, 10]],
        [[15, 14, 13], [18, 17, 16]],
    ]
    rgb = camera_array_to_rgb(pixels, width=3, height=2)
    np.testing.assert_array_equal(
        rgb,
        [[[3, 2, 1], [6, 5, 4], [9, 8, 7]], [[12, 11, 10], [15, 14, 13], [18, 17, 16]]],
    )
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous


def test_truncated_camera_array_fails_closed():
    with pytest.raises(ValueError, match="array size"):
        camera_array_to_rgb([[[1, 2, 3]]], width=3, height=2)
