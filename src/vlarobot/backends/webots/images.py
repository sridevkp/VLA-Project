"""Restore HWC pixel coordinates from the pinned Webots getImageArray grouping."""

import numpy as np


def camera_array_to_rgb(pixels: object, width: int, height: int) -> np.ndarray:
    """Restore linear RGB pixel order, preserving the existing camera acquisition path.

    R2025a's binding groups row-major pixels into (width, height, 3). A transpose
    scrambles that sequence; reshape restores (height, width, 3) pixel coordinates.
    """
    image = np.asarray(pixels, dtype=np.uint8)
    if width <= 0 or height <= 0 or image.shape != (width, height, 3):
        raise ValueError("Webots RGB array size does not match camera dimensions")
    return image.reshape(height, width, 3).copy()
