"""Small external-controller smoke test for Webots."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol

from vlarobot.env import configure_webots_environment


class Camera(Protocol):
    def enable(self, sampling_period: int) -> None: ...

    def getHeight(self) -> int: ...

    def getImage(self) -> bytes | None: ...

    def getWidth(self) -> int: ...


class Robot(Protocol):
    def getBasicTimeStep(self) -> float: ...

    def getDevice(self, name: str) -> Camera | None: ...

    def getTime(self) -> float: ...

    def step(self, duration: int) -> int: ...


@dataclass(frozen=True)
class SmokeResult:
    sim_time: float
    image_width: int
    image_height: int


def create_webots_robot() -> Robot:
    """Load Webots' controller API from WEBOTS_HOME and connect to the running world."""
    configure_webots_environment()
    controller_module = importlib.import_module("controller")
    return controller_module.Robot()


def run_smoke(robot: Robot) -> SmokeResult:
    """Read one camera frame after advancing a connected Webots world ten steps."""
    time_step = int(robot.getBasicTimeStep())
    if time_step <= 0:
        raise RuntimeError(f"Webots returned an invalid basic time step: {time_step}")

    camera = robot.getDevice("Astra rgb")
    if camera is None:
        raise RuntimeError("Required Webots camera not found: Astra rgb")
    camera.enable(time_step)

    for _ in range(10):
        if robot.step(time_step) == -1:
            raise RuntimeError("Webots simulation stopped before completing ten steps.")

    image = camera.getImage()
    if image is None:
        raise RuntimeError("Webots camera did not produce an image after ten steps.")

    width = camera.getWidth()
    height = camera.getHeight()
    if width <= 0 or height <= 0 or len(image) != width * height * 4:
        raise RuntimeError(
            f"Webots returned an invalid RGBA image: {width}x{height}, {len(image)} bytes."
        )

    return SmokeResult(
        sim_time=robot.getTime(),
        image_width=width,
        image_height=height,
    )
