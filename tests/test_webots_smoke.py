import pytest

from vlarobot.backends.webots.smoke import run_smoke


class FakeCamera:
    def __init__(self) -> None:
        self.enabled_with: int | None = None

    def enable(self, sampling_period: int) -> None:
        self.enabled_with = sampling_period

    def getHeight(self) -> int:
        return 2

    def getImage(self) -> bytes:
        return bytes(3 * 2 * 4)

    def getWidth(self) -> int:
        return 3


class FakeRobot:
    def __init__(self, step_result: int = 0) -> None:
        self.camera = FakeCamera()
        self.step_result = step_result
        self.step_count = 0

    def getBasicTimeStep(self) -> float:
        return 32.0

    def getDevice(self, name: str) -> FakeCamera | None:
        return self.camera if name == "Astra rgb" else None

    def getTime(self) -> float:
        return 0.32

    def step(self, duration: int) -> int:
        assert duration == 32
        self.step_count += 1
        return self.step_result


def test_smoke_reads_camera_after_ten_steps() -> None:
    robot = FakeRobot()

    result = run_smoke(robot)

    assert robot.camera.enabled_with == 32
    assert robot.step_count == 10
    assert (result.sim_time, result.image_width, result.image_height) == (0.32, 3, 2)


def test_smoke_reports_missing_camera() -> None:
    class RobotWithoutCamera(FakeRobot):
        def getDevice(self, name: str) -> None:
            return None

    with pytest.raises(RuntimeError, match="Astra rgb"):
        run_smoke(RobotWithoutCamera())


def test_smoke_reports_early_simulation_stop() -> None:
    with pytest.raises(RuntimeError, match="stopped"):
        run_smoke(FakeRobot(step_result=-1))
