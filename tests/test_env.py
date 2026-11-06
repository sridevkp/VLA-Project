import os
import sys
from pathlib import Path

import pytest

from vlarobot.env import configure_webots_environment


def test_configure_webots_environment_sets_home_and_api_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webots_home = tmp_path / "webots"
    controller_path = webots_home / "lib" / "controller" / "python"
    controller_path.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.delenv("WEBOTS_HOME", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    configured_path = configure_webots_environment(webots_home)

    assert configured_path == controller_path
    assert os.environ["WEBOTS_HOME"] == str(webots_home.resolve())
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(controller_path)
    assert sys.path[0] == str(controller_path)


def test_configure_webots_environment_preserves_existing_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webots_home = tmp_path / "webots"
    controller_path = webots_home / "lib" / "controller" / "python"
    controller_path.mkdir(parents=True)
    existing_path = str(tmp_path / "existing")
    monkeypatch.setattr(sys, "path", [existing_path])
    monkeypatch.setenv("PYTHONPATH", existing_path)

    configure_webots_environment(webots_home)

    assert os.environ["PYTHONPATH"].split(os.pathsep) == [
        str(controller_path),
        existing_path,
    ]
    assert sys.path[:2] == [str(controller_path), existing_path]


def test_configure_webots_environment_requires_webots_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEBOTS_HOME", raising=False)

    with pytest.raises(RuntimeError, match="WEBOTS_HOME"):
        configure_webots_environment()
