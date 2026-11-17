import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.safe_ci


def test_controller_has_one_main_guard_and_one_dispatch() -> None:
    path = Path(__file__).parents[1] / "src/vlarobot/backends/webots/controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    calls = [
        node
        for guard in guards
        for node in ast.walk(guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main"
    ]
    assert len(guards) == 1
    assert len(calls) == 1
