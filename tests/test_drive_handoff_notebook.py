"""Current-session replacement cells match instructions and cannot launch training."""

import ast
import json
import re
from pathlib import Path


def test_current_session_cells_are_identical_and_training_is_disabled():
    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/smolvla_drive_handoff_replacements.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    document = (root / "docs/DRIVE_HANDOFF_REPLACEMENT_CELLS.md").read_text()
    examples = re.findall(r"```python\n(.*?)\n```", document, re.S)
    assert len(cells) == len(examples) == 5
    for cell, example in zip(cells, examples, strict=True):
        tree = ast.parse(cell)
        assert ast.dump(tree) == ast.dump(ast.parse(example))
        assert not any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in ("train", "--execute-recovery", "eval-vla")
            for node in ast.walk(tree)
        )
    source = "\n".join(cells)
    assert 'globals()["context"]' in source
    assert 'record["files_sha256"] == gold["files_sha256"]' in source
    assert '"restore-check"' in source
    assert "rmtree" not in source and "SSH_ROUTE_APPROVED" not in source
    assert "pip install" not in source and "uv sync" not in source


def test_packaged_helpers_and_notebook_sha_match_delivery():
    root = Path(__file__).parents[1]
    report = json.loads((root / "docs/evidence/drive-handoff-bundle.json").read_text())
    import hashlib

    for name, expected in report["helper_files_sha256"].items():
        assert hashlib.sha256((root / "tools/recovery" / name).read_bytes()).hexdigest() == expected
    assert report["sha256"] in (root / "docs/DRIVE_HANDOFF_REPLACEMENT_CELLS.md").read_text()
