"""Delivered copy/paste cells match the notebook and cannot launch training via Run All."""

import ast
import json
import re
from pathlib import Path


def test_notebook_cells_match_document_and_training_is_comment_only():
    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/smolvla_account_migration.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    doc = (root / "docs/DURABLE_RECOVERY_PLAN.md").read_text(encoding="utf-8")
    examples = re.findall(r"```python\n(.*?)\n```", doc, re.S)
    assert len(cells) == len(examples) == 8
    for cell, example in zip(cells, examples, strict=True):
        tree = ast.parse(cell)
        assert ast.dump(tree) == ast.dump(ast.parse(example))
        assert not any(
            isinstance(n, ast.Constant) and n.value in ("train", "--execute-recovery", "eval-vla")
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        )
    source = "\n".join(cells)
    assert "--processor-dir" in source and "--session-context" in source
    assert "CONFIRMED_SECOND_COMPUTE_ACCOUNT = False" in source
    assert "SSH_ROUTE_APPROVED = False" in source
    assert "1xSPhmxF4hK0yTXPkPXGS8bziahI5LsVK" not in source
    assert "rmtree" not in source
