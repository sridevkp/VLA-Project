import json
from pathlib import Path

import pytest

from vlarobot.execution.skill_demonstrations import demonstration_decisions


def test_extraction_rejects_non_train_before_opening_raw_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "accepted.json"
    manifest.write_text(
        json.dumps({"episodes": [{"scene": {"metadata": {"split": "validation"}}}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires TRAIN"):
        list(demonstration_decisions(manifest))
