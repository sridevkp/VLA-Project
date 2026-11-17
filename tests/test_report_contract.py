"""Unavailable report text never silently becomes invented J components or Table 5.1 labels."""

import hashlib
import json

import pytest

from vlarobot.eval.report_contract import (
    TABLE51_NAMES,
    ReportDefinitions,
    evaluate_components,
    table51_values,
)


def test_missing_definitions_fail_explicitly():
    with pytest.raises(ValueError, match="no guessed J"):
        evaluate_components(None, {}, {})
    with pytest.raises(ValueError, match="unavailable"):
        table51_values(None, {})


def test_fixture_definition_hash_and_exact_names_are_required(tmp_path):
    pdf = tmp_path / "synthetic-contract.txt"
    pdf.write_text("synthetic test, not Phase I report", encoding="utf-8")
    definitions = tmp_path / "definitions.json"
    data = {
        "report_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "components": {n: "synthetic fixture only" for n in ("Jg", "Jl", "Jr", "Ju")},
        "table51": {
            name: {"definition": "synthetic fixture only", "unit": "fixture"}
            for name in TABLE51_NAMES
        },
    }
    definitions.write_text(json.dumps(data), encoding="utf-8")
    loaded = ReportDefinitions.load(definitions, pdf)
    values = {n: float(i) for i, n in enumerate(loaded.table51_definitions)}
    assert table51_values(loaded, values) == values
    with pytest.raises(ValueError, match="names differ"):
        table51_values(loaded, {"invented": 0.0})
    pdf.write_text("changed fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative report bytes"):
        ReportDefinitions.load(definitions, pdf)


pytestmark = pytest.mark.safe_ci
