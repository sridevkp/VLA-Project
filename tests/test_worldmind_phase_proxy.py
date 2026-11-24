import pytest

from vlarobot.eval.worldmind_phase_proxy import phase_clearance
from vlarobot.eval.worldmind_proxy_policy import GRID, HARD_CHECKS
from vlarobot.risk.gate import CandidateAssessment, RiskGate


@pytest.mark.parametrize(
    ("phase", "plane", "app", "expected"),
    [
        ("CLOSE", 0.02, 0.2, 0),
        ("CLOSE", -0.002, 0.2, None),
        ("CLOSE", 0.02, -0.01, None),
        ("TRANSIT", 0.02, 0.2, None),
    ],
)
def test_phase_contact_control_keeps_table_penetration_and_clutter_vetoes(
    phase,
    plane,
    app,
    expected,
):
    tables = {f"TABLE:foot {i}": 0.2 for i in range(1, 5)}
    tables["TABLE:main frame"] = -0.03
    objects = {"CAN": -0.08, "HAM": 0.2, "APP": app}
    table, object_ = phase_clearance(phase, tables, objects, directional_tabletop_m=plane)
    assessment = CandidateAssessment(
        tuple(zip(HARD_CHECKS, (True, True, table > 0, object_ > 0, True), strict=True)),
        (0.0, 0.0, 0.0, 0.0),
    )
    gate = RiskGate(weights=GRID["equal"], maximize=False, required_constraints=HARD_CHECKS)
    assert gate.select([assessment]).selected_index == expected
