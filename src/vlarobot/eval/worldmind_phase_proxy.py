"""Offline comparison of the declared contact windows with the unchanged v1 hard gate.

This prototype is deliberately not registered in the live arm. It can only replace a
main-frame sphere overlap when an independent directional clearance is positive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

DESCEND_PLACE_WINDOW = frozenset({"DESCEND", "CLOSE", "PLACE_DESCEND", "RELEASE"})
CAN_GRASP_WINDOW = frozenset({"DESCEND", "CLOSE"})


def phase_clearance(
    phase: str,
    table_clearances_m: Mapping[str, float],
    object_clearances_m: Mapping[str, float],
    *,
    directional_tabletop_m: float,
) -> tuple[float, float]:
    """Keep all noncontact checks. A negative directional table bound always vetoes."""
    if (
        set(table_clearances_m) != {"TABLE:main frame", *(f"TABLE:foot {i}" for i in range(1, 5))}
        or set(object_clearances_m) != {"CAN", "HAM", "APP"}
        or not all(
            math.isfinite(value)
            for value in (
                *table_clearances_m.values(),
                *object_clearances_m.values(),
                directional_tabletop_m,
            )
        )
    ):
        raise ValueError("complete finite phase clearance evidence required")
    table = []
    for name, clearance in table_clearances_m.items():
        if name == "TABLE:main frame" and phase in DESCEND_PLACE_WINDOW:
            table.append(directional_tabletop_m)
        else:
            table.append(clearance)
    objects = [
        value
        for name, value in object_clearances_m.items()
        if name != "CAN" or phase not in CAN_GRASP_WINDOW
    ]
    return min(table), min(objects)
