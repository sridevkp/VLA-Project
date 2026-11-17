"""Pure reset comparisons. There are no default or approved physical tolerances."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """Flattened named values with explicit unit-bearing field names."""

    configuration: tuple[tuple[str, str], ...]
    scalars: tuple[tuple[str, float | None], ...]
    orientations: tuple[tuple[str, tuple[float, ...] | None], ...]
    units: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for rows in (self.configuration, self.scalars, self.orientations, self.units):
            if type(rows) is not tuple or any(
                type(row) is not tuple or len(row) != 2 or type(row[0]) is not str for row in rows
            ):
                raise TypeError("State requires immutable named tuples")
        if any(value is not None and type(value) is not tuple for _, value in self.orientations):
            raise TypeError("Orientations must be immutable tuples")
        if any(type(value) is not str for _, value in (*self.configuration, *self.units)):
            raise TypeError("Configuration and units require immutable strings")
        if any(
            value is not None and type(value) not in (int, float, bool) for _, value in self.scalars
        ):
            raise TypeError("Scalar values must be numbers or explicit None")
        if any(
            value is not None and any(type(x) not in (int, float, bool) for x in value)
            for _, value in self.orientations
        ):
            raise TypeError("Quaternion values must be numeric scalars")


@dataclass(frozen=True)
class Bound:
    """Separately approved reference and pairwise ceilings for one named metric."""

    field: str
    reference: float
    pairwise: float
    unit: str


@dataclass(frozen=True)
class Protocol:
    """Caller-supplied criteria; test-only approval IDs confer no live authority."""

    version: str
    approval_id: str
    observability_version: str
    bounds: tuple[Bound, ...]
    historical_omissions: tuple[tuple[str, str], ...]
    quaternion_norm_tolerance: float
    timestamp_tolerance_s: float

    def __post_init__(self) -> None:
        if type(self.bounds) is not tuple or any(type(b) is not Bound for b in self.bounds):
            raise TypeError("Protocol bounds must be immutable typed tuples")
        if type(self.historical_omissions) is not tuple or any(
            type(row) is not tuple or len(row) != 2 or any(type(value) is not str for value in row)
            for row in self.historical_omissions
        ):
            raise TypeError("Omissions require immutable field/reason strings")


@dataclass(frozen=True)
class Admission:
    accepted: bool
    issues: tuple[str, ...]
    differences: tuple[tuple[str, str, float], ...]


def _mapping(items: tuple) -> dict:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("Duplicate field")
    return result


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _angle(a: tuple, b: tuple, norm_tolerance: float) -> float:
    if len(a) != 4 or len(b) != 4 or not all(_number(x) for x in (*a, *b)):
        raise ValueError("Invalid quaternion")
    an, bn = math.hypot(*a), math.hypot(*b)
    if an == 0 or bn == 0 or abs(an - 1) > norm_tolerance or abs(bn - 1) > norm_tolerance:
        raise ValueError("Quaternion norm outside supplied criterion")
    a, b = tuple(x / an for x in a), tuple(x / bn for x in b)
    sign = 1 if sum(x * y for x, y in zip(a, b, strict=True)) >= 0 else -1
    return 4 * math.atan2(
        math.hypot(*(x - sign * y for x, y in zip(a, b, strict=True))),
        math.hypot(*(x + sign * y for x, y in zip(a, b, strict=True))),
    )


def compare(
    reference: State, current: State, previous: tuple[State, ...], protocol: Protocol
) -> Admission:
    """Fail closed on schema gaps, unapproved criteria or unobserved live fields."""
    issues: list[str] = []
    differences: list[tuple[str, str, float]] = []
    try:
        if not all(
            type(x) is str and x.strip()
            for x in (protocol.version, protocol.approval_id, protocol.observability_version)
        ):
            raise ValueError("Missing protocol/approval/observability identity")
        if not all(
            _number(x) and x >= 0
            for x in (protocol.quaternion_norm_tolerance, protocol.timestamp_tolerance_s)
        ):
            raise ValueError("Missing numerical protocol criteria")
        bounds = {b.field: b for b in protocol.bounds}
        omissions = _mapping(protocol.historical_omissions)
        if len(bounds) != len(protocol.bounds) or not all(omissions.values()):
            raise ValueError("Duplicate criterion or undocumented omission")
        configs = [_mapping(s.configuration) for s in (reference, current, *previous)]
        if not configs[0] or any(c != configs[0] for c in configs[1:]):
            raise ValueError("Exact configuration mismatch")
        scalars = [_mapping(s.scalars) for s in (reference, current, *previous)]
        rotations = [_mapping(s.orientations) for s in (reference, current, *previous)]
        units = [_mapping(s.units) for s in (reference, current, *previous)]
        scalar_fields, rotation_fields = set(scalars[0]), set(rotations[0])
        if scalar_fields & rotation_fields or not scalar_fields | rotation_fields:
            raise ValueError("Empty/overlapping state schema")
        fields = scalar_fields | rotation_fields
        if set(bounds) != fields or not set(omissions) <= fields:
            raise ValueError("Missing/extra criteria or omissions")
        if any(u != units[0] for u in units[1:]) or set(units[0]) != fields:
            raise ValueError("State unit/schema mismatch")
        for b in bounds.values():
            if b.unit != units[0][b.field]:
                raise ValueError("Criterion unit mismatch")
            if not b.unit or not all(_number(x) and x >= 0 for x in (b.reference, b.pairwise)):
                raise ValueError("Invalid numeric bound")
        for i, (s, q) in enumerate(zip(scalars, rotations, strict=True)):
            if set(s) != scalar_fields or set(q) != rotation_fields:
                raise ValueError("State schema mismatch")
            for field in fields:
                value = s[field] if field in s else q[field]
                if value is None:
                    if i == 0 and field in omissions:
                        continue
                    raise ValueError(f"Missing field: {field}")
                if i == 0 and field in omissions:
                    raise ValueError("Omission would hide available reference evidence")
                if field in s and not _number(value):
                    raise ValueError(f"Nonfinite field: {field}")
                if field in q:
                    _angle(value, value, protocol.quaternion_norm_tolerance)
        for i, label in [
            (0, "reference"),
            *((i, f"pairwise:{i - 2}") for i in range(2, len(scalars))),
        ]:
            for field in sorted(fields):
                if i == 0 and field in omissions:
                    continue
                delta = (
                    abs(scalars[1][field] - scalars[i][field])
                    if field in scalar_fields
                    else _angle(
                        rotations[1][field], rotations[i][field], protocol.quaternion_norm_tolerance
                    )
                )
                ceiling = bounds[field].reference if i == 0 else bounds[field].pairwise
                differences.append((label, field, delta))
                if delta > ceiling:
                    issues.append(f"{label}:{field} exceeds criterion")
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        issues.append(str(exc))
    return Admission(not issues, tuple(issues), tuple(differences))
