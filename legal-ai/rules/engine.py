"""Typed, deterministic review-rule definitions and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_OPERATORS = {
    "equals", "not_equals", "greater_than", "greater_than_or_equal",
    "less_than", "less_than_or_equal", "contains", "in",
    "is_true", "is_false", "is_not_empty",
}
SEVERITIES = {"info", "low", "medium", "high"}


@dataclass(frozen=True)
class ConcernRule:
    """A configured review criterion; thresholds are policy, not legal facts."""

    rule_id: str
    clause_type: str
    field: str
    operator: str
    finding_type: str
    severity: str
    description: str
    source: str
    value: Any = None
    unit: str | None = None
    exception_field: str | None = "exceptions"

    def __post_init__(self) -> None:
        for name in ("rule_id", "clause_type", "field", "finding_type", "description", "source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if self.operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"unsupported operator: {self.operator}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unsupported severity: {self.severity}")
        if self.operator not in {"is_true", "is_false", "is_not_empty"} and self.value is None:
            raise ValueError(f"value is required for {self.operator}")
        if self.unit is not None and self.unit not in {"day", "month", "year"}:
            raise ValueError("unit must be day, month, or year")

    @property
    def rule_condition(self) -> str:
        symbols = {"equals": "==", "not_equals": "!=", "greater_than": ">",
                   "greater_than_or_equal": ">=", "less_than": "<",
                   "less_than_or_equal": "<=", "contains": "contains",
                   "in": "in", "is_true": "is true", "is_false": "is false",
                   "is_not_empty": "is not empty"}
        expected = "" if self.value is None else f" {self.value}"
        unit = f" {self.unit}s" if self.unit else ""
        return f"{self.field} {symbols[self.operator]}{expected}{unit}".strip()


def _duration_value(value: Any, target_unit: str | None) -> float:
    """Convert extractor Normalizer output or raw duration text to target units."""
    from importlib import import_module
    Normalizer = import_module("legal-ai.extraction.normalizer").Normalizer

    parsed = value
    if isinstance(value, Mapping):
        parsed = value.get("normalized", value)
        if isinstance(parsed, Mapping):
            unit = parsed.get("unit")
            amount = parsed.get("value")
        else:
            unit = amount = None
    else:
        parsed = Normalizer.parse_duration(value)
        unit = parsed.get("unit") if parsed else None
        amount = parsed.get("value") if parsed else None
    if amount is None or unit is None:
        raise ValueError("malformed duration")
    factors = {"day": 1.0, "month": 30.0, "year": 365.0}
    target = target_unit or unit
    return float(amount) * factors[unit] / factors[target]


def rule_matches(rule: ConcernRule, facts: Mapping[str, Any]) -> bool:
    """Return whether one field satisfies a rule. Missing/malformed data is no match."""
    if rule.field not in facts or facts[rule.field] is None:
        return False
    actual = facts[rule.field]
    expected = rule.value
    try:
        if rule.unit:
            actual = _duration_value(actual, rule.unit)
        op = rule.operator
        if op == "equals": return actual == expected
        if op == "not_equals": return actual != expected
        if op == "greater_than": return actual > expected
        if op == "greater_than_or_equal": return actual >= expected
        if op == "less_than": return actual < expected
        if op == "less_than_or_equal": return actual <= expected
        if op == "contains": return expected in actual
        if op == "in": return actual in expected
        if op == "is_true": return actual is True
        if op == "is_false": return actual is False
        if op == "is_not_empty": return bool(actual)
    except (TypeError, ValueError, KeyError):
        return False
    return False
