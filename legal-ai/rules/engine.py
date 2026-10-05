"""Typed, deterministic review-rule definitions and evaluation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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
    raw = value.get("raw") if isinstance(value, Mapping) else value
    parsed = _parse_duration(raw) if isinstance(raw, str) else None
    if isinstance(value, Mapping):
        if parsed is None:
            parsed = value.get("normalized", value)
    if isinstance(parsed, Mapping):
        unit = parsed.get("unit")
        amount = parsed.get("value")
    else:
        unit = amount = None
    if amount is None or unit is None:
        raise ValueError("malformed duration")
    factors = {"day": 1.0, "month": 30.0, "year": 365.0}
    target = target_unit or unit
    return float(amount) * factors[unit] / factors[target]


def _parse_duration(raw: str) -> dict[str, Any] | None:
    """Parse known duration words before digits (e.g. 'ninety' before its '9')."""
    text = raw.casefold().strip()
    if text == "immediately":
        return {"value": 0, "unit": "day"}
    unit = next((name for name in ("year", "month", "day") if name in text), None)
    if unit is None:
        return None
    words = {"twenty-four": 24, "eighteen": 18, "twelve": 12, "ninety": 90,
             "sixty": 60, "thirty": 30, "ten": 10, "nine": 9,
             "eight": 8, "seven": 7, "six": 6, "five": 5,
             "four": 4, "three": 3, "two": 2, "one": 1}
    for word, number in words.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return {"value": number, "unit": unit}
    match = re.search(r"\b(\d+)\b", text)
    return {"value": int(match.group(1)), "unit": unit} if match else None


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


@dataclass(frozen=True)
class ConcernFinding:
    rule_id: str
    clause_type: str
    finding_type: str
    severity: str
    field: str
    extracted_value: Any
    rule_condition: str
    explanation: str
    evidence: Any
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_rules(
    clause_type: str,
    facts: Mapping[str, Any],
    rules: Sequence[ConcernRule],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> list[ConcernFinding]:
    """Evaluate matching criteria and suppress a match when its exception field is populated."""
    findings = []
    for rule in rules:
        if rule.clause_type != clause_type or not rule_matches(rule, facts):
            continue
        if rule.exception_field:
            exceptions = facts.get(rule.exception_field)
            if exceptions:
                continue
        raw = facts[rule.field]
        extracted = raw.get("raw", raw) if isinstance(raw, Mapping) else raw
        findings.append(ConcernFinding(
            rule_id=rule.rule_id,
            clause_type=clause_type,
            finding_type=rule.finding_type,
            severity=rule.severity,
            field=rule.field,
            extracted_value=extracted,
            rule_condition=rule.rule_condition,
            explanation=(f"The extracted value for {rule.field} matches a configured review criterion. "
                         "This is a potential concern according to the configured review policy."),
            evidence=(evidence or {}).get(rule.field, extracted),
            source=rule.source,
        ))
    return findings
