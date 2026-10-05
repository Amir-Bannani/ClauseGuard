"""Tests for typed concern rules and deterministic field evaluation."""

from importlib import import_module

import pytest

ConcernRule = import_module("legal-ai.rules.engine").ConcernRule
rule_matches = import_module("legal-ai.rules.engine").rule_matches


def make_rule(**overrides):
    values = dict(rule_id="demo.duration", clause_type="non_compete", field="duration",
                  operator="greater_than", value=12, unit="month",
                  finding_type="duration_review", severity="high",
                  description="Demonstration review threshold.",
                  source="demo_configuration")
    values.update(overrides)
    return ConcernRule(**values)


def test_valid_rule_and_condition():
    rule = make_rule()
    assert rule.rule_id == "demo.duration"
    assert rule.rule_condition == "duration > 12 months"


@pytest.mark.parametrize("change", [
    {"rule_id": ""}, {"clause_type": ""}, {"field": ""},
    {"operator": "execute"}, {"severity": "critical"},
    {"value": None}, {"source": ""}, {"unit": "fortnight"},
])
def test_invalid_rule_rejected(change):
    with pytest.raises(ValueError):
        make_rule(**change)


def test_numeric_duration_match_and_non_match():
    rule = make_rule()
    assert rule_matches(rule, {"duration": "five (5) years"})
    assert not rule_matches(rule, {"duration": "six months"})


def test_normalized_extractor_duration_is_supported():
    rule = make_rule()
    facts = {"duration": {"raw": "five (5) years", "normalized": {"value": 5, "unit": "year"}}}
    assert rule_matches(rule, facts)


def test_comparison_and_equality_operators():
    assert rule_matches(make_rule(operator="equals", value="worldwide", unit=None),
                        {"duration": "worldwide"})
    assert rule_matches(make_rule(operator="not_equals", value="local", unit=None),
                        {"duration": "worldwide"})
    assert rule_matches(make_rule(operator="less_than", value=24), {"duration": "18 months"})
    assert rule_matches(make_rule(operator="greater_than_or_equal", value=18),
                        {"duration": "18 months"})


def test_membership_boolean_and_nonempty_operators():
    assert rule_matches(make_rule(operator="contains", value="client", unit=None),
                        {"duration": ["client", "vendor"]})
    assert rule_matches(make_rule(operator="in", value=["worldwide", "global"], unit=None),
                        {"duration": "worldwide"})
    assert rule_matches(make_rule(operator="is_true", value=None, unit=None), {"duration": True})
    assert rule_matches(make_rule(operator="is_false", value=None, unit=None), {"duration": False})
    assert rule_matches(make_rule(operator="is_not_empty", value=None, unit=None), {"duration": ["x"]})


def test_missing_null_and_malformed_values_do_not_match():
    rule = make_rule()
    assert not rule_matches(rule, {})
    assert not rule_matches(rule, {"duration": None})
    assert not rule_matches(rule, {"duration": "an extended period"})

