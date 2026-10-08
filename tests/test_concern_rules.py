"""Tests for typed concern rules and deterministic field evaluation."""

from importlib import import_module

import pytest

ConcernRule = import_module("legal-ai.rules.engine").ConcernRule
rule_matches = import_module("legal-ai.rules.engine").rule_matches
evaluate_rules = import_module("legal-ai.rules.engine").evaluate_rules
load_rules = import_module("legal-ai.rules.engine").load_rules
InMemoryRuleStore = import_module("legal-ai.rules.engine").InMemoryRuleStore


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


def test_raw_duration_corrects_ambiguous_digit_in_number_word():
    rule = make_rule(field="notice_period", unit="day", value=30, operator="less_than")
    # The current extractor normalizer can read the "9" inside "ninety";
    # the raw extraction is retained, so rule comparison reparses that source.
    facts = {"notice_period": {"raw": "ninety days", "normalized": {"value": 9, "unit": "day"}}}
    assert not rule_matches(rule, facts)


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


def test_evaluator_emits_structured_finding():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five (5) years"}, [rule])
    assert len(findings) == 1
    result = findings[0].to_dict()
    assert result["rule_id"] == rule.rule_id
    assert result["extracted_value"] == "five (5) years"
    assert result["rule_condition"] == "duration > 12 months"
    assert result["evidence"] == "five (5) years"
    assert "potential concern" in result["explanation"]


def test_evaluator_suppresses_populated_exception_and_wrong_category():
    rule = make_rule()
    assert evaluate_rules("non_compete", {"duration": "five years", "exceptions": ["written consent"]}, [rule]) == []
    assert evaluate_rules("confidentiality", {"duration": "five years"}, [rule]) == []


def test_evaluator_keeps_empty_exceptions_and_supplies_source_evidence():
    rule = make_rule()
    findings = evaluate_rules("non_compete", {"duration": "five years", "exceptions": []}, [rule], evidence={"duration": "five (5) years following termination"})
    assert findings[0].evidence == "five (5) years following termination"


def test_load_rules_returns_configured_criteria():
    rules = load_rules()
    assert len(rules) == 3
    assert all(isinstance(rule, ConcernRule) for rule in rules)
    clause_types = {rule.clause_type for rule in rules}
    assert clause_types == {"non_compete", "termination_notice", "confidentiality"}


def test_in_memory_rule_store_filters_by_clause_type():
    rule_nc1 = make_rule(rule_id="nc.1", clause_type="non_compete")
    rule_nc2 = make_rule(rule_id="nc.2", clause_type="non_compete")
    rule_term = make_rule(rule_id="term.1", clause_type="termination_notice")
    store = InMemoryRuleStore([rule_nc1, rule_nc2, rule_term])

    nc_rules = store.get_rules_for_clause_type("non_compete")
    assert [r.rule_id for r in nc_rules] == ["nc.1", "nc.2"]

    term_rules = store.get_rules_for_clause_type("termination_notice")
    assert [r.rule_id for r in term_rules] == ["term.1"]

    assert store.get_rules_for_clause_type("confidentiality") == []


def test_in_memory_rule_store_defaults_to_all_fixture_rules():
    store = InMemoryRuleStore()
    nc_rules = store.get_rules_for_clause_type("non_compete")
    assert len(nc_rules) == 1
    assert nc_rules[0].rule_id == "demo.non_compete.duration_over_12_months"

