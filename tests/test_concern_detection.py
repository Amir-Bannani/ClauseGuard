"""End-to-end extraction adapter, criteria retrieval, and rule-evaluation tests."""

from importlib import import_module


ConcernDetector = import_module("legal-ai.rules.detector").ConcernDetector


def test_extractor_json_to_retrieved_rule_and_finding():
    clause = "For five (5) years after termination, the employee shall not work for a competitor."
    result = ConcernDetector().detect(
        "non_compete", '{"duration": "five (5) years", "exceptions": []}', clause_text=clause
    ).to_dict()
    assert result["retrieved_rule_ids"] == ["demo.non_compete.duration_over_12_months"]
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["extracted_value"] == "five (5) years"
    assert finding["evidence"] == "five (5) years"
    assert finding["source"] == "demo_configuration"
    assert "configured review policy" in finding["explanation"]


def test_non_trigger_and_exception_produce_no_finding():
    detector = ConcernDetector()
    below_threshold = detector.detect("non_compete", {"duration": "six months", "exceptions": []})
    excepted = detector.detect("non_compete", {"duration": "five years", "exceptions": ["prior consent"]})
    assert below_threshold.findings == []
    assert excepted.findings == []


def test_invalid_extractor_json_is_reported_as_adapter_error():
    import pytest

    with pytest.raises(ValueError, match="valid JSON"):
        ConcernDetector().detect("non_compete", "not JSON")
