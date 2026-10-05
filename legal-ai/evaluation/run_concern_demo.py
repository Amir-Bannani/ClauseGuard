"""Run labeled synthetic criteria-retrieval and rule-evaluation examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from importlib import import_module

metrics = import_module("legal-ai.evaluation.retrieval")
retrieval = import_module("legal-ai.retrieval.contract_retriever")
detector_module = import_module("legal-ai.rules.detector")


def run_demo() -> dict[str, object]:
    """Return metrics for synthetic cases; extraction is supplied as known facts."""
    retriever = retrieval.InMemoryRuleRetriever()
    retrieval_cases = [
        {"clause_type": "non_compete", "query": "duration months", "expected_rule_id": "demo.non_compete.duration_over_12_months"},
        {"clause_type": "termination_notice", "query": "notice period days", "expected_rule_id": "demo.termination_notice.shorter_than_30_days"},
        {"clause_type": "confidentiality", "query": "duration months", "expected_rule_id": "demo.confidentiality.duration_over_60_months"},
    ]
    examples = [
        ("non_compete", {"duration": "five years", "exceptions": []}, True),
        ("non_compete", {"duration": "six months", "exceptions": []}, False),
        ("termination_notice", {"notice_period": "immediately", "conditions": []}, True),
        ("termination_notice", {"notice_period": "ninety days", "conditions": []}, False),
        ("confidentiality", {"duration": "six years", "exceptions": []}, True),
        ("confidentiality", {"duration": "five years", "exceptions": []}, False),
    ]
    detector = detector_module.ConcernDetector(retriever)
    predictions = [bool(detector.detect(category, facts).findings) for category, facts, _ in examples]
    expected = [label for _, _, label in examples]
    return {
        "dataset": "synthetic demonstration cases; extracted facts are supplied",
        "recall_at_1": metrics.recall_at_k(retrieval_cases, retriever, k=1),
        "rule_metrics": metrics.binary_rule_metrics(expected, predictions),
        "cases": len(examples),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
