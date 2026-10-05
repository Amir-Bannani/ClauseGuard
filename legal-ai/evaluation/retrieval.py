"""Metrics for review-criteria retrieval, independent of extraction quality."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def recall_at_k(cases: Sequence[Mapping[str, Any]], retriever: Any, *, k: int = 5) -> float:
    """Fraction of labeled cases whose expected applicable rule appears in top-k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        docs = retriever.retrieve(
            case.get("query", ""), clause_type=case["clause_type"], top_k=k,
            metadata_filter=case.get("metadata_filter"),
        )
        hits += case["expected_rule_id"] in {doc.rule_id for doc in docs}
    return hits / len(cases)


def binary_rule_metrics(expected: Sequence[bool], predicted: Sequence[bool]) -> dict[str, int | float]:
    """Compute confusion counts for rule evaluation on known extracted facts."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have equal lengths")
    tp = fp = tn = fn = 0
    for truth, prediction in zip(expected, predicted):
        if truth and prediction:
            tp += 1
        elif prediction:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(expected) if expected else 0.0}
