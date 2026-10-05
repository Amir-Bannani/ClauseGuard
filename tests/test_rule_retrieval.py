"""Tests for local, metadata-filtered review-criteria retrieval."""

from importlib import import_module

import pytest

retrieval = import_module("legal-ai.retrieval.contract_retriever")
InMemoryRuleRetriever = retrieval.InMemoryRuleRetriever
load_rule_documents = retrieval.load_rule_documents


def test_fixture_documents_include_rule_and_source_metadata():
    docs = load_rule_documents()
    assert len(docs) == 3
    assert all(doc.rule_id and doc.source == "demo_configuration" for doc in docs)
    assert all("Review rationale:" in doc.text for doc in docs)


def test_retrieval_filters_by_clause_type():
    retriever = InMemoryRuleRetriever()
    docs = retriever.retrieve("long duration review", clause_type="non_compete")
    assert [doc.rule_id for doc in docs] == ["demo.non_compete.duration_over_12_months"]


def test_retrieval_applies_metadata_filter():
    docs = InMemoryRuleRetriever().retrieve(
        "notice period", metadata_filter={"field": "notice_period"}
    )
    assert [doc.metadata["clause_type"] for doc in docs] == ["termination_notice"]


def test_top_k_and_zero_top_k():
    retriever = InMemoryRuleRetriever()
    assert len(retriever.retrieve("review", top_k=2)) == 2
    assert retriever.retrieve("review", top_k=0) == []
    with pytest.raises(ValueError):
        retriever.retrieve("review", top_k=-1)


def test_empty_retrieval_result():
    assert InMemoryRuleRetriever([]).retrieve("anything") == []


def test_targeted_retrieval_uses_extracted_facts():
    retriever = InMemoryRuleRetriever()
    docs = retriever.retrieve_for_facts(
        "non_compete", {"duration": {"raw": "five years", "normalized": {"value": 5, "unit": "year"}},
                        "geographical_scope": "worldwide", "exceptions": []}
    )
    assert len(docs) == 1
    assert docs[0].rule.clause_type == "non_compete"
