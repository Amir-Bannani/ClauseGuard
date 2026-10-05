"""Compatibility integration from the current extractor JSON output to findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping

retrieval_module = import_module("legal-ai.retrieval.contract_retriever")
engine_module = import_module("legal-ai.rules.engine")


@dataclass(frozen=True)
class DetectionResult:
    findings: list[Any]
    retrieved_rule_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"findings": [finding.to_dict() for finding in self.findings],
                "retrieved_rule_ids": list(self.retrieved_rule_ids)}


def adapt_extraction_output(output: str | Mapping[str, Any]) -> dict[str, Any]:
    """Accept the extractor's JSON string or an already parsed extraction mapping."""
    if isinstance(output, str):
        try:
            facts = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ValueError("extractor output must be valid JSON") from exc
    else:
        facts = dict(output)
    if not isinstance(facts, dict) or not all(isinstance(key, str) for key in facts):
        raise ValueError("extractor output must be a JSON object with string keys")
    normalizer = import_module("legal-ai.extraction.normalizer").Normalizer
    return normalizer.normalize_extraction("", facts)


class ConcernDetector:
    """Retrieve criteria by clause metadata, then evaluate them deterministically."""

    def __init__(self, retriever: Any | None = None):
        self.retriever = retriever or retrieval_module.InMemoryRuleRetriever()

    def detect(
        self,
        clause_type: str,
        extraction_output: str | Mapping[str, Any],
        *,
        clause_text: str | None = None,
        top_k: int = 5,
    ) -> DetectionResult:
        facts = adapt_extraction_output(extraction_output)
        documents = self.retriever.retrieve_for_facts(clause_type, facts, top_k=top_k)
        evidence = _evidence_by_field(facts, clause_text)
        findings = engine_module.evaluate_rules(
            clause_type, facts, [document.rule for document in documents], evidence=evidence
        )
        return DetectionResult(findings, [document.rule_id for document in documents])


def _evidence_by_field(facts: Mapping[str, Any], clause_text: str | None) -> dict[str, str]:
    evidence = {}
    if not clause_text:
        return evidence
    folded = clause_text.casefold()
    for field, value in facts.items():
        raw = value.get("raw") if isinstance(value, Mapping) else value
        if not isinstance(raw, str) or not raw:
            continue
        start = folded.find(raw.casefold())
        if start >= 0:
            evidence[field] = clause_text[start:start + len(raw)]
    return evidence
