"""Local retrieval of configured review criteria (not legal-document search)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from importlib import import_module

ConcernRule = import_module("legal-ai.rules.engine").ConcernRule


@dataclass(frozen=True)
class RuleDocument:
    rule_id: str
    text: str
    metadata: Mapping[str, Any]
    source: str
    rule: ConcernRule


class RuleRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        clause_type: str | None = None,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RuleDocument]: ...


def load_rule_documents(path: str | Path | None = None) -> list[RuleDocument]:
    """Load the small checked-in configured-criteria knowledge base."""
    fixture = Path(path) if path else Path(__file__).parents[1] / "rules" / "review_rules.json"
    records = json.loads(fixture.read_text(encoding="utf-8"))
    docs = []
    for record in records:
        rule = ConcernRule(**{key: value for key, value in record.items() if key != "rationale"})
        condition = rule.rule_condition
        text = "\n".join((
            f"Rule ID: {rule.rule_id}", f"Clause type: {rule.clause_type}",
            f"Relevant field: {rule.field}", f"Condition: {condition}",
            f"Review criterion: {rule.description}",
            f"Review rationale: {record['rationale']}", f"Severity: {rule.severity}",
            f"Source: {rule.source}",
        ))
        docs.append(RuleDocument(
            rule_id=rule.rule_id,
            text=text,
            metadata={"clause_type": rule.clause_type, "field": rule.field,
                      "finding_type": rule.finding_type, "severity": rule.severity},
            source=rule.source,
            rule=rule,
        ))
    return docs


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


class InMemoryRuleRetriever:
    """Small deterministic retriever; replaceable behind the RuleRetriever protocol."""

    def __init__(self, documents: Sequence[RuleDocument] | None = None):
        self.documents = list(documents if documents is not None else load_rule_documents())

    def retrieve(
        self,
        query: str,
        *,
        clause_type: str | None = None,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RuleDocument]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []
        query_tokens = _tokens(query)
        candidates = []
        for index, document in enumerate(self.documents):
            if clause_type is not None and document.metadata.get("clause_type") != clause_type:
                continue
            if metadata_filter and any(document.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            overlap = len(query_tokens & _tokens(document.text))
            # Metadata applicability is primary; lexical overlap only orders applicable criteria.
            candidates.append((-overlap, index, document))
        candidates.sort(key=lambda row: (row[0], row[1]))
        return [document for _, _, document in candidates[:top_k]]

    def retrieve_for_facts(
        self, clause_type: str, facts: Mapping[str, Any], *, top_k: int = 5
    ) -> list[RuleDocument]:
        """Build a targeted query from normalized extracted facts, not whole documents."""
        parts = [clause_type]
        for field, value in facts.items():
            if value is None or field == "exceptions":
                continue
            if isinstance(value, Mapping):
                value = value.get("raw", value.get("normalized", value))
            parts.append(f"{field} {value}")
        return self.retrieve(" ".join(parts), clause_type=clause_type, top_k=top_k)
