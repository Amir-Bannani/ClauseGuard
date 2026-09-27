"""Data models for document blocks, hierarchy trees, and extracted clauses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentBlock:
    """Represents a line/paragraph extracted from a document with layout metadata."""

    text: str
    page: int = 1
    font_size: float | None = None
    bold: bool = False
    italic: bool = False
    alignment: str = "left"
    indent_left: float = 0.0


@dataclass
class Clause:
    """Hierarchical legal clause tree representation."""

    id: str
    parent_id: str | None = None
    level: int = 1
    title: str | None = None
    text: str = ""
    page_start: int = 1
    page_end: int = 1
    children: list[Clause] = field(default_factory=list)

    def to_dict(self, include_children: bool = True) -> dict[str, Any]:
        """Convert Clause instance to clean JSON dictionary (tree format)."""
        data: dict[str, Any] = {
            "id": self.id,
            "parent_id": self.parent_id,
            "level": self.level,
            "title": self.title,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
        if include_children and self.children:
            data["children"] = [child.to_dict(include_children=True) for child in self.children]
        return data


@dataclass
class ExtractedClause:
    """Flat extracted clause with section context for RAG and downstream analysis."""

    clause_id: str
    section_id: str | None
    section_title: str | None
    number: str | None
    text: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "number": self.number,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
