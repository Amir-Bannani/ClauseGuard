"""ClauseSegmenter: DFS tree walker extracting clean Clause[] array with section context."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from ml.src.models import Clause, ExtractedClause
except ImportError:
    from models import Clause, ExtractedClause

SECTION_TITLE_STRIP_REGEX = re.compile(
    r"^(?:ARTICLE|SECTION|SUB-SECTION|SUBSECTION|CLAUSE|SUB-CLAUSE|SUBCLAUSE)?\s*(?:\d+|[IVXLCDM]+)[\.\:\-]?\s*",
    re.IGNORECASE,
)


class ClauseSegmenter:
    """Walks the Clause hierarchy tree using DFS to separate headings from clauses and produce a clean Clause[] array."""

    def segment_tree(self, tree_nodes: list[Clause]) -> list[ExtractedClause]:
        extracted_clauses: list[ExtractedClause] = []

        def walk(node: Clause, current_section_id: str | None = None, current_section_title: str | None = None):
            # 1. Update section context if node represents a Section / Heading
            if self._is_section(node):
                current_section_id = node.id
                current_section_title = self._extract_section_title(node.text)

            # 2. Extract Clause if node represents an actual contractual clause
            if self._is_clause(node):
                extracted_clauses.append(
                    ExtractedClause(
                        clause_id=node.id,
                        section_id=current_section_id,
                        section_title=current_section_title,
                        number=node.id,
                        text=node.text.strip(),
                        page_start=node.page_start,
                        page_end=node.page_end,
                    )
                )

            # 3. Recursive DFS traversal over child nodes
            for child in node.children:
                walk(child, current_section_id, current_section_title)

        for root_node in tree_nodes:
            walk(root_node)

        return extracted_clauses

    def _is_section(self, node: Clause) -> bool:
        """Determines if a node is a Section/Heading node (e.g. '10. Termination')."""
        if node.level == 1:
            return True
        if node.children and not self._is_clause_text(node.text):
            return True
        return False

    def _is_clause(self, node: Clause) -> bool:
        """Determines if a node represents an actual contractual clause (e.g. '10.1 Either party...')."""
        # A Level 1 section header without specific clause text is a Section, not a Clause
        if node.level == 1 and not (node.id and "." in node.id and node.id != "1."):
            # Check if text is just a short heading title
            text = node.text.strip()
            if len(text) < 60 and not text.endswith("."):
                return False

        return self._is_clause_text(node.text)

    def _is_clause_text(self, text: str) -> bool:
        """Checks if text contains actual contractual prose sentences."""
        clean_text = text.strip()
        if not clean_text:
            return False
        # Clause texts typically have prose sentences or length > 25
        return len(clean_text) > 25 or clean_text.endswith(".") or "shall" in clean_text.lower() or "agrees" in clean_text.lower()

    def _extract_section_title(self, text: str) -> str:
        """Extract clean section title string (e.g. '10. Termination' -> 'Termination')."""
        clean_title = SECTION_TITLE_STRIP_REGEX.sub("", text.strip()).strip()
        return clean_title if clean_title else text.strip()
