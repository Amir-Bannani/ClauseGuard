"""Hierarchy tree builder reconstructing nested contract clause trees with Root Stack Reset."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from pdfParser.src.models import Clause, DocumentBlock
    from pdfParser.src.segmenter import ContractSegmenter, NumberingInfo
except ImportError:
    try:
        from ml.src.models import Clause, DocumentBlock
        from ml.src.segmenter import ContractSegmenter, NumberingInfo
    except ImportError:
        from models import Clause, DocumentBlock
        from segmenter import ContractSegmenter, NumberingInfo


class HierarchyTreeBuilder:
    """Builds hierarchical Clause trees with Root Stack Reset rules and parent_id linking."""

    def __init__(self, segmenter: ContractSegmenter | None = None):
        self.segmenter = segmenter or ContractSegmenter()

    def build_tree(self, classified_blocks: list[tuple[DocumentBlock, str]]) -> list[Clause]:
        root_clauses: list[Clause] = []
        clause_stack: list[Clause] = []
        active_heading: Clause | None = None
        current_clause: Clause | None = None

        clause_counter = 1

        for block, role in classified_blocks:
            text = ContractSegmenter.clean_dangling_tail(block.text.strip())
            page = block.page or 1

            if not text:
                continue

            # Filter isolated noise artifacts like "2-A:"
            if len(text) <= 5 and not any(c.isalpha() for c in text):
                if current_clause:
                    current_clause.text += f" {text}"
                continue

            numbering: NumberingInfo | None = self.segmenter.detect_numbering(text)

            is_root_section = False
            if numbering and numbering.level == 1 and numbering.numbering_type in ("decimal", "named_section"):
                is_root_section = True
            elif role == "HEADING" and (not numbering or numbering.level == 1):
                is_root_section = True

            if role == "HEADING" or is_root_section:
                heading_id = numbering.clean_number if (numbering and numbering.clean_number) else f"heading_{len(root_clauses)+1}"
                title = text

                # ROOT STACK RESET RULE: Top-level Level 1 section nodes set parent_id = None
                parent_id = None if is_root_section else (active_heading.id if active_heading else None)
                level = 1 if is_root_section else (numbering.level if numbering else 1)

                heading_clause = Clause(
                    id=heading_id,
                    parent_id=parent_id,
                    level=level,
                    title=title,
                    text=text,
                    page_start=page,
                    page_end=page,
                )

                if parent_id is None:
                    root_clauses.append(heading_clause)
                    clause_stack = [heading_clause]
                else:
                    parent_clause = self._find_clause_by_id(root_clauses, parent_id)
                    if parent_clause:
                        parent_clause.children.append(heading_clause)
                    else:
                        root_clauses.append(heading_clause)
                    while clause_stack and clause_stack[-1].level >= level:
                        clause_stack.pop()
                    clause_stack.append(heading_clause)

                active_heading = heading_clause
                current_clause = heading_clause

            elif role in ("CLAUSE_START", "LIST_ITEM"):
                raw_id = numbering.clean_number if (numbering and numbering.clean_number) else f"clause_{clause_counter}"
                clause_counter += 1

                level = numbering.level if numbering else 2

                # ROOT STACK RESET for top-level clauses
                if level == 1 and numbering and numbering.numbering_type in ("decimal", "named_section"):
                    parent_id = None
                    clause_id = raw_id
                else:
                    parent_id = self._determine_parent_id(raw_id, level, clause_stack, active_heading)
                    if parent_id and not raw_id.startswith(parent_id) and ("." not in raw_id):
                        clause_id = f"{parent_id}({raw_id})" if not raw_id.startswith("(") else f"{parent_id}{raw_id}"
                    else:
                        clause_id = raw_id

                clause = Clause(
                    id=clause_id,
                    parent_id=parent_id,
                    level=level,
                    title=None,
                    text=text,
                    page_start=page,
                    page_end=page,
                )

                if parent_id is None:
                    root_clauses.append(clause)
                    clause_stack = [clause]
                else:
                    parent_clause = self._find_clause_by_id(root_clauses, parent_id)
                    if parent_clause:
                        parent_clause.children.append(clause)
                    else:
                        root_clauses.append(clause)

                    while clause_stack and clause_stack[-1].level >= level:
                        clause_stack.pop()
                    clause_stack.append(clause)

                current_clause = clause

            elif role == "CONTINUATION":
                if current_clause:
                    current_clause.text += f"\n{text}"
                    current_clause.page_end = page

            elif role == "OTHER":
                if current_clause and len(text) > 0:
                    current_clause.page_end = page

        self._cleanup_tree(root_clauses)
        return root_clauses

    def _determine_parent_id(
        self,
        clause_id: str,
        level: int,
        stack: list[Clause],
        active_heading: Clause | None,
    ) -> str | None:
        if "." in clause_id:
            parts = clause_id.split(".")
            return ".".join(parts[:-1])

        for ancestor in reversed(stack):
            if ancestor.level < level:
                return ancestor.id

        if active_heading and active_heading.level < level:
            return active_heading.id

        return None

    def _find_clause_by_id(self, clauses: list[Clause], clause_id: str) -> Clause | None:
        for c in clauses:
            if c.id == clause_id:
                return c
            found = self._find_clause_by_id(c.children, clause_id)
            if found:
                return found
        return None

    def _cleanup_tree(self, clauses: list[Clause]) -> None:
        for c in clauses:
            c.text = ContractSegmenter.clean_dangling_tail(c.text)
            if c.children:
                self._cleanup_tree(c.children)
