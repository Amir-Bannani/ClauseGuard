"""ClauseSegmenter: DFS tree walker extracting clean Clause[] array with section context."""

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
    from pdfParser.src.models import Clause, ExtractedClause
except ImportError:
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
                heading_line = node.title or (node.text.split("\n")[0] if node.text else "")
                current_section_title = self._extract_section_title(heading_line)

            # 2. Extract Clause if node represents an actual contractual clause
            if self._is_clause(node):
                number_val = None if (node.id and node.id.startswith("heading_")) else node.id
                clean_text = self._clean_clause_text(node.text, number_val, current_section_title)
                extracted_clauses.append(
                    ExtractedClause(
                        clause_id=node.id,
                        section_id=current_section_id,
                        section_title=current_section_title,
                        number=number_val,
                        text=clean_text,
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

    def _clean_clause_text(self, raw_text: str, number: str | None, section_title: str | None) -> str:
        """Strips leading clause numbers and repeating section title lines from contractual prose."""
        text = raw_text.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        if not lines:
            return ""

        # 1. Strip repeating heading line if first line is the section title
        if len(lines) > 1:
            first_line = lines[0]
            clean_first = SECTION_TITLE_STRIP_REGEX.sub("", first_line).strip()
            if section_title and (clean_first.lower() == section_title.lower() or first_line.lower() == section_title.lower()):
                lines = lines[1:]

        text = "\n".join(lines).strip()

        # 2. Strip leading clause number prefix (e.g. "1.1 ", "10.2 ", "1. ", "(a) ") from start of text
        if number:
            pattern = r"^\s*" + re.escape(number) + r"[\.\:\-]?\s*"
            text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()

        return text if text else raw_text.strip()

    def _is_section(self, node: Clause) -> bool:
        """Determines if a node is a Section/Heading node (e.g. '10. Termination' or 'POSITION')."""
        if node.level == 1:
            return True
        if node.children and not self._is_clause_text(node.text):
            return True
        return False

    def _is_clause(self, node: Clause) -> bool:
        """Determines if a node represents an actual contractual clause (numbered or unnumbered)."""
        # Filter unnumbered document preamble headers or signature blocks (heading_*)
        if node.id and node.id.startswith("heading_"):
            text_upper = node.text.upper().strip()
            if (
                "SIGNATURE" in text_upper
                or "EMPLOYER:" in text_upper
                or "EMPLOYEE:" in text_upper
                or "BY SIGNING BELOW" in text_upper
                or text_upper.startswith("EMPLOYMENT AGREEMENT")
                or text_upper.startswith("EMPLOYMENT CONTRACT")
                or text_upper.startswith("THIS AGREEMENT")
                or text_upper.startswith("THIS CONTRACT")
            ):
                return False

            lines = [l for l in node.text.strip().split("\n") if l.strip()]
            if len(lines) <= 1:
                return False

        if node.level == 1 and not (node.id and "." in node.id and node.id != "1."):
            text = node.text.strip()
            first_line = text.split("\n")[0]
            if len(first_line) < 60 and not first_line.endswith(".") and len(text.split("\n")) <= 1:
                return False

        return self._is_clause_text(node.text)

    def _is_clause_text(self, text: str) -> bool:
        """Checks if text contains actual contractual prose sentences."""
        clean_text = text.strip()
        if not clean_text:
            return False
        return len(clean_text) > 25 or clean_text.endswith(".") or "shall" in clean_text.lower() or "agrees" in clean_text.lower()

    def _extract_section_title(self, text: str) -> str:
        """Extract clean section title string (e.g. '10. Termination' -> 'Termination')."""
        clean_title = SECTION_TITLE_STRIP_REGEX.sub("", text.strip()).strip()
        return clean_title if clean_title else text.strip()
