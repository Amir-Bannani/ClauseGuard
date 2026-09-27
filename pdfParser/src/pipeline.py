"""Streamlined Contract Parser Pipeline orchestrating layout extraction, segmentation, tree building, and clause extraction."""

from __future__ import annotations

import os
import statistics
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
    from pdfParser.src.clause_extractor import ClauseSegmenter
    from pdfParser.src.extractor import DocumentExtractor
    from pdfParser.src.models import Clause, DocumentBlock, ExtractedClause
    from pdfParser.src.segmenter import ContractSegmenter
    from pdfParser.src.tree_builder import HierarchyTreeBuilder
except ImportError:
    try:
        from ml.src.clause_extractor import ClauseSegmenter
        from ml.src.extractor import DocumentExtractor
        from ml.src.models import Clause, DocumentBlock, ExtractedClause
        from ml.src.segmenter import ContractSegmenter
        from ml.src.tree_builder import HierarchyTreeBuilder
    except ImportError:
        from clause_extractor import ClauseSegmenter
        from extractor import DocumentExtractor
        from models import Clause, DocumentBlock, ExtractedClause
        from segmenter import ContractSegmenter
        from tree_builder import HierarchyTreeBuilder


def _count_nodes(nodes: list[Clause]) -> int:
    total = 0
    for node in nodes:
        total += 1
        if node.children:
            total += _count_nodes(node.children)
    return total


class ContractParserPipeline:
    """Orchestrates layout extraction, structural segmentation, hierarchy tree building, and clause extraction."""

    def __init__(self):
        self.extractor = DocumentExtractor()
        self.segmenter = ContractSegmenter()
        self.tree_builder = HierarchyTreeBuilder(segmenter=self.segmenter)
        self.clause_segmenter = ClauseSegmenter()

    def parse_file(self, file_path: str) -> dict[str, Any]:
        blocks = self.extractor.extract(file_path)
        return self.parse_blocks(blocks, document_name=os.path.basename(file_path))

    def parse_text(self, text: str, document_name: str = "plain_text") -> dict[str, Any]:
        lines = text.splitlines()
        blocks = [DocumentBlock(text=line.strip(), page=1) for line in lines if line.strip()]
        return self.parse_blocks(blocks, document_name=document_name)

    def parse_blocks(self, blocks: list[DocumentBlock], document_name: str = "document") -> dict[str, Any]:
        if not blocks:
            return {
                "document": document_name,
                "total_blocks": 0,
                "total_nodes": 0,
                "total_clauses": 0,
                "tree": [],
                "clauses": [],
                "debug_blocks": [],
            }

        # 1. Split inline sub-clauses
        expanded_blocks: list[DocumentBlock] = []
        for b in blocks:
            sub_chunks = self.segmenter.split_inline_clauses(b.text)
            if len(sub_chunks) <= 1:
                expanded_blocks.append(b)
            else:
                for chunk in sub_chunks:
                    expanded_blocks.append(
                        DocumentBlock(
                            text=chunk,
                            page=b.page,
                            font_size=b.font_size,
                            bold=b.bold,
                            italic=b.italic,
                            alignment=b.alignment,
                            indent_left=b.indent_left,
                        )
                    )

        # 2. Calculate median font size
        font_sizes = [b.font_size for b in expanded_blocks if b.font_size is not None]
        median_font_size = statistics.median(font_sizes) if font_sizes else 12.0

        # 3. Structural role classification
        classified_blocks: list[tuple[DocumentBlock, str]] = []
        last_role: str | None = None

        for b in expanded_blocks:
            role = self.segmenter.classify_role(b, last_role=last_role, median_font_size=median_font_size)
            classified_blocks.append((b, role))
            if role in ("HEADING", "CLAUSE_START", "CONTINUATION", "LIST_ITEM"):
                last_role = role

        # 4. Tree building
        clause_tree: list[Clause] = self.tree_builder.build_tree(classified_blocks)
        total_nodes = _count_nodes(clause_tree)

        # 5. ClauseSegmenter DFS tree walker: Segment Tree into Clause[] with section_id and section_title
        extracted_clauses: list[ExtractedClause] = self.clause_segmenter.segment_tree(clause_tree)

        debug_blocks = [
            {
                "role": role,
                "text": b.text[:100] + ("…" if len(b.text) > 100 else ""),
                "page": b.page,
            }
            for b, role in classified_blocks
        ]

        return {
            "document": document_name,
            "total_blocks": len(expanded_blocks),
            "total_nodes": total_nodes,
            "total_clauses": len(extracted_clauses),
            "tree": [c.to_dict(include_children=True) for c in clause_tree],
            "clauses": [c.to_dict() for c in extracted_clauses],
            "debug_blocks": debug_blocks,
        }
