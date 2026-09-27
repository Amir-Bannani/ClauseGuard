"""Structural boundary and numbering segmenter for contract text blocks."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from pdfParser.src.models import DocumentBlock
except ImportError:
    try:
        from ml.src.models import DocumentBlock
    except ImportError:
        from models import DocumentBlock


@dataclass
class NumberingInfo:
    raw: str
    level: int
    numbering_type: str  # "decimal", "named_section", "parenthetical", "letter", "roman", "bullet"
    clean_number: str = ""
    prefix: str = ""


# Regex patterns for contract section and clause numbering schemes
NAMED_SECTION_PATTERN = re.compile(
    r"^(?P<prefix>ARTICLE|SECTION|SUB-SECTION|SUBSECTION|CLAUSE|SUB-CLAUSE|SUBCLAUSE|PARAGRAPH|SUB-PART|ITEM|SCHEDULE|EXHIBIT)\s+"
    r"(?P<num>[IVXLCDM\d]+(?:[\.\-][a-zA-Z0-9]+)*)[:\.-]?\s*",
    re.IGNORECASE,
)

DECIMAL_PATTERN = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?|\d+[\.\-][a-zA-Z0-9]+)\.?(?=\s+[A-Z0-9\"'“\(]|\s*$|:|\s+--|\s*—)",
)

PARENTHETICAL_PATTERN = re.compile(
    r"^\((?P<num>[a-z]{1,3}|[A-Z]{1,3}|\d+|[ivxlcdmIVXLCDM]+)\)\s*",
)

RIGHT_PAREN_PATTERN = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*|[a-z]{1,3}|[ivxlcdmIVXLCDM]+)\)\s*",
)

BULLET_PATTERN = re.compile(
    r"^[•\-\*\u2022\u2023\u25b6\u25c0\u25ba\u25c4]\s+",
)

# Intra-block clause boundary regex matching clause headers embedded inside text
INTRA_BLOCK_CLAUSE_REGEX = re.compile(
    r"(?:^|[\.\s\:\-\—\)])\s*"
    r"("
    r"(?:ARTICLE|SECTION|SUB-SECTION|SUBSECTION|CLAUSE|SUB-CLAUSE|SUBCLAUSE|Paragraph|Sub-part|Item)\s+[IVXLCDM\d]+(?:[\.\-][a-zA-Z0-9]+)*[:\.-]?"
    r"|"
    r"\d+\.(?=\s+[A-Z])"
    r"|"
    r"\d+\.\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?"
    r"|"
    r"\d+[\.\-][a-zA-Z0-9]+"
    r"|"
    r"\([a-zA-Z0-9]{1,3}\)"
    r")"
    r"(?=\s+[A-Z0-9\"'“\(]|\s*:|\s*--|\s*—)"
)

# Non-backtracking linear-time regex for trailing dangling layout keywords
DANGLING_TAIL_REGEX = re.compile(
    r"\s+\b(Paragraph|Sub-part|Item|Clause|Sub-clause|Section|ARTICLE)(?:[\s\.\:\-]*[0-9a-zA-Z\(\)]*)*[:\.-]?$",
    re.IGNORECASE,
)

# Signature block regex
SIGNATURE_PATTERN = re.compile(
    r"^(IN\s+WITNESS\s+WHEREOF|FOR\s+THE\s+EMPLOYER|FOR\s+THE\s+EMPLOYEE|BY:|SIGNATURE:|NAME:|TITLE:|DATE:|EXECUTED\s+AS\s+A|SIGNED\s+BY)",
    re.IGNORECASE,
)

ROMAN_NUMERALS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
CONTRACT_KEYWORDS = {"agreement", "witnesseth", "recitals", "definitions", "schedule", "exhibit"}


class ContractSegmenter:
    """Detects structural signals, splits inline sub-clauses, and classifies block roles."""

    @staticmethod
    def detect_numbering(text: str) -> NumberingInfo | None:
        if not text or not text.strip():
            return None

        clean_text = text.strip()

        # 1. Named Section (e.g., ARTICLE 1, SECTION 2:, Clause 3.c, Item 11.a)
        named_match = NAMED_SECTION_PATTERN.match(clean_text)
        if named_match:
            prefix = named_match.group("prefix").upper()
            num_str = named_match.group("num")
            dots = num_str.count(".") + num_str.count("-")
            level = 1 if dots == 0 else dots + 1
            if prefix in ("ARTICLE", "SECTION") and level == 1:
                level = 1
            return NumberingInfo(
                raw=named_match.group(0).strip(),
                level=level,
                numbering_type="named_section",
                clean_number=num_str,
                prefix=prefix,
            )

        # 2. Decimal Numbering (e.g., 1., 1.1, 2.1.1, 1.1(a))
        decimal_match = DECIMAL_PATTERN.match(clean_text)
        if decimal_match:
            num_str = decimal_match.group("num")
            dots = num_str.count(".")
            has_paren = "(" in num_str
            level = (1 if dots == 0 else dots + 1) + (1 if has_paren else 0)
            return NumberingInfo(
                raw=decimal_match.group(0).strip(),
                level=level,
                numbering_type="decimal",
                clean_number=num_str,
            )

        # 3. Parenthetical Enclosed (e.g., (a), (A), (i))
        paren_match = PARENTHETICAL_PATTERN.match(clean_text)
        if paren_match:
            num_str = paren_match.group("num")
            num_lower = num_str.lower()
            if num_lower in ROMAN_NUMERALS:
                level = 4
                num_type = "roman"
            elif num_str.isalpha():
                level = 3
                num_type = "letter"
            else:
                level = 3
                num_type = "parenthetical"
            return NumberingInfo(
                raw=paren_match.group(0).strip(),
                level=level,
                numbering_type=num_type,
                clean_number=num_str,
            )

        # 4. Right Parenthesis (e.g., 1), a), i))
        rparen_match = RIGHT_PAREN_PATTERN.match(clean_text)
        if rparen_match:
            num_str = rparen_match.group("num")
            num_lower = num_str.lower()
            dots = num_str.count(".")
            if dots > 0:
                level = dots + 1
            elif num_lower in ROMAN_NUMERALS:
                level = 4
            elif num_str.isalpha():
                level = 3
            else:
                level = 2
            return NumberingInfo(
                raw=rparen_match.group(0).strip(),
                level=level,
                numbering_type="right_parenthesis",
                clean_number=num_str,
            )

        # 5. Bullet points
        bullet_match = BULLET_PATTERN.match(clean_text)
        if bullet_match:
            return NumberingInfo(
                raw=bullet_match.group(0).strip(),
                level=3,
                numbering_type="bullet",
                clean_number="•",
            )

        return None

    @staticmethod
    def split_inline_clauses(text: str) -> list[str]:
        """Split a concatenated block string into separate sub-clause chunks if multiple clause starts exist."""
        if not text or not text.strip():
            return []

        matches = list(INTRA_BLOCK_CLAUSE_REGEX.finditer(text))
        if not matches or len(matches) <= 1:
            return [text.strip()]

        chunks: list[str] = []
        last_idx = 0
        for m in matches:
            start = m.start(1)
            if start > last_idx:
                prev_chunk = text[last_idx:start].strip()
                if prev_chunk:
                    chunks.append(prev_chunk)
            last_idx = start

        if last_idx < len(text):
            final_chunk = text[last_idx:].strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks if chunks else [text.strip()]

    @staticmethod
    def clean_dangling_tail(text: str) -> str:
        """Strip trailing dangling layout keywords and truncated cross-references in linear time."""
        if not text or len(text) < 5:
            return text
        match = DANGLING_TAIL_REGEX.search(text)
        if match:
            matched_str = match.group(0)
            if len(matched_str) < 40:
                return text[:match.start()].strip()
        return text.strip()

    def classify_role(
        self,
        block: DocumentBlock,
        last_role: str | None = None,
        median_font_size: float = 12.0,
    ) -> str:
        """Classify block into role: HEADING, CLAUSE_START, CONTINUATION, LIST_ITEM, or OTHER."""
        text = block.text.strip()
        if not text:
            return "OTHER"

        if SIGNATURE_PATTERN.search(text) or "____" in text:
            return "OTHER"

        numbering = self.detect_numbering(text)

        # Heading detection heuristic
        body_text = text[len(numbering.raw):].strip() if numbering else text
        is_upper = body_text.isupper() and len(body_text) > 2
        is_large_font = block.font_size and block.font_size > median_font_size * 1.08
        is_bold_title = block.bold and len(body_text) <= 60 and not body_text.endswith(".")
        is_contract_kw = body_text.lower() in CONTRACT_KEYWORDS

        if is_upper or is_large_font or is_bold_title or is_contract_kw:
            if not numbering or numbering.numbering_type == "named_section" or numbering.level == 1:
                return "HEADING"

        if numbering:
            if numbering.numbering_type == "bullet":
                return "LIST_ITEM"
            return "CLAUSE_START"

        if last_role in ("HEADING", "CLAUSE_START", "CONTINUATION", "LIST_ITEM"):
            if not text.isupper() and len(text) > 10:
                return "CONTINUATION"

        return "OTHER"
