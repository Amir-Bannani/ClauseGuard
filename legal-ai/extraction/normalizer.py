"""
ClauseGuard — Duration Normalizer & Metadata Enrichment.
Parses temporal expressions into structured value/unit metadata while keeping raw legal text.
"""

import re
from typing import Dict, Any, Optional


class Normalizer:
    """
    Parses raw duration strings into structured numeric value and temporal unit metadata.
    """

    WORD_TO_NUM = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "twelve": 12, "eighteen": 18, "twenty-four": 24, "thirty": 30,
        "sixty": 60, "ninety": 90
    }

    @classmethod
    def parse_duration(cls, raw_val: Any) -> Optional[Dict[str, Any]]:
        if not raw_val or not isinstance(raw_val, str):
            return None
        text = raw_val.lower().strip()
        if text == "immediately":
            return {"value": 0, "unit": "day"}

        num = None
        match_digit = re.search(r'\b(\d+)\b', text)
        if match_digit:
            num = int(match_digit.group(1))
        else:
            for word, val in cls.WORD_TO_NUM.items():
                if word in text:
                    num = val
                    break
        unit = None
        if "year" in text:
            unit = "year"
        elif "month" in text:
            unit = "month"
        elif "day" in text:
            unit = "day"

        if num is not None and unit is not None:
            return {"value": num, "unit": unit}
        return None

    @classmethod
    def normalize_extraction(cls, category: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(extraction)
        for field in ["duration", "notice_period"]:
            if field in normalized and normalized[field] is not None:
                raw_str = normalized[field]
                parsed = cls.parse_duration(raw_str)
                normalized[field] = {
                    "raw": raw_str,
                    "normalized": parsed
                }
        return normalized
