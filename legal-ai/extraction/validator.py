"""
ClauseGuard — Safe Formatting-Only Extraction Validator.
Validates JSON syntax and repairs formatting without altering legal meaning.
"""

import re
import json
from typing import Dict, Any, List, Tuple


class ExtractionValidator:
    """
    Safe Formatting Repair Layer.
    Splits stringified list items without legal inference.
    """

    @staticmethod
    def validate_and_repair(raw_output: str, expected_schema: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        errors = []
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            errors.append(f"JSON Parse Failure: {str(e)}")
            return False, expected_schema, errors

        if not isinstance(parsed, dict):
            errors.append(f"Root element is {type(parsed).__name__}, expected dict.")
            return False, expected_schema, errors

        validated_dict = {}
        for key, expected_default in expected_schema.items():
            if key not in parsed:
                errors.append(f"Missing schema key '{key}'.")
                validated_dict[key] = expected_default
            else:
                val = parsed[key]
                if isinstance(expected_default, list):
                    if isinstance(val, list):
                        validated_dict[key] = val
                    elif isinstance(val, str) and val.strip():
                        split_items = [x.strip() for x in re.split(r',|\band\b', val) if x.strip()]
                        validated_dict[key] = split_items
                    else:
                        validated_dict[key] = []
                else:
                    validated_dict[key] = val

        return len(errors) == 0, validated_dict, errors
