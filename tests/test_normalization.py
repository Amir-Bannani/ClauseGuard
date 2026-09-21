"""Tests for conservative text normalization."""

from __future__ import annotations

import pytest

from ml.src.normalization import normalize_text


def test_whitespace_collapsing() -> None:
    raw = "  This   is  a\n\ttest \r\n clause.   "
    expected = "this is a test clause."
    assert normalize_text(raw) == expected


def test_casing_normalization() -> None:
    raw = "Governing Law AND Jurisdiction"
    expected = "governing law and jurisdiction"
    assert normalize_text(raw) == expected


def test_unicode_nfkc_normalization() -> None:
    # 'ﬁ' ligature (\uFB01) decomposed to 'fi'
    raw = "The ﬁnancial statement"
    assert normalize_text(raw) == "the financial statement"


def test_typographic_quotes_normalization() -> None:
    raw = "“Double quote” and ‘single quote’ and «guillemets»"
    expected = '"double quote" and \'single quote\' and "guillemets"'
    assert normalize_text(raw) == expected


def test_dash_variants_normalization() -> None:
    # em dash, en dash, minus sign
    raw = "Clause A—Part 1–Subpart 2−Note 3"
    expected = "clause a-part 1-subpart 2-note 3"
    assert normalize_text(raw) == expected


def test_preservation_of_digits_and_legal_tokens() -> None:
    tokens = [
        "30 days",
        "90 days",
        "5 years",
        "10%",
        "Section 12",
        "Article 4.2(b)",
        "Title 17 CFR § 240.12b-2",
        "$1,000,000",
    ]
    for token in tokens:
        normalized = normalize_text(token)
        # Verify that all original digits are preserved in sequence
        original_digits = [c for c in token if c.isdigit()]
        norm_digits = [c for c in normalized if c.isdigit()]
        assert original_digits == norm_digits, f"Failed for token: {token}"


def test_determinism() -> None:
    text = "Section 4.3 — Notice: Within 15 business days, party “A” shall notify party ‘B’."
    first = normalize_text(text)
    for _ in range(10):
        assert normalize_text(text) == first
