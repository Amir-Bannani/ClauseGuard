"""Deterministic, conservative text normalization for dataset audits and exact overlap checks.

This module provides conservative normalization that avoids aggressive rewriting:
- Unicode NFKC normalization
- Case normalization (via casefold)
- Whitespace collapsing
- Conservative typographic quote unification
- Conservative dash variant unification

CRITICAL: Digits and legally meaningful tokens (such as '30 days', '5 years',
'10%', 'Section 12', 'Article 4.2') are strictly preserved.
"""

from __future__ import annotations

import re
import unicodedata

# Typographic quotes to standard ASCII equivalents
QUOTE_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "`": "'",
    "´": "'",
})

# Typographic dash variants to standard ASCII hyphen-minus
DASH_TRANSLATION = str.maketrans({
    "—": "-",  # em dash \u2014
    "–": "-",  # en dash \u2013
    "−": "-",  # minus sign \u2212
    "‒": "-",  # figure dash \u2012
    "―": "-",  # horizontal bar \u2015
})

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Deterministically normalize contract clause text for overlap analysis.

    Parameters
    ----------
    text : str
        Input clause string.

    Returns
    -------
    str
        Conservatively normalized text string with preserved digits,
        unified quotes and dashes, folded casing, and collapsed whitespace.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode NFKC normalization (decomposes ligatures like 'fi' -> 'f' + 'i')
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Unify typographic quotes
    normalized = normalized.translate(QUOTE_TRANSLATION)

    # 3. Unify dash variants
    normalized = normalized.translate(DASH_TRANSLATION)

    # 4. Case folding for robust case-insensitive comparison
    normalized = normalized.casefold()

    # 5. Collapse all whitespace sequences into a single space and strip edges
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()

    return normalized
