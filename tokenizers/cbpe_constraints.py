# tokenizers/cbpe_constraints.py
# -*- coding: utf-8 -*-
"""
Minimal Constrained BPE (CBPE) constraint hook for Devanagari.

This module implements a small set of script-aware rules that can be
plugged into a BPE/Unigram tokenizer trainer to avoid obviously
bad merges for Devanagari (Hindi/Sanskrit).

Inspired by MorphTok's Constrained BPE idea for handling dependent
vowels and script peculiarities in Indic languages. See:

- "MorphTok: Morphologically Grounded Tokenization for Indian Languages"
  (Brahma et al., 2025)
"""

from typing import Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Unicode helpers for Devanagari
# ---------------------------------------------------------------------------

# Basic Devanagari block
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# Dependent vowel signs: U+093E..U+094C (AA..AU) in Devanagari.
DEPENDENT_VOWEL_START = 0x093E
DEPENDENT_VOWEL_END = 0x094C

# Virama (halant) U+094D.
VIRAMA = "\u094D"

# Nukta U+093C
NUKTA = "\u093C"

# Combining marks: U+0900–U+0903 (various signs), U+0951–U+0954 (various marks)
COMBINING_MARK_RANGES = [
    (0x0900, 0x0903),  # Devanagari sign Inverted Candrabindu, Candrabindu, Anusvara, Visarga
    (0x0951, 0x0954),  # Devanagari stress signs and other marks
]

# Set of all Devanagari combining marks (dependent vowels, virama, nukta, other combining marks)
DEVANAGARI_COMBINING = set()

# Add dependent vowels
for cp in range(DEPENDENT_VOWEL_START, DEPENDENT_VOWEL_END + 1):
    DEVANAGARI_COMBINING.add(cp)

# Add virama and nukta
DEVANAGARI_COMBINING.add(ord(VIRAMA))
DEVANAGARI_COMBINING.add(ord(NUKTA))

# Add other combining marks
for start, end in COMBINING_MARK_RANGES:
    for cp in range(start, end + 1):
        DEVANAGARI_COMBINING.add(cp)

def is_devanagari_char(ch: str) -> bool:
    """Return True if ch is in the basic Devanagari block."""
    if not ch:
        return False
    cp = ord(ch)
    return DEVANAGARI_START <= cp <= DEVANAGARI_END

def is_dependent_vowel(ch: str) -> bool:
    """Return True if ch is a Devanagari dependent vowel sign (matra)."""
    if not ch:
        return False
    cp = ord(ch)
    return DEPENDENT_VOWEL_START <= cp <= DEPENDENT_VOWEL_END

def is_virama(ch: str) -> bool:
    """Return True if ch is the Devanagari sign virama (halant)."""
    return ch == VIRAMA

def is_nukta(ch: str) -> bool:
    """Return True if ch is the Devanagari sign nukta."""
    return ch == NUKTA

def is_devanagari_combining(ch: str) -> bool:
    """
    Return True if ch is a Devanagari combining mark.

    Includes:
    - Dependent vowel signs (matras): U+093E–U+094C
    - Virama (halant): U+094D
    - Nukta: U+093C
    - Other combining marks: U+0900–U+0903, U+0951–U+0954
    """
    if not ch:
        return False
    cp = ord(ch)
    return cp in DEVANAGARI_COMBINING

# ---------------------------------------------------------------------------
# CBPE merge constraints
# ---------------------------------------------------------------------------

def cbpe_merge_allowed(left: str, right: str) -> bool:
    """
    Decide whether a BPE merge between symbols `left` and `right` is allowed.

    Both `left` and `right` are assumed to be current BPE *symbols*,
    i.e., substrings over Unicode characters, not full tokens yet.

    MorphTok-style constraints for Devanagari:

    1. Do NOT allow merges where `right` begins with a combining mark:
       - Dependent vowel signs (matras): U+093E–U+094C
       - Virama (halant): U+094D
       - Nukta: U+093C
       - Other combining marks: U+0900–U+0903, U+0951–U+0954

    2. Do NOT allow merges that would create a symbol starting with a
       combining mark (the merged result should not start with one).

    3. Conservative approach: Better to skip some legitimate merges than
       to create tokens that mutilate script structure.

    Examples of disallowed merges:
    - ("क", "ि") - would create token starting with dependent vowel
    - ("त", "्") - would create token starting with virama
    - ("क्", "्ष") - if it would split expected aksharas

    Examples of allowed merges (conservative):
    - ("क", "ा") → "का" - only when it's the intended akshara
    """

    if not left or not right:
        # Degenerate; let caller decide, but generally BPE shouldn't see empty symbols.
        return False

    # Check if right begins with a combining mark - disallow such merges
    if right and is_devanagari_combining(right[0]):
        return False

    # Check if the merged result would start with a combining mark - disallow
    merged = left + right
    if merged and is_devanagari_char(merged[0]) and is_devanagari_combining(merged[0]):
        return False

    # Additional conservative check: if right is a single combining mark, disallow
    # (combining marks should not be standalone tokens)
    if len(right) == 1 and is_devanagari_combining(right):
        return False

    return True

def filter_bpe_merges(
    merges: Iterable[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Filter a sequence of BPE merge candidates (left,right) using
    cbpe_merge_allowed.

    This is a generic helper for a home-grown BPE implementation.

    Example:

        base_merges = [('क', '्'), ('्', 'ष'), ('क्', 'ष'), ('्', 'ा')]
        constrained_merges = filter_bpe_merges(base_merges)

    Any merge that would result in a symbol starting with a Devanagari
    dependent vowel or virama will be removed.
    """

    allowed: List[Tuple[str, str]] = []

    for left, right in merges:
        if cbpe_merge_allowed(left, right):
            allowed.append((left, right))

    return allowed

