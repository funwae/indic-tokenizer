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

# ---------------------------------------------------------------------------
# CBPE merge constraints
# ---------------------------------------------------------------------------

def cbpe_merge_allowed(left: str, right: str) -> bool:
    """
    Decide whether a BPE merge between symbols `left` and `right` is allowed.

    Both `left` and `right` are assumed to be current BPE *symbols*,
    i.e., substrings over Unicode characters, not full tokens yet.

    This is a *minimal* constraint set:

    1. Do NOT allow merges that would produce a symbol starting with a
       dependent vowel sign (matra) or virama. These are combining marks
       and should not appear at the start of a standalone symbol.

       (CBPE idea: avoid tokens that begin with dependent vowels.)

    2. Allow merges that attach a dependent vowel to a preceding consonant
       or cluster (the usual case in Devanagari), as long as the *resulting*
       symbol does NOT start with a dependent vowel/virama.

    You can extend this function with more rules later (e.g., grapheme
    cluster checks, morphology boundaries).
    """

    if not left or not right:
        # Degenerate; let caller decide, but generally BPE shouldn't see empty symbols.
        return False

    merged = left + right
    first_char = merged[0]

    # If the merged symbol is Devanagari and begins with a combining mark
    # (dependent vowel or virama), we disallow it.
    if is_devanagari_char(first_char) and (is_dependent_vowel(first_char) or is_virama(first_char)):
        return False

    # Future extension examples:
    #  - You might disallow merges that create *standalone* dependent vowels,
    #    e.g., if merged is a single dependent vowel with no base.
    #  - You might inspect the *last* char to avoid certain illegal endings.

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

