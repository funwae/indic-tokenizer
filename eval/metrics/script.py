# eval/metrics/script.py
# -*- coding: utf-8 -*-
"""
Script and grapheme adequacy metrics for tokenization evaluation.

Measures how well tokenization respects Devanagari script structure:
grapheme clusters, aksharas, and dependent vowels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from eval.grapheme_violations import violation_rate
from tokenizers.cbpe_constraints import (
    DEVANAGARI_END,
    DEVANAGARI_START,
    is_dependent_vowel,
    is_virama,
)
from tokenizers.grapheme_segmenter import segment_devanagari_graphemes


@dataclass
class ScriptMetrics:
    """Script adequacy metrics for tokenization."""

    grapheme_violation_rate: float
    akshara_integrity_rate: float
    akshara_split_count: int
    dependent_vowel_split_rate: float
    grapheme_aligned_token_rate: float
    single_grapheme_tokens: int
    multi_grapheme_tokens: int
    grapheme_fragment_tokens: int
    devanagari_token_share: float
    mixed_script_token_share: float
    akshara_segmentation_version: str  # "v0_heuristic"


# Unicode ranges for Devanagari
DEVANAGARI_CONSONANT_START = 0x0915  # क
DEVANAGARI_CONSONANT_END = 0x0939  # ह


def is_devanagari_consonant(char: str) -> bool:
    """Check if character is a Devanagari consonant."""
    if not char:
        return False
    cp = ord(char[0])  # Check first character
    return DEVANAGARI_CONSONANT_START <= cp <= DEVANAGARI_CONSONANT_END


def segment_aksharas(text: str) -> List[str]:
    """
    Segment text into aksharas using v0 heuristic.

    Akshara pattern: Base consonant (optionally + virama + consonant) + dependent vowels + diacritics

    This is a v0 heuristic built on top of grapheme clusters. It may not
    handle all complex conjuncts perfectly, but should catch most common cases.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of akshara strings.
    """
    # Start with grapheme clusters
    graphemes = segment_devanagari_graphemes(text, keep_non_devanagari=False)

    if not graphemes:
        return []

    aksharas: List[str] = []
    current_akshara: List[str] = []
    i = 0

    while i < len(graphemes):
        grapheme = graphemes[i]

        # Check if this grapheme starts a new akshara
        # Pattern: consonant or dependent vowel at start
        first_char = grapheme[0] if grapheme else ""

        if is_devanagari_consonant(first_char):
            # Start of new akshara: consonant
            if current_akshara:
                aksharas.append("".join(current_akshara))
                current_akshara = []

            current_akshara.append(grapheme)
            i += 1

            # Look ahead for virama + consonant (conjunct)
            if i < len(graphemes):
                next_grapheme = graphemes[i]
                if next_grapheme and is_virama(next_grapheme[0]):
                    current_akshara.append(next_grapheme)
                    i += 1

                    # Look for following consonant
                    if i < len(graphemes):
                        next_next = graphemes[i]
                        if next_next and is_devanagari_consonant(next_next[0]):
                            current_akshara.append(next_next)
                            i += 1

            # Collect dependent vowels and diacritics
            while i < len(graphemes):
                next_grapheme = graphemes[i]
                if not next_grapheme:
                    break

                first_char = next_grapheme[0]
                if is_dependent_vowel(first_char) or is_virama(first_char):
                    # Part of current akshara
                    current_akshara.append(next_grapheme)
                    i += 1
                elif is_devanagari_consonant(first_char):
                    # New akshara starts
                    break
                else:
                    # Other character (punctuation, space, etc.) - end akshara
                    break

        elif is_dependent_vowel(first_char):
            # Standalone dependent vowel (unusual but possible)
            if current_akshara:
                aksharas.append("".join(current_akshara))
            current_akshara = [grapheme]
            i += 1
        else:
            # Non-Devanagari or punctuation - end current akshara if any
            if current_akshara:
                aksharas.append("".join(current_akshara))
                current_akshara = []
            i += 1

    # Add final akshara
    if current_akshara:
        aksharas.append("".join(current_akshara))

    return aksharas


def akshara_integrity_rate(text: str, tokens: List[str]) -> Tuple[float, int]:
    """
    Calculate akshara integrity rate.

    Integrity = 1 - (split aksharas / total aksharas)
    Higher is better (ideal = 1.0).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    Tuple[float, int]
        (integrity_rate, split_count)
    """
    aksharas = segment_aksharas(text)

    if len(aksharas) == 0:
        return 1.0, 0

    # Reconstruct token text (remove special markers)
    token_text = "".join(tokens).replace("##", "").replace("▁", "").strip()

    # Count aksharas that are split across token boundaries
    split_count = 0
    token_pos = 0

    for akshara in aksharas:
        # Find akshara in token text
        akshara_start = token_text.find(akshara, token_pos)
        if akshara_start == -1:
            # Akshara not found (might be modified by tokenizer)
            # Count as split if it's not in any single token
            found_in_token = any(akshara in t.replace("##", "").replace("▁", "") for t in tokens)
            if not found_in_token:
                split_count += 1
            continue

        akshara_end = akshara_start + len(akshara)

        # Check if akshara spans multiple tokens
        # Simple heuristic: if akshara is longer than average token, likely split
        avg_token_len = (
            sum(len(t.replace("##", "").replace("▁", "")) for t in tokens) / len(tokens)
            if tokens
            else 0
        )

        if avg_token_len > 0 and len(akshara) > avg_token_len * 1.2:
            # Check if akshara appears as substring in any single token
            found_whole = any(
                akshara in t.replace("##", "").replace("▁", "") for t in tokens
            )
            if not found_whole:
                split_count += 1

        token_pos = akshara_end

    integrity_rate = 1.0 - (split_count / len(aksharas)) if len(aksharas) > 0 else 1.0
    return integrity_rate, split_count


def dependent_vowel_split_rate(text: str, tokens: List[str]) -> float:
    """
    Calculate rate of dependent vowel separations from base consonants.

    Lower is better (fewer separations).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        Dependent vowel split rate (0.0-1.0).
    """
    # Count dependent vowels in text
    dependent_vowels = [ch for ch in text if is_dependent_vowel(ch)]

    if len(dependent_vowels) == 0:
        return 0.0

    # Count dependent vowels that appear at token start (separated from base)
    split_count = 0

    for token in tokens:
        token_clean = token.replace("##", "").replace("▁", "").strip()
        if token_clean and is_dependent_vowel(token_clean[0]):
            split_count += 1

    return split_count / len(dependent_vowels) if len(dependent_vowels) > 0 else 0.0


def classify_token_grapheme_alignment(
    token: str, graphemes: List[str]
) -> Tuple[int, int, int]:
    """
    Classify tokens by grapheme alignment.

    Returns counts of:
    - single_grapheme: tokens that are exactly one grapheme
    - multi_grapheme: tokens that are multiple complete graphemes
    - fragment: tokens that are part of a grapheme (violation)

    Parameters
    ----------
    token : str
        Token string.
    graphemes : List[str]
        List of all graphemes in the text.

    Returns
    -------
    Tuple[int, int, int]
        (single_count, multi_count, fragment_count)
    """
    token_clean = token.replace("##", "").replace("▁", "").strip()

    if not token_clean:
        return 0, 0, 0

    # Check if token is exactly one grapheme
    if token_clean in graphemes:
        return 1, 0, 0

    # Check if token is multiple complete graphemes
    # (token contains multiple graphemes as substrings)
    grapheme_matches = sum(1 for g in graphemes if g in token_clean)
    if grapheme_matches > 1:
        # Check if token is composed of complete graphemes
        # Simple check: if token length matches sum of matched grapheme lengths
        matched_graphemes = [g for g in graphemes if g in token_clean]
        total_length = sum(len(g) for g in matched_graphemes)
        if total_length == len(token_clean):
            return 0, 1, 0

    # Otherwise, likely a fragment (violation)
    return 0, 0, 1


def grapheme_aligned_token_rate(text: str, tokens: List[str]) -> Tuple[float, int, int, int]:
    """
    Calculate grapheme-aligned token rate.

    Returns rate and counts of:
    - single_grapheme_tokens: tokens = exactly one grapheme
    - multi_grapheme_tokens: tokens = multiple complete graphemes
    - fragment_tokens: tokens that split graphemes

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    Tuple[float, int, int, int]
        (aligned_rate, single_count, multi_count, fragment_count)
    """
    if len(tokens) == 0:
        return 0.0, 0, 0, 0

    graphemes = segment_devanagari_graphemes(text)

    single_count = 0
    multi_count = 0
    fragment_count = 0

    for token in tokens:
        single, multi, fragment = classify_token_grapheme_alignment(token, graphemes)
        single_count += single
        multi_count += multi
        fragment_count += fragment

    # Aligned rate = (single + multi) / total
    aligned_rate = (single_count + multi_count) / len(tokens) if len(tokens) > 0 else 0.0

    return aligned_rate, single_count, multi_count, fragment_count


def is_pure_devanagari_token(token: str) -> bool:
    """
    Check if token contains only Devanagari characters (plus common punctuation).

    Parameters
    ----------
    token : str
        Token string.

    Returns
    -------
    bool
        True if token is pure Devanagari.
    """
    token_clean = token.replace("##", "").replace("▁", "").strip()

    if not token_clean:
        return False

    # Common punctuation that can appear in Devanagari text
    allowed_punct = {".", ",", "!", "?", ":", ";", "।", "॥", "—", "-", "'", '"', "(", ")"}

    for char in token_clean:
        cp = ord(char)
        is_devanagari = DEVANAGARI_START <= cp <= DEVANAGARI_END
        is_punct = char in allowed_punct or char.isspace()

        if not (is_devanagari or is_punct):
            return False

    return True


def devanagari_token_share(tokens: List[str]) -> float:
    """
    Calculate share of tokens that are pure Devanagari.

    Parameters
    ----------
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        Devanagari token share (0.0-1.0).
    """
    if len(tokens) == 0:
        return 0.0

    devanagari_count = sum(1 for token in tokens if is_pure_devanagari_token(token))
    return devanagari_count / len(tokens)


def mixed_script_token_share(tokens: List[str]) -> float:
    """
    Calculate share of tokens that mix scripts (e.g., Devanagari + Latin).

    Parameters
    ----------
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        Mixed script token share (0.0-1.0).
    """
    if len(tokens) == 0:
        return 0.0

    mixed_count = 0

    for token in tokens:
        token_clean = token.replace("##", "").replace("▁", "").strip()
        if not token_clean:
            continue

        has_devanagari = False
        has_other_script = False

        for char in token_clean:
            cp = ord(char)
            if DEVANAGARI_START <= cp <= DEVANAGARI_END:
                has_devanagari = True
            elif char.isalnum() and not (0x0900 <= cp <= 0x097F):
                # Has non-Devanagari alphanumeric
                has_other_script = True

        if has_devanagari and has_other_script:
            mixed_count += 1

    return mixed_count / len(tokens) if len(tokens) > 0 else 0.0


def evaluate_script(text: str, tokens: List[str], lang: str = "hi") -> ScriptMetrics:
    """
    Evaluate all script adequacy metrics for a tokenization.

    Parameters
    ----------
    text : str
        Original input text.
    tokens : List[str]
        List of tokens from tokenizer.
    lang : str
        Language code (default: "hi").

    Returns
    -------
    ScriptMetrics
        All script adequacy metrics.
    """
    # Grapheme violation rate
    gv_rate = violation_rate(text, tokens)

    # Akshara integrity
    akshara_integrity, akshara_splits = akshara_integrity_rate(text, tokens)

    # Dependent vowel split rate
    dv_split_rate = dependent_vowel_split_rate(text, tokens)

    # Grapheme alignment
    (
        grapheme_aligned_rate,
        single_grapheme,
        multi_grapheme,
        fragment_grapheme,
    ) = grapheme_aligned_token_rate(text, tokens)

    # Script purity
    devanagari_share = devanagari_token_share(tokens)
    mixed_share = mixed_script_token_share(tokens)

    return ScriptMetrics(
        grapheme_violation_rate=gv_rate,
        akshara_integrity_rate=akshara_integrity,
        akshara_split_count=akshara_splits,
        dependent_vowel_split_rate=dv_split_rate,
        grapheme_aligned_token_rate=grapheme_aligned_rate,
        single_grapheme_tokens=single_grapheme,
        multi_grapheme_tokens=multi_grapheme,
        grapheme_fragment_tokens=fragment_grapheme,
        devanagari_token_share=devanagari_share,
        mixed_script_token_share=mixed_share,
        akshara_segmentation_version="v0_heuristic",
    )

