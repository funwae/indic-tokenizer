# eval/grapheme_violations.py
# -*- coding: utf-8 -*-
"""
Grapheme violation detection for Indic Tokenization Lab.

Detects when tokenizers split Unicode grapheme clusters, which is a critical
error for Devanagari and other complex scripts.
"""

from __future__ import annotations

import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Add project root to path and import grapheme_segmenter safely
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_grapheme_seg_spec = importlib.util.spec_from_file_location(
    "tokenizers.grapheme_segmenter",
    project_root / "tokenizers" / "grapheme_segmenter.py"
)
_grapheme_seg_module = importlib.util.module_from_spec(_grapheme_seg_spec)
_grapheme_seg_spec.loader.exec_module(_grapheme_seg_module)
segment_devanagari_graphemes = _grapheme_seg_module.segment_devanagari_graphemes
iter_graphemes = _grapheme_seg_module.iter_graphemes


@dataclass
class Violation:
    """Represents a grapheme violation."""

    token_index: int
    token: str
    grapheme: str
    position_in_text: int
    description: str


def detect_violations(text: str, tokens: List[str]) -> List[Violation]:
    """
    Detect all grapheme violations in tokenized text.

    A violation occurs when a token boundary falls inside a Unicode grapheme cluster.

    Parameters
    ----------
    text : str
        Original text that was tokenized.
    tokens : List[str]
        List of tokens from a tokenizer.

    Returns
    -------
    List[Violation]
        List of detected violations, ordered by position in text.
    """
    violations: List[Violation] = []

    # Get all grapheme clusters and their positions
    graphemes = list(iter_graphemes(text))
    grapheme_positions: List[Tuple[int, int, str]] = []  # (start, end, grapheme)

    pos = 0
    for grapheme in graphemes:
        start = pos
        end = pos + len(grapheme)
        grapheme_positions.append((start, end, grapheme))
        pos = end

    # Reconstruct token boundaries in original text
    token_positions: List[Tuple[int, int, int, str]] = []  # (start, end, token_idx, token)
    text_pos = 0
    for token_idx, token in enumerate(tokens):
        # Find token in text starting from current position
        token_start = text.find(token, text_pos)
        if token_start == -1:
            # Token not found (might be special token or modified)
            # Try to find it without special characters
            token_clean = token.replace("##", "").replace("▁", "").strip()
            if token_clean:
                token_start = text.find(token_clean, text_pos)
            if token_start == -1:
                # Still not found, skip this token
                continue

        token_end = token_start + len(token)
        token_positions.append((token_start, token_end, token_idx, token))
        text_pos = token_end

    # Check each token boundary for violations
    for i in range(len(token_positions) - 1):
        token_start, token_end, token_idx, token = token_positions[i]
        next_token_start, _, next_token_idx, _ = token_positions[i + 1]

        # The boundary is at token_end (which equals next_token_start)
        boundary_pos = token_end

        # Check if this boundary falls inside any grapheme
        for grapheme_start, grapheme_end, grapheme in grapheme_positions:
            if grapheme_start < boundary_pos < grapheme_end:
                # Violation: boundary is inside a grapheme
                violation = Violation(
                    token_index=token_idx,
                    token=token,
                    grapheme=grapheme,
                    position_in_text=boundary_pos,
                    description=f"Token boundary at position {boundary_pos} splits grapheme '{grapheme}'",
                )
                violations.append(violation)
                break

    return violations


def count_violations(text: str, tokens: List[str]) -> int:
    """
    Count the number of grapheme violations.

    Parameters
    ----------
    text : str
        Original text that was tokenized.
    tokens : List[str]
        List of tokens from a tokenizer.

    Returns
    -------
    int
        Number of violations detected.
    """
    return len(detect_violations(text, tokens))


def violation_rate(text: str, tokens: List[str]) -> float:
    """
    Calculate the grapheme violation rate.

    Violation rate = number of violations / number of token boundaries

    Parameters
    ----------
    text : str
        Original text that was tokenized.
    tokens : List[str]
        List of tokens from a tokenizer.

    Returns
    -------
    float
        Violation rate between 0.0 and 1.0 (or higher if multiple violations per boundary).
    """
    if len(tokens) <= 1:
        return 0.0

    num_boundaries = len(tokens) - 1
    if num_boundaries == 0:
        return 0.0

    violations = count_violations(text, tokens)
    return violations / num_boundaries


def generate_violation_report(
    text: str, tokenizer_results: dict[str, List[str]]
) -> str:
    """
    Generate a human-readable violation report for multiple tokenizers.

    Parameters
    ----------
    text : str
        Original text that was tokenized.
    tokenizer_results : dict[str, List[str]]
        Dictionary mapping tokenizer names to their token lists.

    Returns
    -------
    str
        Formatted violation report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Grapheme Violation Report")
    lines.append("=" * 80)
    lines.append(f"\nText: {text}\n")

    for tokenizer_name, tokens in tokenizer_results.items():
        violations = detect_violations(text, tokens)
        violation_count = len(violations)
        rate = violation_rate(text, tokens)

        lines.append(f"\n{tokenizer_name}:")
        lines.append(f"  Tokens: {len(tokens)}")
        lines.append(f"  Violations: {violation_count}")
        lines.append(f"  Violation Rate: {rate:.2%}")

        if violations:
            lines.append("\n  Violation Details:")
            for violation in violations:
                lines.append(
                    f"    - Token #{violation.token_index} '{violation.token}' "
                    f"splits grapheme '{violation.grapheme}' at position {violation.position_in_text}"
                )
        else:
            lines.append("  ✓ No violations detected")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def batch_detect_violations(
    texts: List[str], tokenizer_results: dict[str, List[List[str]]]
) -> dict[str, List[List[Violation]]]:
    """
    Detect violations for multiple texts and tokenizers.

    Parameters
    ----------
    texts : List[str]
        List of original texts.
    tokenizer_results : dict[str, List[List[str]]]
        Dictionary mapping tokenizer names to lists of token lists (one per text).

    Returns
    -------
    dict[str, List[List[Violation]]]
        Dictionary mapping tokenizer names to lists of violation lists (one per text).
    """
    results: dict[str, List[List[Violation]]] = {}

    for tokenizer_name, token_lists in tokenizer_results.items():
        violations_list: List[List[Violation]] = []
        for text, tokens in zip(texts, token_lists):
            violations = detect_violations(text, tokens)
            violations_list.append(violations)
        results[tokenizer_name] = violations_list

    return results

