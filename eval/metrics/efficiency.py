# eval/metrics/efficiency.py
# -*- coding: utf-8 -*-
"""
Efficiency metrics for tokenization evaluation.

Measures sequence-level efficiency: compression, fragmentation, and token economy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from eval.fertility import calculate_chars_per_token, calculate_fertility
from tokenizers.grapheme_segmenter import segment_devanagari_graphemes


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for tokenization."""

    fertility: float  # tokens/word
    chars_per_token: float  # chars/token
    compression_ratio_chars: float  # chars/tokens (higher is better)
    compression_ratio_graphemes: float  # graphemes/tokens (higher is better)
    normalized_sequence_length: Optional[float]  # |t(s)| / |t0(s)| (lower is better, <1 = more efficient)
    proportion_continued_words: float  # fraction of words split into ≥2 tokens (lower is better)
    unk_rate: float  # fraction of <unk> or fallback tokens (lower is better)
    baseline_tokenizer_id: Optional[str]  # Which baseline was used for NSL


def compression_ratio_chars(text: str, tokens: List[str]) -> float:
    """
    Calculate character-based compression ratio.

    CR = chars / tokens
    Higher is better (more characters encoded per token).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        Compression ratio (chars/tokens).
    """
    if len(tokens) == 0:
        return 0.0
    return len(text) / len(tokens)


def compression_ratio_graphemes(text: str, tokens: List[str]) -> float:
    """
    Calculate grapheme-based compression ratio.

    CR = graphemes / tokens
    Higher is better (more graphemes encoded per token).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        Compression ratio (graphemes/tokens).
    """
    if len(tokens) == 0:
        return 0.0

    graphemes = segment_devanagari_graphemes(text)
    return len(graphemes) / len(tokens)


def normalized_sequence_length(
    tokens: List[str], baseline_tokens: Optional[List[str]]
) -> Optional[float]:
    """
    Calculate normalized sequence length relative to baseline.

    NSL = |t(s)| / |t0(s)|
    - NSL < 1: more efficient than baseline
    - NSL = 1: same efficiency as baseline
    - NSL > 1: less efficient than baseline

    Parameters
    ----------
    tokens : List[str]
        Tokens from tokenizer being evaluated.
    baseline_tokens : Optional[List[str]]
        Tokens from baseline tokenizer.

    Returns
    -------
    Optional[float]
        Normalized sequence length, or None if baseline not provided.
    """
    if baseline_tokens is None or len(baseline_tokens) == 0:
        return None

    if len(baseline_tokens) == 0:
        return None

    return len(tokens) / len(baseline_tokens)


def proportion_continued_words(text: str, tokens: List[str]) -> float:
    """
    Calculate proportion of words that are split into multiple tokens.

    PCW = (words split into ≥2 tokens) / (total words)
    Lower is better (fewer fragmented words).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        Proportion of continued words (0.0-1.0).
    """
    from eval.fertility import split_words_script_aware

    words = split_words_script_aware(text)
    words = [w for w in words if w.strip()]

    if len(words) == 0 or len(tokens) == 0:
        return 0.0

    # Clean tokens (remove special markers)
    clean_tokens = [t.replace("##", "").replace("▁", "").strip() for t in tokens]
    clean_tokens = [t for t in clean_tokens if t]

    # Map tokens back to text positions
    # Simple approach: try to find each token in the text sequentially
    text_pos = 0
    token_positions = []  # List of (start, end) positions for each token

    for token in clean_tokens:
        # Find token in text starting from current position
        token_start = text.find(token, text_pos)
        if token_start == -1:
            # Token not found - might be special or modified
            # Try without special characters
            token_clean = token.strip()
            if token_clean:
                token_start = text.find(token_clean, text_pos)
            if token_start == -1:
                # Still not found, skip
                continue

        token_end = token_start + len(token)
        token_positions.append((token_start, token_end))
        text_pos = token_end

    if not token_positions:
        return 0.0

    # Count words that span multiple tokens
    split_count = 0

    for word in words:
        # Find word in text
        word_start = text.find(word)
        if word_start == -1:
            continue

        word_end = word_start + len(word)

        # Count how many tokens this word spans
        tokens_covering_word = [
            (start, end)
            for start, end in token_positions
            if start < word_end and end > word_start
        ]

        if len(tokens_covering_word) > 1:
            # Word spans multiple tokens
            split_count += 1

    return split_count / len(words) if len(words) > 0 else 0.0


def unk_rate(tokens: List[str]) -> float:
    """
    Calculate the rate of unknown/fallback tokens.

    UNK rate = (number of <unk> tokens) / (total tokens)
    Lower is better (fewer unknown tokens).

    Parameters
    ----------
    tokens : List[str]
        List of tokens.

    Returns
    -------
    float
        UNK rate (0.0-1.0).
    """
    if len(tokens) == 0:
        return 0.0

    unk_count = sum(
        1
        for token in tokens
        if token in ("<unk>", "<UNK>", "[UNK]", "") or token.startswith("<unk")
    )
    return unk_count / len(tokens)


def evaluate_efficiency(
    text: str,
    tokens: List[str],
    baseline_tokens: Optional[List[str]] = None,
    baseline_tokenizer_id: Optional[str] = None,
) -> EfficiencyMetrics:
    """
    Evaluate all efficiency metrics for a tokenization.

    Parameters
    ----------
    text : str
        Original input text.
    tokens : List[str]
        List of tokens from tokenizer.
    baseline_tokens : Optional[List[str]]
        Tokens from baseline tokenizer for NSL computation.
    baseline_tokenizer_id : Optional[str]
        ID of baseline tokenizer used (for metadata).

    Returns
    -------
    EfficiencyMetrics
        All efficiency metrics.
    """
    fertility = calculate_fertility(text, tokens)
    chars_per_token = calculate_chars_per_token(text, tokens)
    cr_chars = compression_ratio_chars(text, tokens)
    cr_graphemes = compression_ratio_graphemes(text, tokens)
    nsl = normalized_sequence_length(tokens, baseline_tokens)
    pcw = proportion_continued_words(text, tokens)
    unk = unk_rate(tokens)

    return EfficiencyMetrics(
        fertility=fertility,
        chars_per_token=chars_per_token,
        compression_ratio_chars=cr_chars,
        compression_ratio_graphemes=cr_graphemes,
        normalized_sequence_length=nsl,
        proportion_continued_words=pcw,
        unk_rate=unk,
        baseline_tokenizer_id=baseline_tokenizer_id,
    )

