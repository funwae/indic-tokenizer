# eval/fertility.py
# -*- coding: utf-8 -*-
"""
Fertility metrics for Indic Tokenization Lab.

Fertility measures token efficiency: how many tokens are needed per word.
Lower fertility is generally better (fewer tokens for the same content).
"""

from __future__ import annotations

from typing import List, Optional

import unicodedata


def split_words_script_aware(text: str) -> List[str]:
    """
    Split text into words using script-aware splitting.

    Treats script changes (e.g., Devanagari to Latin) as word boundaries.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of words.
    """
    words: List[str] = []
    current_word: List[str] = []
    current_script: Optional[str] = None

    for char in text:
        if char.isspace():
            if current_word:
                words.append("".join(current_word))
                current_word = []
                current_script = None
            continue

        # Determine script of current character
        script = unicodedata.name(char, "").split()[0] if char else None

        # If script changed and we have a current word, start a new word
        if current_script is not None and script != current_script and current_word:
            words.append("".join(current_word))
            current_word = [char]
            current_script = script
        else:
            current_word.append(char)
            current_script = script

    # Add final word
    if current_word:
        words.append("".join(current_word))

    return words


def calculate_fertility(text: str, tokens: List[str]) -> float:
    """
    Calculate fertility: tokens per word.

    Fertility = number of tokens / number of words

    Lower is better (fewer tokens needed per word).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens from a tokenizer.

    Returns
    -------
    float
        Fertility (tokens per word).
    """
    words = split_words_script_aware(text)
    # Filter out empty words
    words = [w for w in words if w.strip()]

    if len(words) == 0:
        return 0.0

    return len(tokens) / len(words)


def calculate_chars_per_token(text: str, tokens: List[str]) -> float:
    """
    Calculate characters per token.

    Measures packing efficiency: how many characters are encoded per token.
    Higher is better (more content per token).

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens from a tokenizer.

    Returns
    -------
    float
        Characters per token.
    """
    if len(tokens) == 0:
        return 0.0

    # Count characters in original text (excluding special token markers)
    # Remove common tokenizer markers for accurate comparison
    text_clean = text
    return len(text_clean) / len(tokens)


def compare_fertility(
    texts: List[str], tokenizer_results: dict[str, List[List[str]]]
) -> dict[str, dict[str, float]]:
    """
    Compare fertility metrics across multiple tokenizers and texts.

    Parameters
    ----------
    texts : List[str]
        List of original texts.
    tokenizer_results : dict[str, List[List[str]]]
        Dictionary mapping tokenizer names to lists of token lists (one per text).

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary mapping tokenizer names to metric dictionaries:
        {
            'avg_fertility': float,
            'avg_chars_per_token': float,
            'total_tokens': int,
            'total_words': int,
            'total_chars': int
        }
    """
    results: dict[str, dict[str, float]] = {}

    for tokenizer_name, token_lists in tokenizer_results.items():
        total_tokens = 0
        total_words = 0
        total_chars = 0
        fertilities: List[float] = []
        chars_per_token_list: List[float] = []

        for text, tokens in zip(texts, token_lists):
            words = split_words_script_aware(text)
            words = [w for w in words if w.strip()]

            total_tokens += len(tokens)
            total_words += len(words)
            total_chars += len(text)

            fertility = calculate_fertility(text, tokens)
            chars_per_token = calculate_chars_per_token(text, tokens)

            fertilities.append(fertility)
            chars_per_token_list.append(chars_per_token)

        # Calculate averages
        avg_fertility = sum(fertilities) / len(fertilities) if fertilities else 0.0
        avg_chars_per_token = (
            sum(chars_per_token_list) / len(chars_per_token_list)
            if chars_per_token_list
            else 0.0
        )

        results[tokenizer_name] = {
            "avg_fertility": avg_fertility,
            "avg_chars_per_token": avg_chars_per_token,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "total_chars": total_chars,
        }

    return results

