# tokenizers/pretokenizer.py
# -*- coding: utf-8 -*-
"""
Rule-based pretokenizer for Indic Tokenization Lab.

Provides script-aware pretokenization for Hindi/Sanskrit, handling:
- Whitespace and punctuation splitting
- URL, email, hashtag, mention preservation
- Unicode normalization
- Script-aware boundaries
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


# Regex patterns for special tokens
URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
)  # URLs
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")  # Emails
HASHTAG_PATTERN = re.compile(r"#\w+")  # Hashtags
MENTION_PATTERN = re.compile(r"@\w+")  # Mentions (@username)
NUMBER_PATTERN = re.compile(r"\d+[.,]?\d*")  # Numbers (including decimals)


def preserve_special_tokens(text: str) -> List[tuple[str, bool]]:
    """
    Identify special tokens (URLs, emails, hashtags, mentions, numbers) in text.

    Returns a list of (token, is_special) tuples.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[tuple[str, bool]]
        List of (token, is_special) tuples where is_special indicates
        if the token should be preserved as atomic.
    """
    tokens: List[tuple[str, bool]] = []
    last_end = 0

    # Find all special tokens with their positions
    special_matches: List[tuple[int, int, str]] = []

    for pattern in [URL_PATTERN, EMAIL_PATTERN, HASHTAG_PATTERN, MENTION_PATTERN, NUMBER_PATTERN]:
        for match in pattern.finditer(text):
            special_matches.append((match.start(), match.end(), match.group()))

    # Sort by start position
    special_matches.sort(key=lambda x: x[0])

    # Build token list, preserving special tokens
    for start, end, special_text in special_matches:
        # Add text before special token
        if start > last_end:
            tokens.append((text[last_end:start], False))

        # Add special token
        tokens.append((special_text, True))
        last_end = end

    # Add remaining text
    if last_end < len(text):
        tokens.append((text[last_end:], False))

    return tokens


def split_words(text: str) -> List[str]:
    """
    Split text into words using whitespace.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of words.
    """
    return [w for w in text.split() if w.strip()]


def split_punctuation(text: str) -> List[str]:
    """
    Split punctuation from words while preserving it.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of tokens with punctuation separated.
    """
    # Common punctuation marks (including Devanagari punctuation)
    punctuation = r".,!?;:।॥()[]{}'\"-–—…"

    tokens: List[str] = []
    current_token: List[str] = []

    for char in text:
        if char in punctuation or unicodedata.category(char).startswith("P"):
            # Punctuation found
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
            tokens.append(char)
        elif char.isspace():
            # Whitespace
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
        else:
            current_token.append(char)

    # Add final token
    if current_token:
        tokens.append("".join(current_token))

    return [t for t in tokens if t.strip()]


def detect_script(char: str) -> str:
    """
    Detect the script of a character.

    Parameters
    ----------
    char : str
        Single character.

    Returns
    -------
    str
        Script name (e.g., "Devanagari", "Latin", "Common").
    """
    try:
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            return "Devanagari"
        elif "LATIN" in name or char.isascii():
            return "Latin"
        elif unicodedata.category(char).startswith("P"):
            return "Punctuation"
        elif char.isdigit():
            return "Number"
        else:
            return "Other"
    except (ValueError, TypeError):
        return "Unknown"


def split_script_boundaries(text: str) -> List[str]:
    """
    Split text at script boundaries (e.g., Devanagari to Latin).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of tokens split at script boundaries.
    """
    if not text:
        return []

    tokens: List[str] = []
    current_token: List[str] = []
    current_script: str | None = None

    for char in text:
        script = detect_script(char)

        # Punctuation and whitespace don't break script boundaries
        if script in ("Punctuation", "Unknown") or char.isspace():
            current_token.append(char)
            continue

        if current_script is None:
            current_script = script
            current_token.append(char)
        elif script == current_script:
            current_token.append(char)
        else:
            # Script changed
            if current_token:
                tokens.append("".join(current_token))
            current_token = [char]
            current_script = script

    # Add final token
    if current_token:
        tokens.append("".join(current_token))

    return [t for t in tokens if t.strip()]


def pretokenize(
    text: str, lang: str = "hi", normalize: str = "NFC"
) -> List[str]:
    """
    Pretokenize text with script-aware rules.

    This function:
    1. Normalizes Unicode
    2. Preserves special tokens (URLs, emails, hashtags, mentions, numbers)
    3. Splits on whitespace
    4. Separates punctuation
    5. Handles script boundaries

    Parameters
    ----------
    text : str
        Input text to pretokenize.
    lang : str
        Language code (default: "hi").
    normalize : str
        Unicode normalization form: "NFC", "NFD", "NFKC", "NFKD" (default: "NFC").

    Returns
    -------
    List[str]
        List of pretokenized tokens.
    """
    # Step 1: Normalize Unicode
    if normalize == "NFC":
        text = unicodedata.normalize("NFC", text)
    elif normalize == "NFD":
        text = unicodedata.normalize("NFD", text)
    elif normalize == "NFKC":
        text = unicodedata.normalize("NFKC", text)
    elif normalize == "NFKD":
        text = unicodedata.normalize("NFKD", text)

    # Step 2: Preserve special tokens
    special_tokens = preserve_special_tokens(text)
    tokens: List[str] = []

    for token_text, is_special in special_tokens:
        if is_special:
            # Preserve as atomic token
            tokens.append(token_text)
        else:
            # Process regular text
            # Split on whitespace first
            words = split_words(token_text)
            for word in words:
                # Split punctuation
                word_tokens = split_punctuation(word)
                tokens.extend(word_tokens)

    # Step 3: Handle script boundaries (optional, can be aggressive)
    # For now, we'll keep it simple and not split script boundaries
    # as it might be too aggressive for some use cases

    return [t for t in tokens if t.strip()]

