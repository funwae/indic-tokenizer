# tests/test_pretokenizer.py
# -*- coding: utf-8 -*-
"""
Tests for pretokenizer.
"""

import pytest

from tokenizers.pretokenizer import (
    pretokenize,
    preserve_special_tokens,
    split_words,
    split_punctuation,
)


def test_split_words():
    """Test word splitting."""
    text = "यहाँ आपका हिंदी"
    words = split_words(text)
    assert len(words) == 3
    assert "यहाँ" in words
    assert "आपका" in words
    assert "हिंदी" in words


def test_split_punctuation():
    """Test punctuation splitting."""
    text = "यहाँ, आपका!"
    tokens = split_punctuation(text)
    assert "," in tokens or any("," in t for t in tokens)
    assert "!" in tokens or any("!" in t for t in tokens)


def test_preserve_special_tokens_url():
    """Test URL preservation."""
    text = "Visit https://example.com for more"
    tokens = preserve_special_tokens(text)
    # Should preserve URL as atomic token
    url_found = any("https://example.com" in token[0] for token in tokens if token[1])
    assert url_found


def test_preserve_special_tokens_email():
    """Test email preservation."""
    text = "Email: test@example.com"
    tokens = preserve_special_tokens(text)
    # Should preserve email as atomic token
    email_found = any("test@example.com" in token[0] for token in tokens if token[1])
    assert email_found


def test_pretokenize_basic():
    """Test basic pretokenization."""
    text = "यहाँ आपका हिंदी वाक्य जाएगा।"
    tokens = pretokenize(text)
    assert len(tokens) > 0
    # Should split on whitespace and punctuation
    assert len(tokens) >= 5


def test_pretokenize_normalization():
    """Test Unicode normalization."""
    text = "यहाँ आपका"
    tokens_nfc = pretokenize(text, normalize="NFC")
    tokens_nfkc = pretokenize(text, normalize="NFKC")
    # Both should produce valid tokens
    assert len(tokens_nfc) > 0
    assert len(tokens_nfkc) > 0

