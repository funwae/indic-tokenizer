# tests/test_cbpe_constraints.py
# -*- coding: utf-8 -*-
"""
Tests for CBPE constraints.
"""

import pytest

from tokenizers.cbpe_constraints import (
    cbpe_merge_allowed,
    filter_bpe_merges,
    is_devanagari_char,
    is_dependent_vowel,
    is_virama,
)


def test_cbpe_merge_allowed_normal():
    """Test that normal merges are allowed."""
    assert cbpe_merge_allowed("क", "्ष") is True
    assert cbpe_merge_allowed("क", "ा") is True


def test_cbpe_merge_allowed_virama_start():
    """Test that merges starting with virama are disallowed."""
    assert cbpe_merge_allowed("्", "ा") is False
    assert cbpe_merge_allowed("्", "ष") is False


def test_cbpe_merge_allowed_dependent_vowel_start():
    """Test that merges starting with dependent vowel are disallowed."""
    assert cbpe_merge_allowed("ा", "क") is False
    assert cbpe_merge_allowed("ि", "श") is False


def test_cbpe_merge_allowed_empty():
    """Test that empty strings are disallowed."""
    assert cbpe_merge_allowed("", "क") is False
    assert cbpe_merge_allowed("क", "") is False


def test_filter_bpe_merges():
    """Test filtering of BPE merges."""
    merges = [
        ("क", "्ष"),  # Should be allowed
        ("्", "ा"),  # Should be disallowed (virama start)
        ("ा", "क"),  # Should be disallowed (dependent vowel start)
        ("क", "ा"),  # Should be allowed
    ]
    filtered = filter_bpe_merges(merges)
    # Should only contain allowed merges
    assert ("क", "्ष") in filtered
    assert ("क", "ा") in filtered
    assert ("्", "ा") not in filtered
    assert ("ा", "क") not in filtered


def test_is_devanagari_char():
    """Test Devanagari character detection."""
    assert is_devanagari_char("क") is True
    assert is_devanagari_char("ा") is True
    assert is_devanagari_char("a") is False
    assert is_devanagari_char("") is False


def test_is_dependent_vowel():
    """Test dependent vowel detection."""
    assert is_dependent_vowel("ा") is True  # AA
    assert is_dependent_vowel("ि") is True  # I
    assert is_dependent_vowel("क") is False  # Consonant
    assert is_dependent_vowel("") is False


def test_is_virama():
    """Test virama detection."""
    assert is_virama("्") is True
    assert is_virama("क") is False
    assert is_virama("") is False

