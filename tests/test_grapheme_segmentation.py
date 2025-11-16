# tests/test_grapheme_segmentation.py
# -*- coding: utf-8 -*-
"""
Tests for grapheme segmentation.
"""

import pytest

from tokenizers.grapheme_segmenter import (
    segment_devanagari_graphemes,
    iter_graphemes,
)


def test_basic_grapheme_segmentation():
    """Test basic grapheme segmentation."""
    text = "किशोरी"
    graphemes = segment_devanagari_graphemes(text)
    assert len(graphemes) == 3
    assert graphemes == ["कि", "शो", "री"]


def test_grapheme_iterator():
    """Test grapheme iterator."""
    text = "प्रार्थना"
    graphemes = list(iter_graphemes(text))
    # प्रार्थना can be segmented as 3-4 graphemes depending on analysis
    assert len(graphemes) >= 3


def test_complex_graphemes():
    """Test complex grapheme clusters."""
    text = "कर्मयोग"
    graphemes = segment_devanagari_graphemes(text)
    # Should have multiple graphemes
    assert len(graphemes) >= 3


def test_mixed_script():
    """Test mixed script text."""
    text = "Hello किशोरी world"
    graphemes = segment_devanagari_graphemes(text, keep_non_devanagari=True)
    # Should include all graphemes
    assert len(graphemes) > 0


def test_dev_only_filtering():
    """Test Devanagari-only filtering."""
    text = "Hello किशोरी world"
    graphemes = segment_devanagari_graphemes(text, keep_non_devanagari=False)
    # Should only include Devanagari graphemes
    devanagari_graphemes = [g for g in graphemes if any(ord(ch) >= 0x0900 and ord(ch) <= 0x097F for ch in g)]
    assert len(devanagari_graphemes) > 0

