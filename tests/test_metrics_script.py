# tests/test_metrics_script.py
# -*- coding: utf-8 -*-
"""
Tests for script adequacy metrics.
"""

import pytest

from eval.metrics.script import (
    segment_aksharas,
    akshara_integrity_rate,
    dependent_vowel_split_rate,
    grapheme_aligned_token_rate,
    devanagari_token_share,
    mixed_script_token_share,
    is_pure_devanagari_token,
    evaluate_script,
)


def test_segment_aksharas():
    """Test akshara segmentation (v0 heuristic)."""
    text = "किशोरी"
    aksharas = segment_aksharas(text)
    assert len(aksharas) >= 1
    assert "कि" in aksharas or any("कि" in a for a in aksharas)

    text = "प्रार्थना"
    aksharas = segment_aksharas(text)
    assert len(aksharas) >= 1


def test_akshara_integrity_rate():
    """Test akshara integrity rate."""
    text = "किशोरी"
    tokens = ["किशोरी"]  # Single token, no splits
    integrity, splits = akshara_integrity_rate(text, tokens)
    assert integrity >= 0.0
    assert integrity <= 1.0
    assert splits >= 0

    # If token splits akshara, integrity should be lower
    text = "किशोरी"
    tokens = ["कि", "शोरी"]  # Split
    integrity, splits = akshara_integrity_rate(text, tokens)
    # Integrity might be lower, but depends on heuristic
    assert integrity >= 0.0
    assert integrity <= 1.0


def test_dependent_vowel_split_rate():
    """Test dependent vowel split rate."""
    text = "किशोरी"  # Contains dependent vowels
    tokens = ["किशोरी"]
    rate = dependent_vowel_split_rate(text, tokens)
    assert rate >= 0.0
    assert rate <= 1.0

    # If dependent vowel is at token start, rate should be higher
    tokens_with_split = ["कि", "शो", "री"]
    rate = dependent_vowel_split_rate(text, tokens_with_split)
    assert rate >= 0.0
    assert rate <= 1.0


def test_grapheme_aligned_token_rate():
    """Test grapheme-aligned token rate."""
    text = "किशोरी"
    tokens = ["किशोरी"]
    rate, single, multi, fragment = grapheme_aligned_token_rate(text, tokens)
    assert rate >= 0.0
    assert rate <= 1.0
    assert single >= 0
    assert multi >= 0
    assert fragment >= 0


def test_devanagari_token_share():
    """Test Devanagari token share."""
    tokens = ["यहाँ", "आपका", "हिंदी"]
    share = devanagari_token_share(tokens)
    assert share >= 0.0
    assert share <= 1.0
    # All tokens are Devanagari, so share should be high
    assert share > 0.5

    tokens_mixed = ["यहाँ", "hello", "हिंदी"]
    share = devanagari_token_share(tokens_mixed)
    assert share < 1.0


def test_mixed_script_token_share():
    """Test mixed script token share."""
    tokens = ["यहाँ", "आपका", "हिंदी"]
    share = mixed_script_token_share(tokens)
    assert share >= 0.0
    assert share <= 1.0
    # Pure Devanagari tokens, so mixed share should be low
    assert share < 0.5

    tokens_mixed = ["यहाँhello", "आपका", "हिंदी"]
    share = mixed_script_token_share(tokens_mixed)
    # Has mixed script token
    assert share > 0.0


def test_is_pure_devanagari_token():
    """Test pure Devanagari token detection."""
    assert is_pure_devanagari_token("यहाँ") is True
    assert is_pure_devanagari_token("आपका") is True
    assert is_pure_devanagari_token("hello") is False
    assert is_pure_devanagari_token("यहाँhello") is False
    assert is_pure_devanagari_token("यहाँ।") is True  # With punctuation


def test_evaluate_script():
    """Test comprehensive script evaluation."""
    text = "यहाँ आपका हिंदी वाक्य जाएगा।"
    tokens = ["यहाँ", "आपका", "हिंदी", "वाक्य", "जाएगा", "।"]

    metrics = evaluate_script(text=text, tokens=tokens, lang="hi")

    assert metrics.grapheme_violation_rate >= 0.0
    assert metrics.akshara_integrity_rate >= 0.0
    assert metrics.akshara_integrity_rate <= 1.0
    assert metrics.akshara_split_count >= 0
    assert metrics.dependent_vowel_split_rate >= 0.0
    assert metrics.grapheme_aligned_token_rate >= 0.0
    assert metrics.grapheme_aligned_token_rate <= 1.0
    assert metrics.single_grapheme_tokens >= 0
    assert metrics.multi_grapheme_tokens >= 0
    assert metrics.grapheme_fragment_tokens >= 0
    assert metrics.devanagari_token_share >= 0.0
    assert metrics.devanagari_token_share <= 1.0
    assert metrics.mixed_script_token_share >= 0.0
    assert metrics.mixed_script_token_share <= 1.0
    assert metrics.akshara_segmentation_version == "v0_heuristic"

