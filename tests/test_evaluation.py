# tests/test_evaluation.py
# -*- coding: utf-8 -*-
"""
Tests for evaluation metrics.
"""

import pytest

from eval.fertility import calculate_fertility, calculate_chars_per_token
from eval.grapheme_violations import count_violations, violation_rate


def test_calculate_fertility():
    """Test fertility calculation."""
    text = "यहाँ आपका हिंदी"
    tokens = ["यहाँ", "आपका", "हिंदी"]
    fertility = calculate_fertility(text, tokens)
    assert fertility > 0
    assert fertility == pytest.approx(1.0, abs=0.1)  # 3 tokens / 3 words = 1.0


def test_calculate_chars_per_token():
    """Test chars per token calculation."""
    text = "यहाँ आपका हिंदी"
    tokens = ["यहाँ", "आपका", "हिंदी"]
    chars_per_token = calculate_chars_per_token(text, tokens)
    assert chars_per_token > 0
    assert chars_per_token == pytest.approx(len(text) / len(tokens), abs=0.1)


def test_count_violations_no_violations():
    """Test violation counting when there are no violations."""
    text = "किशोरी"
    tokens = ["किशोरी"]  # Single token, no boundaries, no violations
    violations = count_violations(text, tokens)
    assert violations == 0


def test_violation_rate():
    """Test violation rate calculation."""
    text = "किशोरी"
    tokens = ["किशोरी"]
    rate = violation_rate(text, tokens)
    assert rate == 0.0  # No boundaries, no violations


def test_violation_rate_empty():
    """Test violation rate with empty input."""
    text = ""
    tokens = []
    rate = violation_rate(text, tokens)
    assert rate == 0.0


def test_violation_rate_single_token():
    """Test violation rate with single token."""
    text = "किशोरी"
    tokens = ["किशोरी"]
    rate = violation_rate(text, tokens)
    assert rate == 0.0  # No boundaries

