# tests/test_metrics_efficiency.py
# -*- coding: utf-8 -*-
"""
Tests for efficiency metrics.
"""

import pytest

from eval.metrics.efficiency import (
    compression_ratio_chars,
    compression_ratio_graphemes,
    normalized_sequence_length,
    proportion_continued_words,
    unk_rate,
    evaluate_efficiency,
)


def test_compression_ratio_chars():
    """Test character-based compression ratio."""
    text = "यहाँ आपका हिंदी"
    tokens = ["यहाँ", "आपका", "हिंदी"]
    cr = compression_ratio_chars(text, tokens)
    assert cr > 0
    assert cr == pytest.approx(len(text) / len(tokens), abs=0.1)


def test_compression_ratio_graphemes():
    """Test grapheme-based compression ratio."""
    text = "किशोरी"
    tokens = ["किशोरी"]
    cr = compression_ratio_graphemes(text, tokens)
    assert cr > 0


def test_normalized_sequence_length():
    """Test normalized sequence length."""
    tokens = ["यहाँ", "आपका", "हिंदी"]
    baseline_tokens = ["यहाँ", "आपका", "हिंदी"]
    nsl = normalized_sequence_length(tokens, baseline_tokens)
    assert nsl == pytest.approx(1.0, abs=0.01)

    # More tokens than baseline
    tokens_longer = ["यहाँ", "आपका", "हिंदी", "वाक्य"]
    nsl = normalized_sequence_length(tokens_longer, baseline_tokens)
    assert nsl > 1.0

    # Fewer tokens than baseline
    tokens_shorter = ["यहाँ", "आपका"]
    nsl = normalized_sequence_length(tokens_shorter, baseline_tokens)
    assert nsl < 1.0


def test_normalized_sequence_length_no_baseline():
    """Test NSL with no baseline."""
    tokens = ["यहाँ", "आपका"]
    nsl = normalized_sequence_length(tokens, None)
    assert nsl is None


def test_proportion_continued_words():
    """Test proportion of continued words."""
    # Single token per word - should be 0
    text = "यहाँ आपका हिंदी"
    tokens = ["यहाँ", "आपका", "हिंदी"]
    pcw = proportion_continued_words(text, tokens)
    assert pcw == pytest.approx(0.0, abs=0.1)

    # Words split across tokens
    text = "यहाँ आपका"
    tokens = ["यहाँ", "आ", "##प", "##का"]  # "आपका" split
    pcw = proportion_continued_words(text, tokens)
    assert pcw > 0.0
    assert pcw <= 1.0


def test_unk_rate():
    """Test UNK rate calculation."""
    tokens = ["यहाँ", "आपका", "हिंदी"]
    unk = unk_rate(tokens)
    assert unk == 0.0

    tokens_with_unk = ["यहाँ", "<unk>", "हिंदी"]
    unk = unk_rate(tokens_with_unk)
    assert unk == pytest.approx(1.0 / 3.0, abs=0.01)


def test_evaluate_efficiency():
    """Test comprehensive efficiency evaluation."""
    text = "यहाँ आपका हिंदी वाक्य जाएगा।"
    tokens = ["यहाँ", "आपका", "हिंदी", "वाक्य", "जाएगा", "।"]
    baseline_tokens = ["यहाँ", "आपका", "हिंदी", "वाक्य", "जाएगा", "।"]

    metrics = evaluate_efficiency(
        text=text,
        tokens=tokens,
        baseline_tokens=baseline_tokens,
        baseline_tokenizer_id="test_baseline",
    )

    assert metrics.fertility > 0
    assert metrics.chars_per_token > 0
    assert metrics.compression_ratio_chars > 0
    assert metrics.compression_ratio_graphemes > 0
    assert metrics.normalized_sequence_length == pytest.approx(1.0, abs=0.01)
    assert metrics.proportion_continued_words >= 0.0
    assert metrics.unk_rate >= 0.0
    assert metrics.baseline_tokenizer_id == "test_baseline"


def test_evaluate_efficiency_no_baseline():
    """Test efficiency evaluation without baseline."""
    text = "यहाँ आपका हिंदी"
    tokens = ["यहाँ", "आपका", "हिंदी"]

    metrics = evaluate_efficiency(
        text=text,
        tokens=tokens,
        baseline_tokens=None,
        baseline_tokenizer_id=None,
    )

    assert metrics.normalized_sequence_length is None
    assert metrics.baseline_tokenizer_id is None

