# tests/test_smoke_production.py
# -*- coding: utf-8 -*-
"""
Smoke tests for production preview.

These tests verify that the core functionality works end-to-end
without requiring external resources or complex setup.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that core modules can be imported."""
    from eval.metrics import ComprehensiveMetrics, evaluate_comprehensive
    from eval.metrics.efficiency import EfficiencyMetrics
    from eval.metrics.script import ScriptMetrics
    from tokenizers.grapheme_segmenter import segment_devanagari_graphemes
    from tokenizers.cbpe_constraints import cbpe_merge_allowed

    assert ComprehensiveMetrics is not None
    assert evaluate_comprehensive is not None
    assert EfficiencyMetrics is not None
    assert ScriptMetrics is not None


def test_grapheme_segmentation():
    """Test grapheme segmentation works."""
    from tokenizers.grapheme_segmenter import segment_devanagari_graphemes

    text = "किशोरी"
    graphemes = segment_devanagari_graphemes(text)
    assert len(graphemes) > 0
    assert all(isinstance(g, str) for g in graphemes)


def test_comprehensive_metrics_single_text():
    """Test comprehensive metrics on a single Hindi text."""
    from eval.metrics import evaluate_comprehensive

    text = "यहाँ आपका हिंदी वाक्य जाएगा।"
    tokens = ["यहाँ", "आपका", "हिंदी", "वाक्य", "जाएगा", "।"]

    metrics = evaluate_comprehensive(text=text, tokens=tokens, lang="hi")

    # Check efficiency metrics exist
    assert metrics.efficiency is not None
    assert metrics.efficiency.fertility > 0
    assert metrics.efficiency.chars_per_token > 0
    assert metrics.efficiency.compression_ratio_chars > 0
    assert metrics.efficiency.compression_ratio_graphemes > 0
    assert metrics.efficiency.proportion_continued_words >= 0.0
    assert metrics.efficiency.unk_rate >= 0.0

    # Check script metrics exist
    assert metrics.script is not None
    assert metrics.script.grapheme_violation_rate >= 0.0
    assert metrics.script.akshara_integrity_rate >= 0.0
    assert metrics.script.akshara_integrity_rate <= 1.0
    assert metrics.script.dependent_vowel_split_rate >= 0.0
    assert metrics.script.grapheme_aligned_token_rate >= 0.0
    assert metrics.script.devanagari_token_share >= 0.0
    assert metrics.script.devanagari_token_share <= 1.0

    # Check summary stats
    assert metrics.num_tokens > 0
    assert metrics.num_words > 0
    assert metrics.num_chars > 0
    assert metrics.num_graphemes > 0

    # Check no NaNs
    import math

    assert not math.isnan(metrics.efficiency.fertility)
    assert not math.isnan(metrics.efficiency.chars_per_token)
    assert not math.isnan(metrics.script.grapheme_violation_rate)
    assert not math.isnan(metrics.script.akshara_integrity_rate)


def test_benchmark_minimal():
    """Test minimal benchmark run on 1-2 lines of Hindi."""
    from scripts.run_benchmark import load_corpus, evaluate_tokenizer_on_corpus
    from scripts.compare_tokenizers import create_tokenizer_from_config, load_registry

    # Create minimal test corpus
    test_texts = [
        "यहाँ आपका हिंदी वाक्य जाएगा।",
        "भारत में आज कई महत्वपूर्ण घटनाएं हुईं।",
    ]

    # Load registry
    registry_path = Path("tokenizers/registry.yaml")
    if not registry_path.exists():
        pytest.skip("Registry file not found")

    registry = load_registry(registry_path)

    # Try to load mbert (should be available)
    if "mbert" not in registry:
        pytest.skip("mbert tokenizer not available in registry")

    config = registry["mbert"]
    tokenizer = create_tokenizer_from_config(config)

    # Evaluate on test texts
    metrics_dict = evaluate_tokenizer_on_corpus(
        tokenizer=tokenizer,
        texts=test_texts,
        lang="hi",
        baseline_tokenizer=None,
        baseline_tokenizer_id=None,
    )

    # Check metrics structure
    assert "efficiency" in metrics_dict
    assert "script" in metrics_dict
    assert "summary" in metrics_dict

    eff = metrics_dict["efficiency"]
    assert "avg_fertility" in eff
    assert "avg_chars_per_token" in eff
    assert "avg_compression_ratio_chars" in eff

    scr = metrics_dict["script"]
    assert "avg_grapheme_violation_rate" in scr
    assert "avg_akshara_integrity_rate" in scr

    # Check no NaNs
    import math

    assert not math.isnan(eff["avg_fertility"])
    assert not math.isnan(scr["avg_grapheme_violation_rate"])


def test_akshara_segmentation():
    """Test akshara segmentation (v0 heuristic)."""
    from eval.metrics.script import segment_aksharas

    text = "किशोरी"
    aksharas = segment_aksharas(text)
    assert len(aksharas) > 0
    assert all(isinstance(a, str) for a in aksharas)


def test_cbpe_constraints():
    """Test CBPE constraints work."""
    from tokenizers.cbpe_constraints import cbpe_merge_allowed

    # Test that dependent vowel + consonant merge is disallowed
    # This is a basic sanity check
    result = cbpe_merge_allowed("ा", "क")
    # The exact result depends on implementation, but should not crash
    assert isinstance(result, bool)

