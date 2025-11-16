# eval/metrics package
# -*- coding: utf-8 -*-
"""
Comprehensive metrics system for Indic Tokenization Lab.

Phase 1: Efficiency + Script metrics
Phase 2 (Future): Fairness + Morphology metrics
"""

from __future__ import annotations

from typing import List, Optional

from eval.metrics.efficiency import EfficiencyMetrics, evaluate_efficiency
from eval.metrics.script import ScriptMetrics, evaluate_script
from eval.metrics.fairness import FairnessMetrics  # Phase 2 stub
from eval.metrics.morphology import MorphologyMetrics  # Phase 2 stub


__all__ = [
    "ComprehensiveMetrics",
    "EfficiencyMetrics",
    "ScriptMetrics",
    "FairnessMetrics",
    "MorphologyMetrics",
    "evaluate_comprehensive",
]


from dataclasses import dataclass


@dataclass
class ComprehensiveMetrics:
    """Comprehensive metrics for tokenization evaluation (Phase 1)."""

    efficiency: EfficiencyMetrics
    script: ScriptMetrics
    fairness: Optional[FairnessMetrics]  # None in Phase 1
    morphology: Optional[MorphologyMetrics]  # None in Phase 1
    num_tokens: int
    num_words: int
    num_chars: int
    num_graphemes: int
    num_aksharas: int


def evaluate_comprehensive(
    text: str,
    tokens: List[str],
    lang: str = "hi",
    baseline_tokens: Optional[List[str]] = None,
    baseline_tokenizer_id: Optional[str] = None,
) -> ComprehensiveMetrics:
    """
    Evaluate comprehensive metrics for a tokenization (Phase 1).

    Computes efficiency and script metrics. Fairness and morphology
    metrics are deferred to Phase 2.

    Parameters
    ----------
    text : str
        Original input text.
    tokens : List[str]
        List of tokens from tokenizer.
    lang : str
        Language code (default: "hi").
    baseline_tokens : Optional[List[str]]
        Tokens from baseline tokenizer for NSL computation.
    baseline_tokenizer_id : Optional[str]
        ID of baseline tokenizer used (for metadata).

    Returns
    -------
    ComprehensiveMetrics
        Comprehensive metrics including efficiency and script metrics.
    """
    # Count basic statistics
    from eval.fertility import split_words_script_aware
    from tokenizers.grapheme_segmenter import segment_devanagari_graphemes
    from eval.metrics.script import segment_aksharas

    words = split_words_script_aware(text)
    words = [w for w in words if w.strip()]
    graphemes = segment_devanagari_graphemes(text)
    aksharas = segment_aksharas(text) if lang in ("hi", "sa") else []

    # Evaluate efficiency metrics
    efficiency = evaluate_efficiency(
        text=text,
        tokens=tokens,
        baseline_tokens=baseline_tokens,
        baseline_tokenizer_id=baseline_tokenizer_id,
    )

    # Evaluate script metrics
    script = evaluate_script(text=text, tokens=tokens, lang=lang)

    return ComprehensiveMetrics(
        efficiency=efficiency,
        script=script,
        fairness=None,  # Phase 2
        morphology=None,  # Phase 2
        num_tokens=len(tokens),
        num_words=len(words),
        num_chars=len(text),
        num_graphemes=len(graphemes),
        num_aksharas=len(aksharas),
    )

