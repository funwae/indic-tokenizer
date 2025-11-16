# eval/metrics/morphology.py
# -*- coding: utf-8 -*-
"""
Morphology-aware metrics for tokenization evaluation.

Phase 2: Requires MorphTok or similar annotated morphology dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MorphologyMetrics:
    """Morphology metrics require annotated data - Phase 2."""

    pass  # Will be implemented in Phase 2


# Phase 2 planned functions:
# - boundary_precision_recall_f1(gold_boundaries, predicted_boundaries) -> Tuple[float, float, float]
# - morpheme_aligned_token_rate(text, tokens, gold_morphemes) -> float

