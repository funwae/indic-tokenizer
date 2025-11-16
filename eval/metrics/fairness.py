# eval/metrics/fairness.py
# -*- coding: utf-8 -*-
"""
Fairness metrics for cross-language tokenization evaluation.

Implements metrics from Petrov et al. and GPE paper:
- Tokenization Parity (TP): |t(s_A)| / |t(s_B)| for same content in languages A and B
- Tokenization Premium: E[|t(s_lang)|] / E[|t(s_en)|] - how many more tokens a language pays vs English
- Compression Ratio Disparity: ΔCR(lang1, lang2) - difference in compression ratios
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FairnessMetrics:
    """Fairness metrics for cross-language tokenization evaluation."""

    tokenization_parity: float
    """Tokenization parity: |t(s_A)| / |t(s_B)| for same content in languages A and B.
    
    TP ≈ 1 indicates fair tokenization (same number of tokens for same content).
    TP > 1 means language A uses more tokens than language B.
    TP < 1 means language A uses fewer tokens than language B.
    """

    tokenization_premium: float
    """Tokenization premium: E[|t(s_lang)|] / E[|t(s_en)|].
    
    How many more tokens a language pays vs English on average.
    Lower premium is better (fairer tokenization).
    """

    compression_ratio_disparity: float
    """Compression ratio disparity: ΔCR(lang1, lang2).
    
    Difference in compression ratios between two languages.
    Lower disparity is better.
    """


def tokenization_parity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Calculate tokenization parity between two tokenizations of the same content.
    
    Parameters
    ----------
    tokens_a : List[str]
        Tokens from language A tokenizer.
    tokens_b : List[str]
        Tokens from language B tokenizer (same content).
    
    Returns
    -------
    float
        Tokenization parity: |t(s_A)| / |t(s_B)|.
        Returns 0.0 if tokens_b is empty to avoid division by zero.
    """
    if len(tokens_b) == 0:
        return 0.0
    return len(tokens_a) / len(tokens_b)


def tokenization_premium(tokens_lang: List[str], tokens_en: List[str]) -> float:
    """
    Calculate tokenization premium: how many more tokens a language pays vs English.
    
    Parameters
    ----------
    tokens_lang : List[str]
        Tokens from language tokenizer.
    tokens_en : List[str]
        Tokens from English tokenizer (same content).
    
    Returns
    -------
    float
        Tokenization premium: |t(s_lang)| / |t(s_en)|.
        Returns 0.0 if tokens_en is empty to avoid division by zero.
    """
    if len(tokens_en) == 0:
        return 0.0
    return len(tokens_lang) / len(tokens_en)


def compression_ratio_disparity(
    cr_lang1: float, cr_lang2: float
) -> float:
    """
    Calculate compression ratio disparity between two languages.
    
    Parameters
    ----------
    cr_lang1 : float
        Compression ratio for language 1.
    cr_lang2 : float
        Compression ratio for language 2.
    
    Returns
    -------
    float
        Compression ratio disparity: |cr_lang1 - cr_lang2|.
    """
    return abs(cr_lang1 - cr_lang2)


def evaluate_fairness(
    tokens_lang: List[str],
    tokens_en: List[str],
    cr_lang: float,
    cr_en: float,
    baseline_tokens: List[str] | None = None,
) -> FairnessMetrics:
    """
    Evaluate fairness metrics for a language tokenizer compared to English.
    
    Parameters
    ----------
    tokens_lang : List[str]
        Tokens from language tokenizer.
    tokens_en : List[str]
        Tokens from English tokenizer (same content).
    cr_lang : float
        Compression ratio for language.
    cr_en : float
        Compression ratio for English.
    baseline_tokens : List[str], optional
        Baseline tokens for parity calculation (if different from tokens_en).
        If None, uses tokens_en.
    
    Returns
    -------
    FairnessMetrics
        Fairness metrics object.
    """
    # Tokenization parity (compared to baseline or English)
    if baseline_tokens is not None:
        tp = tokenization_parity(tokens_lang, baseline_tokens)
    else:
        tp = tokenization_parity(tokens_lang, tokens_en)
    
    # Tokenization premium (always compared to English)
    premium = tokenization_premium(tokens_lang, tokens_en)
    
    # Compression ratio disparity
    disparity = compression_ratio_disparity(cr_lang, cr_en)
    
    return FairnessMetrics(
        tokenization_parity=tp,
        tokenization_premium=premium,
        compression_ratio_disparity=disparity,
    )

