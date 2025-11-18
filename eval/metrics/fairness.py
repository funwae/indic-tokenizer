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
from typing import List, Dict, Any
import statistics


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


def compute_tp(
    tokenizer: Any,
    en_texts: List[str],
    hi_texts: List[str],
) -> Dict[str, float]:
    """
    Compute tokenization parity (TP) statistics.

    For each pair, computes len(tokens_hi) / len(tokens_en).
    Returns aggregated statistics: mean, median, p10, p90.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer object with tokenize() method.
    en_texts : List[str]
        List of English texts.
    hi_texts : List[str]
        List of Hindi texts (parallel to en_texts).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'mean', 'median', 'p10', 'p90'.
    """
    if len(en_texts) != len(hi_texts):
        raise ValueError("en_texts and hi_texts must have the same length")

    tp_values = []

    for en_text, hi_text in zip(en_texts, hi_texts):
        en_tokens = tokenizer.tokenize(en_text)
        hi_tokens = tokenizer.tokenize(hi_text)

        tp = tokenization_parity(hi_tokens, en_tokens)
        if tp > 0:  # Skip invalid values
            tp_values.append(tp)

    if not tp_values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}

    tp_values_sorted = sorted(tp_values)
    n = len(tp_values_sorted)

    return {
        "mean": statistics.mean(tp_values),
        "median": statistics.median(tp_values),
        "p10": tp_values_sorted[int(n * 0.1)] if n > 0 else 0.0,
        "p90": tp_values_sorted[int(n * 0.9)] if n > 0 else 0.0,
    }


def compute_nsl_cross(
    tokenizer: Any,
    ref_tokenizer: Any,
    texts_lang: List[str],
) -> float:
    """
    Compute Normalized Sequence Length (NSL) vs baseline per language.

    NSL = E[|t(s)|] / E[|t_ref(s)|] for texts in the given language.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer to evaluate (with tokenize() method).
    ref_tokenizer : Any
        Reference/baseline tokenizer (with tokenize() method).
    texts_lang : List[str]
        List of texts in the target language.

    Returns
    -------
    float
        NSL value (average ratio of token counts).
    """
    if not texts_lang:
        return 0.0

    ratios = []

    for text in texts_lang:
        tokens = tokenizer.tokenize(text)
        ref_tokens = ref_tokenizer.tokenize(text)

        if len(ref_tokens) > 0:
            ratio = len(tokens) / len(ref_tokens)
            ratios.append(ratio)

    if not ratios:
        return 0.0

    return statistics.mean(ratios)


def compute_token_tax(
    tokenizer: Any,
    baseline_tokenizer: Any,
    en_texts: List[str],
    hi_texts: List[str],
) -> Dict[str, float]:
    """
    Compute token tax for Hindi vs English relative to baseline.

    Token tax measures how much more tokens a language pays compared to
    a baseline tokenizer. Computes:
    - premium_hi = E[|t(hi)|] / E[|t_base(hi)|]
    - premium_en = E[|t(en)|] / E[|t_base(en)|]
    - tax_ratio = premium_hi / premium_en

    Parameters
    ----------
    tokenizer : Any
        Tokenizer to evaluate (with tokenize() method).
    baseline_tokenizer : Any
        Baseline tokenizer for comparison (with tokenize() method).
    en_texts : List[str]
        List of English texts.
    hi_texts : List[str]
        List of Hindi texts (parallel to en_texts).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'premium_hi', 'premium_en', 'tax_ratio'.
    """
    if len(en_texts) != len(hi_texts):
        raise ValueError("en_texts and hi_texts must have the same length")

    hi_ratios = []
    en_ratios = []

    for hi_text, en_text in zip(hi_texts, en_texts):
        # Hindi
        hi_tokens = tokenizer.tokenize(hi_text)
        hi_base_tokens = baseline_tokenizer.tokenize(hi_text)
        if len(hi_base_tokens) > 0:
            hi_ratios.append(len(hi_tokens) / len(hi_base_tokens))

        # English
        en_tokens = tokenizer.tokenize(en_text)
        en_base_tokens = baseline_tokenizer.tokenize(en_text)
        if len(en_base_tokens) > 0:
            en_ratios.append(len(en_tokens) / len(en_base_tokens))

    premium_hi = statistics.mean(hi_ratios) if hi_ratios else 0.0
    premium_en = statistics.mean(en_ratios) if en_ratios else 0.0

    tax_ratio = premium_hi / premium_en if premium_en > 0 else 0.0

    return {
        "premium_hi": premium_hi,
        "premium_en": premium_en,
        "tax_ratio": tax_ratio,
    }

