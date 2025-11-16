# eval/metrics.py
# -*- coding: utf-8 -*-
"""
Integrated metrics system for Indic Tokenization Lab.

Combines all evaluation metrics (fertility, grapheme violations, etc.)
into a unified evaluation system with scorecard generation.

Phase 1: Now supports comprehensive metrics (efficiency + script).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from eval.fertility import (
    calculate_chars_per_token,
    calculate_fertility,
    compare_fertility,
)
from eval.grapheme_violations import (
    count_violations,
    detect_violations,
    violation_rate,
)
from eval.metrics import (
    ComprehensiveMetrics,
    evaluate_comprehensive,
)


@dataclass
class Metrics:
    """Comprehensive metrics for a single tokenization."""

    fertility: float
    chars_per_token: float
    grapheme_violations: int
    grapheme_violation_rate: float
    num_tokens: int
    num_words: int
    num_chars: int


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple texts."""

    avg_fertility: float
    avg_chars_per_token: float
    total_grapheme_violations: int
    avg_grapheme_violation_rate: float
    total_tokens: int
    total_words: int
    total_chars: int
    num_texts: int


@dataclass
class Scorecard:
    """Scorecard for a tokenizer evaluation."""

    tokenizer_id: str
    tokenizer_name: str
    metrics: Metrics | AggregatedMetrics | ComprehensiveMetrics
    sample_texts: List[str]
    timestamp: str


def evaluate_tokenizer(
    text: str,
    tokenizer,
    lang: str = "hi",
    baseline_tokenizer=None,
    baseline_tokenizer_id: Optional[str] = None,
    use_comprehensive: bool = True,
) -> Metrics | ComprehensiveMetrics:
    """
    Evaluate a single tokenization.

    Parameters
    ----------
    text : str
        Input text to tokenize.
    tokenizer
        Tokenizer object with tokenize() method.
    lang : str
        Language code (default: "hi").
    baseline_tokenizer
        Optional baseline tokenizer for NSL computation.
    baseline_tokenizer_id : Optional[str]
        ID of baseline tokenizer (for metadata).
    use_comprehensive : bool
        If True, return ComprehensiveMetrics (Phase 1). If False, return legacy Metrics.

    Returns
    -------
    Metrics | ComprehensiveMetrics
        Metrics for this tokenization.
    """
    tokens = tokenizer.tokenize(text)

    if use_comprehensive:
        # Use comprehensive metrics (Phase 1)
        baseline_tokens = None
        if baseline_tokenizer is not None:
            baseline_tokens = baseline_tokenizer.tokenize(text)

        return evaluate_comprehensive(
            text=text,
            tokens=tokens,
            lang=lang,
            baseline_tokens=baseline_tokens,
            baseline_tokenizer_id=baseline_tokenizer_id,
        )
    else:
        # Legacy metrics format (backward compatibility)
        words = [w for w in text.split() if w.strip()]

        fertility = calculate_fertility(text, tokens)
        chars_per_token = calculate_chars_per_token(text, tokens)
        grapheme_violations = count_violations(text, tokens)
        grapheme_violation_rate = violation_rate(text, tokens)

        return Metrics(
            fertility=fertility,
            chars_per_token=chars_per_token,
            grapheme_violations=grapheme_violations,
            grapheme_violation_rate=grapheme_violation_rate,
            num_tokens=len(tokens),
            num_words=len(words),
            num_chars=len(text),
        )


def evaluate_batch(
    texts: List[str], tokenizer, lang: str = "hi"
) -> AggregatedMetrics:
    """
    Evaluate tokenizer on a batch of texts.

    Parameters
    ----------
    texts : List[str]
        List of input texts.
    tokenizer
        Tokenizer object with tokenize() method.
    lang : str
        Language code (default: "hi").

    Returns
    -------
    AggregatedMetrics
        Aggregated metrics across all texts.
    """
    all_tokens: List[List[str]] = []
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_violations = 0
    fertilities: List[float] = []
    chars_per_token_list: List[float] = []
    violation_rates: List[float] = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        all_tokens.append(tokens)

        words = [w for w in text.split() if w.strip()]
        total_tokens += len(tokens)
        total_words += len(words)
        total_chars += len(text)

        fertility = calculate_fertility(text, tokens)
        chars_per_token = calculate_chars_per_token(text, tokens)
        violations = count_violations(text, tokens)
        violation_rate_val = violation_rate(text, tokens)

        fertilities.append(fertility)
        chars_per_token_list.append(chars_per_token)
        violation_rates.append(violation_rate_val)
        total_violations += violations

    avg_fertility = sum(fertilities) / len(fertilities) if fertilities else 0.0
    avg_chars_per_token = (
        sum(chars_per_token_list) / len(chars_per_token_list)
        if chars_per_token_list
        else 0.0
    )
    avg_violation_rate = (
        sum(violation_rates) / len(violation_rates) if violation_rates else 0.0
    )

    return AggregatedMetrics(
        avg_fertility=avg_fertility,
        avg_chars_per_token=avg_chars_per_token,
        total_grapheme_violations=total_violations,
        avg_grapheme_violation_rate=avg_violation_rate,
        total_tokens=total_tokens,
        total_words=total_words,
        total_chars=total_chars,
        num_texts=len(texts),
    )


def generate_scorecard(
    tokenizer_results: Dict[str, Metrics | AggregatedMetrics | ComprehensiveMetrics],
    tokenizer_names: Optional[Dict[str, str]] = None,
    sample_texts: Optional[List[str]] = None,
) -> Dict[str, Scorecard]:
    """
    Generate scorecards for multiple tokenizers.

    Parameters
    ----------
    tokenizer_results : Dict[str, Metrics | AggregatedMetrics]
        Dictionary mapping tokenizer IDs to their metrics.
    tokenizer_names : Dict[str, str], optional
        Dictionary mapping tokenizer IDs to display names.
    sample_texts : List[str], optional
        Sample texts used for evaluation.

    Returns
    -------
    Dict[str, Scorecard]
        Dictionary mapping tokenizer IDs to scorecards.
    """
    if tokenizer_names is None:
        tokenizer_names = {tid: tid for tid in tokenizer_results.keys()}

    if sample_texts is None:
        sample_texts = []

    scorecards: Dict[str, Scorecard] = {}
    timestamp = datetime.now().isoformat()

    for tokenizer_id, metrics in tokenizer_results.items():
        tokenizer_name = tokenizer_names.get(tokenizer_id, tokenizer_id)

        scorecard = Scorecard(
            tokenizer_id=tokenizer_id,
            tokenizer_name=tokenizer_name,
            metrics=metrics,
            sample_texts=sample_texts,
            timestamp=timestamp,
        )
        scorecards[tokenizer_id] = scorecard

    return scorecards


def export_scorecard(
    scorecard: Scorecard | Dict[str, Scorecard], format: str = "json"
) -> str:
    """
    Export scorecard to a string format.

    Parameters
    ----------
    scorecard : Scorecard | Dict[str, Scorecard]
        Single scorecard or dictionary of scorecards.
    format : str
        Export format: "json" or "markdown".

    Returns
    -------
    str
        Exported scorecard as string.
    """
    if format == "json":
        if isinstance(scorecard, dict):
            # Convert to serializable format
            data = {
                tid: {
                    "tokenizer_id": sc.tokenizer_id,
                    "tokenizer_name": sc.tokenizer_name,
                    "metrics": asdict(sc.metrics),
                    "sample_texts": sc.sample_texts,
                    "timestamp": sc.timestamp,
                }
                for tid, sc in scorecard.items()
            }
        else:
            data = {
                "tokenizer_id": scorecard.tokenizer_id,
                "tokenizer_name": scorecard.tokenizer_name,
                "metrics": asdict(scorecard.metrics),
                "sample_texts": scorecard.sample_texts,
                "timestamp": scorecard.timestamp,
            }
        return json.dumps(data, indent=2, ensure_ascii=False)

    elif format == "markdown":
        lines = []
        lines.append("# Tokenizer Evaluation Scorecard")
        lines.append("")

        if isinstance(scorecard, dict):
            for tid, sc in scorecard.items():
                lines.append(f"## {sc.tokenizer_name} ({sc.tokenizer_id})")
                lines.append("")
                lines.append(f"**Timestamp:** {sc.timestamp}")
                lines.append("")

                metrics = sc.metrics
                if isinstance(metrics, ComprehensiveMetrics):
                    lines.append("### Comprehensive Metrics (Phase 1)")
                    lines.append("")
                    lines.append("#### Efficiency Metrics")
                    lines.append("")
                    eff = metrics.efficiency
                    lines.append(f"- **Fertility:** {eff.fertility:.3f} tokens/word")
                    lines.append(f"- **Chars per Token:** {eff.chars_per_token:.3f}")
                    lines.append(f"- **Compression Ratio (chars):** {eff.compression_ratio_chars:.3f}")
                    lines.append(f"- **Compression Ratio (graphemes):** {eff.compression_ratio_graphemes:.3f}")
                    if eff.normalized_sequence_length is not None:
                        lines.append(f"- **Normalized Sequence Length:** {eff.normalized_sequence_length:.3f} (baseline: {eff.baseline_tokenizer_id or 'N/A'})")
                    lines.append(f"- **Proportion Continued Words:** {eff.proportion_continued_words:.2%}")
                    lines.append(f"- **UNK Rate:** {eff.unk_rate:.2%}")
                    lines.append("")
                    lines.append("#### Script Metrics")
                    lines.append("")
                    scr = metrics.script
                    lines.append(f"- **Grapheme Violation Rate:** {scr.grapheme_violation_rate:.2%}")
                    lines.append(f"- **Akshara Integrity Rate:** {scr.akshara_integrity_rate:.2%} (v0 heuristic)")
                    lines.append(f"- **Akshara Split Count:** {scr.akshara_split_count}")
                    lines.append(f"- **Dependent Vowel Split Rate:** {scr.dependent_vowel_split_rate:.2%}")
                    lines.append(f"- **Grapheme-Aligned Token Rate:** {scr.grapheme_aligned_token_rate:.2%}")
                    lines.append(f"  - Single grapheme tokens: {scr.single_grapheme_tokens}")
                    lines.append(f"  - Multi grapheme tokens: {scr.multi_grapheme_tokens}")
                    lines.append(f"  - Fragment tokens: {scr.grapheme_fragment_tokens}")
                    lines.append(f"- **Devanagari Token Share:** {scr.devanagari_token_share:.2%}")
                    lines.append(f"- **Mixed Script Token Share:** {scr.mixed_script_token_share:.2%}")
                    lines.append("")
                    lines.append("#### Summary Statistics")
                    lines.append("")
                    lines.append(f"- **Tokens:** {metrics.num_tokens}")
                    lines.append(f"- **Words:** {metrics.num_words}")
                    lines.append(f"- **Chars:** {metrics.num_chars}")
                    lines.append(f"- **Graphemes:** {metrics.num_graphemes}")
                    lines.append(f"- **Aksharas:** {metrics.num_aksharas}")
                    lines.append("")
                    lines.append("*Note: Fairness and Morphology metrics are planned for Phase 2*")
                elif isinstance(metrics, Metrics):
                    lines.append("### Metrics")
                    lines.append("")
                    lines.append(f"- **Fertility:** {metrics.fertility:.3f} tokens/word")
                    lines.append(
                        f"- **Chars per Token:** {metrics.chars_per_token:.3f}"
                    )
                    lines.append(
                        f"- **Grapheme Violations:** {metrics.grapheme_violations}"
                    )
                    lines.append(
                        f"- **Violation Rate:** {metrics.grapheme_violation_rate:.2%}"
                    )
                    lines.append(f"- **Tokens:** {metrics.num_tokens}")
                    lines.append(f"- **Words:** {metrics.num_words}")
                    lines.append(f"- **Chars:** {metrics.num_chars}")
                elif isinstance(metrics, AggregatedMetrics):
                    lines.append("### Aggregated Metrics")
                    lines.append("")
                    lines.append(f"- **Avg Fertility:** {metrics.avg_fertility:.3f} tokens/word")
                    lines.append(
                        f"- **Avg Chars per Token:** {metrics.avg_chars_per_token:.3f}"
                    )
                    lines.append(
                        f"- **Total Grapheme Violations:** {metrics.total_grapheme_violations}"
                    )
                    lines.append(
                        f"- **Avg Violation Rate:** {metrics.avg_grapheme_violation_rate:.2%}"
                    )
                    lines.append(f"- **Total Tokens:** {metrics.total_tokens}")
                    lines.append(f"- **Total Words:** {metrics.total_words}")
                    lines.append(f"- **Total Chars:** {metrics.total_chars}")
                    lines.append(f"- **Number of Texts:** {metrics.num_texts}")

                lines.append("")
        else:
            sc = scorecard
            lines.append(f"## {sc.tokenizer_name} ({sc.tokenizer_id})")
            lines.append("")
            lines.append(f"**Timestamp:** {sc.timestamp}")
            lines.append("")

            metrics = sc.metrics
            if isinstance(metrics, ComprehensiveMetrics):
                lines.append("### Comprehensive Metrics (Phase 1)")
                lines.append("")
                lines.append("#### Efficiency Metrics")
                lines.append("")
                eff = metrics.efficiency
                lines.append(f"- **Fertility:** {eff.fertility:.3f} tokens/word")
                lines.append(f"- **Chars per Token:** {eff.chars_per_token:.3f}")
                lines.append(f"- **Compression Ratio (chars):** {eff.compression_ratio_chars:.3f}")
                lines.append(f"- **Compression Ratio (graphemes):** {eff.compression_ratio_graphemes:.3f}")
                if eff.normalized_sequence_length is not None:
                    lines.append(f"- **Normalized Sequence Length:** {eff.normalized_sequence_length:.3f} (baseline: {eff.baseline_tokenizer_id or 'N/A'})")
                lines.append(f"- **Proportion Continued Words:** {eff.proportion_continued_words:.2%}")
                lines.append(f"- **UNK Rate:** {eff.unk_rate:.2%}")
                lines.append("")
                lines.append("#### Script Metrics")
                lines.append("")
                scr = metrics.script
                lines.append(f"- **Grapheme Violation Rate:** {scr.grapheme_violation_rate:.2%}")
                lines.append(f"- **Akshara Integrity Rate:** {scr.akshara_integrity_rate:.2%} (v0 heuristic)")
                lines.append(f"- **Akshara Split Count:** {scr.akshara_split_count}")
                lines.append(f"- **Dependent Vowel Split Rate:** {scr.dependent_vowel_split_rate:.2%}")
                lines.append(f"- **Grapheme-Aligned Token Rate:** {scr.grapheme_aligned_token_rate:.2%}")
                lines.append(f"  - Single grapheme tokens: {scr.single_grapheme_tokens}")
                lines.append(f"  - Multi grapheme tokens: {scr.multi_grapheme_tokens}")
                lines.append(f"  - Fragment tokens: {scr.grapheme_fragment_tokens}")
                lines.append(f"- **Devanagari Token Share:** {scr.devanagari_token_share:.2%}")
                lines.append(f"- **Mixed Script Token Share:** {scr.mixed_script_token_share:.2%}")
                lines.append("")
                lines.append("#### Summary Statistics")
                lines.append("")
                lines.append(f"- **Tokens:** {metrics.num_tokens}")
                lines.append(f"- **Words:** {metrics.num_words}")
                lines.append(f"- **Chars:** {metrics.num_chars}")
                lines.append(f"- **Graphemes:** {metrics.num_graphemes}")
                lines.append(f"- **Aksharas:** {metrics.num_aksharas}")
                lines.append("")
                lines.append("*Note: Fairness and Morphology metrics are planned for Phase 2*")
            elif isinstance(metrics, Metrics):
                lines.append("### Metrics")
                lines.append("")
                lines.append(f"- **Fertility:** {metrics.fertility:.3f} tokens/word")
                lines.append(f"- **Chars per Token:** {metrics.chars_per_token:.3f}")
                lines.append(f"- **Grapheme Violations:** {metrics.grapheme_violations}")
                lines.append(
                    f"- **Violation Rate:** {metrics.grapheme_violation_rate:.2%}"
                )
                lines.append(f"- **Tokens:** {metrics.num_tokens}")
                lines.append(f"- **Words:** {metrics.num_words}")
                lines.append(f"- **Chars:** {metrics.num_chars}")

        lines.append("")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unsupported format: {format}")

