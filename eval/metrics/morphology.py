# eval/metrics/morphology.py
# -*- coding: utf-8 -*-
"""
Morphology metrics for tokenization evaluation.

Implements metrics inspired by MorphTok and EvalTok:
- Boundary precision/recall/F1 over morpheme boundaries
- Morpheme-aligned token rate (tokens that exactly match morphemes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Any
import csv


@dataclass
class MorphologyMetrics:
    """Morphology metrics for tokenization evaluation."""

    boundary_precision: float
    """Precision of token boundaries matching morpheme boundaries."""

    boundary_recall: float
    """Recall of token boundaries matching morpheme boundaries."""

    boundary_f1: float
    """F1 score of token boundaries matching morpheme boundaries."""

    morpheme_aligned_token_rate: float
    """Proportion of tokens that exactly match a single morpheme."""

    num_morphemes: int
    """Total number of morphemes in the text."""

    num_tokens: int
    """Total number of tokens."""


def extract_morpheme_boundaries(annotated_text: str) -> List[int]:
    """
    Extract morpheme boundary positions from annotated text.

    Format: "word1|morpheme1+morpheme2 word2|morpheme3"
    Returns list of character positions where morpheme boundaries occur.

    Parameters
    ----------
    annotated_text : str
        Text with morpheme boundaries marked (format: word|morpheme1+morpheme2).

    Returns
    -------
    List[int]
        List of character positions where morpheme boundaries occur.
    """
    boundaries: List[int] = []
    pos = 0

    # Split by words (whitespace)
    words = annotated_text.split()

    for word_annotation in words:
        if '|' not in word_annotation:
            # No morpheme annotation, skip
            pos += len(word_annotation) + 1  # +1 for space
            continue

        # Extract word and morpheme annotation
        parts = word_annotation.split('|', 1)
        if len(parts) != 2:
            pos += len(word_annotation) + 1
            continue

        word, morpheme_str = parts

        # Find morpheme boundaries within the word
        # Morphemes are separated by '+'
        morphemes = morpheme_str.split('+')

        if len(morphemes) > 1:
            # Calculate boundary positions within the word
            morpheme_start = 0
            for i, morpheme in enumerate(morphemes[:-1]):  # All except last
                morpheme_start += len(morpheme)
                # Boundary is at the end of this morpheme
                boundaries.append(pos + morpheme_start)

        pos += len(word) + 1  # +1 for space

    return sorted(set(boundaries))  # Remove duplicates and sort


def extract_token_boundaries(text: str, tokens: List[str]) -> List[int]:
    """
    Extract token boundary positions from tokenized text.

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.

    Returns
    -------
    List[int]
        List of character positions where token boundaries occur.
    """
    boundaries: List[int] = []
    pos = 0

    # Reconstruct text from tokens to find boundaries
    # This is approximate - assumes tokens appear in order in the text
    for i, token in enumerate(tokens):
        # Clean token (remove special markers like </w>, ##, etc.)
        clean_token = token.replace('</w>', '').replace('##', '').replace('▁', '').strip()

        if not clean_token:
            continue

        # Find token in text starting from current position
        token_pos = text.find(clean_token, pos)
        if token_pos == -1:
            # Token not found, skip
            continue

        # Boundary is at the end of this token
        boundary = token_pos + len(clean_token)
        if boundary < len(text):
            boundaries.append(boundary)

        pos = token_pos + len(clean_token)

    return sorted(set(boundaries))


def boundary_precision_recall_f1(
    gold_boundaries: List[int],
    predicted_boundaries: List[int],
) -> Tuple[float, float, float]:
    """
    Calculate boundary precision, recall, and F1.

    Parameters
    ----------
    gold_boundaries : List[int]
        Gold standard morpheme boundary positions.
    predicted_boundaries : List[int]
        Predicted token boundary positions.

    Returns
    -------
    Tuple[float, float, float]
        (precision, recall, f1)
    """
    gold_set = set(gold_boundaries)
    pred_set = set(predicted_boundaries)

    # True positives: boundaries that are in both sets
    tp = len(gold_set & pred_set)

    # Precision: tp / (tp + fp) = tp / len(pred_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0

    # Recall: tp / (tp + fn) = tp / len(gold_set)
    recall = tp / len(gold_set) if len(gold_set) > 0 else 0.0

    # F1: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def morpheme_aligned_token_rate(
    text: str,
    tokens: List[str],
    gold_morphemes: List[str],
) -> float:
    """
    Calculate the proportion of tokens that exactly match a single morpheme.

    Parameters
    ----------
    text : str
        Original text.
    tokens : List[str]
        List of tokens.
    gold_morphemes : List[str]
        List of gold standard morphemes.

    Returns
    -------
    float
        Proportion of tokens that exactly match a single morpheme.
    """
    if len(tokens) == 0:
        return 0.0

    # Clean tokens (remove special markers)
    clean_tokens = [
        t.replace('</w>', '').replace('##', '').replace('▁', '').strip()
        for t in tokens
    ]
    clean_tokens = [t for t in clean_tokens if t]

    # Create set of gold morphemes (normalized)
    gold_set = set(m.strip() for m in gold_morphemes if m.strip())

    # Count tokens that match a morpheme exactly
    matched = sum(1 for token in clean_tokens if token in gold_set)

    return matched / len(clean_tokens) if len(clean_tokens) > 0 else 0.0


def evaluate_morphology(
    text: str,
    tokens: List[str],
    annotated_text: str,
) -> MorphologyMetrics:
    """
    Evaluate morphology metrics for a tokenization.

    Parameters
    ----------
    text : str
        Original text (without morpheme annotations).
    tokens : List[str]
        List of tokens from tokenizer.
    annotated_text : str
        Text with morpheme boundaries marked (format: word|morpheme1+morpheme2).

    Returns
    -------
    MorphologyMetrics
        Morphology metrics object.
    """
    # Extract morpheme boundaries
    gold_boundaries = extract_morpheme_boundaries(annotated_text)

    # Extract token boundaries
    predicted_boundaries = extract_token_boundaries(text, tokens)

    # Calculate boundary precision/recall/F1
    precision, recall, f1 = boundary_precision_recall_f1(
        gold_boundaries, predicted_boundaries
    )

    # Extract morphemes from annotated text
    morphemes: List[str] = []
    for word_annotation in annotated_text.split():
        if '|' not in word_annotation:
            continue
        parts = word_annotation.split('|', 1)
        if len(parts) == 2:
            morpheme_str = parts[1]
            # Split by '+' to get individual morphemes
            morphemes.extend(morpheme_str.split('+'))

    # Calculate morpheme-aligned token rate
    aligned_rate = morpheme_aligned_token_rate(text, tokens, morphemes)

    return MorphologyMetrics(
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        morpheme_aligned_token_rate=aligned_rate,
        num_morphemes=len(morphemes),
        num_tokens=len(tokens),
    )


def load_morphology_tsv(tsv_path: str) -> List[Dict[str, str]]:
    """
    Load morphology-annotated dataset from TSV file.

    Format: id, text, morphemes (space-separated)

    Parameters
    ----------
    tsv_path : str
        Path to TSV file.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries with keys: 'id', 'text', 'morphemes'.
    """
    examples = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            examples.append({
                'id': row.get('id', ''),
                'text': row.get('text', ''),
                'morphemes': row.get('morphemes', ''),
            })
    return examples


def compute_boundary_f1(
    tokenizer: Any,
    examples: List[Dict[str, str]],
) -> Dict[str, float]:
    """
    Compute boundary precision, recall, and F1 for a tokenizer.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer object with tokenize() method.
    examples : List[Dict[str, str]]
        List of examples with 'text' and 'morphemes' keys.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'precision', 'recall', 'f1' keys.
    """
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for example in examples:
        text = example['text']
        morphemes_str = example['morphemes']

        # Tokenize
        tokens = tokenizer.tokenize(text)

        # Convert morpheme string to boundary positions
        # Morphemes are space-separated, so we need to map them to character positions
        gold_boundaries = _morphemes_to_boundaries(text, morphemes_str)

        # Extract token boundaries
        predicted_boundaries = extract_token_boundaries(text, tokens)

        # Compute precision/recall/F1
        precision, recall, f1 = boundary_precision_recall_f1(
            gold_boundaries, predicted_boundaries
        )

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    return {
        'precision': sum(all_precisions) / len(all_precisions) if all_precisions else 0.0,
        'recall': sum(all_recalls) / len(all_recalls) if all_recalls else 0.0,
        'f1': sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
    }


def _morphemes_to_boundaries(text: str, morphemes_str: str) -> List[int]:
    """
    Convert space-separated morpheme string to boundary positions.

    Parameters
    ----------
    text : str
        Original text.
    morphemes_str : str
        Space-separated morphemes.

    Returns
    -------
    List[int]
        List of character positions where morpheme boundaries occur.
    """
    morphemes = [m.strip() for m in morphemes_str.split() if m.strip()]
    boundaries = []
    pos = 0

    for i, morpheme in enumerate(morphemes[:-1]):  # All except last
        # Find morpheme in text starting from current position
        morpheme_pos = text.find(morpheme, pos)
        if morpheme_pos == -1:
            # Morpheme not found, skip
            continue

        # Boundary is at the end of this morpheme
        boundary = morpheme_pos + len(morpheme)
        if boundary < len(text):
            boundaries.append(boundary)

        pos = morpheme_pos + len(morpheme)

    return sorted(set(boundaries))


def compute_morpheme_alignment(
    tokenizer: Any,
    examples: List[Dict[str, str]],
) -> Dict[str, float]:
    """
    Compute morpheme alignment metrics.

    Returns:
    - % of tokens that exactly match a gold morpheme span
    - % of morphemes covered by a single token

    Parameters
    ----------
    tokenizer : Any
        Tokenizer object with tokenize() method.
    examples : List[Dict[str, str]]
        List of examples with 'text' and 'morphemes' keys.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'token_match_rate' and 'morpheme_coverage_rate' keys.
    """
    total_tokens = 0
    matched_tokens = 0
    total_morphemes = 0
    covered_morphemes = 0

    for example in examples:
        text = example['text']
        morphemes_str = example['morphemes']
        morphemes = [m.strip() for m in morphemes_str.split() if m.strip()]

        # Tokenize
        tokens = tokenizer.tokenize(text)

        # Clean tokens
        clean_tokens = [
            t.replace('</w>', '').replace('##', '').replace('▁', '').strip()
            for t in tokens
        ]
        clean_tokens = [t for t in clean_tokens if t]

        # Create set of gold morphemes
        gold_set = set(m for m in morphemes if m)

        # Count tokens that match morphemes exactly
        for token in clean_tokens:
            total_tokens += 1
            if token in gold_set:
                matched_tokens += 1

        # Count morphemes covered by single tokens
        for morpheme in morphemes:
            total_morphemes += 1
            # Check if morpheme appears as a single token
            if morpheme in clean_tokens:
                covered_morphemes += 1

    return {
        'token_match_rate': matched_tokens / total_tokens if total_tokens > 0 else 0.0,
        'morpheme_coverage_rate': covered_morphemes / total_morphemes if total_morphemes > 0 else 0.0,
    }


def compute_morph_fragmentation(
    tokenizer: Any,
    examples: List[Dict[str, str]],
) -> float:
    """
    Compute average number of tokens per morpheme (morphological fragmentation).

    Similar to fertility but at morpheme level.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer object with tokenize() method.
    examples : List[Dict[str, str]]
        List of examples with 'text' and 'morphemes' keys.

    Returns
    -------
    float
        Average tokens per morpheme.
    """
    total_tokens = 0
    total_morphemes = 0

    for example in examples:
        text = example['text']
        morphemes_str = example['morphemes']
        morphemes = [m.strip() for m in morphemes_str.split() if m.strip()]

        # Tokenize
        tokens = tokenizer.tokenize(text)

        # Clean tokens
        clean_tokens = [
            t.replace('</w>', '').replace('##', '').replace('▁', '').strip()
            for t in tokens
        ]
        clean_tokens = [t for t in clean_tokens if t]

        total_tokens += len(clean_tokens)
        total_morphemes += len(morphemes)

    return total_tokens / total_morphemes if total_morphemes > 0 else 0.0
