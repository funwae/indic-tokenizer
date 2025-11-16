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
from typing import List, Set, Tuple


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
