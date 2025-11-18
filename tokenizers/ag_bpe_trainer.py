# tokenizers/ag_bpe_trainer.py
# -*- coding: utf-8 -*-
"""
Attention-Guided BPE (AG-BPE) Trainer.

Uses language model attention patterns to guide BPE merge decisions,
prioritizing merges between tokens that have high co-attention or
mutual information in the LM.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tokenizers.grapheme_segmenter import segment_devanagari_graphemes
from tokenizers.cbpe_constraints import cbpe_merge_allowed

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

Word = Tuple[str, ...]  # tuple of symbol strings
Vocab = Dict[Word, int]
Pair = Tuple[str, str]


def extract_attention_patterns(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
) -> Dict[Tuple[str, str], float]:
    """
    Extract attention patterns from LM for text samples.

    Computes co-attention scores for token pairs: how much attention
    tokens pay to each other across the corpus.

    Parameters
    ----------
    model : torch.nn.Module
        Trained language model.
    tokenizer : Any
        Tokenizer with encode() method.
    texts : List[str]
        List of text samples.
    device : str
        Device to run model on.

    Returns
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping (token1, token2) -> co-attention score.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for attention extraction")

    model.eval()
    co_attention = defaultdict(float)
    pair_counts = defaultdict(int)

    with torch.no_grad():
        for text in texts:
            try:
                # Tokenize
                token_ids = tokenizer.encode(text)
                if len(token_ids) < 2:
                    continue

                # Create input
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

                # Forward pass to get attention
                # This is a simplified version - actual implementation depends on model architecture
                # For transformer models, we'd extract attention weights from attention layers
                outputs = model(input_ids)

                # Extract attention weights (simplified - would need model-specific extraction)
                # For now, return empty dict as placeholder
                # In real implementation, extract from model.blocks[i].attention.attn_weights

            except Exception as e:
                print(f"  Warning: Failed to process text: {e}")
                continue

    # Normalize co-attention scores
    normalized = {}
    for pair, score in co_attention.items():
        count = pair_counts.get(pair, 1)
        normalized[pair] = score / count if count > 0 else 0.0

    return normalized


def compute_mutual_information(
    pair_freqs: Dict[Pair, int],
    left_freqs: Dict[str, int],
    right_freqs: Dict[str, int],
    total_pairs: int,
) -> Dict[Pair, float]:
    """
    Compute mutual information for token pairs.

    MI(x, y) = log(P(x,y) / (P(x) * P(y)))

    Parameters
    ----------
    pair_freqs : Dict[Pair, int]
        Frequency of each pair.
    left_freqs : Dict[str, int]
        Frequency of left tokens.
    right_freqs : Dict[str, int]
        Frequency of right tokens.
    total_pairs : int
        Total number of pairs.

    Returns
    -------
    Dict[Pair, float]
        Dictionary mapping pairs to mutual information scores.
    """
    import math

    mi_scores = {}

    for (left, right), pair_freq in pair_freqs.items():
        left_freq = left_freqs.get(left, 0)
        right_freq = right_freqs.get(right, 0)

        if pair_freq == 0 or left_freq == 0 or right_freq == 0:
            mi_scores[(left, right)] = 0.0
            continue

        # Probabilities
        p_pair = pair_freq / total_pairs
        p_left = left_freq / total_pairs
        p_right = right_freq / total_pairs

        # Mutual information
        if p_left * p_right > 0:
            mi = math.log(p_pair / (p_left * p_right))
            mi_scores[(left, right)] = mi
        else:
            mi_scores[(left, right)] = 0.0

    return mi_scores


def train_ag_bpe_tokenizer(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    min_pair_frequency: int = 2,
    attention_model_path: Optional[Path] = None,
    attention_weight: float = 0.5,
    mi_weight: float = 0.3,
    frequency_weight: float = 0.2,
    max_lines: Optional[int] = None,
    dev_only: bool = True,
) -> None:
    """
    Train an Attention-Guided BPE tokenizer.

    Combines:
    - Pair frequency (standard BPE)
    - Mutual information (statistical association)
    - Attention patterns (semantic association, if LM available)

    Parameters
    ----------
    corpus_path : Path
        Path to training corpus.
    output_dir : Path
        Output directory for tokenizer.
    vocab_size : int
        Target vocabulary size.
    min_pair_frequency : int
        Minimum pair frequency to consider.
    attention_model_path : Path, optional
        Path to trained LM for attention extraction.
    attention_weight : float
        Weight for attention scores (0.0-1.0).
    mi_weight : float
        Weight for mutual information (0.0-1.0).
    frequency_weight : float
        Weight for frequency (0.0-1.0).
    max_lines : int, optional
        Maximum lines to process.
    dev_only : bool
        Only keep Devanagari graphemes.
    """
    print("Training Attention-Guided BPE tokenizer...")
    print(f"  Corpus: {corpus_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Attention weight: {attention_weight}")
    print(f"  MI weight: {mi_weight}")
    print(f"  Frequency weight: {frequency_weight}")

    # Normalize weights
    total_weight = attention_weight + mi_weight + frequency_weight
    if total_weight > 0:
        attention_weight /= total_weight
        mi_weight /= total_weight
        frequency_weight /= total_weight

    # Load corpus and segment into graphemes
    print("\n1. Loading and segmenting corpus...")
    words: List[Word] = []
    word_end_symbol = "</w>"

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue

            # Segment into graphemes
            graphemes = segment_devanagari_graphemes(line)
            if dev_only:
                # Filter to only Devanagari graphemes
                from tokenizers.cbpe_constraints import is_devanagari_char
                graphemes = [g for g in graphemes if any(is_devanagari_char(c) for c in g)]

            if graphemes:
                word = tuple(graphemes) + (word_end_symbol,)
                words.append(word)

    print(f"  Loaded {len(words)} words")

    # Initialize vocabulary
    print("\n2. Initializing vocabulary...")
    vocab: Vocab = {}
    for word in words:
        for symbol in word:
            if symbol not in vocab:
                vocab[symbol] = len(vocab)

    print(f"  Initial vocab size: {len(vocab)}")

    # Count pair frequencies
    print("\n3. Counting pair frequencies...")
    pair_freqs: Counter[Pair] = Counter()
    left_freqs: Counter[str] = Counter()
    right_freqs: Counter[str] = Counter()

    for word in words:
        for i in range(len(word) - 1):
            left, right = word[i], word[i + 1]
            pair = (left, right)
            pair_freqs[pair] += 1
            left_freqs[left] += 1
            right_freqs[right] += 1

    total_pairs = sum(pair_freqs.values())
    print(f"  Total pairs: {total_pairs}")
    print(f"  Unique pairs: {len(pair_freqs)}")

    # Compute mutual information
    print("\n4. Computing mutual information...")
    mi_scores = compute_mutual_information(
        dict(pair_freqs),
        dict(left_freqs),
        dict(right_freqs),
        total_pairs,
    )
    print(f"  Computed MI for {len(mi_scores)} pairs")

    # Extract attention patterns (if LM available)
    attention_scores: Dict[Pair, float] = {}
    if attention_model_path and attention_model_path.exists() and TORCH_AVAILABLE:
        print("\n5. Extracting attention patterns...")
        try:
            # Load model and tokenizer
            from models.tiny_lm import create_tiny_lm, TinyLMConfig
            # This is a placeholder - would need actual model loading
            print("  ⚠️  Attention extraction not yet implemented (requires model loading)")
        except Exception as e:
            print(f"  ⚠️  Failed to extract attention: {e}")
    else:
        print("\n5. Skipping attention extraction (no model provided)")

    # BPE training loop
    print("\n6. Training BPE with attention/MI guidance...")
    merges: List[Pair] = []
    num_merges = 0
    target_merges = vocab_size - len(vocab)

    while num_merges < target_merges:
        # Compute scores for all valid pairs
        pair_scores: Dict[Pair, float] = {}

        for pair, freq in pair_freqs.items():
            if freq < min_pair_frequency:
                continue

            if not cbpe_merge_allowed(pair[0], pair[1]):
                continue

            # Combined score
            freq_score = freq / total_pairs if total_pairs > 0 else 0.0
            mi_score = mi_scores.get(pair, 0.0)
            attn_score = attention_scores.get(pair, 0.0)

            # Normalize scores (simple min-max normalization)
            # In practice, you'd want more sophisticated normalization
            combined_score = (
                frequency_weight * freq_score +
                mi_weight * max(0, mi_score) +  # MI can be negative
                attention_weight * attn_score
            )

            pair_scores[pair] = combined_score

        if not pair_scores:
            print("  No more valid pairs to merge")
            break

        # Select best pair
        best_pair = max(pair_scores.items(), key=lambda x: x[1])[0]
        merges.append(best_pair)
        num_merges += 1

        if num_merges % 100 == 0:
            print(f"  Merges: {num_merges}/{target_merges}")

        # Apply merge and update frequencies
        new_pair_freqs: Counter[Pair] = Counter()
        new_left_freqs: Counter[str] = Counter()
        new_right_freqs: Counter[str] = Counter()

        merged_symbol = best_pair[0] + best_pair[1]

        for word in words:
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(merged_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # Update word
            word_tuple = tuple(new_word)
            words[words.index(word)] = word_tuple

            # Update frequencies
            for j in range(len(word_tuple) - 1):
                left, right = word_tuple[j], word_tuple[j + 1]
                pair = (left, right)
                new_pair_freqs[pair] += 1
                new_left_freqs[left] += 1
                new_right_freqs[right] += 1

        pair_freqs = new_pair_freqs
        left_freqs = new_left_freqs
        right_freqs = new_right_freqs
        total_pairs = sum(pair_freqs.values())

        # Add merged symbol to vocab
        if merged_symbol not in vocab:
            vocab[merged_symbol] = len(vocab)

    print(f"\n✓ Training complete: {num_merges} merges")

    # Save tokenizer
    print("\n7. Saving tokenizer...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vocab
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({token: idx for token, idx in sorted(vocab.items(), key=lambda x: x[1])}, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Vocab saved: {len(vocab)} tokens")

    # Save merges
    merges_path = output_dir / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left} {right}\n")
    print(f"  ✓ Merges saved: {len(merges)} rules")

    # Save config
    config = {
        "type": "ag_bpe",
        "lang": "hi",
        "script": "Deva",
        "word_end_symbol": word_end_symbol,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "attention_weight": attention_weight,
        "mi_weight": mi_weight,
        "frequency_weight": frequency_weight,
        "attention_model": str(attention_model_path) if attention_model_path else None,
        "description": "Attention-Guided BPE tokenizer with mutual information weighting.",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Config saved")

    print(f"\n✓ Tokenizer saved to {output_dir}")


def main():
    """Main entry point for AG-BPE training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Attention-Guided BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input corpus file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min-pair-frequency", type=int, default=2, help="Min pair frequency")
    parser.add_argument("--attention-model", type=str, help="Path to LM for attention extraction")
    parser.add_argument("--attention-weight", type=float, default=0.5, help="Attention weight")
    parser.add_argument("--mi-weight", type=float, default=0.3, help="MI weight")
    parser.add_argument("--frequency-weight", type=float, default=0.2, help="Frequency weight")
    parser.add_argument("--max-lines", type=int, help="Max lines to process")
    parser.add_argument("--dev-only", action="store_true", help="Only Devanagari graphemes")

    args = parser.parse_args()

    train_ag_bpe_tokenizer(
        corpus_path=Path(args.input),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        attention_model_path=Path(args.attention_model) if args.attention_model else None,
        attention_weight=args.attention_weight,
        mi_weight=args.mi_weight,
        frequency_weight=args.frequency_weight,
        max_lines=args.max_lines,
        dev_only=args.dev_only,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

