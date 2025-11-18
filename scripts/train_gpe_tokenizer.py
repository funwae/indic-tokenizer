#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_gpe_tokenizer.py

Train a **Grapheme Pair Encoding (GPE)-style** BPE tokenizer for Hindi using:

  - Unicode extended grapheme clusters as base units.

  - A simple Sennrich-style BPE loop over those graphemes.

  - A minimal CBPE constraint hook to avoid illegal Devanagari merges.

Usage example:

  python scripts/train_gpe_tokenizer.py \
      --input data/hindi/corpus.txt \
      --output-dir models/gpe_hi_v0 \
      --vocab-size 32000 \
      --min-pair-frequency 2 \
      --max-lines 500000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Add project root to path to import our tokenizers package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizers.grapheme_segmenter import segment_devanagari_graphemes
from tokenizers.cbpe_constraints import cbpe_merge_allowed

Word = Tuple[str, ...]  # tuple of symbol strings, e.g. ("क", "्षा", "</w>")
Vocab = Dict[Word, int]
Pair = Tuple[str, str]

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def iter_lines(path: Path, max_lines: int | None = None) -> Iterable[str]:
    """
    Yield lines from a UTF-8 text file.

    Each line is treated as a sequence of whitespace-separated "words".
    """
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            count += 1
            if max_lines is not None and count >= max_lines:
                break

def build_vocab_from_corpus(
    input_path: Path,
    max_lines: int | None = None,
    lowercase: bool = False,
    dev_only: bool = False,
    word_end_symbol: str = "</w>",
) -> Vocab:
    """
    Build a word->frequency vocab where each word is a tuple of grapheme symbols
    plus a word-end sentinel.

        "किशोरी" -> ("कि", "शो", "री", "</w>")

    Parameters
    ----------
    input_path : Path
        Corpus file.
    max_lines : int or None
        Optional limit on number of lines to read (for faster experiments).
    lowercase : bool
        If True, lowercase text before processing (primarily affects Latin segments).
    dev_only : bool
        If True, only keep graphemes containing Devanagari code points.
    word_end_symbol : str
        Sentinel appended to each word.

    Returns
    -------
    Vocab
        Mapping of word-as-tuple-of-symbols to frequency.
    """
    vocab: Vocab = {}
    for line in iter_lines(input_path, max_lines=max_lines):
        if lowercase:
            line = line.lower()
        # Simple whitespace split into surface words
        for raw_word in line.split():
            if not raw_word:
                continue
            # Grapheme segmentation (GPE pretokenization)
            graphemes = segment_devanagari_graphemes(
                raw_word,
                keep_non_devanagari=not dev_only,
            )
            if not graphemes:
                continue
            symbols = graphemes + [word_end_symbol]
            key = tuple(symbols)
            vocab[key] = vocab.get(key, 0) + 1
    return vocab

# ---------------------------------------------------------------------------
# BPE utilities (Sennrich-style, but over graphemes)
# ---------------------------------------------------------------------------

def get_pair_stats(vocab: Vocab) -> Counter[Pair]:
    """
    Count frequency of each adjacent pair of symbols across vocab.

    This is the usual BPE "pair stats" step.
    """
    pairs: Counter[Pair] = Counter()
    for word, freq in vocab.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs

def merge_vocab(pair: Pair, vocab: Vocab) -> Vocab:
    """
    Apply a single BPE merge operation `pair` to the entire vocab.

    Replace every occurrence of (left, right) with left+right as a new symbol.
    """
    left, right = pair
    merged_symbol = left + right  # GPE-style: concatenated grapheme symbols
    new_vocab: Vocab = {}
    for word, freq in vocab.items():
        if len(word) < 2:
            new_vocab[word] = freq
            continue
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                # Merge this pair into a single symbol
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = new_vocab.get(tuple(new_word), 0) + freq
    return new_vocab

def train_bpe(
    vocab: Vocab,
    target_vocab_size: int,
    min_pair_frequency: int,
    word_end_symbol: str = "</w>",
) -> Tuple[List[Pair], List[str]]:
    """
    Train a GPE-style BPE tokenizer over grapheme-based vocab.

    Parameters
    ----------
    vocab : Vocab
        Word->frequency map where words are tuples of grapheme symbols + </w>.
    target_vocab_size : int
        Desired approximate vocabulary size (including base symbols + merged symbols).
    min_pair_frequency : int
        Stop when the most frequent pair has frequency < this value.
    word_end_symbol : str
        Sentinel symbol that marks end-of-word; we avoid merging across it.

    Returns
    -------
    merges : List[Pair]
        List of merges in order of application (left, right).
    symbols : List[str]
        Final list of symbols observed after training (base + merged).
    """
    # Initial symbol set
    symbols = set()  # type: set[str]
    for word in vocab.keys():
        symbols.update(word)
    merges: List[Pair] = []
    # Number of merges we can perform before hitting the target size
    max_merges = max(target_vocab_size - len(symbols), 0)

    for merge_idx in range(max_merges):
        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break

        # Find the best valid pair (may need to skip invalid ones)
        best_pair = None
        best_freq = 0

        for pair, freq in pair_stats.most_common():
            if freq < min_pair_frequency:
                # No sufficiently frequent pairs left
                break

            left, right = pair

            # Never merge across word boundary sentinel
            if left == word_end_symbol or right == word_end_symbol:
                continue

            # Apply CBPE constraints (e.g., avoid symbols starting with matras/virama)
            if not cbpe_merge_allowed(left, right):
                continue

            # This pair is valid
            best_pair = pair
            best_freq = freq
            break

        if best_pair is None:
            # No valid pairs found
            break

        # Perform the merge
        vocab = merge_vocab(best_pair, vocab)
        left, right = best_pair
        merged_symbol = left + right
        symbols.add(merged_symbol)
        merges.append(best_pair)

        # Optional: simple progress print
        if (merge_idx + 1) % 1000 == 0:
            print(
                f"[BPE] Performed {merge_idx + 1} merges "
                f"(current vocab size ~ {len(symbols)})"
            )

    return merges, sorted(symbols)

# ---------------------------------------------------------------------------
# Vocab / merges export
# ---------------------------------------------------------------------------

def build_token_to_id(
    symbols: List[str],
    add_special_tokens: bool = True,
) -> Dict[str, int]:
    """
    Build a token->id mapping.

    We optionally add a small set of special tokens first, then assign IDs
    to all symbols in a deterministic (sorted) order.
    """
    token_to_id: Dict[str, int] = {}
    next_id = 0
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    if add_special_tokens:
        for tok in special_tokens:
            token_to_id[tok] = next_id
            next_id += 1

    for sym in symbols:
        if sym in token_to_id:
            continue
        token_to_id[sym] = next_id
        next_id += 1

    return token_to_id

def save_tokenizer(
    output_dir: Path,
    merges: List[Pair],
    symbols: List[str],
    lang: str = "hi",
    script: str = "Deva",
    word_end_symbol: str = "</w>",
) -> None:
    """
    Save vocab, merges, and config into `output_dir`.

    Files:
        - vocab.json
        - merges.txt
        - config.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build vocab
    token_to_id = build_token_to_id(symbols)

    # Save vocab.json
    vocab_path = output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)

    # Save merges.txt in a HuggingFace/BPE-compatible format
    merges_path = output_dir / "merges.txt"
    with merges_path.open("w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for left, right in merges:
            f.write(f"{left} {right}\n")

    # Save a small config.json with metadata
    config = {
        "type": "gpe_bpe",
        "lang": lang,
        "script": script,
        "word_end_symbol": word_end_symbol,
        "vocab_size": len(token_to_id),
        "num_merges": len(merges),
        "description": (
            "Grapheme Pair Encoding-style BPE trained over Unicode grapheme "
            "clusters with constrained merges for Devanagari."
        ),
    }

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[GPE] Saved tokenizer to {output_dir}")
    print(f"       vocab_size = {len(token_to_id)}")
    print(f"       num_merges = {len(merges)}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPE-style grapheme-BPE tokenizer for Hindi."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input corpus text file (UTF-8).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save tokenizer artifacts (vocab, merges, config).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size (approximate, including merged symbols).",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency to consider a BPE merge.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on number of lines to read from corpus.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase corpus before processing (mainly for Latin segments).",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Only keep graphemes that contain Devanagari code points.",
    )
    parser.add_argument(
        "--word-end-symbol",
        type=str,
        default="</w>",
        help="Word-end sentinel symbol to append to each word.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Training profile preset (e.g., 'hi_v1' for Hindi v1 settings).",
    )

    args = parser.parse_args()

    # Apply profile presets
    if args.profile == "hi_v1":
        # Hindi v1 profile: optimized for Devanagari tokenization
        if args.vocab_size == 32000:  # Only override if not explicitly set
            args.vocab_size = 32000
        if args.min_pair_frequency == 2:  # Only override if not explicitly set
            args.min_pair_frequency = 2
        args.dev_only = True  # Force dev-only for Hindi
        if args.max_lines is None:
            args.max_lines = 500000  # Default to 500k lines for hi_v1

    return args

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    print(f"[GPE] Building grapheme-based vocab from {input_path} ...")
    vocab = build_vocab_from_corpus(
        input_path=input_path,
        max_lines=args.max_lines,
        lowercase=args.lowercase,
        dev_only=args.dev_only,
        word_end_symbol=args.word_end_symbol,
    )
    print(f"[GPE] Vocab contains {len(vocab)} unique word types.")

    print(
        f"[GPE] Training BPE over graphemes with target vocab size "
        f"{args.vocab_size}, min_pair_frequency={args.min_pair_frequency} ..."
    )
    merges, symbols = train_bpe(
        vocab=vocab,
        target_vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        word_end_symbol=args.word_end_symbol,
    )

    print(f"[GPE] Learned {len(merges)} merges.")
    print(f"[GPE] Final symbol set size: {len(symbols)}")

    save_tokenizer(
        output_dir=output_dir,
        merges=merges,
        symbols=symbols,
        lang="hi",
        script="Deva",
        word_end_symbol=args.word_end_symbol,
    )

if __name__ == "__main__":
    main()

