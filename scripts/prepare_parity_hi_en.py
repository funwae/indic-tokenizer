#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare parallel Hindi-English corpus for fairness/parity evaluation.

Processes IIT Bombay English-Hindi parallel corpus and creates a JSONL file
with sentence pairs for tokenization parity evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_parallel_files(en_path: Path, hi_path: Path) -> List[Tuple[str, str]]:
    """
    Load parallel English-Hindi sentence pairs from two files.

    Parameters
    ----------
    en_path : Path
        Path to English file (one sentence per line).
    hi_path : Path
        Path to Hindi file (one sentence per line).

    Returns
    -------
    List[Tuple[str, str]]
        List of (english_text, hindi_text) pairs.
    """
    pairs = []

    with open(en_path, 'r', encoding='utf-8') as en_f, \
         open(hi_path, 'r', encoding='utf-8') as hi_f:

        for en_line, hi_line in zip(en_f, hi_f):
            en_text = en_line.strip()
            hi_text = hi_line.strip()

            # Skip empty lines
            if not en_text or not hi_text:
                continue

            pairs.append((en_text, hi_text))

    return pairs


def filter_length_ratio(pairs: List[Tuple[str, str]], max_ratio: float = 3.0) -> List[Tuple[str, str]]:
    """
    Filter pairs with extreme length ratios.

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        List of (english_text, hindi_text) pairs.
    max_ratio : float
        Maximum length ratio (default: 3.0, i.e., 3:1 or 1:3).

    Returns
    -------
    List[Tuple[str, str]]
        Filtered pairs.
    """
    filtered = []

    for en_text, hi_text in pairs:
        en_len = len(en_text)
        hi_len = len(hi_text)

        if en_len == 0 or hi_len == 0:
            continue

        ratio = max(en_len / hi_len, hi_len / en_len)

        if ratio <= max_ratio:
            filtered.append((en_text, hi_text))

    return filtered


def sample_pairs(pairs: List[Tuple[str, str]], max_pairs: int, random_seed: int = 42) -> List[Tuple[str, str]]:
    """
    Sample a subset of pairs deterministically.

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        List of pairs.
    max_pairs : int
        Maximum number of pairs to sample.
    random_seed : int
        Random seed for deterministic sampling.

    Returns
    -------
    List[Tuple[str, str]]
        Sampled pairs.
    """
    import random

    if len(pairs) <= max_pairs:
        return pairs

    random.seed(random_seed)
    return random.sample(pairs, max_pairs)


def save_jsonl(pairs: List[Tuple[str, str]], output_path: Path):
    """
    Save pairs as JSONL file.

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        List of (english_text, hindi_text) pairs.
    output_path : Path
        Output JSONL file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for en_text, hi_text in pairs:
            record = {
                "en": en_text,
                "hi": hi_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare parallel Hindi-English corpus for parity evaluation"
    )
    parser.add_argument(
        "--en-file",
        type=str,
        required=True,
        help="Path to English corpus file (one sentence per line)",
    )
    parser.add_argument(
        "--hi-file",
        type=str,
        required=True,
        help="Path to Hindi corpus file (one sentence per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50000,
        help="Maximum number of pairs to sample (default: 50000)",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help="Maximum length ratio for filtering (default: 3.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )

    args = parser.parse_args()

    en_path = Path(args.en_file)
    hi_path = Path(args.hi_file)
    output_path = Path(args.output)

    if not en_path.exists():
        print(f"Error: English file not found: {en_path}", file=sys.stderr)
        return 1

    if not hi_path.exists():
        print(f"Error: Hindi file not found: {hi_path}", file=sys.stderr)
        return 1

    print(f"Loading parallel corpus...")
    print(f"  English: {en_path}")
    print(f"  Hindi: {hi_path}")

    pairs = load_parallel_files(en_path, hi_path)
    print(f"  Loaded {len(pairs)} pairs")

    print(f"Filtering by length ratio (max: {args.max_length_ratio})...")
    pairs = filter_length_ratio(pairs, max_ratio=args.max_length_ratio)
    print(f"  After filtering: {len(pairs)} pairs")

    print(f"Sampling up to {args.max_pairs} pairs...")
    pairs = sample_pairs(pairs, max_pairs=args.max_pairs, random_seed=args.random_seed)
    print(f"  Sampled: {len(pairs)} pairs")

    print(f"Saving to {output_path}...")
    save_jsonl(pairs, output_path)
    print(f"✓ Saved {len(pairs)} pairs to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

