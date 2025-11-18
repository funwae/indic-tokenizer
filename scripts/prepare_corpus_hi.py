#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Hindi corpus for tokenizer training.

Filters, normalizes, and processes raw Hindi text from IndicNLP or other sources.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def is_devanagari_char(char: str) -> bool:
    """Check if character is Devanagari."""
    return '\u0900' <= char <= '\u097F'


def devanagari_ratio(text: str) -> float:
    """Calculate ratio of Devanagari characters in text."""
    if not text:
        return 0.0
    devanagari_count = sum(1 for c in text if is_devanagari_char(c))
    return devanagari_count / len(text)


def normalize_text(text: str) -> str:
    """
    Normalize text for training.

    - Unicode NFC normalization
    - Remove excessive whitespace
    - Strip control characters (except newlines/tabs)
    """
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize whitespace (multiple spaces to single, preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single

    return text.strip()


def filter_line(line: str, min_length: int = 10, min_devanagari_ratio: float = 0.5) -> bool:
    """
    Filter line based on criteria.

    Parameters
    ----------
    line : str
        Input line.
    min_length : int
        Minimum line length in characters.
    min_devanagari_ratio : float
        Minimum ratio of Devanagari characters (0.0-1.0).

    Returns
    -------
    bool
        True if line passes filters, False otherwise.
    """
    if len(line) < min_length:
        return False

    ratio = devanagari_ratio(line)
    if ratio < min_devanagari_ratio:
        return False

    return True


def process_corpus(
    input_path: Path,
    output_path: Path,
    max_lines: int = 500000,
    min_length: int = 10,
    min_devanagari_ratio: float = 0.5,
    random_seed: int = 42,
) -> int:
    """
    Process corpus: filter, normalize, shuffle, and write.

    Uses reservoir sampling to efficiently sample lines without loading
    the entire corpus into memory.

    Parameters
    ----------
    input_path : Path
        Input corpus file path.
    output_path : Path
        Output corpus file path.
    max_lines : int
        Maximum number of lines to output.
    min_length : int
        Minimum line length.
    min_devanagari_ratio : float
        Minimum Devanagari character ratio.
    random_seed : int
        Random seed for deterministic shuffling.

    Returns
    -------
    int
        Number of lines written.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    import random
    random.seed(random_seed)

    # Reservoir sampling: maintain a random sample of max_lines
    reservoir = []
    total_processed = 0
    total_filtered = 0

    print(f"Reading corpus from {input_path}...")
    print(f"Using reservoir sampling to keep up to {max_lines} lines in memory...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines, kept {len(reservoir):,} in reservoir...")

            # Normalize
            normalized = normalize_text(line)

            # Filter
            if filter_line(normalized, min_length, min_devanagari_ratio):
                total_filtered += 1

                # Reservoir sampling algorithm
                if len(reservoir) < max_lines:
                    # Fill reservoir until it's full
                    reservoir.append(normalized)
                else:
                    # Randomly replace an element with probability max_lines / total_filtered
                    j = random.randint(0, total_filtered - 1)
                    if j < max_lines:
                        reservoir[j] = normalized

            total_processed = line_num

    print(f"  Total lines read: {total_processed:,}")
    print(f"  Lines after filtering: {total_filtered:,}")
    print(f"  Lines in reservoir: {len(reservoir):,}")

    # Shuffle the reservoir deterministically
    random.shuffle(reservoir)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in reservoir:
            f.write(line + '\n')

    print(f"✓ Wrote {len(reservoir)} lines to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return len(reservoir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Hindi corpus for tokenizer training"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="indicnlp",
        help="Source identifier (for documentation, not used in processing)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input corpus file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output corpus file path",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500000,
        help="Maximum number of lines to output (default: 500000)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum line length in characters (default: 10)",
    )
    parser.add_argument(
        "--min-devanagari-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of Devanagari characters (default: 0.5)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling (default: 42)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        lines_written = process_corpus(
            input_path=input_path,
            output_path=output_path,
            max_lines=args.max_lines,
            min_length=args.min_length,
            min_devanagari_ratio=args.min_devanagari_ratio,
            random_seed=args.random_seed,
        )
        print(f"\n✓ Corpus preparation complete: {lines_written} lines")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

