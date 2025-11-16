#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Hindi corpus for tokenizer training.

Downloads and prepares Hindi text from multiple sources:
- Wikipedia Hindi dump
- News articles (if available)
- Combines and normalizes text

Usage:
  python scripts/prepare_hindi_corpus.py \
      --output data/hindi/corpus/training.txt \
      --max-lines 500000 \
      --sources wikipedia,news
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional
import unicodedata

try:
    import requests
except ImportError:
    requests = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def normalize_text(text: str) -> str:
    """
    Normalize text for training.
    
    - Unicode NFC normalization
    - Remove excessive whitespace
    - Basic cleaning
    """
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def download_wikipedia_sample(output_path: Path, max_lines: int = 100000) -> int:
    """
    Download a sample of Hindi Wikipedia articles.
    
    Uses HuggingFace datasets library if available, otherwise provides instructions.
    """
    if load_dataset is None:
        print("Warning: 'datasets' library not installed. Install with: pip install datasets")
        print("Alternative: Download Wikipedia dump manually from:")
        print("  https://dumps.wikimedia.org/hiwiki/latest/")
        return 0
    
    try:
        print("Downloading Hindi Wikipedia dataset...")
        dataset = load_dataset("wikipedia", "20220301.hi", split="train", streaming=True)
        
        lines_written = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                text = example.get('text', '')
                if not text or len(text.strip()) < 50:  # Skip very short articles
                    continue
                
                # Normalize and write
                normalized = normalize_text(text)
                if normalized:
                    f.write(normalized + '\n')
                    lines_written += 1
                    
                    if lines_written >= max_lines:
                        break
                    
                    if lines_written % 1000 == 0:
                        print(f"  Processed {lines_written} lines...")
        
        print(f"✓ Downloaded {lines_written} lines from Wikipedia")
        return lines_written
        
    except Exception as e:
        print(f"Error downloading Wikipedia: {e}")
        print("You can manually download from: https://dumps.wikimedia.org/hiwiki/latest/")
        return 0


def combine_existing_corpora(sources: List[Path], output_path: Path) -> int:
    """
    Combine existing corpus files.
    
    Parameters
    ----------
    sources : List[Path]
        List of input corpus file paths.
    output_path : Path
        Output file path.
    
    Returns
    -------
    int
        Number of lines written.
    """
    lines_written = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for source_path in sources:
            if not source_path.exists():
                print(f"Warning: {source_path} not found, skipping")
                continue
            
            print(f"Processing {source_path}...")
            with open(source_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    normalized = normalize_text(line)
                    if normalized and len(normalized) > 10:  # Skip very short lines
                        out_f.write(normalized + '\n')
                        lines_written += 1
                        
                        if lines_written % 1000 == 0:
                            print(f"  Processed {lines_written} lines...")
    
    print(f"✓ Combined {lines_written} lines from {len(sources)} sources")
    return lines_written


def create_synthetic_corpus(output_path: Path, max_lines: int = 10000) -> int:
    """
    Create a small synthetic corpus from existing demo/eval data.
    
    This is a fallback if no external corpus is available.
    """
    print("Creating synthetic corpus from existing data...")
    
    sources = [
        Path("data/hindi/demo/news_small.txt"),
        Path("data/hindi/demo/mixed_small.txt"),
        Path("data/hindi/eval_sets/news_headlines.txt"),
        Path("data/hindi/eval_sets/literature.txt"),
        Path("data/hindi/eval_sets/conversational.txt"),
        Path("data/hindi/curated_examples.jsonl"),
    ]
    
    lines_written = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for source in sources:
            if not source.exists():
                continue
            
            with open(source, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    # Handle JSONL format
                    if source.suffix == '.jsonl':
                        import json
                        try:
                            data = json.loads(line)
                            text = data.get('text', '') or data.get('sentence', '')
                        except:
                            continue
                    else:
                        text = line
                    
                    normalized = normalize_text(text)
                    if normalized and len(normalized) > 10:
                        out_f.write(normalized + '\n')
                        lines_written += 1
                        
                        if lines_written >= max_lines:
                            break
            
            if lines_written >= max_lines:
                break
    
    print(f"✓ Created synthetic corpus with {lines_written} lines")
    return lines_written


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Hindi corpus for tokenizer training"
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
        default=100000,
        help="Maximum number of lines to include (default: 100000)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="wikipedia",
        help="Comma-separated list of sources: wikipedia,news,existing,synthetic",
    )
    parser.add_argument(
        "--existing-files",
        type=str,
        nargs='+',
        help="Paths to existing corpus files to combine",
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sources = [s.strip() for s in args.sources.split(',')]
    total_lines = 0
    
    # Download/prepare from different sources
    if 'wikipedia' in sources:
        wiki_lines = download_wikipedia_sample(output_path, args.max_lines)
        total_lines += wiki_lines
    
    if 'existing' in sources and args.existing_files:
        existing_paths = [Path(f) for f in args.existing_files]
        existing_lines = combine_existing_corpora(existing_paths, output_path)
        total_lines += existing_lines
    
    if 'synthetic' in sources and total_lines < args.max_lines:
        remaining = args.max_lines - total_lines
        synthetic_lines = create_synthetic_corpus(output_path, remaining)
        total_lines += synthetic_lines
    
    if total_lines == 0:
        print("Warning: No corpus lines were generated!")
        print("Creating minimal synthetic corpus as fallback...")
        total_lines = create_synthetic_corpus(output_path, 1000)
    
    print(f"\n✓ Corpus preparation complete: {total_lines} lines written to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    if total_lines < 10000:
        print("\nWarning: Corpus is quite small. For better tokenizer quality, aim for 50K+ lines.")
        print("Consider downloading Wikipedia dump manually or using larger datasets.")


if __name__ == "__main__":
    main()

