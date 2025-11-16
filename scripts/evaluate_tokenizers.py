#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate_tokenizers.py

Comprehensive evaluation script for tokenizers with full metrics and scorecard generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from eval.metrics import (
    AggregatedMetrics,
    evaluate_batch,
    evaluate_tokenizer,
    export_scorecard,
    generate_scorecard,
    Metrics,
)
from scripts.compare_tokenizers import (
    BaseTokenizer,
    create_tokenizer_from_config,
    load_registry,
)


def load_texts_from_file(file_path: Path) -> List[str]:
    """
    Load texts from a file (one per line or JSONL).

    Parameters
    ----------
    file_path : Path
        Path to text file.

    Returns
    -------
    List[str]
        List of texts.
    """
    texts: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        if file_path.suffix == ".jsonl":
            # JSONL format
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "text" in data:
                        texts.append(data["text"])
                    elif isinstance(data, str):
                        texts.append(data)
                except json.JSONDecodeError:
                    continue
        else:
            # Plain text, one per line
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def evaluate_single_text(
    text: str,
    tokenizer_ids: List[str],
    registry: Dict[str, Dict[str, Any]],
    lang: str = "hi",
) -> Dict[str, Metrics]:
    """
    Evaluate multiple tokenizers on a single text.

    Parameters
    ----------
    text : str
        Input text.
    tokenizer_ids : List[str]
        List of tokenizer IDs to evaluate.
    registry : Dict[str, Dict[str, Any]]
        Tokenizer registry.
    lang : str
        Language code.

    Returns
    -------
    Dict[str, Metrics]
        Dictionary mapping tokenizer IDs to metrics.
    """
    results: Dict[str, Metrics] = {}

    for tid in tokenizer_ids:
        if tid not in registry:
            print(f"Warning: Tokenizer '{tid}' not found in registry, skipping.", file=sys.stderr)
            continue

        try:
            cfg = registry[tid]
            tokenizer = create_tokenizer_from_config(cfg)
            metrics = evaluate_tokenizer(text, tokenizer, lang=lang)
            results[tid] = metrics
        except Exception as e:
            print(f"Error evaluating {tid}: {e}", file=sys.stderr)
            continue

    return results


def evaluate_batch_texts(
    texts: List[str],
    tokenizer_ids: List[str],
    registry: Dict[str, Dict[str, Any]],
    lang: str = "hi",
) -> Dict[str, AggregatedMetrics]:
    """
    Evaluate multiple tokenizers on a batch of texts.

    Parameters
    ----------
    texts : List[str]
        List of input texts.
    tokenizer_ids : List[str]
        List of tokenizer IDs to evaluate.
    registry : Dict[str, Dict[str, Any]]
        Tokenizer registry.
    lang : str
        Language code.

    Returns
    -------
    Dict[str, AggregatedMetrics]
        Dictionary mapping tokenizer IDs to aggregated metrics.
    """
    results: Dict[str, AggregatedMetrics] = {}

    for tid in tokenizer_ids:
        if tid not in registry:
            print(f"Warning: Tokenizer '{tid}' not found in registry, skipping.", file=sys.stderr)
            continue

        try:
            cfg = registry[tid]
            tokenizer = create_tokenizer_from_config(cfg)
            metrics = evaluate_batch(texts, tokenizer, lang=lang)
            results[tid] = metrics
        except Exception as e:
            print(f"Error evaluating {tid}: {e}", file=sys.stderr)
            continue

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive tokenizer evaluation with metrics and scorecards."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to evaluate.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file with texts (one per line or JSONL).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to evaluation dataset file.",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        help="Comma-separated list of tokenizer IDs, or 'all' for all tokenizers.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="tokenizers/registry.yaml",
        help="Path to tokenizer registry YAML.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Language code (default: hi).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for scorecard (JSON or Markdown).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine input texts
    texts: List[str] = []

    if args.text:
        texts = [args.text]
    elif args.file:
        texts = load_texts_from_file(Path(args.file))
    elif args.dataset:
        texts = load_texts_from_file(Path(args.dataset))
    else:
        print("Error: Must provide --text, --file, or --dataset.", file=sys.stderr)
        sys.exit(1)

    if not texts:
        print("Error: No texts to evaluate.", file=sys.stderr)
        sys.exit(1)

    # Load registry
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)

    # Determine tokenizers
    if args.tokenizers:
        if args.tokenizers.lower() == "all":
            tokenizer_ids = list(registry.keys())
        else:
            tokenizer_ids = [tid.strip() for tid in args.tokenizers.split(",") if tid.strip()]
    else:
        tokenizer_ids = list(registry.keys())

    # Get tokenizer names for scorecard
    tokenizer_names = {
        tid: registry[tid].get("display_name", tid) for tid in tokenizer_ids if tid in registry
    }

    # Evaluate
    print(f"Evaluating {len(tokenizer_ids)} tokenizer(s) on {len(texts)} text(s)...")

    if len(texts) == 1:
        # Single text evaluation
        results = evaluate_single_text(texts[0], tokenizer_ids, registry, args.lang)
        scorecards = generate_scorecard(results, tokenizer_names, texts)
    else:
        # Batch evaluation
        results = evaluate_batch_texts(texts, tokenizer_ids, registry, args.lang)
        scorecards = generate_scorecard(results, tokenizer_names, texts[:5])  # Sample texts

    # Export scorecard
    output_str = export_scorecard(scorecards, format=args.format)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"\nScorecard saved to: {args.output}")
    else:
        print("\n" + output_str)


if __name__ == "__main__":
    main()

