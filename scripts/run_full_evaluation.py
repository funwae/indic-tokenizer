#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_full_evaluation.py

Run comprehensive evaluation on all tokenizers using all curated examples and eval datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from eval.metrics import AggregatedMetrics, evaluate_batch, generate_scorecard, export_scorecard
from scripts.compare_tokenizers import create_tokenizer_from_config, load_registry


def load_curated_examples(file_path: Path) -> List[str]:
    """Load texts from curated examples JSONL file."""
    texts: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "text" in data:
                    texts.append(data["text"])
            except json.JSONDecodeError:
                continue
    return texts


def load_eval_dataset(file_path: Path) -> List[str]:
    """Load texts from evaluation dataset (one per line)."""
    texts: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full evaluation on all tokenizers and datasets."
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="tokenizers/registry.yaml",
        help="Path to tokenizer registry.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scorecards",
        help="Output directory for scorecards.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Language code (default: hi).",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load registry
    registry = load_registry(registry_path)
    tokenizer_ids = list(registry.keys())
    tokenizer_names = {
        tid: registry[tid].get("display_name", tid) for tid in tokenizer_ids
    }

    print(f"Running full evaluation on {len(tokenizer_ids)} tokenizer(s)...")
    print(f"Tokenizers: {', '.join(tokenizer_ids)}")

    # Evaluate on curated examples
    curated_path = Path(f"data/{args.lang}/curated_examples.jsonl")
    if curated_path.exists():
        print(f"\nEvaluating on curated examples: {curated_path}")
        texts = load_curated_examples(curated_path)
        print(f"  Loaded {len(texts)} examples")

        results: Dict[str, AggregatedMetrics] = {}
        for tid in tokenizer_ids:
            try:
                cfg = registry[tid]
                tokenizer = create_tokenizer_from_config(cfg)
                metrics = evaluate_batch(texts, tokenizer, lang=args.lang)
                results[tid] = metrics
                print(f"  ✓ {tid}")
            except Exception as e:
                print(f"  ✗ {tid}: {e}")

        scorecards = generate_scorecard(results, tokenizer_names, texts[:5])
        output_file = output_dir / f"{args.lang}_curated_scorecard.md"
        with output_file.open("w", encoding="utf-8") as f:
            f.write(export_scorecard(scorecards, format="markdown"))
        print(f"  Scorecard saved: {output_file}")

    # Evaluate on eval datasets
    eval_sets_dir = Path(f"data/{args.lang}/eval_sets")
    if eval_sets_dir.exists():
        for dataset_file in eval_sets_dir.glob("*.txt"):
            print(f"\nEvaluating on dataset: {dataset_file.name}")
            texts = load_eval_dataset(dataset_file)
            print(f"  Loaded {len(texts)} texts")

            results: Dict[str, AggregatedMetrics] = {}
            for tid in tokenizer_ids:
                try:
                    cfg = registry[tid]
                    tokenizer = create_tokenizer_from_config(cfg)
                    metrics = evaluate_batch(texts, tokenizer, lang=args.lang)
                    results[tid] = metrics
                    print(f"  ✓ {tid}")
                except Exception as e:
                    print(f"  ✗ {tid}: {e}")

            scorecards = generate_scorecard(results, tokenizer_names, texts[:5])
            dataset_name = dataset_file.stem
            output_file = output_dir / f"{args.lang}_{dataset_name}_scorecard.md"
            with output_file.open("w", encoding="utf-8") as f:
                f.write(export_scorecard(scorecards, format="markdown"))
            print(f"  Scorecard saved: {output_file}")

    print(f"\n✓ Full evaluation complete. Scorecards saved to: {output_dir}")


if __name__ == "__main__":
    main()

