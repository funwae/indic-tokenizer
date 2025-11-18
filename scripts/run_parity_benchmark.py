#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run tokenization parity benchmark for fairness evaluation.

Computes tokenization parity, NSL, and token tax metrics across
multiple tokenizers using parallel Hindi-English corpus.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics.fairness import compute_tp, compute_nsl_cross, compute_token_tax
from eval.metrics.efficiency import evaluate_efficiency


def load_parallel_corpus(jsonl_path: Path) -> tuple[List[str], List[str]]:
    """
    Load parallel corpus from JSONL file.

    Format: Each line is a JSON object with "en" and "hi" keys.
    """
    en_texts = []
    hi_texts = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                en_texts.append(record.get("en", ""))
                hi_texts.append(record.get("hi", ""))
            except json.JSONDecodeError:
                continue

    return en_texts, hi_texts


def tokenize_with_tokenizer(tokenizer: Any, text: str) -> List[str]:
    """Tokenize text using a tokenizer object."""
    return tokenizer.tokenize(text)


def run_parity_benchmark(
    tokenizers: Dict[str, Any],
    baseline_tokenizer_id: str,
    en_texts: List[str],
    hi_texts: List[str],
    output_dir: Path,
):
    """
    Run parity benchmark across tokenizers.

    Parameters
    ----------
    tokenizers : Dict[str, Any]
        Dictionary mapping tokenizer_id -> tokenizer object.
    baseline_tokenizer_id : str
        ID of baseline tokenizer for NSL and token tax computation.
    en_texts : List[str]
        List of English texts.
    hi_texts : List[str]
        List of Hindi texts (parallel to en_texts).
    output_dir : Path
        Output directory for results.
    """
    if baseline_tokenizer_id not in tokenizers:
        raise ValueError(f"Baseline tokenizer '{baseline_tokenizer_id}' not found")

    baseline_tokenizer = tokenizers[baseline_tokenizer_id]

    print("=== Tokenization Parity Benchmark ===\n")
    print(f"Tokenizers: {list(tokenizers.keys())}")
    print(f"Baseline: {baseline_tokenizer_id}")
    print(f"Parallel sentences: {len(en_texts)}\n")

    results = {}

    for tokenizer_id, tokenizer in tokenizers.items():
        print(f"Evaluating {tokenizer_id}...")

        # Compute TP statistics
        tp_stats = compute_tp(tokenizer, en_texts, hi_texts)

        # Compute NSL vs baseline for Hindi and English
        nsl_hi = compute_nsl_cross(tokenizer, baseline_tokenizer, hi_texts)
        nsl_en = compute_nsl_cross(tokenizer, baseline_tokenizer, en_texts)

        # Compute token tax
        token_tax = compute_token_tax(tokenizer, baseline_tokenizer, en_texts, hi_texts)

        # Compute compression ratios
        hi_crs = []
        en_crs = []
        for hi_text, en_text in zip(hi_texts, en_texts):
            hi_tokens = tokenizer.tokenize(hi_text)
            en_tokens = tokenizer.tokenize(en_text)
            if len(hi_tokens) > 0:
                hi_crs.append(len(hi_text) / len(hi_tokens))
            if len(en_tokens) > 0:
                en_crs.append(len(en_text) / len(en_tokens))

        avg_cr_hi = sum(hi_crs) / len(hi_crs) if hi_crs else 0.0
        avg_cr_en = sum(en_crs) / len(en_crs) if en_crs else 0.0

        results[tokenizer_id] = {
            "tokenization_parity": tp_stats,
            "nsl_hi_vs_baseline": nsl_hi,
            "nsl_en_vs_baseline": nsl_en,
            "token_tax": token_tax,
            "avg_compression_ratio_hi": avg_cr_hi,
            "avg_compression_ratio_en": avg_cr_en,
        }

        print(f"  TP (mean): {tp_stats['mean']:.3f}")
        print(f"  NSL (hi): {nsl_hi:.3f}")
        print(f"  NSL (en): {nsl_en:.3f}")
        print(f"  Token Tax (hi/en): {token_tax['tax_ratio']:.3f}")
        print()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved to {json_path}")

    # Save Markdown table
    md_path = output_dir / "results.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Tokenization Parity Benchmark Results\n\n")
        f.write(f"**Baseline**: {baseline_tokenizer_id}\n")
        f.write(f"**Parallel sentences**: {len(en_texts)}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Tokenizer | TP (mean) | TP (median) | NSL (hi) | NSL (en) | Token Tax |\n")
        f.write("|-----------|-----------|-------------|----------|----------|-----------|\n")

        for tokenizer_id, result in results.items():
            tp = result["tokenization_parity"]
            f.write(f"| {tokenizer_id} | {tp['mean']:.3f} | {tp['median']:.3f} | "
                   f"{result['nsl_hi_vs_baseline']:.3f} | {result['nsl_en_vs_baseline']:.3f} | "
                   f"{result['token_tax']['tax_ratio']:.3f} |\n")

        f.write("\n## Detailed Results\n\n")
        for tokenizer_id, result in results.items():
            f.write(f"### {tokenizer_id}\n\n")
            f.write(f"- **Tokenization Parity**: mean={result['tokenization_parity']['mean']:.3f}, "
                   f"median={result['tokenization_parity']['median']:.3f}, "
                   f"p10={result['tokenization_parity']['p10']:.3f}, "
                   f"p90={result['tokenization_parity']['p90']:.3f}\n")
            f.write(f"- **NSL (Hindi vs baseline)**: {result['nsl_hi_vs_baseline']:.3f}\n")
            f.write(f"- **NSL (English vs baseline)**: {result['nsl_en_vs_baseline']:.3f}\n")
            f.write(f"- **Token Tax**: premium_hi={result['token_tax']['premium_hi']:.3f}, "
                   f"premium_en={result['token_tax']['premium_en']:.3f}, "
                   f"tax_ratio={result['token_tax']['tax_ratio']:.3f}\n")
            f.write(f"- **Compression Ratio (Hindi)**: {result['avg_compression_ratio_hi']:.3f}\n")
            f.write(f"- **Compression Ratio (English)**: {result['avg_compression_ratio_en']:.3f}\n\n")

    print(f"✓ Markdown report saved to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tokenization parity benchmark")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with parallel corpus",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        required=True,
        help="Comma-separated list of tokenizer IDs",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="gpt4o_tok",
        help="Baseline tokenizer ID for NSL and token tax (default: gpt4o_tok)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scorecards/parity_hi_en",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load parallel corpus
    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        print(f"Error: Input file not found: {jsonl_path}", file=sys.stderr)
        return 1

    en_texts, hi_texts = load_parallel_corpus(jsonl_path)
    print(f"Loaded {len(en_texts)} parallel sentence pairs\n")

    # Load tokenizers from registry
    from scripts.compare_tokenizers import load_registry, create_tokenizer_from_config

    registry = load_registry(Path("tokenizers/registry.yaml"))
    tokenizer_ids = [tid.strip() for tid in args.tokenizers.split(',')]

    tokenizers = {}
    for tid in tokenizer_ids:
        if tid not in registry:
            print(f"Warning: Tokenizer '{tid}' not found in registry, skipping", file=sys.stderr)
            continue
        try:
            tokenizers[tid] = create_tokenizer_from_config(registry[tid])
        except Exception as e:
            print(f"Warning: Failed to load tokenizer '{tid}': {e}", file=sys.stderr)

    if not tokenizers:
        print("Error: No tokenizers loaded", file=sys.stderr)
        return 1

    if args.baseline not in tokenizers:
        print(f"Error: Baseline tokenizer '{args.baseline}' not found", file=sys.stderr)
        return 1

    # Run benchmark
    output_dir = Path(args.output_dir)
    run_parity_benchmark(
        tokenizers=tokenizers,
        baseline_tokenizer_id=args.baseline,
        en_texts=en_texts,
        hi_texts=hi_texts,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

