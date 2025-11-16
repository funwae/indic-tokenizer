#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_benchmark.py

Run comprehensive benchmark evaluation on tokenizers.

Usage:
  # Using config file (recommended)
  indic-benchmark --config configs/hi_demo.yaml --output-dir scorecards/hi_demo

  # Using command-line arguments (legacy)
  indic-benchmark \
      --corpus data/hindi/eval_sets/news_headlines.txt \
      --tokenizers indicbert,mbert,gpe_hi_v0 \
      --lang hi \
      --baseline-tokenizer mbert \
      --output-dir scorecards/benchmark_news
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Import transformers BEFORE our tokenizers package to avoid naming conflict
# This must happen before any imports from our local tokenizers package
try:
    from transformers import AutoTokenizer  # noqa: F401
except ImportError:
    pass  # Will be handled later

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics import ComprehensiveMetrics, evaluate_comprehensive
from scripts.compare_tokenizers import (
    create_tokenizer_from_config,
    load_registry,
)


def load_corpus(corpus_path: Path) -> List[str]:
    """
    Load corpus from file (one text per line).

    Parameters
    ----------
    corpus_path : Path
        Path to corpus file.

    Returns
    -------
    List[str]
        List of texts.
    """
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def evaluate_tokenizer_on_corpus(
    tokenizer,
    texts: List[str],
    lang: str,
    baseline_tokenizer=None,
    baseline_tokenizer_id: Optional[str] = None,
) -> Dict[str, any]:
    """
    Evaluate tokenizer on corpus and return aggregated metrics.

    Parameters
    ----------
    tokenizer
        Tokenizer object.
    texts : List[str]
        List of texts to evaluate.
    lang : str
        Language code.
    baseline_tokenizer
        Optional baseline tokenizer for NSL.
    baseline_tokenizer_id : Optional[str]
        Baseline tokenizer ID.

    Returns
    -------
    Dict[str, any]
        Aggregated metrics dictionary.
    """
    all_metrics: List[ComprehensiveMetrics] = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        baseline_tokens = None
        if baseline_tokenizer is not None:
            baseline_tokens = baseline_tokenizer.tokenize(text)

        metrics = evaluate_comprehensive(
            text=text,
            tokens=tokens,
            lang=lang,
            baseline_tokens=baseline_tokens,
            baseline_tokenizer_id=baseline_tokenizer_id,
        )
        all_metrics.append(metrics)

    # Aggregate metrics
    if not all_metrics:
        return {}

    # Aggregate efficiency metrics
    fertilities = [m.efficiency.fertility for m in all_metrics]
    chars_per_token_list = [m.efficiency.chars_per_token for m in all_metrics]
    cr_chars_list = [m.efficiency.compression_ratio_chars for m in all_metrics]
    cr_graphemes_list = [m.efficiency.compression_ratio_graphemes for m in all_metrics]
    nsl_list = [m.efficiency.normalized_sequence_length for m in all_metrics if m.efficiency.normalized_sequence_length is not None]
    pcw_list = [m.efficiency.proportion_continued_words for m in all_metrics]
    unk_list = [m.efficiency.unk_rate for m in all_metrics]

    # Aggregate script metrics
    gv_rates = [m.script.grapheme_violation_rate for m in all_metrics]
    akshara_integrity_list = [m.script.akshara_integrity_rate for m in all_metrics]
    akshara_splits = [m.script.akshara_split_count for m in all_metrics]
    dv_split_rates = [m.script.dependent_vowel_split_rate for m in all_metrics]
    grapheme_aligned_rates = [m.script.grapheme_aligned_token_rate for m in all_metrics]
    devanagari_shares = [m.script.devanagari_token_share for m in all_metrics]
    mixed_shares = [m.script.mixed_script_token_share for m in all_metrics]

    # Summary stats
    total_tokens = sum(m.num_tokens for m in all_metrics)
    total_words = sum(m.num_words for m in all_metrics)
    total_chars = sum(m.num_chars for m in all_metrics)
    total_graphemes = sum(m.num_graphemes for m in all_metrics)
    total_aksharas = sum(m.num_aksharas for m in all_metrics)

    return {
        "efficiency": {
            "avg_fertility": sum(fertilities) / len(fertilities) if fertilities else 0.0,
            "avg_chars_per_token": sum(chars_per_token_list) / len(chars_per_token_list) if chars_per_token_list else 0.0,
            "avg_compression_ratio_chars": sum(cr_chars_list) / len(cr_chars_list) if cr_chars_list else 0.0,
            "avg_compression_ratio_graphemes": sum(cr_graphemes_list) / len(cr_graphemes_list) if cr_graphemes_list else 0.0,
            "avg_normalized_sequence_length": sum(nsl_list) / len(nsl_list) if nsl_list else None,
            "avg_proportion_continued_words": sum(pcw_list) / len(pcw_list) if pcw_list else 0.0,
            "avg_unk_rate": sum(unk_list) / len(unk_list) if unk_list else 0.0,
            "baseline_tokenizer_id": baseline_tokenizer_id,
        },
        "script": {
            "avg_grapheme_violation_rate": sum(gv_rates) / len(gv_rates) if gv_rates else 0.0,
            "avg_akshara_integrity_rate": sum(akshara_integrity_list) / len(akshara_integrity_list) if akshara_integrity_list else 0.0,
            "total_akshara_splits": sum(akshara_splits),
            "avg_dependent_vowel_split_rate": sum(dv_split_rates) / len(dv_split_rates) if dv_split_rates else 0.0,
            "avg_grapheme_aligned_token_rate": sum(grapheme_aligned_rates) / len(grapheme_aligned_rates) if grapheme_aligned_rates else 0.0,
            "avg_devanagari_token_share": sum(devanagari_shares) / len(devanagari_shares) if devanagari_shares else 0.0,
            "avg_mixed_script_token_share": sum(mixed_shares) / len(mixed_shares) if mixed_shares else 0.0,
            "akshara_segmentation_version": "v0_heuristic",
        },
        "summary": {
            "total_tokens": total_tokens,
            "total_words": total_words,
            "total_chars": total_chars,
            "total_graphemes": total_graphemes,
            "total_aksharas": total_aksharas,
            "num_texts": len(all_metrics),
        },
        "fairness": None,  # Phase 2
        "morphology": None,  # Phase 2
    }


def load_config(config_path: Path) -> Dict[str, any]:
    """
    Load benchmark configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to config YAML file.

    Returns
    -------
    Dict[str, any]
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark evaluation on tokenizers"
    )

    # Config file (takes precedence)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark config YAML file (optional, overrides other args)",
    )

    # Command-line arguments (legacy, used if --config not provided)
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus file (one text per line) - ignored if --config provided",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        default=None,
        help="Comma-separated list of tokenizer IDs from registry - ignored if --config provided",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Language code (default: hi) - ignored if --config provided",
    )
    parser.add_argument(
        "--baseline-tokenizer",
        type=str,
        default=None,
        help="Tokenizer ID to use as baseline for NSL computation (optional) - ignored if --config provided",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (creates results.json and results.md)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (legacy, use --output-dir instead)",
    )

    parser.add_argument(
        "--registry",
        type=str,
        default="tokenizers/registry.yaml",
        help="Path to tokenizer registry YAML file",
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        config = load_config(config_path)

    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output:
        # Legacy: use output file's parent directory
        output_dir = Path(args.output).parent
    elif config and "output_dir" in config:
        output_dir = Path(config["output_dir"])
    else:
        print("Error: --output-dir or --output must be specified", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine parameters from config or args
    if config:
        corpora = config.get("corpora", [])
        tokenizer_ids = config.get("tokenizers", [])
        lang = config.get("lang", "hi")
        baseline_tokenizer_id = config.get("baseline_tokenizer_id", None)
    else:
        # Use command-line arguments
        if not args.corpus:
            print("Error: --corpus required when --config not provided", file=sys.stderr)
            sys.exit(1)
        if not args.tokenizers:
            print("Error: --tokenizers required when --config not provided", file=sys.stderr)
            sys.exit(1)
        corpora = [args.corpus]
        tokenizer_ids = [tid.strip() for tid in args.tokenizers.split(",")]
        lang = args.lang
        baseline_tokenizer_id = args.baseline_tokenizer

    # Load registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"Error: Registry file not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    registry = load_registry(registry_path)

    # Validate tokenizers
    missing = [tid for tid in tokenizer_ids if tid not in registry]
    if missing:
        print(f"Error: Tokenizers not found in registry: {missing}", file=sys.stderr)
        sys.exit(1)

    # Load baseline tokenizer if specified
    baseline_tokenizer = None
    if baseline_tokenizer_id:
        if baseline_tokenizer_id not in registry:
            print(
                f"Error: Baseline tokenizer not found in registry: {baseline_tokenizer_id}",
                file=sys.stderr,
            )
            sys.exit(1)
        baseline_config = registry[baseline_tokenizer_id]
        baseline_tokenizer = create_tokenizer_from_config(baseline_config)
        print(f"Using baseline tokenizer: {baseline_tokenizer_id}")

    # Evaluate each corpus
    all_results = {}
    tokenizer_names = {}

    for corpus_path_str in corpora:
        corpus_path = Path(corpus_path_str)
        if not corpus_path.exists():
            print(f"Warning: Corpus file not found: {corpus_path}, skipping", file=sys.stderr)
            continue

        texts = load_corpus(corpus_path)
        print(f"\nLoaded {len(texts)} texts from {corpus_path}")

        corpus_results = {}

        for tokenizer_id in tokenizer_ids:
            print(f"\nEvaluating tokenizer: {tokenizer_id}")
            try:
                config_entry = registry[tokenizer_id]
                tokenizer = create_tokenizer_from_config(config_entry)
                tokenizer_names[tokenizer_id] = config_entry.get("display_name", tokenizer_id)

                metrics_dict = evaluate_tokenizer_on_corpus(
                    tokenizer=tokenizer,
                    texts=texts,
                    lang=lang,
                    baseline_tokenizer=baseline_tokenizer,
                    baseline_tokenizer_id=baseline_tokenizer_id,
                )
                corpus_results[tokenizer_id] = metrics_dict
            except Exception as e:
                print(f"Error evaluating {tokenizer_id}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)

        all_results[corpus_path.stem] = corpus_results

    # Prepare output data
    output_data = {
        "config": str(config_path) if config_path else None,
        "lang": lang,
        "baseline_tokenizer": baseline_tokenizer_id,
        "timestamp": datetime.now().isoformat(),
        "tokenizer_names": tokenizer_names,
        "corpora": {corpus: {"path": str(Path(corpus)), "num_texts": len(load_corpus(Path(corpus)))} for corpus in corpora if Path(corpus).exists()},
        "results": all_results,
    }

    # Write JSON output
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Generate and write Markdown summary
    from eval.metrics import generate_scorecard, export_scorecard

    # Create scorecards for each corpus/tokenizer combination
    scorecards_dict = {}
    for corpus_name, corpus_results in all_results.items():
        for tokenizer_id, metrics_dict in corpus_results.items():
            # Convert metrics dict to ComprehensiveMetrics-like object for scorecard
            # For now, we'll create a simplified scorecard from the aggregated metrics
            scorecard_key = f"{corpus_name}_{tokenizer_id}"
            # Note: This is a simplified version - full scorecard generation would need per-text metrics
            # For now, we'll generate a summary markdown

    # Generate Markdown summary
    md_lines = []
    md_lines.append("# Benchmark Results")
    md_lines.append("")
    md_lines.append(f"**Timestamp:** {output_data['timestamp']}")
    md_lines.append(f"**Language:** {lang}")
    if baseline_tokenizer_id:
        md_lines.append(f"**Baseline Tokenizer:** {baseline_tokenizer_id}")
    md_lines.append("")

    for corpus_name, corpus_results in all_results.items():
        md_lines.append(f"## Corpus: {corpus_name}")
        md_lines.append("")
        md_lines.append("| Tokenizer | Fertility | Chars/Token | CR (chars) | CR (graphemes) | Grapheme Violation Rate | Akshara Integrity |")
        md_lines.append("|-----------|-----------|-------------|------------|----------------|------------------------|-------------------|")

        for tokenizer_id, metrics_dict in corpus_results.items():
            eff = metrics_dict.get("efficiency", {})
            scr = metrics_dict.get("script", {})
            tokenizer_name = tokenizer_names.get(tokenizer_id, tokenizer_id)

            md_lines.append(
                f"| {tokenizer_name} | "
                f"{eff.get('avg_fertility', 0):.3f} | "
                f"{eff.get('avg_chars_per_token', 0):.3f} | "
                f"{eff.get('avg_compression_ratio_chars', 0):.3f} | "
                f"{eff.get('avg_compression_ratio_graphemes', 0):.3f} | "
                f"{scr.get('avg_grapheme_violation_rate', 0):.2%} | "
                f"{scr.get('avg_akshara_integrity_rate', 0):.2%} |"
            )

        md_lines.append("")

    md_path = output_dir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n✓ Benchmark results written to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")
    print(f"\nEvaluated {len(tokenizer_ids)} tokenizer(s) on {len(corpora)} corpus/corpora")


if __name__ == "__main__":
    main()

