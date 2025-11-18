#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run morphology evaluation benchmark.

Evaluates tokenizers on morphology-annotated dataset and computes
boundary F1, morpheme alignment, and fragmentation metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics.morphology import (
    compute_boundary_f1,
    compute_morpheme_alignment,
    compute_morph_fragmentation,
    load_morphology_tsv,
)


def tokenize_with_tokenizer(tokenizer: Any, text: str) -> List[str]:
    """Tokenize text using a tokenizer object."""
    return tokenizer.tokenize(text)


def run_morphology_eval(
    tokenizers: Dict[str, Any],
    examples: List[Dict[str, str]],
    output_dir: Path,
):
    """
    Run morphology evaluation across tokenizers.

    Parameters
    ----------
    tokenizers : Dict[str, Any]
        Dictionary mapping tokenizer_id -> tokenizer object.
    examples : List[Dict[str, str]]
        List of examples with 'id', 'text', 'morphemes' keys.
    output_dir : Path
        Output directory for results.
    """
    print("=== Morphology Evaluation ===\n")
    print(f"Tokenizers: {list(tokenizers.keys())}")
    print(f"Annotated sentences: {len(examples)}\n")

    results = {}

    for tokenizer_id, tokenizer in tokenizers.items():
        print(f"Evaluating {tokenizer_id}...")

        # Compute boundary F1
        boundary_f1 = compute_boundary_f1(tokenizer, examples)

        # Compute morpheme alignment
        alignment = compute_morpheme_alignment(tokenizer, examples)

        # Compute fragmentation
        fragmentation = compute_morph_fragmentation(tokenizer, examples)

        results[tokenizer_id] = {
            "boundary_precision": boundary_f1["precision"],
            "boundary_recall": boundary_f1["recall"],
            "boundary_f1": boundary_f1["f1"],
            "token_match_rate": alignment["token_match_rate"],
            "morpheme_coverage_rate": alignment["morpheme_coverage_rate"],
            "avg_tokens_per_morpheme": fragmentation,
        }

        print(f"  Boundary F1: {boundary_f1['f1']:.3f}")
        print(f"  Token Match Rate: {alignment['token_match_rate']:.3f}")
        print(f"  Morpheme Coverage Rate: {alignment['morpheme_coverage_rate']:.3f}")
        print(f"  Avg Tokens/Morpheme: {fragmentation:.3f}")
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
        f.write("# Morphology Evaluation Results\n\n")
        f.write(f"**Annotated sentences**: {len(examples)}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Tokenizer | Boundary F1 | Token Match | Morpheme Coverage | Tokens/Morpheme |\n")
        f.write("|-----------|-------------|-------------|-------------------|-----------------|\n")

        for tokenizer_id, result in results.items():
            f.write(f"| {tokenizer_id} | {result['boundary_f1']:.3f} | "
                   f"{result['token_match_rate']:.3f} | {result['morpheme_coverage_rate']:.3f} | "
                   f"{result['avg_tokens_per_morpheme']:.3f} |\n")

        f.write("\n## Detailed Results\n\n")
        for tokenizer_id, result in results.items():
            f.write(f"### {tokenizer_id}\n\n")
            f.write(f"- **Boundary Precision**: {result['boundary_precision']:.3f}\n")
            f.write(f"- **Boundary Recall**: {result['boundary_recall']:.3f}\n")
            f.write(f"- **Boundary F1**: {result['boundary_f1']:.3f}\n")
            f.write(f"- **Token Match Rate**: {result['token_match_rate']:.3f}\n")
            f.write(f"- **Morpheme Coverage Rate**: {result['morpheme_coverage_rate']:.3f}\n")
            f.write(f"- **Avg Tokens per Morpheme**: {result['avg_tokens_per_morpheme']:.3f}\n\n")

    print(f"✓ Markdown report saved to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run morphology evaluation")
    parser.add_argument(
        "--input",
        type=str,
        default="data/hindi/morph_gold/hi_morph_gold.tsv",
        help="Input TSV file with morphology annotations",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        default="mbert,indicbert,gpe_cbpe_hi_v1,gpt4o_tok,llama3_8b_tok",
        help="Comma-separated list of tokenizer IDs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scorecards/morph_hi",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load morphology dataset
    tsv_path = Path(args.input)
    if not tsv_path.exists():
        print(f"Error: Input file not found: {tsv_path}", file=sys.stderr)
        return 1

    examples = load_morphology_tsv(str(tsv_path))
    print(f"Loaded {len(examples)} annotated sentences\n")

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

    # Run evaluation
    output_dir = Path(args.output_dir)
    run_morphology_eval(
        tokenizers=tokenizers,
        examples=examples,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

