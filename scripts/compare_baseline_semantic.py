#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline and semantic tokenizers.

Loads evaluation results from both tokenizers and generates a comparison report.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_file: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation results from JSON file."""
    if not results_file.exists():
        return None
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(
    baseline_results: Dict[str, Any],
    semantic_results: Dict[str, Any],
    output_path: Path,
):
    """
    Compare baseline and semantic tokenizer results.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline evaluation results.
    semantic_results : Dict[str, Any]
        Semantic tokenizer evaluation results.
    output_path : Path
        Path to save comparison report.
    """
    print("=== Baseline vs Semantic Tokenizer Comparison ===\n")

    comparison = {
        "baseline_tokenizer": "gpe_cbpe_hi_v1",
        "semantic_tokenizer": "ag_bpe_hi_v1",
        "comparison": {},
    }

    # Compare efficiency/script metrics
    baseline_eff = baseline_results.get("efficiency_script", {})
    semantic_eff = semantic_results.get("efficiency_script", {})

    if baseline_eff and semantic_eff:
        print("Efficiency & Script Metrics:")
        comparison["comparison"]["efficiency_script"] = {
            "baseline": baseline_eff,
            "semantic": semantic_eff,
        }
        # Add delta calculations here

    # Compare parity
    baseline_parity = baseline_results.get("parity", {})
    semantic_parity = semantic_results.get("parity", {})

    if baseline_parity and semantic_parity:
        print("Parity Metrics:")
        comparison["comparison"]["parity"] = {
            "baseline": baseline_parity,
            "semantic": semantic_parity,
        }

    # Compare morphology
    baseline_morph = baseline_results.get("morphology", {})
    semantic_morph = semantic_results.get("morphology", {})

    if baseline_morph and semantic_morph:
        print("Morphology Metrics:")
        comparison["comparison"]["morphology"] = {
            "baseline": baseline_morph,
            "semantic": semantic_morph,
        }

    # Compare tiny LM
    baseline_lm = baseline_results.get("tiny_lm", {})
    semantic_lm = semantic_results.get("tiny_lm", {})

    if baseline_lm and semantic_lm:
        print("Tiny LM Perplexity:")
        comparison["comparison"]["tiny_lm"] = {
            "baseline": baseline_lm,
            "semantic": semantic_lm,
        }

    # Save comparison
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Comparison saved to {output_path}")

    # Generate Markdown report
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline vs Semantic Tokenizer Comparison\n\n")
        f.write("## Summary\n\n")
        f.write("Comparison between GPE+CBPE baseline and Attention-Guided BPE semantic tokenizer.\n\n")
        f.write("## Results\n\n")
        f.write("*(Detailed comparison will be generated when both evaluations are complete)*\n")

    print(f"✓ Markdown report saved to {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare baseline and semantic tokenizers")
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="scorecards/baseline_gpe_cbpe",
        help="Baseline evaluation results directory",
    )
    parser.add_argument(
        "--semantic-dir",
        type=str,
        default="scorecards/semantic_tokenizer",
        help="Semantic tokenizer evaluation results directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scorecards/comparison_baseline_semantic.json",
        help="Output comparison file",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    semantic_dir = Path(args.semantic_dir)

    # Load evaluation summaries
    baseline_summary = load_results(baseline_dir / "evaluation_summary.json")
    semantic_summary = load_results(semantic_dir / "evaluation_summary.json")

    if not baseline_summary:
        print(f"Error: Baseline results not found: {baseline_dir / 'evaluation_summary.json'}")
        return 1

    if not semantic_summary:
        print(f"Error: Semantic results not found: {semantic_dir / 'evaluation_summary.json'}")
        print("  Run: python scripts/run_semantic_evaluation.py --tokenizer-id ag_bpe_hi_v1")
        return 1

    # Load detailed results
    baseline_results = {}
    semantic_results = {}

    # Load efficiency/script
    baseline_eff_file = baseline_dir / "efficiency_script" / "results.json"
    if baseline_eff_file.exists():
        baseline_results["efficiency_script"] = load_results(baseline_eff_file)

    semantic_eff_file = semantic_dir / "efficiency_script" / "results.json"
    if semantic_eff_file.exists():
        semantic_results["efficiency_script"] = load_results(semantic_eff_file)

    # Load parity
    baseline_parity_file = baseline_dir / "parity" / "results.json"
    if baseline_parity_file.exists():
        baseline_results["parity"] = load_results(baseline_parity_file)

    semantic_parity_file = semantic_dir / "parity" / "results.json"
    if semantic_parity_file.exists():
        semantic_results["parity"] = load_results(semantic_parity_file)

    # Load morphology
    baseline_morph_file = baseline_dir / "morphology" / "results.json"
    if baseline_morph_file.exists():
        baseline_results["morphology"] = load_results(baseline_morph_file)

    semantic_morph_file = semantic_dir / "morphology" / "results.json"
    if semantic_morph_file.exists():
        semantic_results["morphology"] = load_results(semantic_morph_file)

    # Load tiny LM
    baseline_lm_file = baseline_dir / "tiny_lm" / "results.json"
    if baseline_lm_file.exists():
        baseline_results["tiny_lm"] = load_results(baseline_lm_file)

    semantic_lm_file = semantic_dir / "tiny_lm" / "results.json"
    if semantic_lm_file.exists():
        semantic_results["tiny_lm"] = load_results(semantic_lm_file)

    # Compare
    compare_results(
        baseline_results,
        semantic_results,
        Path(args.output),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

