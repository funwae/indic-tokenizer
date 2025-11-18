#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run comprehensive evaluation on semantic/fractal tokenizer.

Uses the same evaluation framework as baseline evaluation to ensure
fair comparison.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Reuse baseline evaluation script
from scripts.run_baseline_evaluation import main as baseline_main


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation on semantic tokenizer")
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        required=True,
        help="Semantic tokenizer ID (e.g., ag_bpe_hi_v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scorecards/semantic_tokenizer",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-tiny-lm",
        action="store_true",
        help="Skip tiny LM evaluation",
    )

    args = parser.parse_args()

    # Use baseline evaluation script with semantic tokenizer
    import sys as sys_module
    sys_module.argv = [
        "run_baseline_evaluation.py",
        "--tokenizer-id", args.tokenizer_id,
        "--output-dir", args.output_dir,
    ]
    if args.skip_tiny_lm:
        sys_module.argv.append("--skip-tiny-lm")

    return baseline_main()


if __name__ == "__main__":
    sys.exit(main())

