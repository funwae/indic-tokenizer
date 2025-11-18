#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full evaluation stack on baseline GPE+CBPE tokenizer.

Comprehensive evaluation including:
1. Efficiency + Script metrics vs GPT-4o / Llama-3
2. Parity vs English on IITB parallel corpus
3. Morphology metrics
4. Tiny LM perplexity
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_efficiency_script_eval(tokenizer_id: str, output_dir: Path):
    """Run efficiency and script metrics evaluation."""
    print(f"\n=== 1. Efficiency + Script Metrics ===")

    # Use existing benchmark script
    import subprocess

    # Create a temporary config for this evaluation
    config_content = f"""lang: hi
baseline_tokenizer_id: {tokenizer_id}
tokenizers:
  - id: {tokenizer_id}
  - id: gpt4o_tok
  - id: llama3_8b_tok
  - id: mbert
  - id: indicbert
corpora:
  - path: data/hindi/demo/news_small.txt
    name: Hindi News (Small)
  - path: data/hindi/demo/mixed_small.txt
    name: Hindi Mixed (Small)
"""

    temp_config = output_dir / "temp_benchmark_config.yaml"
    with open(temp_config, "w", encoding="utf-8") as f:
        f.write(config_content)

    # Run benchmark
    result = subprocess.run([
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--config", str(temp_config),
        "--output-dir", str(output_dir / "efficiency_script"),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Efficiency/script evaluation complete")
        results_file = output_dir / "efficiency_script" / "results.json"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
    else:
        print(f"  ⚠️  Efficiency/script evaluation failed: {result.stderr}")
        return None

    # Clean up temp config
    if temp_config.exists():
        temp_config.unlink()

    return None


def run_parity_eval(tokenizer_id: str, output_dir: Path):
    """Run parity evaluation on parallel corpus."""
    print(f"\n=== 2. Parity vs English ===")

    parallel_corpus_path = project_root / "data" / "parallel" / "hi_en.txt"
    if not parallel_corpus_path.exists():
        print(f"  ⚠️  Parallel corpus not found: {parallel_corpus_path}")
        print(f"  Skipping parity evaluation. See docs/32-parity-datasets.md for setup.")
        return None

    # Use existing parity benchmark script
    import subprocess
    result = subprocess.run([
        sys.executable,
        str(project_root / "scripts" / "run_parity_benchmark.py"),
        "--input", str(parallel_corpus_path),
        "--tokenizers", f"{tokenizer_id},gpt4o_tok,llama3_8b_tok,mbert",
        "--baseline", "gpt4o_tok",
        "--output-dir", str(output_dir / "parity"),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Parity evaluation complete")
        return output_dir / "parity" / "results.json"
    else:
        print(f"  ⚠️  Parity evaluation failed: {result.stderr}")
        return None


def run_morphology_eval(tokenizer_id: str, output_dir: Path):
    """Run morphology evaluation."""
    print(f"\n=== 3. Morphology Metrics ===")

    morph_corpus_path = project_root / "data" / "hindi" / "morph_gold" / "hi_morph_gold.tsv"
    if not morph_corpus_path.exists():
        print(f"  ⚠️  Morphology corpus not found: {morph_corpus_path}")
        print(f"  Skipping morphology evaluation.")
        return None

    # Use existing morphology eval script
    import subprocess
    result = subprocess.run([
        sys.executable,
        str(project_root / "scripts" / "run_morphology_eval.py"),
        "--input", str(morph_corpus_path),
        "--tokenizers", f"{tokenizer_id},mbert,indicbert",
        "--output-dir", str(output_dir / "morphology"),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Morphology evaluation complete")
        return output_dir / "morphology" / "results.json"
    else:
        print(f"  ⚠️  Morphology evaluation failed: {result.stderr}")
        return None


def run_tiny_lm_eval(tokenizer_id: str, output_dir: Path):
    """Run tiny LM perplexity evaluation."""
    print(f"\n=== 4. Tiny LM Perplexity ===")

    model_dir = project_root / "models" / "tiny_lm_hi" / tokenizer_id
    if not model_dir.exists():
        print(f"  ⚠️  Tiny LM model not found: {model_dir}")
        print(f"  Train with: python scripts/train_tiny_lm.py --tokenizer-id {tokenizer_id} ...")
        print(f"  Skipping tiny LM evaluation.")
        return None

    eval_corpus_path = project_root / "data" / "hindi" / "processed" / "hi_eval_small.txt"
    if not eval_corpus_path.exists():
        # Fallback to demo corpus
        eval_corpus_path = project_root / "data" / "hindi" / "demo" / "news_small.txt"

    if not eval_corpus_path.exists():
        print(f"  ⚠️  Evaluation corpus not found")
        print(f"  Skipping tiny LM evaluation.")
        return None

    # Use existing tiny LM eval script
    import subprocess
    result = subprocess.run([
        sys.executable,
        str(project_root / "scripts" / "eval_tiny_lm.py"),
        "--model-dir", str(model_dir),
        "--tokenizer-id", tokenizer_id,
        "--eval-corpus", str(eval_corpus_path),
        "--output", str(output_dir / "tiny_lm" / "results.json"),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Tiny LM evaluation complete")
        return output_dir / "tiny_lm" / "results.json"
    else:
        print(f"  ⚠️  Tiny LM evaluation failed: {result.stderr}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run full evaluation stack on baseline tokenizer")
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="gpe_cbpe_hi_v1",
        help="Tokenizer ID to evaluate (default: gpe_cbpe_hi_v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scorecards/baseline_gpe_cbpe",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-tiny-lm",
        action="store_true",
        help="Skip tiny LM evaluation (if model not trained)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Baseline Evaluation: {args.tokenizer_id}")
    print("=" * 70)

    results = {
        "tokenizer_id": args.tokenizer_id,
        "timestamp": datetime.now().isoformat(),
        "evaluations": {},
    }

    # 1. Efficiency + Script metrics
    try:
        eff_results = run_efficiency_script_eval(args.tokenizer_id, output_dir)
        results["evaluations"]["efficiency_script"] = {
            "status": "completed",
            "results_file": str(output_dir / "efficiency_script_results.json"),
        }
    except Exception as e:
        print(f"  ✗ Efficiency/script evaluation failed: {e}")
        results["evaluations"]["efficiency_script"] = {"status": "failed", "error": str(e)}

    # 2. Parity evaluation
    try:
        parity_file = run_parity_eval(args.tokenizer_id, output_dir)
        if parity_file:
            results["evaluations"]["parity"] = {
                "status": "completed",
                "results_file": str(parity_file),
            }
        else:
            results["evaluations"]["parity"] = {"status": "skipped"}
    except Exception as e:
        print(f"  ✗ Parity evaluation failed: {e}")
        results["evaluations"]["parity"] = {"status": "failed", "error": str(e)}

    # 3. Morphology evaluation
    try:
        morph_file = run_morphology_eval(args.tokenizer_id, output_dir)
        if morph_file:
            results["evaluations"]["morphology"] = {
                "status": "completed",
                "results_file": str(morph_file),
            }
        else:
            results["evaluations"]["morphology"] = {"status": "skipped"}
    except Exception as e:
        print(f"  ✗ Morphology evaluation failed: {e}")
        results["evaluations"]["morphology"] = {"status": "failed", "error": str(e)}

    # 4. Tiny LM evaluation
    if not args.skip_tiny_lm:
        try:
            lm_file = run_tiny_lm_eval(args.tokenizer_id, output_dir)
            if lm_file:
                results["evaluations"]["tiny_lm"] = {
                    "status": "completed",
                    "results_file": str(lm_file),
                }
            else:
                results["evaluations"]["tiny_lm"] = {"status": "skipped"}
        except Exception as e:
            print(f"  ✗ Tiny LM evaluation failed: {e}")
            results["evaluations"]["tiny_lm"] = {"status": "failed", "error": str(e)}
    else:
        results["evaluations"]["tiny_lm"] = {"status": "skipped", "reason": "user_requested"}

    # Save summary
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    for eval_name, eval_status in results["evaluations"].items():
        status = eval_status.get("status", "unknown")
        print(f"  {eval_name}: {status}")
    print(f"\n✓ Summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

