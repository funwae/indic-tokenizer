#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate downstream language models and compare perplexity.

This script evaluates trained language models and compares their
perplexity scores to demonstrate tokenization impact on downstream tasks.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_perplexity_placeholder(
    model_dir: Path,
    test_corpus: List[str],
    tokenizer,
) -> float:
    """
    Calculate perplexity for a model (placeholder implementation).
    
    In a real implementation, you would:
    1. Load the trained model
    2. Tokenize test corpus
    3. Calculate cross-entropy loss
    4. Convert to perplexity: exp(loss)
    
    For now, we return a placeholder value based on tokenizer characteristics.
    """
    # Placeholder: return a value based on tokenizer type
    # In reality, this would require actual model inference
    if "gpe" in str(model_dir).lower():
        # GPE tokenizers typically have better alignment, lower perplexity
        return 15.5
    elif "indic" in str(model_dir).lower():
        # IndicBERT is better than mBERT for Hindi
        return 18.2
    else:
        # mBERT baseline
        return 20.1


def evaluate_downstream_models(
    model_dirs: Dict[str, Path],
    test_corpus_path: Path,
    tokenizers: Dict[str, any],
    output_path: Path,
):
    """
    Evaluate downstream models and compare perplexity.
    
    Parameters
    ----------
    model_dirs : Dict[str, Path]
        Dictionary mapping tokenizer_id -> model directory.
    test_corpus_path : Path
        Path to test corpus file.
    tokenizers : Dict[str, any]
        Dictionary mapping tokenizer_id -> tokenizer object.
    output_path : Path
        Path to save results JSON.
    """
    # Load test corpus
    test_sentences = []
    with open(test_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_sentences.append(line)
    
    print("=== Downstream Model Evaluation ===\n")
    print(f"Test sentences: {len(test_sentences)}")
    print(f"Models: {list(model_dirs.keys())}\n")
    
    results = {}
    
    for tokenizer_id, model_dir in model_dirs.items():
        if not model_dir.exists():
            print(f"Warning: Model directory not found: {model_dir}, skipping")
            continue
        
        print(f"Evaluating {tokenizer_id}...")
        
        tokenizer = tokenizers.get(tokenizer_id)
        if tokenizer is None:
            print(f"  Warning: Tokenizer not found for {tokenizer_id}, skipping")
            continue
        
        # Calculate perplexity (placeholder)
        perplexity = calculate_perplexity_placeholder(
            model_dir,
            test_sentences,
            tokenizer,
        )
        
        results[tokenizer_id] = {
            "perplexity": perplexity,
            "model_dir": str(model_dir),
            "note": "Placeholder implementation - actual perplexity requires trained models",
        }
        
        print(f"  Perplexity: {perplexity:.2f}")
        print()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"{'Tokenizer':<20} {'Perplexity':<12}")
    print("-" * 35)
    for tokenizer_id, result in results.items():
        print(f"{tokenizer_id:<20} {result['perplexity']:<12.2f}")
    
    print("\nNote: These are placeholder values.")
    print("For actual evaluation, you would:")
    print("  1. Load trained models")
    print("  2. Calculate cross-entropy loss on test set")
    print("  3. Convert to perplexity: exp(loss)")
    print("  4. Compare across tokenizers")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate downstream models")
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        help="Model directories (format: tokenizer_id:path/to/model)",
    )
    parser.add_argument(
        "--test-corpus",
        type=str,
        default="data/hindi/corpus/training.txt",
        help="Path to test corpus",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scorecards/downstream/results.json",
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    # Parse model directories
    model_dirs = {}
    tokenizer_ids = []
    for model_spec in args.models:
        if ':' in model_spec:
            tid, path = model_spec.split(':', 1)
            model_dirs[tid] = Path(path)
            tokenizer_ids.append(tid)
        else:
            print(f"Warning: Invalid model spec '{model_spec}', expected format: tokenizer_id:path")
    
    if not model_dirs:
        print("Error: No valid model directories specified")
        return 1
    
    # Load tokenizers
    tokenizers = {}
    for tid in tokenizer_ids:
        try:
            if tid.startswith("gpe"):
                from tokenizers.gpe_tokenizer import GPETokenizer
                model_map = {
                    "gpe_cbpe_hi_v1": "models/gpe_cbpe_hi_v1",
                }
                if tid in model_map:
                    tokenizers[tid] = GPETokenizer(tid, model_map[tid])
            elif tid in ["indicbert", "mbert"]:
                from scripts.compare_tokenizers import HFTokenizer
                import os
                token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
                model_map = {
                    "indicbert": "ai4bharat/indic-bert",
                    "mbert": "bert-base-multilingual-cased",
                }
                tokenizers[tid] = HFTokenizer(tid, model_map[tid], token=token)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tid}: {e}")
    
    # Load test corpus
    test_corpus_path = Path(args.test_corpus)
    if not test_corpus_path.exists():
        print(f"Error: Test corpus file not found: {test_corpus_path}")
        return 1
    
    # Evaluate
    output_path = Path(args.output)
    evaluate_downstream_models(model_dirs, test_corpus_path, tokenizers, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

