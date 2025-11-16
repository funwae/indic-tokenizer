#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fairness benchmark script for cross-language tokenization evaluation.

Compares tokenization fairness metrics (parity, premium, disparity) across
different tokenizers using parallel Hindi-English corpus.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics.fairness import evaluate_fairness, tokenization_parity, tokenization_premium
from eval.metrics.efficiency import evaluate_efficiency


def load_parallel_corpus(corpus_path: Path) -> List[Tuple[str, str]]:
    """
    Load parallel Hindi-English corpus.
    
    Format: Tab-separated Hindi-English sentence pairs, one per line.
    """
    pairs = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def tokenize_with_tokenizer(tokenizer, text: str) -> List[str]:
    """Tokenize text using a tokenizer object."""
    return tokenizer.tokenize(text)


def benchmark_fairness(
    tokenizers: Dict[str, any],
    parallel_corpus: List[Tuple[str, str]],
    output_path: Path,
):
    """
    Benchmark fairness metrics across tokenizers.
    
    Parameters
    ----------
    tokenizers : Dict[str, any]
        Dictionary mapping tokenizer_id -> tokenizer object.
    parallel_corpus : List[Tuple[str, str]]
        List of (hindi_text, english_text) pairs.
    output_path : Path
        Path to save results JSON.
    """
    results = {}
    
    print("=== Fairness Benchmark ===\n")
    print(f"Tokenizers: {list(tokenizers.keys())}")
    print(f"Parallel sentences: {len(parallel_corpus)}\n")
    
    for tokenizer_id, tokenizer in tokenizers.items():
        print(f"Evaluating {tokenizer_id}...")
        
        # Aggregate metrics across corpus
        total_premium = 0.0
        total_parity = 0.0
        total_cr_hi = 0.0
        total_cr_en = 0.0
        count = 0
        
        sentence_results = []
        
        for hi_text, en_text in parallel_corpus:
            # Tokenize both languages
            hi_tokens = tokenize_with_tokenizer(tokenizer, hi_text)
            en_tokens = tokenize_with_tokenizer(tokenizer, en_text)
            
            # Calculate compression ratios
            hi_cr = len(hi_text) / len(hi_tokens) if len(hi_tokens) > 0 else 0.0
            en_cr = len(en_text) / len(en_tokens) if len(en_tokens) > 0 else 0.0
            
            # Calculate fairness metrics
            premium = tokenization_premium(hi_tokens, en_tokens)
            parity = tokenization_parity(hi_tokens, en_tokens)
            disparity = abs(hi_cr - en_cr)
            
            total_premium += premium
            total_parity += parity
            total_cr_hi += hi_cr
            total_cr_en += en_cr
            count += 1
            
            sentence_results.append({
                'hindi_text': hi_text,
                'english_text': en_text,
                'hindi_tokens': len(hi_tokens),
                'english_tokens': len(en_tokens),
                'tokenization_premium': premium,
                'tokenization_parity': parity,
                'compression_ratio_disparity': disparity,
                'hindi_cr': hi_cr,
                'english_cr': en_cr,
            })
        
        # Calculate averages
        avg_premium = total_premium / count if count > 0 else 0.0
        avg_parity = total_parity / count if count > 0 else 0.0
        avg_cr_hi = total_cr_hi / count if count > 0 else 0.0
        avg_cr_en = total_cr_en / count if count > 0 else 0.0
        avg_disparity = abs(avg_cr_hi - avg_cr_en)
        
        results[tokenizer_id] = {
            'average_tokenization_premium': avg_premium,
            'average_tokenization_parity': avg_parity,
            'average_compression_ratio_disparity': avg_disparity,
            'average_hindi_cr': avg_cr_hi,
            'average_english_cr': avg_cr_en,
            'sentence_results': sentence_results,
        }
        
        print(f"  Average Tokenization Premium: {avg_premium:.3f}")
        print(f"  Average Tokenization Parity: {avg_parity:.3f}")
        print(f"  Average CR Disparity: {avg_disparity:.3f}")
        print()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Tokenizer':<20} {'Premium':<10} {'Parity':<10} {'CR Disparity':<15}")
    print("-" * 60)
    for tokenizer_id, result in results.items():
        print(f"{tokenizer_id:<20} {result['average_tokenization_premium']:<10.3f} "
              f"{result['average_tokenization_parity']:<10.3f} "
              f"{result['average_compression_ratio_disparity']:<15.3f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark fairness metrics")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/parallel/hi_en.txt",
        help="Path to parallel corpus file",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs='+',
        default=["gpt4o", "gpt4o_mini"],
        help="Tokenizer IDs to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scorecards/fairness_benchmark/results.json",
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    # Load parallel corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        return 1
    
    parallel_corpus = load_parallel_corpus(corpus_path)
    print(f"Loaded {len(parallel_corpus)} parallel sentences\n")
    
    # Load tokenizers
    # Note: This is a simplified version - in production, use the registry system
    # For now, we'll use direct imports to avoid naming conflicts
    tokenizers = {}
    
    for tid in args.tokenizers:
        try:
            if tid.startswith("gpt"):
                from scripts.compare_tokenizers import OpenAITokenizer
                model_map = {
                    "gpt4o": "gpt-4o",
                    "gpt4o_mini": "gpt-4o-mini",
                }
                tokenizers[tid] = OpenAITokenizer(tid, model_map[tid])
            elif tid.startswith("llama"):
                from tokenizers.llama_tokenizer import LlamaTokenizer
                import os
                token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
                model_map = {
                    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
                }
                tokenizers[tid] = LlamaTokenizer(tid, model_map[tid], token=token)
            else:
                print(f"Warning: Unknown tokenizer {tid}, skipping")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tid}: {e}")
    
    if not tokenizers:
        print("Error: No tokenizers loaded")
        return 1
    
    # Run benchmark
    output_path = Path(args.output)
    benchmark_fairness(tokenizers, parallel_corpus, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

