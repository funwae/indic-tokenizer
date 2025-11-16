#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphology evaluation script for tokenization.

Evaluates tokenizers on morphology-annotated dataset and computes
boundary precision/recall/F1 and morpheme-aligned token rate.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.metrics.morphology import evaluate_morphology


def load_annotated_dataset(dataset_path: Path) -> List[Tuple[str, str]]:
    """
    Load morphology-annotated dataset.
    
    Format: Each line contains a sentence with morpheme boundaries marked.
    Format: "word1|morpheme1+morpheme2 word2|morpheme3"
    Lines starting with '#' are comments.
    
    Returns list of (original_text, annotated_text) pairs.
    """
    pairs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Extract original text (remove morpheme annotations)
            # Format: "word1|morpheme1+morpheme2 word2|morpheme3"
            # Original: "word1 word2"
            original_parts = []
            for word_annotation in line.split():
                if '|' in word_annotation:
                    word = word_annotation.split('|')[0]
                else:
                    word = word_annotation
                original_parts.append(word)
            original_text = ' '.join(original_parts)
            
            pairs.append((original_text, line))
    
    return pairs


def tokenize_with_tokenizer(tokenizer, text: str) -> List[str]:
    """Tokenize text using a tokenizer object."""
    return tokenizer.tokenize(text)


def evaluate_morphology_dataset(
    tokenizers: Dict[str, any],
    dataset: List[Tuple[str, str]],
    output_path: Path,
):
    """
    Evaluate morphology metrics across tokenizers.
    
    Parameters
    ----------
    tokenizers : Dict[str, any]
        Dictionary mapping tokenizer_id -> tokenizer object.
    dataset : List[Tuple[str, str]]
        List of (original_text, annotated_text) pairs.
    output_path : Path
        Path to save results JSON.
    """
    results = {}
    
    print("=== Morphology Evaluation ===\n")
    print(f"Tokenizers: {list(tokenizers.keys())}")
    print(f"Annotated sentences: {len(dataset)}\n")
    
    for tokenizer_id, tokenizer in tokenizers.items():
        print(f"Evaluating {tokenizer_id}...")
        
        sentence_results = []
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_aligned_rate = 0.0
        count = 0
        
        for original_text, annotated_text in dataset:
            # Tokenize
            tokens = tokenize_with_tokenizer(tokenizer, original_text)
            
            # Evaluate morphology
            metrics = evaluate_morphology(original_text, tokens, annotated_text)
            
            sentence_results.append({
                'text': original_text,
                'num_tokens': metrics.num_tokens,
                'num_morphemes': metrics.num_morphemes,
                'boundary_precision': metrics.boundary_precision,
                'boundary_recall': metrics.boundary_recall,
                'boundary_f1': metrics.boundary_f1,
                'morpheme_aligned_token_rate': metrics.morpheme_aligned_token_rate,
            })
            
            total_precision += metrics.boundary_precision
            total_recall += metrics.boundary_recall
            total_f1 += metrics.boundary_f1
            total_aligned_rate += metrics.morpheme_aligned_token_rate
            count += 1
        
        # Calculate averages
        avg_precision = total_precision / count if count > 0 else 0.0
        avg_recall = total_recall / count if count > 0 else 0.0
        avg_f1 = total_f1 / count if count > 0 else 0.0
        avg_aligned_rate = total_aligned_rate / count if count > 0 else 0.0
        
        results[tokenizer_id] = {
            'average_boundary_precision': avg_precision,
            'average_boundary_recall': avg_recall,
            'average_boundary_f1': avg_f1,
            'average_morpheme_aligned_token_rate': avg_aligned_rate,
            'sentence_results': sentence_results,
        }
        
        print(f"  Average Boundary Precision: {avg_precision:.3f}")
        print(f"  Average Boundary Recall: {avg_recall:.3f}")
        print(f"  Average Boundary F1: {avg_f1:.3f}")
        print(f"  Average Morpheme-Aligned Token Rate: {avg_aligned_rate:.3f}")
        print()
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Tokenizer':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Aligned Rate':<15}")
    print("-" * 75)
    for tokenizer_id, result in results.items():
        print(f"{tokenizer_id:<20} {result['average_boundary_precision']:<12.3f} "
              f"{result['average_boundary_recall']:<12.3f} "
              f"{result['average_boundary_f1']:<12.3f} "
              f"{result['average_morpheme_aligned_token_rate']:<15.3f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate morphology metrics")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/hindi/morphology/annotated.txt",
        help="Path to morphology-annotated dataset",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs='+',
        default=["gpe_cbpe_hi_v1", "indicbert", "mbert"],
        help="Tokenizer IDs to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scorecards/morphology_benchmark/results.json",
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        return 1
    
    dataset = load_annotated_dataset(dataset_path)
    print(f"Loaded {len(dataset)} annotated sentences\n")
    
    # Load tokenizers
    # Note: Using direct imports to avoid naming conflicts
    tokenizers = {}
    
    for tid in args.tokenizers:
        try:
            if tid.startswith("gpe"):
                from tokenizers.gpe_tokenizer import GPETokenizer
                model_map = {
                    "gpe_cbpe_hi_v1": "models/gpe_cbpe_hi_v1",
                }
                if tid in model_map:
                    tokenizers[tid] = GPETokenizer(tid, model_map[tid])
            elif tid in ["indicbert", "mbert"]:
                # Use HFTokenizer via compare_tokenizers
                from scripts.compare_tokenizers import HFTokenizer
                import os
                token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
                model_map = {
                    "indicbert": "ai4bharat/indic-bert",
                    "mbert": "bert-base-multilingual-cased",
                }
                tokenizers[tid] = HFTokenizer(tid, model_map[tid], token=token)
            else:
                print(f"Warning: Unknown tokenizer {tid}, skipping")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tid}: {e}")
    
    if not tokenizers:
        print("Error: No tokenizers loaded")
        return 1
    
    # Run evaluation
    output_path = Path(args.output)
    evaluate_morphology_dataset(tokenizers, dataset, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

