#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a small downstream language model as a proxy task.

This is a minimal implementation to demonstrate that better tokenization
leads to better downstream performance. Uses small models and limited data.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_minimal_lm_config(vocab_size: int, hidden_size: int = 128, num_layers: int = 2) -> Dict:
    """
    Create a minimal transformer LM configuration.
    
    This is a placeholder - in a real implementation, you would use
    transformers library to create and train a model.
    """
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": 2,
        "intermediate_size": hidden_size * 4,
        "max_position_embeddings": 512,
        "model_type": "gpt2",  # Use GPT-2 architecture as base
    }


def train_lm_with_tokenizer(
    tokenizer_id: str,
    tokenizer,
    corpus_path: Path,
    output_dir: Path,
    max_sentences: int = 10000,
):
    """
    Train a language model using a specific tokenizer.
    
    This is a placeholder implementation. In a real scenario, you would:
    1. Load and tokenize the corpus
    2. Create a small transformer model
    3. Train the model
    4. Save the model and tokenizer
    
    For now, we just create a config file and document the process.
    """
    print(f"Training LM with {tokenizer_id} tokenizer...")
    
    # Load corpus
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            line = line.strip()
            if line:
                sentences.append(line)
    
    print(f"  Loaded {len(sentences)} sentences")
    
    # Tokenize sample to get vocab size
    sample_text = sentences[0] if sentences else ""
    sample_tokens = tokenizer.tokenize(sample_text)
    vocab_size = len(set(tokenizer.tokenize(" ".join(sentences[:100]))))  # Approximate
    
    print(f"  Estimated vocab size: {vocab_size}")
    
    # Create model config
    config = create_minimal_lm_config(vocab_size=vocab_size)
    
    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Create a placeholder training log
    training_log = {
        "tokenizer_id": tokenizer_id,
        "num_sentences": len(sentences),
        "vocab_size": vocab_size,
        "status": "placeholder",
        "note": "This is a placeholder implementation. In production, you would train an actual model here.",
    }
    
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"  ✓ Config saved to {config_path}")
    print(f"  ✓ Training log saved to {log_path}")
    print(f"  Note: This is a placeholder. Actual model training would happen here.")
    
    return output_dir


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train downstream LM (placeholder)")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer ID to use",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/hindi/corpus/training.txt",
        help="Path to training corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for model",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=10000,
        help="Maximum number of sentences to use",
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizers = {}
    try:
        if args.tokenizer.startswith("gpe"):
            from tokenizers.gpe_tokenizer import GPETokenizer
            model_map = {
                "gpe_cbpe_hi_v1": "models/gpe_cbpe_hi_v1",
            }
            if args.tokenizer in model_map:
                tokenizers[args.tokenizer] = GPETokenizer(args.tokenizer, model_map[args.tokenizer])
        elif args.tokenizer in ["indicbert", "mbert"]:
            from scripts.compare_tokenizers import HFTokenizer
            import os
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
            model_map = {
                "indicbert": "ai4bharat/indic-bert",
                "mbert": "bert-base-multilingual-cased",
            }
            tokenizers[args.tokenizer] = HFTokenizer(args.tokenizer, model_map[args.tokenizer], token=token)
        else:
            print(f"Error: Unknown tokenizer {args.tokenizer}")
            return 1
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1
    
    if args.tokenizer not in tokenizers:
        print(f"Error: Tokenizer {args.tokenizer} not loaded")
        return 1
    
    tokenizer = tokenizers[args.tokenizer]
    
    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        return 1
    
    # Train model
    output_dir = Path(args.output_dir)
    train_lm_with_tokenizer(
        args.tokenizer,
        tokenizer,
        corpus_path,
        output_dir,
        max_sentences=args.max_sentences,
    )
    
    print(f"\n✓ Training complete (placeholder). Model config saved to {output_dir}")
    print("\nNote: This is a placeholder implementation.")
    print("For actual training, you would:")
    print("  1. Use transformers library to create a GPT-2 style model")
    print("  2. Train the model on tokenized corpus")
    print("  3. Evaluate perplexity on test set")
    print("  4. Compare results across tokenizers")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

