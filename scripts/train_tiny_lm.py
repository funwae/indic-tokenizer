#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a tiny Hindi language model for downstream evaluation.

Uses the tiny_lm.py model architecture (~1-3M parameters) to demonstrate
tokenization impact on downstream tasks.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.tiny_lm import create_tiny_lm


class TokenizedDataset(Dataset):
    """Dataset for tokenized sequences."""

    def __init__(self, sequences: List[List[int]], max_len: int = 256):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        # Pad or truncate to max_len
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long) if TORCH_AVAILABLE else seq


def train_lm_with_tokenizer(
    tokenizer_id: str,
    tokenizer: Any,
    corpus_path: Path,
    output_dir: Path,
    max_sentences: int = 10000,
    steps: int = 50000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_seq_len: int = 256,
):
    """Train a tiny language model using a specific tokenizer."""
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Creating placeholder configuration.")
        return _create_placeholder_config(tokenizer_id, tokenizer, corpus_path, output_dir, max_sentences)

    print(f"Training LM with {tokenizer_id} tokenizer...")

    # Load and tokenize corpus
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            line = line.strip()
            if line:
                sentences.append(line)

    print(f"  Loaded {len(sentences)} sentences")

    # Tokenize all sentences
    print("  Tokenizing corpus...")
    sequences = []
    vocab_set = set()

    for sentence in sentences:
        try:
            token_ids = tokenizer.encode(sentence)
            if len(token_ids) > 0:
                sequences.append(token_ids)
                vocab_set.update(token_ids)
        except Exception as e:
            print(f"    Warning: Failed to tokenize sentence: {e}")
            continue

    vocab_size = len(vocab_set)
    if vocab_size == 0:
        vocab_size = 1000  # Fallback

    print(f"  Tokenized {len(sequences)} sequences")
    print(f"  Vocabulary size: {vocab_size}")

    # Create model
    model = create_tiny_lm(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_seq_len=max_seq_len,
    )

    num_params = model.get_num_params()
    print(f"  Model parameters: {num_params:,} (~{num_params/1e6:.2f}M)")

    # Create dataset and dataloader
    dataset = TokenizedDataset(sequences, max_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    print(f"  Training on device: {device}")
    print(f"  Training for {steps} steps...")

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0

    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break

            batch = batch.to(device)

            # Create input and target (shift by 1 for next-token prediction)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # Forward pass
            logits = model(input_ids)

            # Reshape for loss computation
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            # Compute loss
            loss = criterion(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 1000 == 0:
                avg_loss = total_loss / 1000
                print(f"    Step {step}/{steps}, Loss: {avg_loss:.4f}")
                total_loss = 0.0

    print(f"  ✓ Training complete")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Model saved to {model_path}")

    # Save config
    config = {
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 2,
        "max_seq_len": max_seq_len,
        "num_params": num_params,
        "training_steps": steps,
    }
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config saved to {config_path}")

    return output_dir


def _create_placeholder_config(
    tokenizer_id: str,
    tokenizer: Any,
    corpus_path: Path,
    output_dir: Path,
    max_sentences: int,
):
    """Create placeholder configuration when PyTorch is not available."""
    print(f"Creating placeholder config for {tokenizer_id} tokenizer...")

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

    # Estimate vocab size
    vocab_set = set()
    for sentence in sentences[:100]:
        try:
            tokens = tokenizer.tokenize(sentence)
            vocab_set.update(tokens)
        except:
            pass

    vocab_size = len(vocab_set) if vocab_set else 1000

    print(f"  Estimated vocab size: {vocab_size}")

    # Create model config
    model = create_tiny_lm(vocab_size=vocab_size)
    num_params = model.get_num_params()

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 2,
        "max_seq_len": 256,
        "num_params": num_params,
        "status": "placeholder",
        "note": "PyTorch not available. Install PyTorch to train actual model.",
    }
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)

    print(f"  ✓ Placeholder config saved to {config_path}")
    print(f"  Note: Install PyTorch to train actual model.")

    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train tiny Hindi LM")
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        required=True,
        help="Tokenizer ID (e.g., mbert, gpe_cbpe_hi_v1)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/hindi/processed/gpe_cbpe_hi_corpus.txt",
        help="Path to training corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for model (e.g., models/tiny_lm_hi/<tokenizer-id>)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Number of training steps (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=10000,
        help="Maximum sentences to use (default: 10000)",
    )

    args = parser.parse_args()

    # Load tokenizer from registry
    from scripts.compare_tokenizers import load_registry, create_tokenizer_from_config

    registry = load_registry(Path("tokenizers/registry.yaml"))
    if args.tokenizer_id not in registry:
        print(f"Error: Tokenizer '{args.tokenizer_id}' not found in registry", file=sys.stderr)
        return 1

    try:
        tokenizer = create_tokenizer_from_config(registry[args.tokenizer_id])
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        return 1

    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}", file=sys.stderr)
        return 1

    # Train model
    output_dir = Path(args.output_dir)
    train_lm_with_tokenizer(
        args.tokenizer_id,
        tokenizer,
        corpus_path,
        output_dir,
        max_sentences=args.max_sentences,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

