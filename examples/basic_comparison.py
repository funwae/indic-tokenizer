#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic comparison example for Indic Tokenization Lab.

This example shows how to compare tokenizers on a simple Hindi text.
"""

from pathlib import Path
from scripts.compare_tokenizers import load_registry, create_tokenizer_from_config

# Load registry
registry = load_registry(Path("tokenizers/registry.yaml"))

# Text to tokenize
text = "यहाँ आपका हिंदी वाक्य जाएगा।"

# Compare tokenizers
tokenizer_ids = ["indicbert", "mbert"]  # Add more as available

print("Tokenizer Comparison")
print("=" * 60)
print(f"Text: {text}\n")

for tid in tokenizer_ids:
    if tid not in registry:
        print(f"Warning: {tid} not found, skipping")
        continue

    try:
        cfg = registry[tid]
        tokenizer = create_tokenizer_from_config(cfg)
        tokens = tokenizer.tokenize(text)
        stats = tokenizer.stats(text)

        print(f"{tokenizer.display_name}:")
        print(f"  Tokens: {stats.num_tokens}")
        print(f"  Chars/token: {stats.chars_per_token:.2f}")
        print(f"  Tokens: {tokens}\n")
    except Exception as e:
        print(f"Error with {tid}: {e}\n")

