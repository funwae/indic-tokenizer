#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation example for Indic Tokenization Lab.

This example shows how to evaluate a tokenizer with comprehensive metrics.
"""

from pathlib import Path
from scripts.compare_tokenizers import load_registry, create_tokenizer_from_config
from eval.metrics import evaluate_tokenizer, export_scorecard, generate_scorecard

# Load registry
registry = load_registry(Path("tokenizers/registry.yaml"))

# Text to evaluate
text = "यहाँ आपका हिंदी वाक्य जाएगा।"

# Evaluate tokenizer
tokenizer_id = "indicbert"  # Change as needed

if tokenizer_id not in registry:
    print(f"Error: {tokenizer_id} not found in registry")
    exit(1)

cfg = registry[tokenizer_id]
tokenizer = create_tokenizer_from_config(cfg)

# Evaluate
metrics = evaluate_tokenizer(text, tokenizer, lang="hi")

print("Evaluation Results")
print("=" * 60)
print(f"Text: {text}\n")
print(f"Tokenizer: {tokenizer.display_name}\n")
print("Metrics:")
print(f"  Fertility: {metrics.fertility:.3f} tokens/word")
print(f"  Chars per token: {metrics.chars_per_token:.3f}")
print(f"  Grapheme violations: {metrics.grapheme_violations}")
print(f"  Violation rate: {metrics.grapheme_violation_rate:.2%}")
print(f"  Total tokens: {metrics.num_tokens}")
print(f"  Total words: {metrics.num_words}")

# Generate scorecard
scorecards = generate_scorecard(
    {tokenizer_id: metrics},
    {tokenizer_id: tokenizer.display_name},
    [text]
)

# Export as markdown
print("\n" + export_scorecard(scorecards, format="markdown"))

