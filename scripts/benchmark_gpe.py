#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script for GPE+CBPE tokenizer that works around import conflicts.

This script directly imports tokenizers to avoid the naming conflict with
HuggingFace's tokenizers library.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizers.gpe_tokenizer import GPETokenizer
from eval.metrics import evaluate_comprehensive
import json

# Load tokenizers
print("Loading tokenizers...")
gpe = GPETokenizer('gpe_cbpe_hi_v1', 'models/gpe_cbpe_hi_v1')
print("✓ GPE+CBPE v1 loaded")

# Test texts
test_texts = [
    'भारत में आज कई महत्वपूर्ण घटनाएं हुईं।',
    'राजधानी दिल्ली में मौसम सुहावना है।',
    'नई दिल्ली में आज एक बड़ी बैठक हुई।',
]

print("\n=== Benchmark Results ===\n")

results = []
for text in test_texts:
    tokens = gpe.tokenize(text)
    metrics = evaluate_comprehensive(text, tokens, lang='hi', baseline_tokens=None)
    
    results.append({
        'text': text,
        'num_tokens': metrics.num_tokens,
        'fertility': metrics.efficiency.fertility,
        'chars_per_token': metrics.efficiency.chars_per_token,
        'compression_ratio_chars': metrics.efficiency.compression_ratio_chars,
        'grapheme_violation_rate': metrics.script.grapheme_violation_rate,
        'akshara_integrity_rate': metrics.script.akshara_integrity_rate,
        'dependent_vowel_split_rate': metrics.script.dependent_vowel_split_rate,
    })
    
    print(f"Text: {text}")
    print(f"  Tokens: {metrics.num_tokens}")
    print(f"  Fertility: {metrics.efficiency.fertility:.3f}")
    print(f"  Chars/Token: {metrics.efficiency.chars_per_token:.3f}")
    print(f"  CR (chars): {metrics.efficiency.compression_ratio_chars:.3f}")
    print(f"  Grapheme Violation: {metrics.script.grapheme_violation_rate:.2%}")
    print(f"  Akshara Integrity: {metrics.script.akshara_integrity_rate:.2%}")
    print(f"  Dependent Vowel Split: {metrics.script.dependent_vowel_split_rate:.2%}")
    print()

# Summary
avg_fertility = sum(r['fertility'] for r in results) / len(results)
avg_chars_per_token = sum(r['chars_per_token'] for r in results) / len(results)
avg_gv = sum(r['grapheme_violation_rate'] for r in results) / len(results)
avg_ai = sum(r['akshara_integrity_rate'] for r in results) / len(results)

print("=== Summary ===")
print(f"Average Fertility: {avg_fertility:.3f}")
print(f"Average Chars/Token: {avg_chars_per_token:.3f}")
print(f"Average Grapheme Violation Rate: {avg_gv:.2%}")
print(f"Average Akshara Integrity: {avg_ai:.2%}")

# Save results
output_dir = Path('scorecards/hi_benchmark')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'gpe_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Results saved to {output_dir / 'gpe_results.json'}")

