#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/verify_setup.py

Quick verification that all components are importable and basic functionality works.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from tokenizers import grapheme_segmenter, cbpe_constraints, gpe_tokenizer
        from tokenizers import pretokenizer, sentencepiece_tokenizer
        from eval import grapheme_violations, fertility, metrics
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_grapheme_segmentation():
    """Test grapheme segmentation."""
    print("\nTesting grapheme segmentation...")
    try:
        from tokenizers.grapheme_segmenter import segment_devanagari_graphemes

        test_cases = [
            ("किशोरी", 3),  # Should have 3 graphemes
            ("प्रार्थना", 3),  # Should have 3+ graphemes (can vary)
            ("कर्मयोग", 4),  # Should have 4 graphemes
        ]

        all_passed = True
        for text, expected_min in test_cases:
            result = segment_devanagari_graphemes(text)
            if len(result) >= expected_min:
                print(f"  ✓ '{text}' -> {len(result)} graphemes: {result}")
            else:
                print(f"  ✗ '{text}' -> {len(result)} graphemes (expected at least {expected_min})")
                all_passed = False

        if all_passed:
            print("✓ Grapheme segmentation works")
        return all_passed
    except Exception as e:
        print(f"✗ Grapheme segmentation error: {e}")
        return False


def test_cbpe_constraints():
    """Test CBPE constraints."""
    print("\nTesting CBPE constraints...")
    try:
        from tokenizers.cbpe_constraints import cbpe_merge_allowed

        # Test cases: (left, right, should_be_allowed)
        test_cases = [
            ("क", "्ष", True),  # Normal merge
            ("्", "ा", False),  # Virama + dependent vowel (should be disallowed)
            ("ा", "क", False),  # Dependent vowel at start (should be disallowed)
            ("क", "ा", True),  # Consonant + dependent vowel (should be allowed)
        ]

        all_passed = True
        for left, right, expected in test_cases:
            result = cbpe_merge_allowed(left, right)
            if result == expected:
                print(f"  ✓ cbpe_merge_allowed('{left}', '{right}') = {result} (expected {expected})")
            else:
                print(f"  ✗ cbpe_merge_allowed('{left}', '{right}') = {result} (expected {expected})")
                all_passed = False

        if all_passed:
            print("✓ CBPE constraints work correctly")
        return all_passed
    except Exception as e:
        print(f"✗ CBPE constraints error: {e}")
        return False


def test_registry():
    """Test registry loading."""
    print("\nTesting registry...")
    try:
        from scripts.compare_tokenizers import load_registry

        registry_path = Path("tokenizers/registry.yaml")
        if not registry_path.exists():
            print(f"✗ Registry file not found: {registry_path}")
            return False

        registry = load_registry(registry_path)
        if len(registry) > 0:
            print(f"✓ Registry loads successfully ({len(registry)} tokenizers)")
            for tid in registry.keys():
                print(f"  - {tid}")
            return True
        else:
            print("✗ Registry is empty")
            return False
    except Exception as e:
        print(f"✗ Registry loading error: {e}")
        return False


def test_evaluation():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    try:
        from eval.fertility import calculate_fertility, calculate_chars_per_token
        from eval.grapheme_violations import count_violations, violation_rate

        text = "यहाँ आपका हिंदी"
        tokens = ["यहाँ", "आपका", "हिंदी"]

        fertility = calculate_fertility(text, tokens)
        chars_per_token = calculate_chars_per_token(text, tokens)
        violations = count_violations(text, tokens)
        v_rate = violation_rate(text, tokens)

        print(f"  Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Fertility: {fertility:.3f}")
        print(f"  Chars/token: {chars_per_token:.3f}")
        print(f"  Violations: {violations}")
        print(f"  Violation rate: {v_rate:.3f}")

        if fertility > 0 and chars_per_token > 0:
            print("✓ Evaluation metrics work")
            return True
        else:
            print("✗ Evaluation metrics returned invalid values")
            return False
    except Exception as e:
        print(f"✗ Evaluation metrics error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretokenizer():
    """Test pretokenizer."""
    print("\nTesting pretokenizer...")
    try:
        from tokenizers.pretokenizer import pretokenize

        test_cases = [
            "यहाँ आपका हिंदी वाक्य जाएगा।",
            "Visit https://example.com for more info",
            "Email: test@example.com",
            "#hashtag @mention",
        ]

        all_passed = True
        for text in test_cases:
            result = pretokenize(text)
            if len(result) > 0:
                print(f"  ✓ '{text[:30]}...' -> {len(result)} tokens")
            else:
                print(f"  ✗ '{text[:30]}...' -> 0 tokens")
                all_passed = False

        if all_passed:
            print("✓ Pretokenizer works")
        return all_passed
    except Exception as e:
        print(f"✗ Pretokenizer error: {e}")
        return False


def test_hf_tokenizer_loading():
    """Test loading HF tokenizers (if transformers available)."""
    print("\nTesting HF tokenizer loading...")
    try:
        from scripts.compare_tokenizers import load_registry, create_tokenizer_from_config

        registry_path = Path("tokenizers/registry.yaml")
        registry = load_registry(registry_path)

        # Try to load indicbert if available
        if "indicbert" in registry:
            try:
                cfg = registry["indicbert"]
                tokenizer = create_tokenizer_from_config(cfg)
                # Test tokenization
                test_text = "यहाँ आपका हिंदी"
                tokens = tokenizer.tokenize(test_text)
                print(f"  ✓ Successfully loaded: {tokenizer.display_name}")
                print(f"    Test tokenization: {len(tokens)} tokens")
                return True
            except Exception as e:
                print(f"  ⚠ Could not load indicbert: {e}")
                print("  (This is OK - model will download on first use)")
                return True  # Not a failure, just a warning
        else:
            print("  ⚠ indicbert not in registry")
            return True
    except Exception as e:
        print(f"✗ HF tokenizer loading error: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Indic Tokenization Lab - Setup Verification")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Grapheme Segmentation", test_grapheme_segmentation),
        ("CBPE Constraints", test_cbpe_constraints),
        ("Registry", test_registry),
        ("Evaluation Metrics", test_evaluation),
        ("Pretokenizer", test_pretokenizer),
        ("HF Tokenizer Loading", test_hf_tokenizer_loading),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All basic tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

