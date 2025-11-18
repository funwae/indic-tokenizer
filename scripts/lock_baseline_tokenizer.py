#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lock in baseline GPE+CBPE Hindi tokenizer variants.

This script verifies and documents trained tokenizer models, ensuring they
are ready for comprehensive evaluation as baselines.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_tokenizer_model(model_dir: Path) -> Dict[str, Any]:
    """
    Verify a tokenizer model is complete and ready.

    Parameters
    ----------
    model_dir : Path
        Path to model directory.

    Returns
    -------
    Dict[str, Any]
        Model metadata and verification status.
    """
    status = {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "has_vocab": False,
        "has_merges": False,
        "has_config": False,
        "vocab_size": 0,
        "num_merges": 0,
        "ready": False,
    }

    if not model_dir.exists():
        return status

    # Check vocab.json
    vocab_path = model_dir / "vocab.json"
    if vocab_path.exists():
        status["has_vocab"] = True
        try:
            with vocab_path.open("r", encoding="utf-8") as f:
                vocab = json.load(f)
                status["vocab_size"] = len(vocab)
        except Exception as e:
            status["vocab_error"] = str(e)

    # Check merges.txt
    merges_path = model_dir / "merges.txt"
    if merges_path.exists():
        status["has_merges"] = True
        try:
            with merges_path.open("r", encoding="utf-8") as f:
                merges = [line.strip() for line in f if line.strip()]
                status["num_merges"] = len(merges)
        except Exception as e:
            status["merges_error"] = str(e)

    # Check config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        status["has_config"] = True
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                status["config"] = config
        except Exception as e:
            status["config_error"] = str(e)

    # Model is ready if it has all required files
    status["ready"] = (
        status["has_vocab"] and
        status["has_merges"] and
        status["has_config"]
    )

    return status


def lock_baseline_variants() -> Dict[str, Any]:
    """
    Lock in baseline GPE+CBPE tokenizer variants.

    Returns
    -------
    Dict[str, Any]
        Status of baseline variants.
    """
    results = {}

    # Check v1
    v1_dir = project_root / "models" / "gpe_cbpe_hi_v1"
    results["gpe_cbpe_hi_v1"] = verify_tokenizer_model(v1_dir)

    # Check v2 (optional)
    v2_dir = project_root / "models" / "gpe_cbpe_hi_v2"
    if v2_dir.exists():
        results["gpe_cbpe_hi_v2"] = verify_tokenizer_model(v2_dir)

    return results


def main():
    """Main entry point."""
    print("=== Locking Baseline GPE+CBPE Tokenizer Variants ===\n")

    results = lock_baseline_variants()

    all_ready = True
    for variant_id, status in results.items():
        print(f"{variant_id}:")
        print(f"  Model directory: {status['model_dir']}")
        print(f"  Exists: {status['exists']}")
        print(f"  Has vocab.json: {status['has_vocab']} (vocab_size: {status['vocab_size']})")
        print(f"  Has merges.txt: {status['has_merges']} (num_merges: {status['num_merges']})")
        print(f"  Has config.json: {status['has_config']}")
        print(f"  Ready: {status['ready']}")

        if not status['ready']:
            all_ready = False
            print(f"  ⚠️  WARNING: {variant_id} is not ready for evaluation")
        else:
            print(f"  ✓ {variant_id} is ready")
        print()

    if all_ready:
        print("✓ All baseline variants are locked and ready for evaluation")
        return 0
    else:
        print("⚠️  Some variants are not ready. Train missing models before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

