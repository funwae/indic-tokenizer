#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify HuggingFace authentication works.

This script tests authentication with gated models like ai4bharat/indic-bert.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_hf_auth():
    """Test HuggingFace authentication."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    
    if not token:
        print("Error: HUGGING_FACE_HUB_TOKEN or HF_TOKEN environment variable not set")
        print("Set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'")
        return False
    
    try:
        from transformers import AutoTokenizer
        
        print("Testing HuggingFace authentication...")
        print(f"Token: {token[:10]}...{token[-4:]}")
        print()
        
        # Test with ai4bharat/indic-bert (gated model)
        print("Loading ai4bharat/indic-bert tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indic-bert",
            token=token
        )
        print("✓ Tokenizer loaded successfully!")
        print()
        
        # Test tokenization
        test_text = "यहाँ आपका हिंदी वाक्य जाएगा।"
        print(f"Testing tokenization on: {test_text}")
        tokens = tokenizer.tokenize(test_text)
        print(f"✓ Tokenization successful!")
        print(f"  Number of tokens: {len(tokens)}")
        print(f"  First 10 tokens: {tokens[:10]}")
        print()
        
        print("✓ All tests passed! HuggingFace authentication is working.")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you've accepted the model license:")
        print("   https://huggingface.co/ai4bharat/indic-bert")
        print("2. Verify your token is correct")
        print("3. Check your internet connection")
        return False


if __name__ == "__main__":
    success = test_hf_auth()
    sys.exit(0 if success else 1)

