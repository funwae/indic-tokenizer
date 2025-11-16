#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to set up HuggingFace authentication.

This script helps you configure HuggingFace authentication for accessing
gated models like ai4bharat/indic-bert.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import login, whoami
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)


def main():
    """Main entry point."""
    print("=" * 60)
    print("HuggingFace Authentication Setup")
    print("=" * 60)
    print()

    # Check if already logged in
    try:
        user_info = whoami()
        print(f"✓ Already logged in as: {user_info.get('name', 'Unknown')}")
        print(f"  Full name: {user_info.get('fullname', 'N/A')}")
        print()

        response = input("Do you want to login with a different account? (y/N): ")
        if response.lower() != 'y':
            print("Keeping current authentication.")
            return
    except Exception:
        print("Not currently logged in.")
        print()

    # Get token
    print("To get your HuggingFace token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Name it (e.g., 'indic-tokenizer-lab')")
    print("4. Select 'Read' permissions")
    print("5. Copy the token")
    print()

    token = input("Paste your HuggingFace token here: ").strip()

    if not token:
        print("Error: No token provided.")
        sys.exit(1)

    if not token.startswith("hf_"):
        print("Warning: Token should start with 'hf_'. Continuing anyway...")
        print()

    # Login
    try:
        print("Logging in...")
        login(token=token)
        print("✓ Successfully logged in!")
        print()

        # Verify
        user_info = whoami()
        print(f"Logged in as: {user_info.get('name', 'Unknown')}")
        print()
        print("You can now use gated models like ai4bharat/indic-bert.")
        print()
        print("To test, run:")
        print('  python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained(\'ai4bharat/indic-bert\'); print(\'✓ Success!\')"')

    except Exception as e:
        print(f"Error: Failed to login: {e}")
        print()
        print("Make sure:")
        print("1. The token is correct")
        print("2. You have internet connection")
        print("3. You've accepted the model license (for gated models)")
        sys.exit(1)


if __name__ == "__main__":
    main()

