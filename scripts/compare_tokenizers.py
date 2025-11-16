#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_tokenizers.py

Compare how different tokenizers segment a given text.

Usage examples:

  python scripts/compare_tokenizers.py \
      --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
      --tokenizers indicbert,generic_bert

  python scripts/compare_tokenizers.py \
      --file data/hindi/eval_sets/news_headlines.txt \
      --tokenizers indicbert \
      --json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # pip install pyyaml

try:
    from transformers import AutoTokenizer  # pip install transformers
except ImportError as e:
    AutoTokenizer = None  # type: ignore

try:
    from tokenizers.gpe_tokenizer import GPETokenizer
except ImportError:
    GPETokenizer = None  # type: ignore

try:
    from tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
except ImportError:
    SentencePieceTokenizer = None  # type: ignore

# ---------------------------------------------------------------------------
# Core tokenizer abstraction
# ---------------------------------------------------------------------------

@dataclass
class TokenizationStats:
    tokens: List[str]
    num_tokens: int
    chars: int
    chars_per_token: float

class BaseTokenizer:
    """
    Minimal tokenizer interface.

    Concrete subclasses must implement:
        - tokenize(text: str) -> List[str]
        - encode(text: str) -> List[int]
    """

    def __init__(self, tokenizer_id: str, display_name: Optional[str] = None):
        self.tokenizer_id = tokenizer_id
        self.display_name = display_name or tokenizer_id

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def stats(self, text: str) -> TokenizationStats:
        tokens = self.tokenize(text)
        chars = len(text)
        num_tokens = len(tokens)
        chars_per_token = chars / num_tokens if num_tokens > 0 else 0.0

        return TokenizationStats(
            tokens=tokens,
            num_tokens=num_tokens,
            chars=chars,
            chars_per_token=chars_per_token,
        )

class HFTokenizer(BaseTokenizer):
    """
    HuggingFace tokenizer adapter.

    Uses transformers.AutoTokenizer under the hood.
    """

    def __init__(self, tokenizer_id: str, model_name: str, display_name: Optional[str] = None):
        super().__init__(tokenizer_id, display_name)
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed. "
                "Install with: pip install transformers"
            )
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> List[str]:
        # Using the fast tokenizer's tokenize method
        return self._tokenizer.tokenize(text)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

# Optional: you can add an OpenAI/TikToken-based tokenizer later
class OpenAITokenizer(BaseTokenizer):
    """
    Placeholder for an OpenAI/TikToken-based tokenizer.

    Implement if you want to compare against OpenAI's tokenization
    (e.g., gpt-4.1-mini / gpt-4o). For now, this is a stub.
    """

    def __init__(self, tokenizer_id: str, model: str, display_name: Optional[str] = None):
        super().__init__(tokenizer_id, display_name)
        self._model = model
        try:
            import tiktoken  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "tiktoken is not installed. Install with: pip install tiktoken"
            ) from e
        self._enc = tiktoken.encoding_for_model(model)

    def tokenize(self, text: str) -> List[str]:
        # tiktoken exposes only IDs; we approximate "tokens" as the decoded pieces
        token_ids = self.encode(text)
        return [self._enc.decode([tid]) for tid in token_ids]

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load tokenizers/registry.yaml and return a dict mapping
    tokenizer_id -> config dict.

    Example registry.yaml:

    tokenizers:
      - id: indicbert
        type: hf
        model_name: "ai4bharat/indic-bert"
        display_name: "AI4Bharat IndicBERT"

      - id: generic_bert
        type: hf
        model_name: "bert-base-multilingual-cased"
        display_name: "mBERT"

      - id: openai_gpt4o
        type: openai
        model: "gpt-4.1-mini"
        display_name: "OpenAI GPT-4.1-mini"
    """

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tokenizers_cfg = data.get("tokenizers", [])
    registry: Dict[str, Dict[str, Any]] = {}

    for cfg in tokenizers_cfg:
        tid = cfg.get("id")
        if not tid:
            continue
        registry[tid] = cfg

    if not registry:
        raise ValueError(f"No tokenizers defined in registry: {registry_path}")

    return registry

def create_tokenizer_from_config(cfg: Dict[str, Any]) -> BaseTokenizer:
    ttype = cfg.get("type")
    tid = cfg.get("id")
    display_name = cfg.get("display_name") or tid

    if ttype == "hf":
        model_name = cfg["model_name"]
        return HFTokenizer(
            tokenizer_id=tid,
            model_name=model_name,
            display_name=display_name,
        )
    elif ttype == "openai":
        model = cfg["model"]
        return OpenAITokenizer(
            tokenizer_id=tid,
            model=model,
            display_name=display_name,
        )
    elif ttype == "custom_gpe":
        if GPETokenizer is None:
            raise RuntimeError(
                "GPETokenizer is not available. "
                "Make sure tokenizers.gpe_tokenizer can be imported."
            )
        model_path = cfg["model_path"]
        return GPETokenizer(
            tokenizer_id=tid,
            model_path=model_path,
            display_name=display_name,
        )
    elif ttype == "sentencepiece":
        if SentencePieceTokenizer is None:
            raise RuntimeError(
                "SentencePieceTokenizer is not available. "
                "Make sure tokenizers.sentencepiece_tokenizer can be imported."
            )
        model_path = cfg["model_path"]
        return SentencePieceTokenizer(
            tokenizer_id=tid,
            model_path=model_path,
            display_name=display_name,
        )
    else:
        raise ValueError(f"Unknown tokenizer type '{ttype}' for id '{tid}'")

# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    tokenizer_id: str
    name: str
    tokens: List[str]
    num_tokens: int
    chars: int
    chars_per_token: float

def compare_text(
    text: str,
    tokenizer_ids: List[str],
    registry: Dict[str, Dict[str, Any]],
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []

    for tid in tokenizer_ids:
        if tid not in registry:
            raise KeyError(f"Tokenizer id '{tid}' not found in registry")

        cfg = registry[tid]
        tokenizer = create_tokenizer_from_config(cfg)
        stats = tokenizer.stats(text)

        rows.append(
            ComparisonRow(
                tokenizer_id=tid,
                name=tokenizer.display_name,
                tokens=stats.tokens,
                num_tokens=stats.num_tokens,
                chars=stats.chars,
                chars_per_token=stats.chars_per_token,
            )
        )

    return rows

def print_table(text: str, rows: List[ComparisonRow]) -> None:
    print("=== Indic Tokenization Lab — Comparison ===")
    print()
    print("Input text:")
    print(text)
    print()

    header = f"{'Tokenizer':30} {'Tokens':>8} {'Chars':>8} {'Chars/tok':>10}"
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row.name[:28]:30} "
            f"{row.num_tokens:8d} "
            f"{row.chars:8d} "
            f"{row.chars_per_token:10.2f}"
        )

    print()
    print("Tokens (per tokenizer):")
    for row in rows:
        print()
        print(f"[{row.name}] ({row.num_tokens} tokens)")
        print(" ".join(f"[{t}]" for t in row.tokens))

def rows_to_json(text: str, rows: List[ComparisonRow]) -> Dict[str, Any]:
    return {
        "text": text,
        "results": [
            {
                "tokenizerId": row.tokenizer_id,
                "tokenizerName": row.name,
                "tokens": row.tokens,
                "stats": {
                    "numTokens": row.num_tokens,
                    "chars": row.chars,
                    "charsPerToken": row.chars_per_token,
                },
            }
            for row in rows
        ],
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare how different tokenizers segment Indic text."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Text to tokenize. If omitted, use --file or stdin.",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to a UTF-8 text file whose content will be tokenized.",
    )

    parser.add_argument(
        "--registry",
        type=str,
        default="tokenizers/registry.yaml",
        help="Path to tokenizer registry YAML (default: tokenizers/registry.yaml).",
    )

    parser.add_argument(
        "--tokenizers",
        type=str,
        help=(
            "Comma-separated list of tokenizer ids to use. "
            "If omitted, all tokenizers in the registry will be used."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of a human-readable table.",
    )

    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Determine input text
    text: Optional[str] = None
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        # Read from stdin if available
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()

    if not text:
        print("Error: no input text provided. Use --text, --file, or pipe stdin.", file=sys.stderr)
        sys.exit(1)

    registry_path = Path(args.registry)
    registry = load_registry(registry_path)

    if args.tokenizers:
        tokenizer_ids = [tid.strip() for tid in args.tokenizers.split(",") if tid.strip()]
    else:
        tokenizer_ids = list(registry.keys())

    rows = compare_text(text, tokenizer_ids, registry)

    if args.json:
        payload = rows_to_json(text, rows)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_table(text, rows)

if __name__ == "__main__":
    main()

