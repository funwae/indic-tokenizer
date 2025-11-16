#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_sentencepiece_baseline.py

Train a SentencePiece BPE/Unigram baseline tokenizer for Hindi/Sanskrit.

This provides an alternative to GPE for comparison purposes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import sentencepiece as spm

from tokenizers.pretokenizer import pretokenize


def train_sentencepiece_model(
    input_path: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    pretokenize_input: bool = True,
    normalization_rule_name: str = "nmt_nfkc_cf",
    lang: str = "hi",
) -> None:
    """
    Train a SentencePiece model.

    Parameters
    ----------
    input_path : Path
        Path to input corpus file.
    output_dir : Path
        Directory to save model files.
    vocab_size : int
        Vocabulary size (default: 32000).
    model_type : str
        Model type: "bpe" or "unigram" (default: "bpe").
    pretokenize_input : bool
        Whether to apply pretokenization before training (default: True).
    normalization_rule_name : str
        SentencePiece normalization rule (default: "nmt_nfkc_cf").
        Use "identity" to disable normalization.
    lang : str
        Language code (default: "hi").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # If pretokenizing, create a temporary pretokenized file
    if pretokenize_input:
        print("[SP] Pretokenizing input corpus...")
        pretokenized_path = output_dir / "pretokenized_corpus.txt"
        with input_path.open("r", encoding="utf-8") as infile, pretokenized_path.open(
            "w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                tokens = pretokenize(line, lang=lang, normalize="NFC")
                # Write pretokenized line (space-separated tokens)
                outfile.write(" ".join(tokens) + "\n")
        training_input = str(pretokenized_path)
    else:
        training_input = str(input_path)

    # SentencePiece training parameters
    model_prefix = str(output_dir / "sp_model")
    normalization_rule = (
        "identity" if pretokenize_input else normalization_rule_name
    )  # Use identity if we already pretokenized

    print(f"[SP] Training SentencePiece {model_type} model...")
    print(f"     Input: {training_input}")
    print(f"     Vocab size: {vocab_size}")
    print(f"     Model type: {model_type}")

    spm.SentencePieceTrainer.train(
        input=training_input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        normalization_rule_name=normalization_rule,
        character_coverage=0.9995,
        byte_fallback=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        remove_extra_whitespaces=False,
    )

    # Save config
    config = {
        "type": "sentencepiece_baseline",
        "lang": lang,
        "script": "Deva",
        "vocab_size": vocab_size,
        "model_type": model_type,
        "pretokenized": pretokenize_input,
        "normalization_rule": normalization_rule,
        "model_file": "sp_model.model",
        "vocab_file": "sp_model.vocab",
        "description": (
            f"SentencePiece {model_type} baseline trained with "
            f"{'pretokenization' if pretokenize_input else 'raw text'}"
        ),
    }

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[SP] Model saved to {output_dir}")
    print(f"     Model file: {model_prefix}.model")
    print(f"     Vocab file: {model_prefix}.vocab")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece baseline tokenizer for Hindi/Sanskrit."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input corpus text file (UTF-8).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save model files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bpe", "unigram"],
        default="bpe",
        help="Model type: bpe or unigram (default: bpe).",
    )
    parser.add_argument(
        "--pretokenize",
        action="store_true",
        default=True,
        help="Apply pretokenization before training (default: True).",
    )
    parser.add_argument(
        "--no-pretokenize",
        dest="pretokenize",
        action="store_false",
        help="Do not apply pretokenization.",
    )
    parser.add_argument(
        "--normalization-rule",
        type=str,
        default="nmt_nfkc_cf",
        help="SentencePiece normalization rule (default: nmt_nfkc_cf). Use 'identity' to disable.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Language code (default: hi).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    train_sentencepiece_model(
        input_path=input_path,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        pretokenize_input=args.pretokenize,
        normalization_rule_name=args.normalization_rule,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()

