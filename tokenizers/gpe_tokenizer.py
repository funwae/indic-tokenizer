# tokenizers/gpe_tokenizer.py
# -*- coding: utf-8 -*-
"""
GPE Tokenizer adapter for Indic Tokenization Lab.

Loads a trained GPE (Grapheme Pair Encoding) tokenizer and implements
the BaseTokenizer interface for use with compare_tokenizers.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tokenizers.grapheme_segmenter import segment_devanagari_graphemes


class GPETokenizer:
    """
    GPE tokenizer that loads vocab and merges from a trained model directory.

    Implements the BaseTokenizer interface for compatibility with
    compare_tokenizers.py.
    """

    def __init__(
        self,
        tokenizer_id: str,
        model_path: str | Path,
        display_name: Optional[str] = None,
    ):
        """
        Initialize GPE tokenizer from a model directory.

        Parameters
        ----------
        tokenizer_id : str
            Unique identifier for this tokenizer.
        model_path : str or Path
            Path to directory containing vocab.json, merges.txt, config.json.
        display_name : str, optional
            Human-readable name. Defaults to tokenizer_id.
        """
        self.tokenizer_id = tokenizer_id
        self.display_name = display_name or tokenizer_id
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GPE tokenizer model directory not found: {self.model_path}"
            )

        # Load vocab
        vocab_path = self.model_path / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.json not found in model directory: {self.model_path}"
            )
        with vocab_path.open("r", encoding="utf-8") as f:
            self.token_to_id: Dict[str, int] = json.load(f)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        # Load merges
        merges_path = self.model_path / "merges.txt"
        if not merges_path.exists():
            raise FileNotFoundError(
                f"merges.txt not found in model directory: {self.model_path}"
            )
        self.merges: List[Tuple[str, str]] = self._load_merges(merges_path)

        # Load config (optional)
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.word_end_symbol = self.config.get("word_end_symbol", "</w>")
        else:
            self.config = {}
            self.word_end_symbol = "</w>"

        # Get <unk> token ID
        self.unk_id = self.token_to_id.get("<unk>", 0)

    def _load_merges(self, merges_path: Path) -> List[Tuple[str, str]]:
        """
        Load BPE merges from merges.txt file.

        Format: one merge per line as "left right" (skipping version header).
        """
        merges: List[Tuple[str, str]] = []
        with merges_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return merges

    def _apply_bpe(
        self, word_graphemes: List[str], word_end_symbol: str = "</w>"
    ) -> List[str]:
        """
        Apply BPE merges to a sequence of graphemes.

        Parameters
        ----------
        word_graphemes : List[str]
            List of grapheme clusters for a single word.
        word_end_symbol : str
            Word-end sentinel to append.

        Returns
        -------
        List[str]
            List of BPE tokens after applying all merges.
        """
        # Start with graphemes + word-end symbol
        symbols = word_graphemes + [word_end_symbol]

        # Apply each merge in order
        for left, right in self.merges:
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                    # Merge this pair
                    new_symbols.append(left + right)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        List[str]
            List of token strings.
        """
        tokens: List[str] = []
        # Split text into words (whitespace-separated)
        words = text.split()

        for word in words:
            # Segment word into graphemes
            graphemes = segment_devanagari_graphemes(word, keep_non_devanagari=True)
            if not graphemes:
                # Empty word, skip
                continue

            # Apply BPE merges
            word_tokens = self._apply_bpe(graphemes, self.word_end_symbol)
            tokens.extend(word_tokens)

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Parameters
        ----------
        text : str
            Input text to encode.

        Returns
        -------
        List[int]
            List of token IDs.
        """
        tokens = self.tokenize(text)
        ids: List[int] = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_id)
            ids.append(token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to text.

        Parameters
        ----------
        ids : List[int]
            List of token IDs.

        Returns
        -------
        str
            Decoded text.
        """
        tokens: List[str] = []
        for token_id in ids:
            token = self.id_to_token.get(token_id, "<unk>")
            tokens.append(token)

        # Join tokens and remove word-end symbols
        text = "".join(tokens)
        text = text.replace(self.word_end_symbol, " ")
        return text.strip()

    def stats(self, text: str):
        """
        Compute tokenization statistics for text.

        This method matches the interface expected by compare_tokenizers.py.
        Returns an object compatible with TokenizationStats dataclass.
        """
        tokens = self.tokenize(text)
        chars = len(text)
        num_tokens = len(tokens)
        chars_per_token = chars / num_tokens if num_tokens > 0 else 0.0

        # Return a simple object that matches TokenizationStats interface
        # Using a simple class to avoid circular imports
        from types import SimpleNamespace
        return SimpleNamespace(
            tokens=tokens,
            num_tokens=num_tokens,
            chars=chars,
            chars_per_token=chars_per_token,
        )

