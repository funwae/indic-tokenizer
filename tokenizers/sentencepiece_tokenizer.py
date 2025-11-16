# tokenizers/sentencepiece_tokenizer.py
# -*- coding: utf-8 -*-
"""
SentencePiece tokenizer adapter for Indic Tokenization Lab.

Wraps SentencePiece models for use in the comparison system.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import sentencepiece as spm


class SentencePieceTokenizer:
    """
    SentencePiece tokenizer adapter.

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
        Initialize SentencePiece tokenizer from a model directory.

        Parameters
        ----------
        tokenizer_id : str
            Unique identifier for this tokenizer.
        model_path : str or Path
            Path to directory containing sp_model.model and config.json.
        display_name : str, optional
            Human-readable name. Defaults to tokenizer_id.
        """
        self.tokenizer_id = tokenizer_id
        self.display_name = display_name or tokenizer_id
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model directory not found: {self.model_path}"
            )

        # Load config
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Find model file
        model_file = self.model_path / "sp_model.model"
        if not model_file.exists():
            # Try alternative name
            model_file = self.model_path / (self.config.get("model_file", "sp_model.model"))
            if not model_file.exists():
                raise FileNotFoundError(
                    f"SentencePiece model file not found in: {self.model_path}"
                )

        # Load SentencePiece model
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_file))

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
        return self._sp.encode(text, out_type=str)

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
        return self._sp.encode(text, out_type=int)

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
        return self._sp.decode(ids)

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

        return SimpleNamespace(
            tokens=tokens,
            num_tokens=num_tokens,
            chars=chars,
            chars_per_token=chars_per_token,
        )

