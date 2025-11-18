# tokenizers/tiktoken_tokenizer.py
# -*- coding: utf-8 -*-
"""
Tiktoken tokenizer adapter for Indic Tokenization Lab.

Supports OpenAI GPT-4o and other tiktoken-based tokenizers.
"""

from __future__ import annotations

from typing import List, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore


class TiktokenTokenizer:
    """
    Tiktoken tokenizer adapter for OpenAI models.

    Implements the BaseTokenizer interface for compatibility with
    compare_tokenizers.py.
    """

    def __init__(
        self,
        tokenizer_id: str,
        encoding_name: str = "o200k_base",
        display_name: Optional[str] = None,
    ):
        """
        Initialize Tiktoken tokenizer.

        Parameters
        ----------
        tokenizer_id : str
            Unique identifier for this tokenizer.
        encoding_name : str
            Tiktoken encoding name (default: "o200k_base" for GPT-4o).
        display_name : str, optional
            Human-readable name. Defaults to tokenizer_id.
        """
        if tiktoken is None:
            raise RuntimeError(
                "tiktoken library is not installed. "
                "Install with: pip install tiktoken"
            )

        self.tokenizer_id = tokenizer_id
        self.display_name = display_name or tokenizer_id
        self.encoding_name = encoding_name

        # Get encoding
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tiktoken encoding '{encoding_name}': {e}"
            ) from e

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
        # Tiktoken exposes only IDs; we decode each token ID to get the string representation
        token_ids = self.encode(text)
        tokens: List[str] = []
        for tid in token_ids:
            try:
                # Decode single token
                token_bytes = self._enc.decode_single_token_bytes(tid)
                token_str = token_bytes.decode('utf-8', errors='replace')
                tokens.append(token_str)
            except Exception:
                # Fallback: use the token ID as string representation
                tokens.append(f"<token_{tid}>")
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
        return self._enc.encode(text)

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
        return self._enc.decode(ids)

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
        from types import SimpleNamespace
        return SimpleNamespace(
            tokens=tokens,
            num_tokens=num_tokens,
            chars=chars,
            chars_per_token=chars_per_token,
        )

