# tokenizers/llama3_tokenizer.py
# -*- coding: utf-8 -*-
"""
Llama-3 tokenizer adapter for Indic Tokenization Lab.

Loads Llama-3 tokenizers via HuggingFace transformers and implements
the BaseTokenizer interface for use with compare_tokenizers.py.
"""

from __future__ import annotations

from typing import List, Optional

# Import transformers BEFORE our tokenizers package to avoid naming conflict
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore


class Llama3Tokenizer:
    """
    Llama-3 tokenizer adapter using HuggingFace transformers.

    Implements the BaseTokenizer interface for compatibility with
    compare_tokenizers.py.
    """

    def __init__(
        self,
        tokenizer_id: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        display_name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize Llama-3 tokenizer from HuggingFace model.

        Parameters
        ----------
        tokenizer_id : str
            Unique identifier for this tokenizer.
        model_name : str
            HuggingFace model name (e.g., "meta-llama/Meta-Llama-3-8B").
        display_name : str, optional
            Human-readable name. Defaults to tokenizer_id.
        token : str, optional
            HuggingFace authentication token for gated models.
        """
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers library is not installed. "
                "Install with: pip install transformers"
            )

        self.tokenizer_id = tokenizer_id
        self.display_name = display_name or tokenizer_id
        self.model_name = model_name

        # Get token from environment if not provided
        if token is None:
            import os
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

        # Load tokenizer
        try:
            if token:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=token,
                    use_fast=True,
                )
            else:
                # Will use cached token from huggingface-cli login if available
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Llama-3 tokenizer {model_name}: {e}. "
                "Make sure you have access to the model and are authenticated."
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
        return self._tokenizer.tokenize(text)

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
        return self._tokenizer.encode(text, add_special_tokens=False)

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
        return self._tokenizer.decode(ids, skip_special_tokens=True)

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

