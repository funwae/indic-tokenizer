# tokenizers/ag_bpe_tokenizer.py
# -*- coding: utf-8 -*-
"""
Attention-Guided BPE (AG-BPE) Tokenizer adapter.

Loads a trained AG-BPE tokenizer and implements the BaseTokenizer interface.
AG-BPE tokenizers use the same format as GPE tokenizers (vocab.json, merges.txt).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from tokenizers.gpe_tokenizer import GPETokenizer


class AGBPETokenizer(GPETokenizer):
    """
    Attention-Guided BPE tokenizer adapter.

    Inherits from GPETokenizer since AG-BPE uses the same format
    (vocab.json, merges.txt, config.json). The difference is in
    how the merges were selected (attention-guided vs frequency-only).
    """

    def __init__(
        self,
        tokenizer_id: str,
        model_path: str | Path,
        display_name: Optional[str] = None,
    ):
        """
        Initialize AG-BPE tokenizer from a model directory.

        Parameters
        ----------
        tokenizer_id : str
            Unique identifier for this tokenizer.
        model_path : str or Path
            Path to directory containing vocab.json, merges.txt, config.json.
        display_name : str, optional
            Human-readable name. Defaults to tokenizer_id.
        """
        super().__init__(tokenizer_id, model_path, display_name)

        # Load AG-BPE specific config if available
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                self.attention_weight = config.get("attention_weight", 0.0)
                self.mi_weight = config.get("mi_weight", 0.0)
                self.frequency_weight = config.get("frequency_weight", 1.0)

