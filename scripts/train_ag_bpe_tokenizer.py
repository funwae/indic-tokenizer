#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Attention-Guided BPE (AG-BPE) tokenizer.

Usage:
  python scripts/train_ag_bpe_tokenizer.py \
      --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
      --output-dir models/ag_bpe_hi_v1 \
      --vocab-size 32000 \
      --attention-weight 0.5 \
      --mi-weight 0.3 \
      --frequency-weight 0.2
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizers.ag_bpe_trainer import train_ag_bpe_tokenizer, main

if __name__ == "__main__":
    sys.exit(main())

