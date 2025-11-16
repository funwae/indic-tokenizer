# Naming Conflict Workaround

## Issue

There is a naming conflict between our local `tokenizers/` package and HuggingFace's `tokenizers` library (a dependency of `transformers`). When `transformers` is imported, it tries to import `tokenizers` and may find our local package instead, causing import errors.

## Current Status

The GPE tokenizer can be loaded and used directly via Python imports, but the `compare_tokenizers.py` script has import issues when both transformers and our tokenizers package are needed.

## Workaround

For now, use the tokenizer directly in Python scripts:

```python
import sys
sys.path.insert(0, '/path/to/indic-tokenizer')

from tokenizers.gpe_tokenizer import GPETokenizer

tokenizer = GPETokenizer('gpe_cbpe_hi_v1', 'models/gpe_cbpe_hi_v1')
tokens = tokenizer.tokenize('भारत में आज कई महत्वपूर्ण घटनाएं हुईं।')
```

## Long-term Solution

Consider renaming the `tokenizers/` package to `indic_tokenizers/` or `lab_tokenizers/` to avoid the conflict entirely.

