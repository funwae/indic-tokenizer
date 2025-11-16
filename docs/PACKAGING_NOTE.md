# Packaging Note: Naming Conflict

## Issue

There is a naming conflict between our local `tokenizers/` package directory and the HuggingFace `tokenizers` library. This can cause import errors when using HuggingFace tokenizers.

## Workaround

When importing from HuggingFace's `tokenizers` library, use absolute imports:

```python
from tokenizers import Tokenizer as HFTokenizer  # HuggingFace tokenizers
from tokenizers.gpe_tokenizer import GPETokenizer  # Our tokenizers
```

Or use import aliases to avoid conflicts.

## Future Fix

Consider renaming our `tokenizers/` package to `indic_tokenizers/` or `tokenizer_impl/` to avoid this conflict.

