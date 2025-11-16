# Fixing the Tokenizers Naming Conflict

## Problem

Our local `tokenizers/` package directory conflicts with HuggingFace's `tokenizers` library. When Python tries to import `tokenizers`, it finds our local package instead of the HuggingFace library.

## Solution Options

### Option 1: Rename Our Package (Recommended for Long-term)

Rename `tokenizers/` to `indic_tokenizers/` or `tokenizer_impl/`:

```bash
mv tokenizers indic_tokenizers
# Update all imports in the codebase
find . -name "*.py" -exec sed -i 's/from tokenizers\./from indic_tokenizers./g' {} \;
find . -name "*.py" -exec sed -i 's/import tokenizers\./import indic_tokenizers./g' {} \;
```

### Option 2: Use Absolute Imports with sys.path Manipulation

Modify scripts to import transformers before adding our package to path:

```python
# Import transformers first (before our tokenizers package)
from transformers import AutoTokenizer

# Then add our package to path
import sys
sys.path.insert(0, 'path/to/project')
from tokenizers.gpe_tokenizer import GPETokenizer
```

### Option 3: Install as Package with Different Name

Update `pyproject.toml` to use a different package name:

```toml
[project]
name = "indic-tokenization-lab"

[tool.setuptools]
packages = ["indic_tokenizers", "eval", "scripts"]
```

## Current Workaround

For now, when testing HuggingFace models, run from a directory where our `tokenizers/` package isn't in the Python path, or use the test script from `/tmp`.

## Testing Authentication

The authentication itself works! You can verify with:

```bash
export HUGGING_FACE_HUB_TOKEN="your_token"
cd /tmp  # Avoid naming conflict
python3 -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', token='your_token'); print('✓ Works!')"
```

## Status

- ✅ Authentication is working (token is valid)
- ✅ User is logged in as "bohselecta"  
- ⚠️ Naming conflict prevents direct use in our scripts
- 🔧 Need to implement one of the solutions above

