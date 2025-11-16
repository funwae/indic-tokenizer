# Quick Start Guide

This guide will get you up and running with the Indic Tokenization Lab quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Clone or Navigate to Project

```bash
cd indic-tokenization-lab
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install:
- `transformers` - For HuggingFace tokenizers
- `tokenizers` - HF tokenizers library
- `sentencepiece` - SentencePiece models
- `pyyaml` - YAML parsing
- `regex` - Unicode regex for grapheme clustering

### 4. Verify Installation

```bash
python3 scripts/verify_setup.py
```

This will test that all components are importable and basic functionality works.

## Basic Usage

### Compare Tokenizers

Compare how different tokenizers segment Hindi text:

```bash
python3 scripts/compare_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert
```

### Evaluate with Metrics

Get comprehensive metrics for tokenizers:

```bash
python3 scripts/evaluate_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert \
  --output scorecards/test.json
```

### Test Grapheme Segmentation

Test the grapheme segmenter:

```bash
python3 -m tokenizers.grapheme_segmenter "किशोरी"
python3 -m tokenizers.grapheme_segmenter "प्रार्थना"
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=tokenizers --cov=eval
```

## Training Tokenizers

### Train a GPE Tokenizer

You'll need a Hindi corpus file first:

```bash
python3 scripts/train_gpe_tokenizer.py \
  --input data/hindi/corpus.txt \
  --output-dir models/gpe_hi_v0 \
  --vocab-size 32000 \
  --min-pair-frequency 2 \
  --max-lines 100000
```

### Train a SentencePiece Baseline

```bash
python3 scripts/train_sentencepiece_baseline.py \
  --input data/hindi/corpus.txt \
  --output-dir models/sp_hi_baseline \
  --vocab-size 32000 \
  --model-type bpe \
  --pretokenize
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

1. Make sure virtual environment is activated
2. Install dependencies: `pip install -r requirements.txt`
3. Run from project root directory

### Tokenizer Download Issues

HF tokenizers will download models on first use. This may take time and require internet connection.

### Missing Corpus

Training scripts require corpus files. You can:
- Use provided eval datasets for testing
- Download Hindi corpora (check licensing)
- Create small test corpus from curated examples

## Next Steps

- Read `docs/PROJECT_SPECS.md` for complete documentation
- Check `examples/` directory for code examples
- See `docs/01-roadmap.md` for development roadmap

## Getting Help

- Check `docs/` for detailed documentation
- Review `examples/` for usage patterns
- See `docs/50-contributing-guidelines.md` for contribution info

