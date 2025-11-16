# Indic Tokenization Lab

**Build the best open tokenizer stack for Devanagari (Hindi + Sanskrit first), with morphology-aware segmentation, script-aware constraints, and a transparent evaluation lab.**

## Overview

This project addresses the tokenization challenges faced by Devanagari scripts (Hindi, Sanskrit) in modern LLM tokenizers. Standard BPE/WordPiece tokenizers often misalign splits, create high fertility (2-4× more tokens than English), and ignore rich morphology and sandhi patterns.

## Production Preview: Indic Tokenization Lab v0.1

**What's included:**
- ✅ GPE+CBPE Hindi tokenizer (Grapheme Pair Encoding with Constrained BPE)
- ✅ Phase 1 comprehensive metrics (efficiency + script adequacy)
- ✅ Demo benchmark + scorecards
- ✅ CLI tools: `indic-compare` and `indic-benchmark`

**Quick Demo:**
```bash
# Install
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Compare tokenizers
indic-compare --text "यहाँ आपका हिंदी वाक्य जाएगा।" --tokenizers mbert,gpe_hi_v0

# Run demo benchmark
indic-benchmark --config configs/hi_demo.yaml --output-dir scorecards/hi_demo

# View results
cat scorecards/hi_demo/results.md
```

See `scorecards/hi_demo/results.md` for example output.

---

## Quick Start

### Installation

**Requirements:**
- Python 3.8+
- ~500MB disk space for models
- No GPU required (CPU-only evaluation)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Set up HuggingFace authentication (for gated models like ai4bharat/indic-bert)
pip install huggingface_hub
python scripts/setup_hf_auth.py
# Or manually: huggingface-cli login
```

**Note:** Some models (like `ai4bharat/indic-bert`) are gated and require authentication. See `docs/HUGGINGFACE_AUTH.md` for setup instructions.

### Basic Usage

#### Compare Tokenizers

```bash
# Using CLI entrypoint (recommended)
indic-compare --text "यहाँ आपका हिंदी वाक्य जाएगा।" --tokenizers mbert,gpe_hi_v0

# Or using Python script directly
python scripts/compare_tokenizers.py \
    --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
    --tokenizers mbert,gpe_hi_v0

# Compare from a file
indic-compare --file data/hindi/demo/news_small.txt --tokenizers mbert,gpe_hi_v0 --json
```

#### Evaluate Tokenizers with Metrics

```bash
# Evaluate on single text with full metrics
python scripts/evaluate_tokenizers.py \
    --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
    --tokenizers indicbert,mbert,gpe_hi_v0 \
    --output scorecard.json

# Evaluate on dataset
python scripts/evaluate_tokenizers.py \
    --dataset data/hindi/eval_sets/news_headlines.txt \
    --tokenizers all \
    --output scorecards/news_evaluation.md \
    --format markdown
```

#### Train a GPE Tokenizer

```bash
# Train a GPE tokenizer (requires corpus)
python scripts/train_gpe_tokenizer.py \
    --input data/hindi/corpus.txt \
    --output-dir models/gpe_hi_v0 \
    --vocab-size 32000 \
    --min-pair-frequency 2
```

#### Run Benchmark Evaluation

```bash
# Using config file (recommended)
indic-benchmark --config configs/hi_demo.yaml --output-dir scorecards/hi_demo

# Using command-line arguments
indic-benchmark \
    --corpus data/hindi/demo/news_small.txt \
    --tokenizers mbert,gpe_hi_v0 \
    --lang hi \
    --baseline-tokenizer mbert \
    --output-dir scorecards/demo

# View results
cat scorecards/hi_demo/results.md
```

### Python API Examples

See the `examples/` directory for Python code examples:

- `basic_comparison.py` - Simple tokenizer comparison
- `evaluate_tokenizer.py` - Evaluation with metrics

## Project Structure

```
indic-tokenization-lab/
├── docs/              # Comprehensive documentation
├── data/              # Corpora and evaluation sets
├── tokenizers/        # Tokenizer implementations
├── eval/              # Evaluation scripts and metrics
├── playground/        # Web UI playground (future)
└── scripts/           # Utility scripts
```

## Documentation

See the `docs/` folder for comprehensive documentation:

- **Quick Start** (`docs/QUICK_START.md`) - Get started quickly
- **Project Specs** (`docs/PROJECT_SPECS.md`) - Complete technical specifications
- **Vision & Scope** (`docs/00-vision.md`) - Project goals and philosophy
- **Roadmap** (`docs/01-roadmap.md`) - Development milestones
- **Linguistics** (`docs/10-*.md`) - Devanagari, Hindi, Sanskrit notes
- **Architecture** (`docs/20-*.md`) - Tokenizer design and training
- **Evaluation** (`docs/22-evaluation-metrics.md`) - Metrics and benchmarks
- **Integration** (`docs/40-*.md`) - HuggingFace and LLM integration

## Key Features

- **Script-aware tokenization** respecting Devanagari grapheme clusters
- **Morphology-aware pre-segmentation** for Hindi
- **Sandhi-aware splitting** for Sanskrit
- **Constrained BPE (CBPE)** preventing script violations
- **Comprehensive evaluation** with intrinsic, human, and downstream metrics
- **Playground UI** for interactive tokenizer comparison (planned)

## Contributing

See `docs/50-contributing-guidelines.md` for how to contribute. We welcome:

- Failure examples (real Hindi/Sanskrit snippets where tokenizers fail)
- New tokenizer adapters
- Evaluation improvements
- Documentation enhancements

## License

[To be determined - suggest MIT or Apache 2.0]

## References

- MorphTok: Morphologically Grounded Tokenization for Indian Languages (Brahma et al., 2025)
- AI4Bharat IndicNLP Library
- Krutrim Tokenizer
- Saṃsādhanī Sanskrit Tools

See `docs/02-research-survey.md` for a comprehensive list of references.

