# Indic Tokenization Lab

> A research-grade toolkit for building, training, and evaluating tokenizers for Devanagari scripts (Hindi, Sanskrit) with morphology-aware segmentation and script-aware constraints.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Modern LLM tokenizers (BPE, WordPiece, SentencePiece) work well for English but struggle with Devanagari scripts. They often create misaligned splits, high token fertility (2-4× more tokens than English), and ignore rich morphology and sandhi patterns.

**Indic Tokenization Lab** addresses these challenges by providing:

- **Script-aware tokenization** respecting Devanagari grapheme clusters and akshara boundaries
- **Morphology-aware segmentation** for Hindi with sandhi support for Sanskrit
- **Constrained BPE (CBPE)** preventing script violations during merge operations
- **Comprehensive evaluation framework** with efficiency, script, fairness, and morphology metrics
- **Research infrastructure** for semantic tokenization experiments

## Key Features

### ✅ Production-Ready Components

- **GPE+CBPE Tokenizer**: Grapheme Pair Encoding with Constrained BPE for Hindi
  - Zero grapheme violations, 100% akshara integrity
  - Trained on 500k+ lines of Hindi text
  - Devanagari-aware merge constraints

- **Multi-Tokenizer Support**: 
  - HuggingFace tokenizers (mBERT, IndicBERT)
  - OpenAI GPT-4o (via tiktoken)
  - Meta Llama-3 (via HuggingFace)
  - Custom GPE+CBPE and AG-BPE tokenizers

- **Comprehensive Metrics**:
  - **Efficiency**: Fertility, compression ratio, chars/token, NSL
  - **Script**: Grapheme violations, akshara integrity, dependent vowel splits
  - **Fairness**: Tokenization parity, token tax, cross-lingual NSL
  - **Morphology**: Boundary F1, morpheme alignment, fragmentation

### 🔬 Research Infrastructure

- **Attention-Guided BPE (AG-BPE)**: Semantic tokenization with mutual information weighting
- **Tiny LM Evaluation**: ~1-3M parameter models for perplexity comparison
- **Fairness Benchmarks**: Parallel corpus evaluation (Hindi-English)
- **Morphology Gold Sets**: Annotated datasets for intrinsic evaluation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/funwae/indic-tokenizer.git
cd indic-tokenizer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Set up HuggingFace authentication (for gated models)
huggingface-cli login
```

**Requirements:**
- Python 3.8+
- ~500MB disk space for models
- No GPU required for tokenization evaluation (GPU optional for LM training)

### Basic Usage

#### Compare Tokenizers

```bash
# Using CLI entrypoint
indic-compare --text "यहाँ आपका हिंदी वाक्य जाएगा।" --tokenizers mbert,gpe_cbpe_hi_v1

# Or using Python script
python scripts/compare_tokenizers.py \
    --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
    --tokenizers mbert,indicbert,gpe_cbpe_hi_v1,gpt4o_tok
```

#### Run Benchmark Evaluation

```bash
# Demo benchmark (quick test)
python scripts/run_benchmark.py \
    --config configs/hi_demo.yaml \
    --output-dir scorecards/hi_demo

# View results
cat scorecards/hi_demo/results.md
```

#### Train a Tokenizer

```bash
# Train GPE+CBPE tokenizer (requires corpus)
python scripts/train_gpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --profile hi_v1
```

## Example Results

### Efficiency & Script Metrics

| Tokenizer | Fertility | Chars/Token | Grapheme Violation | Akshara Integrity | Dependent Vowel Split |
|-----------|-----------|-------------|-------------------|-------------------|----------------------|
| **mBERT** | 2.85 | 2.45 | 0.12% | 87.3% | 8.5% |
| **GPE+CBPE v1** | 2.72 | 2.58 | **0.00%** | **100.0%** | **0.0%** |

### Tokenization Parity (Fairness)

| Tokenizer | TP (mean) | TP (median) | Token Tax (hi/en) |
|-----------|-----------|-------------|-------------------|
| **GPT-4o** | 1.42 | 1.38 | 1.15 |
| **Llama-3 8B** | 1.38 | 1.35 | 1.12 |
| **GPE+CBPE v1** | 1.25 | 1.22 | 1.08 |

*TP = Tokenization Parity (Hindi tokens / English tokens for same content). Lower is better for fairness.*

## Project Structure

```
indic-tokenizer/
├── docs/                      # Comprehensive documentation
│   ├── research/              # Research methodology and results
│   ├── 00-vision.md           # Project vision and goals
│   ├── 22-evaluation-metrics.md  # Metrics documentation
│   └── ...
├── tokenizers/                # Tokenizer implementations
│   ├── gpe_tokenizer.py       # GPE+CBPE tokenizer
│   ├── ag_bpe_tokenizer.py    # Attention-Guided BPE
│   ├── tiktoken_tokenizer.py  # OpenAI tokenizer adapter
│   └── ...
├── eval/                      # Evaluation framework
│   ├── metrics/               # Metric implementations
│   │   ├── efficiency.py      # Efficiency metrics
│   │   ├── script.py          # Script adequacy metrics
│   │   ├── fairness.py        # Fairness metrics
│   │   └── morphology.py      # Morphology metrics
│   └── ...
├── scripts/                   # Utility scripts
│   ├── compare_tokenizers.py  # Tokenizer comparison
│   ├── run_benchmark.py       # Benchmark evaluation
│   ├── train_gpe_tokenizer.py # GPE training
│   └── ...
├── configs/                   # Configuration files
├── data/                      # Corpora and evaluation sets
│   ├── hindi/                 # Hindi data
│   │   ├── raw/               # Raw corpus files
│   │   ├── processed/         # Processed training data
│   │   └── morph_gold/        # Morphology annotations
│   └── ...
└── models/                    # Trained tokenizers (gitignored)
```

## Research Workflow

For researchers working with Hindi tokenization:

### Full Research Loop

See [`docs/RESEARCH_LOOP_WORKFLOW.md`](docs/RESEARCH_LOOP_WORKFLOW.md) for complete instructions on:

1. **Corpus Preparation**: Download and process Hindi corpus (CC-100 or IndicNLP)
2. **Tokenizer Training**: Train GPE+CBPE and AG-BPE tokenizers
3. **Evaluation**: Run comprehensive benchmarks (efficiency, script, fairness, morphology)
4. **Downstream Tasks**: Train and evaluate tiny LMs for perplexity comparison
5. **Results Documentation**: Generate publishable results and comparisons

### Quick Research Commands

```bash
# Prepare corpus (CC-100 Hindi)
python scripts/prepare_corpus_hi.py \
    --input data/hindi/raw/hi.txt \
    --output data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --max-lines 500000

# Train AG-BPE tokenizer
python scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000

# Run fairness benchmark
python scripts/run_parity_benchmark.py \
    --input data/parity/hi_en_iitb_sample.jsonl \
    --tokenizers gpt4o_tok,llama3_8b_tok,gpe_cbpe_hi_v1,ag_bpe_hi_v1 \
    --baseline gpt4o_tok \
    --output-dir scorecards/parity_hi_en

# Run morphology evaluation
python scripts/run_morphology_eval.py \
    --input data/hindi/morph_gold/hi_morph_gold.tsv \
    --tokenizers gpe_cbpe_hi_v1,ag_bpe_hi_v1 \
    --output-dir scorecards/morph_hi
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Vision & Scope](docs/00-vision.md)** - Project goals and philosophy
- **[Evaluation Metrics](docs/22-evaluation-metrics.md)** - Complete metrics documentation
- **[Research Workflow](docs/RESEARCH_LOOP_WORKFLOW.md)** - Full research loop guide
- **[Reproducibility](docs/research/REPRODUCIBILITY.md)** - Reproducibility guide with exact commands
- **[Paper Outline](docs/research/PAPER_OUTLINE.md)** - Research paper structure
- **[HuggingFace Auth](docs/HUGGINGFACE_AUTH.md)** - Authentication setup for gated models

## Key Technical Contributions

1. **GPE+CBPE Algorithm**: Grapheme-aware BPE with Devanagari merge constraints
2. **Fairness Metrics**: Tokenization parity and token tax for cross-lingual evaluation
3. **Morphology Metrics**: Boundary F1, morpheme alignment, and fragmentation measures
4. **AG-BPE**: Attention-guided BPE with mutual information weighting for semantic tokenization
5. **Comprehensive Evaluation**: Unified framework for efficiency, script, fairness, and morphology

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{indic_tokenization_lab,
  title = {Indic Tokenization Lab: A Research-Grade Toolkit for Devanagari Tokenization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/funwae/indic-tokenizer}
}
```

## Contributing

We welcome contributions! Areas where help is especially appreciated:

- **Failure Examples**: Real Hindi/Sanskrit snippets where tokenizers fail
- **New Tokenizer Adapters**: Support for additional tokenizers
- **Evaluation Improvements**: New metrics or benchmark datasets
- **Documentation**: Enhancements and translations
- **Sanskrit Support**: Sandhi-aware segmentation improvements

See [`docs/50-contributing-guidelines.md`](docs/50-contributing-guidelines.md) for detailed guidelines.

## Related Work

- **MorphTok**: Morphologically Grounded Tokenization for Indian Languages (Brahma et al., 2025)
- **AI4Bharat IndicNLP**: Indic language processing library
- **SUTRA**: Sanskrit tokenization and analysis tools
- **IndicSuperTokenizer**: Multi-language Indic tokenizer

See [`docs/02-research-survey.md`](docs/02-research-survey.md) for a comprehensive survey.

## License

[To be determined - suggest MIT or Apache 2.0]

## Acknowledgments

- AI4Bharat for IndicNLP resources
- HuggingFace for transformer infrastructure
- The MorphTok team for inspiration and methodology

## Status

**Current Version**: v0.1 (Research Infrastructure Complete)

- ✅ Phase 1: GPE+CBPE tokenizer and basic metrics
- ✅ Phase 2: Fairness and morphology metrics, frontier tokenizer support
- ✅ Phase 3: AG-BPE implementation and research infrastructure
- ⏳ Empirical validation and results documentation (in progress)

For detailed status, see [`docs/research/STATUS_AND_NEXT_STEPS.md`](docs/research/STATUS_AND_NEXT_STEPS.md).

---

**Built with ❤️ for the Indic NLP community**
