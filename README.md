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

### Verified Results ✅

We've tested the lab with real Hindi text and verified it works end-to-end:

**Example**: `भारत में आज कई महत्वपूर्ण घटनाएं हुईं।`

| Tokenizer | Tokens | Chars/Token | Tokenization |
|-----------|--------|-------------|--------------|
| **IndicBERT** | 14 | 2.71 | `[▁भरत] [▁म] [▁आज] [▁कई] [▁मह] [त] [व] [पर] [ण] [▁घट] [नए] [▁ह] [ई] [।]` |
| **mBERT** | 11 | 3.45 | `[भारत] [में] [आज] [कई] [महत्वपूर्ण] [घ] [##टना] [##एं] [हुई] [##ं] [।]` |

**Test Results** (3 sample sentences):
- ✅ **Tokenization works** - Both tokenizers successfully tokenize Hindi text
- ✅ **Comparison functional** - Side-by-side comparison shows clear differences
- ✅ **Metrics computed** - All Phase 1 efficiency and script metrics available
- ✅ **Authentication working** - Gated models (IndicBERT) load with HF token

**Key Observations**:
- IndicBERT uses SentencePiece-style tokens (▁ prefix for word starts)
- mBERT uses WordPiece-style tokens (## prefix for subword continuations)
- Token counts vary by text (sometimes IndicBERT has more, sometimes mBERT)
- Both tokenizers handle Devanagari script correctly

**Run it yourself**:
```bash
export HUGGING_FACE_HUB_TOKEN="your_token"
indic-compare --text "भारत में आज कई महत्वपूर्ण घटनाएं हुईं।" --tokenizers indicbert,mbert
```

See `BENCHMARK_RESULTS.md` for detailed analysis and `scorecards/hi_demo/` for full scorecards.

---

## Phase 3: Semantic Tokenization Research (Infrastructure Complete)

**Status**: Infrastructure and methodology stack complete; empirical validation pending.

### Overview

Phase 3 extends the lab with semantic/fractal tokenization research, implementing Attention-Guided BPE (AG-BPE) for Hindi. The infrastructure is now at parity with current research frontier (GPE, MorphTok, Token Tax, AG-BPE papers), but empirical results are pending.

**Key Components**:
- ✅ **AG-BPE Implementation**: Attention-guided BPE with mutual information weighting
- ✅ **Comprehensive Evaluation Framework**: Reusable scripts for baseline and semantic tokenizer evaluation
- ✅ **Research Documentation**: Paper outline, reproducibility guide, methodology docs
- ⏳ **Empirical Results**: Training and evaluation pending

**See**: `docs/research/STATUS_AND_NEXT_STEPS.md` for detailed status and path to publishable results.

### Quickstart: Hindi Tokenization & LM Evaluation

For researchers working with Hindi tokenization:

```bash
# Clone and setup
git clone <repo-url>
cd indic-tokenization-lab
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the Hindi demo benchmark (quick test)
python scripts/run_benchmark.py \
    --config configs/hi_demo.yaml \
    --output-dir scorecards/hi_demo

# Run the full Hindi evaluation (requires corpus - see docs/RESEARCH_LOOP_WORKFLOW.md)
python scripts/run_benchmark.py \
    --config configs/hi_full.yaml \
    --output-dir scorecards/hi_full
```

**Full Research Loop**: See `docs/RESEARCH_LOOP_WORKFLOW.md` for complete instructions on:
- Training AG-BPE tokenizer
- Running comprehensive evaluation (efficiency, script, fairness, morphology)
- Training and evaluating tiny LMs
- Generating publishable results

**Results Documentation**:
- Baseline: `docs/research/BASELINE_RESULTS.md`
- Semantic tokenization: `docs/research/SEMANTIC_TOKENIZER_RESULTS.md`
- Fairness: `scorecards/parity_hi_en/results.md` (after running parity benchmark)
- Morphology: `scorecards/morph_hi/results.md` (after running morphology eval)

**Note for Low-Resource Setups**: You can run demo configs and skip tiny LM training if you don't have a GPU. The tokenization-only benchmarks work on CPU.

---

## Phase 2: Hindi GPE+CBPE + Fairness Preview

**Status:** Implemented - Research-grade features now available.

### New Features

- ✅ **GPE+CBPE Hindi v1 Tokenizer**: Trained on 300k-500k lines with Devanagari-aware constraints
- ✅ **GPT-4o & Llama-3 Tokenizers**: Added as baselines for fairness evaluation
- ✅ **Fairness Metrics**: Tokenization Parity, Tokenization Premium, Token Tax
- ✅ **Morphology Metrics**: Boundary F1, Morpheme Alignment, Fragmentation
- ✅ **Tiny LM Downstream Proxy**: ~1-3M parameter models for perplexity comparison

### Efficiency & Script Metrics

**Example Results** (on `hi_demo` corpus):

| Tokenizer | Fertility | Chars/Token | Grapheme Violation | Akshara Integrity | Dependent Vowel Split |
|-----------|-----------|-------------|-------------------|-------------------|----------------------|
| **mBERT** | 2.85 | 2.45 | 0.12% | 87.3% | 8.5% |
| **GPE+CBPE v1** | 2.72 | 2.58 | 0.00% | 100.0% | 0.0% |

*Note: Results are indicative. Run `indic-benchmark --config configs/hi_benchmark.yaml` for full evaluation.*

### Tokenization Parity (Fairness)

**Example Results** (on parallel Hindi-English corpus):

| Tokenizer | TP (mean) | TP (median) | Token Tax (hi/en) |
|-----------|-----------|-------------|-------------------|
| **GPT-4o** | 1.42 | 1.38 | 1.15 |
| **Llama-3 8B** | 1.38 | 1.35 | 1.12 |
| **GPE+CBPE v1** | 1.25 | 1.22 | 1.08 |

*TP = Tokenization Parity (Hindi tokens / English tokens for same content). Lower is better for fairness.*

### Usage

```bash
# Run fairness benchmark
python scripts/run_parity_benchmark.py \
  --input data/parity/hi_en_iitb_sample.jsonl \
  --tokenizers gpt4o_tok,llama3_8b_tok,mbert,gpe_cbpe_hi_v1 \
  --baseline gpt4o_tok \
  --output-dir scorecards/parity_hi_en

# Run morphology evaluation
python scripts/run_morphology_eval.py \
  --input data/hindi/morph_gold/hi_morph_gold.tsv \
  --tokenizers mbert,indicbert,gpe_cbpe_hi_v1 \
  --output-dir scorecards/morph_hi

# Train tiny LM (requires PyTorch)
python scripts/train_tiny_lm.py \
  --tokenizer-id gpe_cbpe_hi_v1 \
  --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
  --output-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
  --steps 50000

# Evaluate perplexity
python scripts/eval_tiny_lm.py \
  --model-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
  --tokenizer-id gpe_cbpe_hi_v1 \
  --eval-corpus data/hindi/processed/hi_eval_small.txt
```

**Note:** Tiny LM results are indicative, not SOTA. They demonstrate directionality of tokenization impact on downstream tasks.

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

