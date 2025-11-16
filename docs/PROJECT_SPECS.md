# Indic Tokenization Lab - Project Specifications

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Data Structures](#data-structures)
5. [API Reference](#api-reference)
6. [File Structure](#file-structure)
7. [Usage Guide](#usage-guide)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Tokenizer Types](#tokenizer-types)
10. [Training Pipeline](#training-pipeline)
11. [Configuration](#configuration)
12. [Dependencies](#dependencies)

---

## Project Overview

The **Indic Tokenization Lab** is a comprehensive toolkit for building, training, and evaluating tokenizers for Devanagari scripts (Hindi and Sanskrit). It addresses the tokenization challenges faced by Indic languages in modern LLM tokenizers, providing:

- **Grapheme Pair Encoding (GPE)** - BPE over Unicode grapheme clusters
- **Constrained BPE (CBPE)** - Script-aware merge constraints
- **Comprehensive Evaluation** - Metrics for fertility, grapheme violations, and more
- **Multiple Tokenizer Support** - HuggingFace, SentencePiece, GPE, and custom tokenizers
- **Complete Lab Infrastructure** - Training, evaluation, comparison, and scorecard generation

### Key Features

- ✅ Script-aware tokenization respecting Devanagari grapheme clusters
- ✅ Morphology-aware pre-segmentation (foundation for future work)
- ✅ Sandhi-aware splitting support (Sanskrit)
- ✅ Constrained BPE preventing script violations
- ✅ Comprehensive evaluation with intrinsic metrics
- ✅ Multiple tokenizer comparison and benchmarking
- ✅ Scorecard generation for research and reporting

---

## Architecture

### Layered Design

The tokenization system follows a three-layer architecture:

```
┌─────────────────────────────────────────┐
│  L2: Subword Layer (BPE/CBPE/GPE)      │
│  - Constrained BPE merges               │
│  - Grapheme-based BPE (GPE)             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  L1: Morphology Layer (Future)          │
│  - Hindi morphology segmentation        │
│  - Sanskrit sandhi splitting            │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  L0: Script Layer                       │
│  - Unicode grapheme segmentation        │
│  - Script-aware pretokenization         │
│  - Special token preservation           │
└─────────────────────────────────────────┘
```

### Component Interaction

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Training   │────▶│  Tokenizers  │────▶│  Evaluation  │
│   Scripts    │     │   (Registry) │     │   Metrics    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       │                    │                     │
       ▼                    ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Models/    │     │  Comparison  │     │  Scorecards  │
│   Artifacts  │     │    Scripts   │     │   (Output)   │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Components

### 1. Tokenizers

#### 1.1 Grapheme Segmenter (`tokenizers/grapheme_segmenter.py`)

**Purpose:** Split text into Unicode extended grapheme clusters (UAX #29).

**Key Functions:**
- `iter_graphemes(text: str) -> Iterator[str]` - Generator for grapheme clusters
- `segment_devanagari_graphemes(text: str, keep_non_devanagari: bool = True) -> List[str]` - Segment text into graphemes

**Implementation:**
- Uses `regex` library with `\X` pattern for UAX #29 compliance
- Handles Devanagari script boundaries
- CLI support for debugging: `python -m tokenizers.grapheme_segmenter "text"`

#### 1.2 Pretokenizer (`tokenizers/pretokenizer.py`)

**Purpose:** Rule-based pretokenization with script-aware rules.

**Key Functions:**
- `pretokenize(text: str, lang: str = "hi", normalize: str = "NFC") -> List[str]` - Main pretokenization function
- `preserve_special_tokens(text: str) -> List[tuple[str, bool]]` - Identify and preserve URLs, emails, hashtags, mentions, numbers
- `split_words(text: str) -> List[str]` - Whitespace splitting
- `split_punctuation(text: str) -> List[str]` - Punctuation separation
- `split_script_boundaries(text: str) -> List[str]` - Script-aware splitting

**Features:**
- Unicode normalization (NFC/NFKC)
- Special token preservation (URLs, emails, hashtags, mentions, numbers)
- Script-aware boundaries
- Devanagari punctuation handling

#### 1.3 CBPE Constraints (`tokenizers/cbpe_constraints.py`)

**Purpose:** Constrained BPE merge rules for Devanagari.

**Key Functions:**
- `cbpe_merge_allowed(left: str, right: str) -> bool` - Check if merge is allowed
- `filter_bpe_merges(merges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]` - Filter merges by constraints

**Constraints:**
- Prevents tokens starting with dependent vowels (matras)
- Prevents tokens starting with virama (halant)
- Ensures script-aware tokenization

#### 1.4 GPE Tokenizer (`tokenizers/gpe_tokenizer.py`)

**Purpose:** Load and use trained GPE tokenizers.

**Key Methods:**
- `tokenize(text: str) -> List[str]` - Tokenize text
- `encode(text: str) -> List[int]` - Encode to token IDs
- `decode(ids: List[int]) -> str` - Decode from token IDs
- `stats(text: str)` - Compute tokenization statistics

**Model Format:**
- `vocab.json` - Token to ID mapping
- `merges.txt` - BPE merge rules
- `config.json` - Model metadata

#### 1.5 SentencePiece Tokenizer (`tokenizers/sentencepiece_tokenizer.py`)

**Purpose:** Wrapper for SentencePiece models.

**Key Methods:**
- Same interface as GPE Tokenizer
- Uses SentencePiece library for tokenization

**Model Format:**
- `sp_model.model` - SentencePiece model file
- `sp_model.vocab` - Vocabulary file
- `config.json` - Model metadata

### 2. Training Scripts

#### 2.1 GPE Trainer (`scripts/train_gpe_tokenizer.py`)

**Purpose:** Train Grapheme Pair Encoding tokenizers.

**Features:**
- Grapheme-based pre-tokenization
- Sennrich-style BPE training
- CBPE constraint integration
- Configurable vocabulary size and merge frequency

**Usage:**
```bash
python scripts/train_gpe_tokenizer.py \
  --input data/hindi/corpus.txt \
  --output-dir models/gpe_hi_v0 \
  --vocab-size 32000 \
  --min-pair-frequency 2
```

**Output:**
- `vocab.json` - Token vocabulary
- `merges.txt` - BPE merge rules
- `config.json` - Model configuration

#### 2.2 SentencePiece Baseline Trainer (`scripts/train_sentencepiece_baseline.py`)

**Purpose:** Train SentencePiece BPE/Unigram baseline tokenizers.

**Features:**
- Optional pretokenization
- BPE or Unigram model types
- Configurable normalization
- HuggingFace-compatible output

**Usage:**
```bash
python scripts/train_sentencepiece_baseline.py \
  --input data/hindi/corpus.txt \
  --output-dir models/sp_hi_baseline \
  --vocab-size 32000 \
  --model-type bpe \
  --pretokenize
```

### 3. Evaluation System

#### 3.1 Grapheme Violations (`eval/grapheme_violations.py`)

**Purpose:** Detect when tokenizers split grapheme clusters.

**Key Functions:**
- `detect_violations(text: str, tokens: List[str]) -> List[Violation]` - Find all violations
- `count_violations(text: str, tokens: List[str]) -> int` - Count violations
- `violation_rate(text: str, tokens: List[str]) -> float` - Calculate violation rate
- `generate_violation_report(text: str, tokenizer_results: dict) -> str` - Generate report

**Violation Detection:**
- Compares token boundaries with grapheme cluster boundaries
- Reports position and affected grapheme
- Calculates violation rate: `violations / token_boundaries`

#### 3.2 Fertility Metrics (`eval/fertility.py`)

**Purpose:** Measure token efficiency.

**Key Functions:**
- `calculate_fertility(text: str, tokens: List[str]) -> float` - Tokens per word
- `calculate_chars_per_token(text: str, tokens: List[str]) -> float` - Characters per token
- `compare_fertility(texts: List[str], tokenizer_results: dict) -> dict` - Batch comparison

**Metrics:**
- **Fertility:** `num_tokens / num_words` (lower is better)
- **Chars per Token:** `num_chars / num_tokens` (higher is better)

#### 3.3 Integrated Metrics (`eval/metrics.py`)

**Purpose:** Unified evaluation system.

**Key Functions:**
- `evaluate_tokenizer(text: str, tokenizer, lang: str = "hi") -> Metrics` - Single text evaluation
- `evaluate_batch(texts: List[str], tokenizer, lang: str = "hi") -> AggregatedMetrics` - Batch evaluation
- `generate_scorecard(tokenizer_results: dict, ...) -> dict` - Generate scorecards
- `export_scorecard(scorecard, format: str = "json") -> str` - Export to JSON/Markdown

**Data Structures:**
- `Metrics` - Single text metrics
- `AggregatedMetrics` - Batch metrics
- `Scorecard` - Complete evaluation results

### 4. Comparison Scripts

#### 4.1 Basic Comparison (`scripts/compare_tokenizers.py`)

**Purpose:** Simple tokenizer comparison.

**Features:**
- Compare multiple tokenizers on same text
- Display tokens and statistics
- JSON output support
- Registry-based tokenizer loading

**Usage:**
```bash
python scripts/compare_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert,gpe_hi_v0
```

#### 4.2 Comprehensive Evaluation (`scripts/evaluate_tokenizers.py`)

**Purpose:** Full evaluation with metrics and scorecards.

**Features:**
- Single text or batch evaluation
- All metrics (fertility, violations, chars/token)
- Scorecard generation (JSON/Markdown)
- Dataset evaluation support

**Usage:**
```bash
python scripts/evaluate_tokenizers.py \
  --dataset data/hindi/eval_sets/news_headlines.txt \
  --tokenizers all \
  --output scorecards/news_eval.md \
  --format markdown
```

#### 4.3 Full Evaluation (`scripts/run_full_evaluation.py`)

**Purpose:** Run comprehensive evaluation on all datasets.

**Features:**
- Evaluates all tokenizers in registry
- Processes all curated examples and eval datasets
- Generates scorecards for each dataset
- Batch processing

**Usage:**
```bash
python scripts/run_full_evaluation.py \
  --lang hi \
  --output-dir scorecards
```

---

## Data Structures

### Metrics

```python
@dataclass
class Metrics:
    fertility: float                    # tokens/word
    chars_per_token: float              # chars/token
    grapheme_violations: int            # number of violations
    grapheme_violation_rate: float      # violation rate (0.0-1.0)
    num_tokens: int                     # total tokens
    num_words: int                      # total words
    num_chars: int                      # total characters
```

### AggregatedMetrics

```python
@dataclass
class AggregatedMetrics:
    avg_fertility: float                # average tokens/word
    avg_chars_per_token: float          # average chars/token
    total_grapheme_violations: int      # total violations
    avg_grapheme_violation_rate: float  # average violation rate
    total_tokens: int                   # total tokens
    total_words: int                    # total words
    total_chars: int                    # total characters
    num_texts: int                      # number of texts evaluated
```

### Scorecard

```python
@dataclass
class Scorecard:
    tokenizer_id: str                   # tokenizer identifier
    tokenizer_name: str                 # display name
    metrics: Metrics | AggregatedMetrics # evaluation metrics
    sample_texts: List[str]             # sample texts used
    timestamp: str                      # ISO timestamp
```

### Violation

```python
@dataclass
class Violation:
    token_index: int                    # index of violating token
    token: str                          # token text
    grapheme: str                       # affected grapheme
    position_in_text: int               # position in original text
    description: str                    # violation description
```

---

## API Reference

### Tokenizer Interface

All tokenizers implement a common interface:

```python
class BaseTokenizer:
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens."""
        ...

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    def stats(self, text: str) -> TokenizationStats:
        """Compute tokenization statistics."""
        ...
```

### Registry System

Tokenizers are registered in `tokenizers/registry.yaml`:

```yaml
tokenizers:
  - id: tokenizer_id
    type: hf | openai | custom_gpe | sentencepiece
    model_name: "model/name"  # for HF
    model_path: "path/to/model"  # for custom
    display_name: "Display Name"
```

### Evaluation API

```python
# Single text evaluation
from eval.metrics import evaluate_tokenizer
metrics = evaluate_tokenizer(text, tokenizer, lang="hi")

# Batch evaluation
from eval.metrics import evaluate_batch
metrics = evaluate_batch(texts, tokenizer, lang="hi")

# Scorecard generation
from eval.metrics import generate_scorecard, export_scorecard
scorecards = generate_scorecard(results, tokenizer_names, sample_texts)
markdown = export_scorecard(scorecards, format="markdown")
```

---

## File Structure

```
indic-tokenization-lab/
├── docs/                          # Documentation
│   ├── 00-vision.md              # Project vision
│   ├── 01-roadmap.md             # Development roadmap
│   ├── 02-research-survey.md     # Research references
│   ├── 10-linguistics-devanagari.md
│   ├── 11-linguistics-hindi-morphology.md
│   ├── 12-linguistics-sanskrit-sandhi.md
│   ├── 20-tokenizer-architecture.md
│   ├── 21-training-pipeline.md
│   ├── 22-evaluation-metrics.md
│   ├── 23-datasets-and-licensing.md
│   ├── 30-playground-ui-spec.md
│   ├── 31-gpe-prototype-plan.md
│   ├── 40-integration-hf-and-llms.md
│   ├── 50-contributing-guidelines.md
│   ├── 99-notes-open-questions.md
│   ├── BUILD_PLAN.md
│   └── PROJECT_SPECS.md          # This file
│
├── data/                          # Data and examples
│   ├── hindi/
│   │   ├── curated_examples.jsonl
│   │   ├── eval_sets/
│   │   │   ├── news_headlines.txt
│   │   │   ├── literature.txt
│   │   │   └── conversational.txt
│   │   └── README.md
│   └── sanskrit/
│       ├── curated_examples.jsonl
│       ├── eval_sets/
│       │   ├── classical.txt
│       │   └── sandhi_examples.txt
│       └── README.md
│
├── eval/                          # Evaluation modules
│   ├── __init__.py
│   ├── grapheme_violations.py
│   ├── fertility.py
│   ├── metrics.py
│   └── README.md
│
├── examples/                      # Usage examples
│   ├── basic_comparison.py
│   ├── evaluate_tokenizer.py
│   └── README.md
│
├── models/                        # Trained models (gitignored)
│   └── .gitkeep
│
├── playground/                    # Web UI (future)
│   └── README.md
│
├── scripts/                       # Utility scripts
│   ├── compare_tokenizers.py
│   ├── evaluate_tokenizers.py
│   ├── run_full_evaluation.py
│   ├── train_gpe_tokenizer.py
│   └── train_sentencepiece_baseline.py
│
├── scorecards/                    # Evaluation results
│   └── README.md
│
├── tokenizers/                    # Tokenizer implementations
│   ├── __init__.py
│   ├── cbpe_constraints.py
│   ├── gpe_tokenizer.py
│   ├── grapheme_segmenter.py
│   ├── pretokenizer.py
│   ├── sentencepiece_tokenizer.py
│   └── registry.yaml
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Usage Guide

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Operations

#### 1. Compare Tokenizers

```bash
python scripts/compare_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert
```

#### 2. Evaluate with Metrics

```bash
python scripts/evaluate_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers all \
  --output scorecard.json
```

#### 3. Train GPE Tokenizer

```bash
python scripts/train_gpe_tokenizer.py \
  --input data/hindi/corpus.txt \
  --output-dir models/gpe_hi_v0 \
  --vocab-size 32000
```

#### 4. Train SentencePiece Baseline

```bash
python scripts/train_sentencepiece_baseline.py \
  --input data/hindi/corpus.txt \
  --output-dir models/sp_hi_baseline \
  --vocab-size 32000 \
  --model-type bpe
```

#### 5. Full Evaluation

```bash
python scripts/run_full_evaluation.py \
  --lang hi \
  --output-dir scorecards
```

### Python API

```python
from scripts.compare_tokenizers import load_registry, create_tokenizer_from_config
from eval.metrics import evaluate_tokenizer

# Load tokenizer
registry = load_registry(Path("tokenizers/registry.yaml"))
cfg = registry["indicbert"]
tokenizer = create_tokenizer_from_config(cfg)

# Tokenize
tokens = tokenizer.tokenize("यहाँ आपका हिंदी वाक्य जाएगा।")

# Evaluate
metrics = evaluate_tokenizer("यहाँ आपका हिंदी वाक्य जाएगा।", tokenizer)
print(f"Fertility: {metrics.fertility:.3f}")
print(f"Violations: {metrics.grapheme_violations}")
```

---

## Evaluation Metrics

### Intrinsic Metrics

1. **Fertility** (`tokens / words`)
   - Lower is better
   - Measures token efficiency
   - Target: < 2.0 for Hindi

2. **Chars per Token** (`chars / tokens`)
   - Higher is better
   - Measures packing efficiency
   - Target: > 3.0 for Hindi

3. **Grapheme Violation Rate** (`violations / boundaries`)
   - Lower is better (0.0 is ideal)
   - Measures script awareness
   - Target: 0.0 for proper tokenizers

### Evaluation Workflow

1. **Single Text Evaluation**
   - Tokenize text
   - Calculate metrics
   - Detect violations
   - Generate report

2. **Batch Evaluation**
   - Process multiple texts
   - Aggregate metrics
   - Calculate averages
   - Generate scorecard

3. **Dataset Evaluation**
   - Load dataset
   - Evaluate all tokenizers
   - Compare results
   - Export scorecards

---

## Tokenizer Types

### 1. HuggingFace Tokenizers (`type: hf`)

- Uses `transformers.AutoTokenizer`
- Supports any HF model
- Examples: IndicBERT, mBERT, IndicBART

### 2. OpenAI Tokenizers (`type: openai`)

- Uses `tiktoken` library
- Supports GPT models
- Requires `tiktoken` package

### 3. GPE Tokenizers (`type: custom_gpe`)

- Custom Grapheme Pair Encoding
- Trained with `train_gpe_tokenizer.py`
- Grapheme-aware BPE

### 4. SentencePiece Tokenizers (`type: sentencepiece`)

- SentencePiece models
- Trained with `train_sentencepiece_baseline.py`
- BPE or Unigram variants

---

## Training Pipeline

### GPE Training

1. **Corpus Preparation**
   - Load corpus file
   - Optional: filter, normalize

2. **Grapheme Segmentation**
   - Segment each word into graphemes
   - Add word-end sentinel

3. **BPE Training**
   - Count pair frequencies
   - Apply CBPE constraints
   - Perform merges iteratively
   - Save vocab and merges

4. **Model Export**
   - Generate vocab.json
   - Generate merges.txt
   - Save config.json

### SentencePiece Training

1. **Corpus Preparation**
   - Load corpus file
   - Optional: pretokenize

2. **SentencePiece Training**
   - Configure model type (BPE/Unigram)
   - Set vocabulary size
   - Train model

3. **Model Export**
   - Save .model file
   - Save .vocab file
   - Save config.json

---

## Configuration

### Registry Configuration

Tokenizers are configured in `tokenizers/registry.yaml`:

```yaml
tokenizers:
  - id: unique_id
    type: hf | openai | custom_gpe | sentencepiece
    model_name: "model/name"      # for HF
    model_path: "path/to/model"   # for custom
    display_name: "Display Name"
```

### Training Configuration

GPE training parameters:
- `--vocab-size`: Target vocabulary size (default: 32000)
- `--min-pair-frequency`: Minimum pair frequency (default: 2)
- `--max-lines`: Limit corpus size (optional)
- `--lowercase`: Lowercase text (optional)
- `--dev-only`: Only Devanagari graphemes (optional)

SentencePiece training parameters:
- `--vocab-size`: Vocabulary size (default: 32000)
- `--model-type`: bpe or unigram (default: bpe)
- `--pretokenize`: Apply pretokenization (default: True)
- `--normalization-rule`: SP normalization rule

---

## Dependencies

### Core Dependencies

- `transformers>=4.20.0` - HuggingFace tokenizers
- `tokenizers>=0.13.0` - HF tokenizers library
- `sentencepiece>=0.1.96` - SentencePiece models
- `pyyaml>=6.0` - YAML parsing
- `regex>=2023.0.0` - Unicode regex (grapheme clustering)

### Optional Dependencies

- `tiktoken>=0.5.0` - OpenAI tokenizers (optional)

### Development Dependencies

- `pytest>=7.0.0` - Testing
- `black>=23.0.0` - Code formatting
- `mypy>=1.0.0` - Type checking
- `ruff>=0.1.0` - Linting

---

## Data Formats

### Curated Examples (JSONL)

```json
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "lang": "hi",
  "domain": "news",
  "complexity": "medium",
  "tags": ["compound"],
  "description": "Description of example",
  "source": "example"
}
```

### Evaluation Datasets (TXT)

Plain text files with one text per line:

```
यहाँ आपका हिंदी वाक्य जाएगा।
रेलगाड़ी स्टेशन पर आ गई।
...
```

### Model Artifacts

**GPE Model:**
- `vocab.json` - Token to ID mapping
- `merges.txt` - BPE merge rules
- `config.json` - Model metadata

**SentencePiece Model:**
- `sp_model.model` - SentencePiece model
- `sp_model.vocab` - Vocabulary
- `config.json` - Model metadata

### Scorecards

**JSON Format:**
```json
{
  "tokenizer_id": {
    "tokenizer_id": "...",
    "tokenizer_name": "...",
    "metrics": {...},
    "sample_texts": [...],
    "timestamp": "..."
  }
}
```

**Markdown Format:**
- Human-readable tables
- Metric summaries
- Comparison across tokenizers

---

## Performance Considerations

### Training

- GPE training: CPU-bound, scales with corpus size
- SentencePiece training: Fast, optimized C++ backend
- Recommended: Start with small corpus (100K-1M lines) for testing

### Evaluation

- Grapheme violation detection: O(n*m) where n=texts, m=tokens
- Fertility calculation: O(n) per text
- Batch evaluation: Parallelizable across tokenizers

### Memory

- Tokenizer models: ~10-100MB per model
- Evaluation: Minimal memory overhead
- Training: Scales with vocabulary size

---

## Future Extensions

### Planned Features

1. **Morphology Layer (L1)**
   - Hindi morphological segmentation
   - Sanskrit sandhi splitting
   - Morphology-aware constraints

2. **Playground UI**
   - Web-based tokenizer comparison
   - Interactive visualization
   - Real-time evaluation

3. **Downstream Evaluation**
   - LM perplexity experiments
   - MT BLEU scores
   - Task-specific benchmarks

4. **Additional Scripts**
   - Marathi support
   - Nepali support
   - Other Indic scripts

### Research Directions

- Morphology-aware tokenization
- Sandhi-aware Sanskrit tokenization
- Multi-script code-mixed text
- Low-resource language support

---

## Contributing

See `docs/50-contributing-guidelines.md` for:
- How to contribute
- Code style guidelines
- Testing requirements
- Documentation standards

---

## License

MIT License - See `LICENSE` file for details.

---

## References

- MorphTok: Morphologically Grounded Tokenization for Indian Languages (Brahma et al., 2025)
- GPE: Egalitarian Language Representation in Language Models (Velayuthan & Sarveswaran, 2024/2025)
- AI4Bharat IndicNLP Library
- Unicode Text Segmentation (UAX #29)

See `docs/02-research-survey.md` for comprehensive references.

---

## Version History

- **v0.1.0** (Current)
  - GPE prototype implementation
  - Evaluation infrastructure
  - SentencePiece baseline
  - Comprehensive metrics
  - Scorecard generation

---

## Support

For issues, questions, or contributions:
- Check documentation in `docs/`
- Review examples in `examples/`
- See contributing guidelines in `docs/50-contributing-guidelines.md`

---

*Last updated: 2025*

