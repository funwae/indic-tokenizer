# Reproducibility Guide

This document ensures all experiments in Phase 3 are reproducible.

---

## Fixed Random Seeds

All scripts use fixed random seeds for reproducibility:

- **Corpus shuffling**: `random_seed=42` (in `prepare_corpus_hi.py`)
- **BPE training**: Deterministic (no randomness in merge selection)
- **Model training**: `torch.manual_seed(42)` (when PyTorch available)
- **Data sampling**: `random_seed=42` (in `prepare_parity_hi_en.py`)

---

## Training Procedures

### Baseline GPE+CBPE Tokenizer

**Command**:
```bash
python scripts/train_gpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --dev-only \
    --max-lines 500000 \
    --profile hi_v1
```

**Parameters**:
- Vocab size: 32,000
- Min pair frequency: 2
- Dev-only: True (only Devanagari graphemes)
- Max lines: 500,000
- Random seed: N/A (deterministic BPE)

### Attention-Guided BPE Tokenizer

**Command**:
```bash
python scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2 \
    --dev-only \
    --max-lines 500000
```

**Parameters**:
- Vocab size: 32,000
- Attention weight: 0.5
- MI weight: 0.3
- Frequency weight: 0.2
- Min pair frequency: 2
- Dev-only: True
- Max lines: 500,000

---

## Evaluation Procedures

### Full Intrinsic Metrics Benchmark

**Command**:
```bash
python scripts/run_benchmark.py \
    --config configs/hi_full.yaml \
    --output-dir scorecards/hi_full
```

**Outputs**:
- `results.json` (all metrics)
- `results.md` (human-readable summary)

### Fairness/Token Tax Benchmark

**Command**:
```bash
python scripts/run_parity_benchmark.py \
    --input data/parity/hi_en_iitb_sample.jsonl \
    --tokenizers gpt4o_tok,llama3_8b_tok,gpe_cbpe_hi_v1,ag_bpe_hi_v1 \
    --baseline gpt4o_tok \
    --output-dir scorecards/parity_hi_en
```

**Outputs**:
- `results.json`
- `results.md`

### Morphology Evaluation

**Command**:
```bash
python scripts/run_morphology_eval.py \
    --input data/hindi/morph_gold/hi_morph_gold.tsv \
    --tokenizers gpe_cbpe_hi_v1,ag_bpe_hi_v1,gpt4o_tok,llama3_8b_tok \
    --output-dir scorecards/morph_hi
```

**Outputs**:
- `results.json`
- `results.md`

### Tiny LM Training

**Command** (for each tokenizer):
```bash
python scripts/train_tiny_lm.py \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --steps 50000 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt
```

### Tiny LM Evaluation

**Command**:
```bash
python scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt
```

**Output**: Perplexity score printed to stdout (record for documentation)

---

## Corpus Preparation

### Hindi Training Corpus

**Command**:
```bash
python scripts/prepare_corpus_hi.py \
    --input data/hindi/raw/indicnlp_hi.txt \
    --output data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --max-lines 500000 \
    --min-length 10 \
    --min-devanagari-ratio 0.5 \
    --random-seed 42
```

**Parameters**:
- Max lines: 500,000
- Min length: 10 characters
- Min Devanagari ratio: 0.5
- Random seed: 42

### Parallel Corpus (Parity Evaluation)

**Command**:
```bash
python scripts/prepare_parity_hi_en.py \
    --input-hi data/parity/iitb_train.hi \
    --input-en data/parity/iitb_train.en \
    --output data/parity/hi_en_iitb_sample.jsonl \
    --max-pairs 50000
```

**Parameters**:
- Max pairs: 50,000
- Max length ratio: 3.0 (default)
- Random seed: 42 (default)

---

## Model Checkpoints

### Tokenizer Models

- **Baseline**: `models/gpe_cbpe_hi_v1/`
  - `vocab.json`
  - `merges.txt`
  - `config.json`

- **Semantic**: `models/ag_bpe_hi_v1/`
  - `vocab.json`
  - `merges.txt`
  - `config.json`

### Tiny LM Models

- **Baseline**: `models/tiny_lm_hi/gpe_cbpe_hi_v1/`
  - `model.pt`
  - `config.json`

- **Semantic**: `models/tiny_lm_hi/ag_bpe_hi_v1/`
  - `model.pt`
  - `config.json`

---

## Environment

### Python Version
- Python 3.8+ (tested with Python 3.12.3)

### Dependencies
- See `requirements.txt`
- Key packages: `transformers>=4.20.0`, `tiktoken>=0.5.0`, `regex>=2023.0.0`, `pyyaml>=6.0`
- For tiny LM training: `torch>=2.0.0` (optional, GPU recommended)

### System Requirements
- **CPU-only evaluation**: No GPU required for tokenization benchmarks
- **Tiny LM training**: GPU recommended (8-12GB VRAM), but can run on CPU (slower)
- **Disk space**: ~500MB for models, ~1GB for corpora, ~2-5GB for tiny LM checkpoints
- **Memory**: 8GB RAM minimum, 16GB recommended for corpus processing

---

## Verification

To verify reproducibility:

1. **Check fixed seeds**: All scripts use `random_seed=42`
2. **Verify corpus**: Same corpus → same tokenizer
3. **Compare results**: Re-run evaluation → same results
4. **Check versions**: Document package versions in `requirements.txt`

---

## Notes

- Some randomness may remain in model training (PyTorch operations)
- For exact reproducibility, set `torch.manual_seed(42)` and `torch.use_deterministic_algorithms(True)`
- Corpus preparation is deterministic with fixed seed
- BPE training is deterministic (no randomness in merge selection)

