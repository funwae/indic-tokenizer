# Handoff to India Team

This document provides a quick start guide for the team in India to continue the Indic Tokenization Lab project.

## Repository

**GitHub**: https://github.com/funwae/indic-tokenizer.git

## Current Status

### ✅ Completed
- Phase 2 infrastructure: Evaluation metrics, benchmarks, morphology evaluation
- Phase 3 infrastructure: AG-BPE trainer, evaluation scripts, documentation templates
- Corpus preparation: Hindi corpus processed (500k lines, 111MB)
- Training scripts: GPE+CBPE and AG-BPE trainers ready

### 🔄 Pending (Next Steps)
- **Train tokenizers** on full corpus (see `docs/TRAINING_ON_DESKTOP.md`)
- **Run evaluation suite** on trained tokenizers
- **Fill in results documentation** with actual metrics
- **Train tiny LMs** for perplexity comparison
- **Run downstream benchmarks** (optional)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/funwae/indic-tokenizer.git
cd indic-tokenizer
```

### 2. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Corpus
The processed corpus should be available at:
- `data/hindi/processed/gpe_cbpe_hi_corpus.txt` (111MB, 500k lines)

If not present, you can:
- Download CC-100 Hindi corpus (21GB) to `data/hindi/raw/hi.txt`
- Process it using: `python3 scripts/prepare_corpus_hi.py --input data/hindi/raw/hi.txt --output data/hindi/processed/gpe_cbpe_hi_corpus.txt --max-lines 500000`

### 4. Train Tokenizers

**Option A: Automated (Recommended)**
```bash
./scripts/train_all_tokenizers.sh
```

**Option B: Manual**
```bash
# Train GPE+CBPE
python3 scripts/train_gpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2

# Train AG-BPE
python3 scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2
```

**Expected time**: 10-30 minutes for processed corpus (500k lines)

### 5. Run Evaluation Suite

After training, run comprehensive evaluation:

```bash
# Baseline evaluation (GPE+CBPE)
python3 scripts/run_baseline_evaluation.py \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --output-dir scorecards/baseline_gpe_cbpe

# Semantic tokenizer evaluation (AG-BPE)
python3 scripts/run_semantic_evaluation.py \
    --tokenizer-id ag_bpe_hi_v1 \
    --output-dir scorecards/semantic_ag_bpe
```

### 6. Train Tiny LMs (for perplexity comparison)

```bash
# Train tiny LM for baseline
python3 scripts/train_tiny_lm.py \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --max-sentences 10000

# Train tiny LM for AG-BPE
python3 scripts/train_tiny_lm.py \
    --tokenizer-id ag_bpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/ag_bpe_hi_v1 \
    --max-sentences 10000

# Evaluate perplexity
python3 scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --eval-corpus data/hindi/demo/news_small.txt

python3 scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/ag_bpe_hi_v1 \
    --tokenizer-id ag_bpe_hi_v1 \
    --eval-corpus data/hindi/demo/news_small.txt
```

### 7. Fill in Results Documentation

Update the following documents with actual results:

- `docs/research/BASELINE_RESULTS.md` - GPE+CBPE evaluation results
- `docs/research/SEMANTIC_TOKENIZER_RESULTS.md` - AG-BPE evaluation results
- `docs/research/DOWNSTREAM_BENCHMARK_RESULTS.md` - Downstream task results (if applicable)

## Key Documentation

- **Training Guide**: `docs/TRAINING_ON_DESKTOP.md` - Complete training instructions
- **Reproducibility**: `docs/research/REPRODUCIBILITY.md` - Fixed seeds and exact commands
- **Research Workflow**: `docs/RESEARCH_LOOP_WORKFLOW.md` - Full research loop process
- **Project Overview**: `README.md` - Project introduction and quickstart

## Project Structure

```
indic-tokenizer/
├── scripts/              # Training and evaluation scripts
├── tokenizers/          # Tokenizer implementations
├── eval/                # Evaluation metrics
├── models/              # Trained tokenizers and LMs
├── data/                # Corpora and datasets
├── docs/                # Documentation
│   ├── research/        # Research results templates
│   └── TRAINING_ON_DESKTOP.md  # Training guide
└── scorecards/          # Evaluation results
```

## Tokenizer Types

- **GPE+CBPE** (`gpe_cbpe_hi_v1`): Baseline grapheme-aware BPE with constraints
- **AG-BPE** (`ag_bpe_hi_v1`): Attention-guided BPE with mutual information
- **Frontier tokenizers**: GPT-4o, Llama-3, mBERT, IndicBERT (for comparison)

## Evaluation Metrics

1. **Efficiency + Script**: Token count, fertility, compression ratio, grapheme violations
2. **Fairness/Parity**: Tokenization parity vs English, token tax, NSL cross-language
3. **Morphology**: Boundary precision/recall, morpheme alignment, fragmentation
4. **Downstream Proxy**: Tiny LM perplexity

## Support

For questions or issues:
1. Check documentation in `docs/` directory
2. Review `docs/RESEARCH_LOOP_WORKFLOW.md` for workflow guidance
3. Check `docs/research/REPRODUCIBILITY.md` for exact commands

## Next Milestones

1. ✅ Train both tokenizers on full corpus
2. ✅ Run complete evaluation suite
3. ✅ Document results in research templates
4. ⏳ Compare baseline vs semantic tokenizer
5. ⏳ Run downstream benchmarks (optional)
6. ⏳ Finalize paper/blog post

---

**Last Updated**: November 2024
**Status**: Ready for training and evaluation phase

