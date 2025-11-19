# Training Tokenizers on Desktop Machine

This guide provides step-by-step instructions for training GPE+CBPE and AG-BPE tokenizers on a faster desktop machine.

## Prerequisites

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/funwae/indic-tokenizer.git
   cd indic-tokenizer
   ```

2. **Set up Python environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Transfer corpus file** (if needed):
   - Source: `data/hindi/processed/gpe_cbpe_hi_corpus.txt` (111MB, 500k lines)
   - Or use the full raw corpus: `data/hindi/raw/hi.txt` (21GB)
   - Ensure the file is in the same relative path: `data/hindi/processed/gpe_cbpe_hi_corpus.txt`

## Training Commands

### 1. Train GPE+CBPE Tokenizer

```bash
# Activate virtual environment
source .venv/bin/activate

# Train on processed corpus (500k lines, ~10-30 minutes)
python3 scripts/train_gpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2

# OR train on full raw corpus (21GB, ~2-4 hours)
python3 scripts/train_gpe_tokenizer.py \
    --input data/hindi/raw/hi.txt \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2
```

**Expected output:**
- `models/gpe_cbpe_hi_v1/vocab.json` (~32k entries)
- `models/gpe_cbpe_hi_v1/merges.txt` (~32k merge rules)
- `models/gpe_cbpe_hi_v1/config.json`

**Verification:**
```bash
# Check vocab size
python3 -c "import json; print(len(json.load(open('models/gpe_cbpe_hi_v1/vocab.json'))))"
# Should be close to 32000

# Check merges count
wc -l models/gpe_cbpe_hi_v1/merges.txt
# Should be ~32000 lines
```

### 2. Train AG-BPE Tokenizer

```bash
# Train AG-BPE (uses mutual information + optional attention)
python3 scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2

# Note: --attention-model can be provided if you have a trained LM,
# but it's optional - AG-BPE works with just MI + frequency
```

**Expected output:**
- `models/ag_bpe_hi_v1/vocab.json`
- `models/ag_bpe_hi_v1/merges.txt`
- `models/ag_bpe_hi_v1/config.json` (with attention/MI weights)

**Verification:**
```bash
python3 -c "import json; print(len(json.load(open('models/ag_bpe_hi_v1/vocab.json'))))"
wc -l models/ag_bpe_hi_v1/merges.txt
```

## Training Time Estimates

| Corpus | Size | Lines | Estimated Time (Desktop) |
|--------|------|-------|-------------------------|
| Processed | 111MB | 500k | 10-30 minutes |
| Full Raw | 21GB | ~50M | 2-4 hours |

*Times vary based on CPU speed, RAM, and disk I/O.*

## After Training

Once training completes, you can:

1. **Verify tokenizers work**:
   ```bash
   python3 scripts/compare_tokenizers.py \
       --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
       --tokenizers gpe_cbpe_hi_v1,ag_bpe_hi_v1
   ```

2. **Run evaluation suite** (see next section)

3. **Commit models to git** (if desired, though they're large):
   ```bash
   git add models/gpe_cbpe_hi_v1/ models/ag_bpe_hi_v1/
   git commit -m "Add trained GPE+CBPE and AG-BPE tokenizers"
   ```

## Next Steps: Running Evaluation

After training, run the full evaluation suite:

```bash
# 1. Baseline evaluation (GPE+CBPE)
python3 scripts/run_baseline_evaluation.py \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --output-dir scorecards/baseline_gpe_cbpe

# 2. Semantic tokenizer evaluation (AG-BPE)
python3 scripts/run_semantic_evaluation.py \
    --tokenizer-id ag_bpe_hi_v1 \
    --output-dir scorecards/semantic_ag_bpe

# 3. Train tiny LMs for perplexity comparison
python3 scripts/train_tiny_lm.py \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --max-sentences 10000

python3 scripts/train_tiny_lm.py \
    --tokenizer-id ag_bpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/ag_bpe_hi_v1 \
    --max-sentences 10000

# 4. Evaluate tiny LMs
python3 scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --eval-corpus data/hindi/demo/news_small.txt

python3 scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/ag_bpe_hi_v1 \
    --tokenizer-id ag_bpe_hi_v1 \
    --eval-corpus data/hindi/demo/news_small.txt
```

## Troubleshooting

**Out of memory errors:**
- Use `--max-lines` to limit corpus size for testing
- Process corpus in chunks (modify script if needed)

**Slow training:**
- Ensure you're using the processed corpus (500k lines) not the full 21GB
- Check CPU usage - training is CPU-bound
- Consider using fewer merges for testing: `--vocab-size 16000`

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version: `python3 --version` (should be 3.8+)

## Quick Reference

**Training GPE+CBPE:**
```bash
python3 scripts/train_gpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2
```

**Training AG-BPE:**
```bash
python3 scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2
```

