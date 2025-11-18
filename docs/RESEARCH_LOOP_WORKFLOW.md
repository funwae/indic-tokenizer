# Full Research Loop Workflow for Hindi Tokenization

This document outlines the complete workflow for running the full research loop: training AG-BPE, evaluating against baselines, and generating publishable results.

## Prerequisites

- Ubuntu with CUDA working (`nvidia-smi` OK)
- Python 3.8+ with virtual environment
- All dependencies installed (`pip install -r requirements.txt`)
- Raw Hindi corpus (IndicNLP or similar)

## Step 0: Environment Setup and Smoke Test

```bash
cd /path/to/indic-tokenization-lab

# Activate venv
source .venv/bin/activate

# Verify dependencies
pip list | grep -E "(transformers|tokenizers|sentencepiece|tiktoken)"

# Run smoke test
python scripts/run_benchmark.py \
    --config configs/hi_demo.yaml \
    --output-dir scorecards/hi_demo_smoke
```

Expected: Benchmark completes and writes `scorecards/hi_demo_smoke/results.md`.

## Step 1: Prepare Full Hindi Corpus

**Skip if `data/hindi/processed/gpe_cbpe_hi_corpus.txt` already exists.**

### 1.1 Obtain Raw Corpus

Download IndicNLP Hindi corpus or similar:
- Place raw text file at: `data/hindi/raw/indicnlp_hi.txt`
- Should be UTF-8 encoded Hindi text (one sentence per line recommended)

### 1.2 Process Corpus

```bash
python scripts/prepare_corpus_hi.py \
    --input data/hindi/raw/indicnlp_hi.txt \
    --output data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --max-lines 500000
```

### 1.3 Verify

```bash
head -n 3 data/hindi/processed/gpe_cbpe_hi_corpus.txt
wc -l data/hindi/processed/gpe_cbpe_hi_corpus.txt
```

Expected: Clean Hindi text, ~300k-500k lines.

## Step 2: Train AG-BPE Tokenizer

### 2.1 Train AG-BPE

```bash
python scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2 \
    --max-lines 500000 \
    --dev-only
```

**Note**: If no attention model is provided, AG-BPE will use only MI and frequency (still semantic-aware).

### 2.2 Verify Training

```bash
ls models/ag_bpe_hi_v1/
# Should contain: vocab.json, merges.txt, config.json
```

### 2.3 Quick Test

```bash
python scripts/compare_tokenizers.py \
    --text "भारत में आज कई महत्वपूर्ण घटनाएं हुईं।" \
    --tokenizers gpe_cbpe_hi_v1,ag_bpe_hi_v1,gpt4o_tok,llama3_8b_tok
```

Expected: Clean table showing different token counts.

## Step 3: Run Full Intrinsic Metrics

### 3.1 Update Config

Ensure `configs/hi_full.yaml` includes:
- All tokenizers: `mbert`, `indicbert`, `gpe_cbpe_hi_v1`, `ag_bpe_hi_v1`, `gpt4o_tok`, `llama3_8b_tok`
- Corpus path: `data/hindi/processed/gpe_cbpe_hi_corpus.txt`

### 3.2 Run Benchmark

```bash
python scripts/run_benchmark.py \
    --config configs/hi_full.yaml \
    --output-dir scorecards/hi_full
```

### 3.3 Check Results

```bash
cat scorecards/hi_full/results.md
```

Expected: Metrics for all tokenizers including `ag_bpe_hi_v1`.

## Step 4: Run Fairness/Token Tax Benchmark

### 4.1 Prepare Parity Dataset

**Skip if `data/parity/hi_en_iitb_sample.jsonl` exists.**

```bash
python scripts/prepare_parity_hi_en.py \
    --input-hi data/parity/iitb_train.hi \
    --input-en data/parity/iitb_train.en \
    --output data/parity/hi_en_iitb_sample.jsonl \
    --max-pairs 50000
```

### 4.2 Run Parity Benchmark

```bash
python scripts/run_parity_benchmark.py \
    --input data/parity/hi_en_iitb_sample.jsonl \
    --tokenizers gpt4o_tok,llama3_8b_tok,gpe_cbpe_hi_v1,ag_bpe_hi_v1 \
    --baseline gpt4o_tok \
    --output-dir scorecards/parity_hi_en
```

### 4.3 Check Results

```bash
cat scorecards/parity_hi_en/results.md
```

Expected: TP, NSL, token tax for each tokenizer.

## Step 5: Run Morphology Evaluation

### 5.1 Verify Gold Set

```bash
ls data/hindi/morph_gold/hi_morph_gold.tsv
```

### 5.2 Run Morphology Eval

```bash
python scripts/run_morphology_eval.py \
    --input data/hindi/morph_gold/hi_morph_gold.tsv \
    --tokenizers gpe_cbpe_hi_v1,ag_bpe_hi_v1,gpt4o_tok,llama3_8b_tok \
    --output-dir scorecards/morph_hi
```

### 5.3 Check Results

```bash
cat scorecards/morph_hi/results.md
```

Expected: Boundary F1, alignment, fragmentation for each tokenizer.

## Step 6: Train & Evaluate Tiny LMs

### 6.1 Prepare Eval Corpus

Create held-out eval set:

```bash
# Use last 10k lines as eval (adjust based on corpus size)
tail -n 10000 data/hindi/processed/gpe_cbpe_hi_corpus.txt > \
    data/hindi/processed/hi_eval_small.txt
```

### 6.2 Train Tiny LMs

**Baseline (GPE+CBPE)**:
```bash
python scripts/train_tiny_lm.py \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --steps 50000 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt
```

**AG-BPE**:
```bash
python scripts/train_tiny_lm.py \
    --tokenizer-id ag_bpe_hi_v1 \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/ag_bpe_hi_v1 \
    --steps 50000 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt
```

**Optional: Frontier tokenizers**:
```bash
python scripts/train_tiny_lm.py \
    --tokenizer-id gpt4o_tok \
    --corpus data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/tiny_lm_hi/gpt4o_tok \
    --steps 50000 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt
```

### 6.3 Evaluate Perplexity

```bash
python scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/gpe_cbpe_hi_v1 \
    --tokenizer-id gpe_cbpe_hi_v1 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt

python scripts/eval_tiny_lm.py \
    --model-dir models/tiny_lm_hi/ag_bpe_hi_v1 \
    --tokenizer-id ag_bpe_hi_v1 \
    --eval-corpus data/hindi/processed/hi_eval_small.txt
```

**Record perplexity numbers** for each tokenizer.

## Step 7: Fill in Research Documentation

### 7.1 Baseline Results

Update `docs/research/BASELINE_RESULTS.md`:
- Summary table: `gpe_cbpe_hi_v1` vs GPT-4o/Llama-3
- Metrics: fertility, NSL, CR, token tax, script metrics
- Note: This is the "good citizen" Devanagari baseline

### 7.2 Semantic Tokenizer Results

Update `docs/research/SEMANTIC_TOKENIZER_RESULTS.md`:
- **Intrinsic metrics**: Compare `gpe_cbpe_hi_v1` vs `ag_bpe_hi_v1`
  - Fertility, NSL, CR
  - Akshara integrity, dependent-vowel split rate
  - TP/Token tax vs GPT-4o baseline
- **Morphology metrics**: Boundary F1, alignment, fragmentation
- **Tiny LM results**: Perplexity for both tokenizers
- **Headline**: "AG-BPE reduces token count by ~X% vs GPE+CBPE while maintaining akshara integrity and improving tiny LM perplexity by Y%."

### 7.3 Downstream Benchmark

Update `docs/research/DOWNSTREAM_BENCHMARK_RESULTS.md`:
- If MT/classification task implemented: BLEU/accuracy deltas
- Otherwise: Mark as TODO

### 7.4 Reproducibility

Update `docs/research/REPRODUCIBILITY.md`:
- Python version
- GPU requirements (8-12GB recommended)
- Exact commands from Steps 1-6
- Fixed seeds (if applicable)

## Step 8: Final Polish

### 8.1 Update README

Add "Quickstart for Hindi" section to `README.md`:

```markdown
## Quickstart: Hindi Tokenization & LM Evaluation

```bash
git clone <repo-url>
cd indic-tokenization-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Hindi demo benchmark
python scripts/run_benchmark.py \
    --config configs/hi_demo.yaml \
    --output-dir scorecards/hi_demo

# Run the full Hindi evaluation (including AG-BPE)
python scripts/run_benchmark.py \
    --config configs/hi_full.yaml \
    --output-dir scorecards/hi_full
```

See:
- Baseline: `docs/research/BASELINE_RESULTS.md`
- Semantic tokenization: `docs/research/SEMANTIC_TOKENIZER_RESULTS.md`
- Fairness: `scorecards/parity_hi_en/results.md`
- Morphology: `scorecards/morph_hi/results.md`
```

### 8.2 Optional: Low-Resource Setup Note

For setups without GPU:
- Can run demo configs
- Skip tiny LM training
- Use CPU for tokenization-only benchmarks

### 8.3 Commit and Push

```bash
git status
git add .
git commit -m "Add AG-BPE Hindi tokenizer + full evaluation results"
git push origin main
```

## Success Criteria

AG-BPE Hindi is successful if:

1. **Efficiency**: ~3-10% lower fertility/NSL vs GPT-4o/Llama-3
2. **Morphology**: Equal or better boundary F1 vs GPE+CBPE baseline
3. **Downstream**: Modest perplexity improvement (even 2-5% is meaningful)

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'tokenizers.grapheme_segmenter'`:
- Ensure project root is in `sys.path`
- Check that `tokenizers/grapheme_segmenter.py` exists

### CUDA Out of Memory

For tiny LM training:
- Reduce batch size
- Reduce model size (d_model, n_layers)
- Use gradient accumulation

### Corpus Not Found

- Verify corpus path in config files
- Run `prepare_corpus_hi.py` first
- Check file permissions

## Next Steps

Once results are documented:
1. Review `docs/research/PAPER_OUTLINE.md`
2. Prepare publication materials
3. Share with collaborators

