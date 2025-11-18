# Phase 3 Status: Infrastructure Complete, Empirical Results Pending

**Date**: November 2024
**Status**: Infrastructure and methodology stack complete; empirical validation pending

---

## Current Status: Infrastructure Parity with Research Frontier

### What We Have Built

Our lab now implements a comprehensive tokenization research stack that aligns with current frontier work:

#### 1. GPE+CBPE Baseline (Grapheme + Constrained BPE)

**Alignment with Research**:
- **GPE Paper**: "It All Begins with Tokenizers" / Grapheme Pair Encoding
- **MorphTok**: Morphologically Grounded Tokenization for Indian Languages

**Our Implementation**:
- ✅ Grapheme-based pre-tokenization for Devanagari
- ✅ Constrained BPE (CBPE) with Devanagari combining-mark constraints
- ✅ Akshara integrity and dependent-vowel split metrics
- ✅ Hindi-focused corpus and morphology gold set

**Status**: Baseline `gpe_cbpe_hi_v1` is **locked and documented** (`docs/research/BASELINE_RESULTS.md`)

#### 2. Fairness & Token Tax Framework

**Alignment with Research**:
- **Petrov et al.**: "Language Model Tokenizers Introduce Unfairness"
- **Token Tax Paper**: Tokenization premium analysis across languages

**Our Implementation**:
- ✅ Parity dataset (IIT Bombay EN–HI parallel corpus)
- ✅ Fairness metrics: Tokenization Parity (TP), NSL cross-lingual, Token Tax
- ✅ Support for GPT-4o and Llama-3 tokenizers (frontier baselines)
- ✅ Framework to compute token tax for Hindi vs English

**Status**: Metrics implemented, evaluation scripts ready

#### 3. Morphology Evaluation

**Alignment with Research**:
- **MorphTok**: Morphologically segmented datasets and metrics

**Our Implementation**:
- ✅ Hindi morphology gold set (~150 sentences)
- ✅ Boundary F1, morpheme alignment, fragmentation metrics
- ✅ Evaluation scripts ready

**Status**: Gold set created, metrics implemented

#### 4. Attention-Guided BPE (AG-BPE)

**Alignment with Research**:
- **Recent AG-BPE Work**: HuggingFace blog + papers on semantic tokenization
- **Approach**: Hybrid scoring (frequency + attention + mutual information)

**Our Implementation**:
- ✅ AG-BPE trainer with attention/MI weighting
- ✅ Tokenizer adapter and registry integration
- ✅ Training pipeline ready

**Status**: Implementation complete, **training pending**

#### 5. Downstream Proxy (Tiny LM)

**Alignment with Research**:
- **"BPE is Suboptimal for LM Pretraining"**: Tokenization impact on perplexity
- **MorphTok**: Downstream task evaluation

**Our Implementation**:
- ✅ Tiny LM architecture (~1-3M params)
- ✅ Training and evaluation scripts
- ✅ Framework to compare perplexity across tokenizers

**Status**: Scripts ready, **model training pending**

---

## What's Missing: Empirical Results

### Current Gap

We have built the **infrastructure and methodology** to match current research, but we **haven't yet run the key experiments** that would produce publishable results.

### Missing Numbers

The following comparisons are **not yet documented**:

1. **Efficiency Metrics**:
   - Fertility/NSL/token tax: `ag_bpe_hi_v1` vs `gpe_cbpe_hi_v1` vs GPT-4o/Llama-3
   - Compression ratios on Hindi corpora

2. **Morphology Metrics**:
   - Boundary F1: AG-BPE vs baseline vs frontier tokenizers
   - Morpheme alignment rates
   - Fragmentation comparison

3. **Downstream Performance**:
   - Tiny LM perplexity: baseline vs AG-BPE vs frontier tokenizers
   - MT BLEU scores (if implemented)

### Why This Matters

Recent AG-BPE papers show claims like:
- AG-BPE vocabularies that are smaller but **beat GPT-4 tokenizers in compression**
- More semantic subwords with better morphological alignment
- Improved downstream task performance

To demonstrate similar results for Hindi, we need:
- **Trained AG-BPE tokenizer** (`ag_bpe_hi_v1`)
- **Comprehensive evaluation** across all metrics
- **Comparison tables** showing deltas vs baselines

---

## Path to Publishable Results

### One-Pager Action Plan

To close the gap and produce publishable results:

#### Step 1: Train AG-BPE Tokenizer

```bash
python scripts/train_ag_bpe_tokenizer.py \
    --input data/hindi/processed/gpe_cbpe_hi_corpus.txt \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2 \
    --dev-only \
    --max-lines 500000
```

#### Step 2: Run Comprehensive Evaluation

**Efficiency + Script Metrics**:
```bash
python scripts/run_baseline_evaluation.py \
    --tokenizer-id ag_bpe_hi_v1 \
    --output-dir scorecards/ag_bpe_hi_v1
```

**Parity Benchmark**:
```bash
python scripts/run_parity_benchmark.py \
    --tokenizers ag_bpe_hi_v1,gpe_cbpe_hi_v1,gpt4o_tok,llama3_8b_tok \
    --baseline gpt4o_tok
```

**Morphology Evaluation**:
```bash
python scripts/run_morphology_eval.py \
    --input data/hindi/morph_gold/hi_morph_gold.tsv \
    --tokenizers ag_bpe_hi_v1,gpe_cbpe_hi_v1,mbert,indicbert \
    --output-dir scorecards/morphology_ag_bpe
```

**Tiny LM Perplexity**:
```bash
# Train tiny LMs with each tokenizer
python scripts/train_tiny_lm.py --tokenizer-id gpe_cbpe_hi_v1 ...
python scripts/train_tiny_lm.py --tokenizer-id ag_bpe_hi_v1 ...
python scripts/train_tiny_lm.py --tokenizer-id mbert ...

# Evaluate perplexity
python scripts/eval_tiny_lm.py --models ...
```

#### Step 3: Compare and Document

```bash
python scripts/compare_baseline_semantic.py \
    --baseline-dir scorecards/baseline_gpe_cbpe \
    --semantic-dir scorecards/ag_bpe_hi_v1 \
    --output scorecards/comparison_final.json
```

#### Step 4: Fill in Results Templates

- Update `docs/research/SEMANTIC_TOKENIZER_RESULTS.md` with actual numbers
- Update `docs/research/DOWNSTREAM_BENCHMARK_RESULTS.md` with perplexity/BLEU scores
- Generate comparison tables showing deltas

---

## Success Criteria for Publishable Results

If AG-BPE Hindi achieves:

1. **Efficiency**:
   - ~3-10% lower fertility/NSL vs GPT-4o/Llama-3
   - Lower token tax (Hindi premium reduced)

2. **Morphology**:
   - Equal or better boundary F1 vs GPE+CBPE baseline
   - Better morpheme alignment than frontier tokenizers

3. **Downstream**:
   - Modest perplexity improvement in tiny LM (even 2-5% is meaningful)
   - Better or equal performance vs baseline

Then we have **publishable results**, especially framed as:
> "Semantic + Grapheme + CBPE Tokenization for Hindi/Indic: Combining AG-BPE with Script-Aware Constraints"

---

## Comparison with Research Frontier

### Infrastructure/Design: ✅ At Parity

- **GPE+CBPE**: Matches MorphTok/GPE paper recommendations
- **Fairness Framework**: Matches Token Tax paper methodology
- **Morphology Metrics**: Matches MorphTok evaluation
- **AG-BPE Implementation**: Aligns with recent semantic tokenization work
- **Downstream Proxy**: Matches "BPE is Suboptimal" evaluation approach

### Empirical Results: ⏳ Pending

- **Baseline**: Documented (GPE+CBPE v1)
- **Semantic Tokenizer**: Implemented but not yet trained/evaluated
- **Comparisons**: Framework ready, numbers not yet computed

---

## Next Steps

1. **Immediate**: Train `ag_bpe_hi_v1` tokenizer
2. **Short-term**: Run full evaluation stack
3. **Medium-term**: Fill in results templates with actual numbers
4. **Long-term**: If results are positive, prepare for publication

---

## References

- **GPE**: "It All Begins with Tokenizers" / Grapheme Pair Encoding
- **MorphTok**: "MorphTok: Morphologically Grounded Tokenization for Indian Languages" (Brahma et al., 2025)
- **Token Tax**: "Token Tax: Language Model Tokenizers Introduce Unfairness" (Petrov et al.)
- **AG-BPE**: Recent HuggingFace blog + papers on semantic tokenization
- **BPE Suboptimal**: "BPE is Suboptimal for LM Pretraining" (work on tokenization impact)

