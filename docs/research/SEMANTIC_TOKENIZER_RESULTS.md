# Semantic Tokenizer Results

**Status**: Pending evaluation
**Tokenizer**: `ag_bpe_hi_v1` (Attention-Guided BPE)
**Baseline**: `gpe_cbpe_hi_v1` (GPE+CBPE)

---

## Methodology

### Attention-Guided BPE (AG-BPE)

The AG-BPE tokenizer combines three signals to guide BPE merge decisions:

1. **Pair Frequency** (standard BPE): How often pairs occur in corpus
2. **Mutual Information**: Statistical association between tokens
3. **Attention Patterns** (if LM available): Semantic association from language model

**Scoring Function**:
```
score(pair) = w_freq * freq_score + w_mi * mi_score + w_attn * attn_score
```

Where:
- `w_freq` = frequency weight (default: 0.2)
- `w_mi` = mutual information weight (default: 0.3)
- `w_attn` = attention weight (default: 0.5)

---

## Success Criteria

For AG-BPE to be considered successful:

### 1. Script & Morphology
- ✅ Grapheme violation ≤ baseline (0%)
- ✅ Akshara integrity ≥ baseline (100%)
- ✅ Morphology F1 ≥ baseline (0.469)

### 2. Efficiency
- ✅ Fertility < baseline
- ✅ NSL < baseline
- ✅ Token tax < baseline

### 3. Downstream
- ✅ Perplexity ≤ baseline

---

## Evaluation Results

**Status**: Pending - AG-BPE tokenizer training and evaluation not yet completed.

**Next Steps**:
1. Train `ag_bpe_hi_v1` tokenizer using `scripts/train_ag_bpe_tokenizer.py`
2. Run comprehensive evaluation using `scripts/run_semantic_evaluation.py`
3. Compare with baseline using `scripts/compare_baseline_semantic.py`
4. Fill in results below with actual numbers

*(Results will be populated after running comprehensive evaluation)*

### Efficiency & Script Metrics

| Metric | Baseline (GPE+CBPE) | Semantic (AG-BPE) | Delta |
|--------|---------------------|-------------------|-------|
| Fertility | TBD | TBD | TBD |
| Chars/Token | TBD | TBD | TBD |
| Grapheme Violation | 0% | TBD | TBD |
| Akshara Integrity | 100% | TBD | TBD |
| Morphology F1 | 0.469 | TBD | TBD |

### Tokenization Parity

| Metric | Baseline | Semantic | Delta |
|--------|----------|----------|-------|
| TP (mean) | TBD | TBD | TBD |
| Token Tax | TBD | TBD | TBD |

### Tiny LM Perplexity

| Tokenizer | Perplexity | Delta |
|-----------|------------|-------|
| Baseline | TBD | - |
| Semantic | TBD | TBD |

---

## Analysis

*(Analysis will be added after evaluation)*

---

## Novel Contributions

If success criteria are met, this demonstrates:

1. **Semantic tokenization for Indic languages**: First application of attention-guided tokenization to Hindi
2. **Improved efficiency**: Better fertility/NSL than baseline while maintaining script awareness
3. **Better morphology**: Improved morphological alignment through semantic guidance
4. **Fair evaluation framework**: Comprehensive metrics comparing against frontier tokenizers

---

## Next Steps

1. Train AG-BPE tokenizer on full corpus
2. Run comprehensive evaluation
3. Compare with baseline and document results
4. If successful, proceed to downstream benchmark

---

## References

- AG-BPE: "Attention-Guided BPE for Subword Tokenization"
- Baseline: `docs/research/BASELINE_RESULTS.md`
- Evaluation Framework: `docs/22-evaluation-metrics.md`

