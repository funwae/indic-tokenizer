# Downstream Benchmark Results

**Status**: Pending evaluation
**Task**: Hindi-English Machine Translation (or Text Classification)
**Tokenizers**: Baseline (GPE+CBPE), Semantic (AG-BPE), Reference (mBERT/IndicBERT)

---

## Task Description

### Machine Translation (Selected)

**Dataset**: IITB Hindi-English Parallel Corpus (subset)
**Direction**: Hindi → English
**Model**: Small encoder-decoder transformer (~2-4M parameters)
**Evaluation Metric**: BLEU score

### Alternative: Text Classification

**Dataset**: Hindi text classification dataset
**Task**: Sentiment analysis or topic classification
**Model**: Small encoder-only transformer
**Evaluation Metric**: Accuracy, F1 score

---

## Results

*(Results will be populated after running downstream evaluation)*

### Machine Translation (BLEU)

| Tokenizer | BLEU Score | Delta vs Baseline |
|-----------|------------|-------------------|
| Baseline (GPE+CBPE) | TBD | - |
| Semantic (AG-BPE) | TBD | TBD |
| mBERT | TBD | TBD |
| IndicBERT | TBD | TBD |

### Text Classification (if used)

| Tokenizer | Accuracy | F1 Score | Delta vs Baseline |
|-----------|----------|----------|-------------------|
| Baseline | TBD | TBD | - |
| Semantic | TBD | TBD | TBD |

---

## Analysis

*(Analysis will be added after evaluation)*

### Key Findings

- Tokenization impact on downstream performance
- Correlation between intrinsic metrics and downstream metrics
- Trade-offs between efficiency and task performance

---

## Next Steps

1. Prepare dataset (train/dev/test split)
2. Implement small seq2seq model for MT
3. Train models with each tokenizer
4. Evaluate and compare results
5. Document findings

---

## References

- IITB Parallel Corpus: http://www.cfilt.iitb.ac.in/iitb_parallel/
- BLEU: "BLEU: a Method for Automatic Evaluation of Machine Translation"
- Evaluation Framework: `docs/22-evaluation-metrics.md`

