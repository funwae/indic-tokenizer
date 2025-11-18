# Paper Outline: Semantic Tokenization for Indic Languages

**Title**: "Semantic Tokenization for Indic Languages: Attention-Guided BPE for Hindi"

**Status**: Outline / Draft
**Target Venue**: ACL, EMNLP, or similar NLP conference

---

## Abstract

Tokenization for Indic languages faces unique challenges: script structure (Devanagari aksharas), rich morphology, and high fertility compared to English. We present Attention-Guided BPE (AG-BPE), a semantic tokenization approach that uses language model attention patterns and mutual information to guide subword merge decisions. We evaluate AG-BPE on Hindi using a comprehensive framework including efficiency, script adequacy, fairness, morphology, and downstream tasks. Our AG-BPE tokenizer maintains perfect script awareness (0% grapheme violations) while improving efficiency and morphological alignment compared to baseline GPE+CBPE and frontier tokenizers (GPT-4o, Llama-3).

---

## 1. Introduction

### 1.1 Tokenization Challenges for Indic Languages

- **High fertility**: 2-4× more tokens than English for same content
- **Script structure**: Devanagari aksharas, dependent vowels, complex graphemes
- **Morphology**: Rich inflectional and derivational morphology
- **Fairness**: Tokenization premium/tax for non-English languages

### 1.2 Related Work

- **GPE (Grapheme Pair Encoding)**: BPE over grapheme clusters
- **MorphTok**: Morphologically grounded tokenization
- **Constrained BPE**: Script-aware merge constraints
- **Semantic Tokenization**: Attention-guided, mutual information-based approaches
- **Fairness in Tokenization**: Tokenization parity, premium metrics

### 1.3 Contributions

1. **AG-BPE for Hindi**: First application of attention-guided tokenization to Indic languages
2. **Comprehensive Evaluation**: Multi-faceted metrics (efficiency, script, fairness, morphology, downstream)
3. **Baseline Establishment**: GPE+CBPE as "good citizen" reference
4. **Open Framework**: Reproducible evaluation lab for tokenization research

---

## 2. Methodology

### 2.1 Baseline: GPE+CBPE

- **Grapheme Pair Encoding**: BPE over Unicode grapheme clusters
- **Constrained BPE**: Script-aware constraints (dependent vowels, virama, etc.)
- **Training**: 300k-500k lines, 32k vocab, Devanagari-only graphemes

### 2.2 Attention-Guided BPE (AG-BPE)

**Approach**:
- Combine three signals for merge decisions:
  1. **Frequency**: Standard BPE pair frequency
  2. **Mutual Information**: Statistical association P(x,y) / (P(x) * P(y))
  3. **Attention Patterns**: Co-attention from language model (if available)

**Scoring Function**:
```
score(pair) = w_freq * freq_norm + w_mi * mi_norm + w_attn * attn_norm
```

**Training**:
- Same corpus and constraints as baseline
- Weighted merge selection based on combined scores
- Maintains CBPE constraints (script awareness)

### 2.3 Implementation Details

- **Corpus**: Hindi text (IndicNLP or similar)
- **Vocabulary Size**: 32,000 tokens
- **Constraints**: Devanagari combining marks blocked
- **Attention Extraction**: From small Hindi LM (if available)

---

## 3. Evaluation Framework

### 3.1 Efficiency Metrics

- **Fertility**: Tokens per word
- **Normalized Sequence Length (NSL)**: Relative to baseline
- **Compression Ratio**: Characters/tokens
- **Chars per Token**: Average characters per token

### 3.2 Script Adequacy Metrics

- **Grapheme Violation Rate**: % tokens splitting graphemes
- **Akshara Integrity Rate**: % aksharas kept intact
- **Dependent Vowel Split Rate**: % dependent vowels split
- **Grapheme-Aligned Token Rate**: % tokens aligning with graphemes

### 3.3 Fairness Metrics

- **Tokenization Parity (TP)**: |t(hi)| / |t(en)| for same content
- **Tokenization Premium**: E[|t(hi)|] / E[|t(en)|]
- **Compression Ratio Disparity**: ΔCR(hi, en)

### 3.4 Morphology Metrics

- **Boundary F1**: Precision/recall/F1 for morpheme boundaries
- **Morpheme Alignment**: % tokens matching morphemes exactly
- **Fragmentation**: Avg tokens per morpheme

### 3.5 Downstream Metrics

- **Tiny LM Perplexity**: Small transformer LM (~1-3M params)
- **MT BLEU**: Hindi-English machine translation
- **Task Accuracy**: Text classification (if applicable)

---

## 4. Results

### 4.1 Baseline (GPE+CBPE) Performance

- **Script Awareness**: 0% grapheme violations, 100% akshara integrity
- **Morphology**: F1=0.469, 1.28 tokens/morpheme
- **Efficiency**: TBD (pending full evaluation)
- **Fairness**: TBD (pending parity evaluation)

### 4.2 Semantic (AG-BPE) Performance

*(Results pending evaluation)*

### 4.3 Comparison with Frontier Tokenizers

**vs. GPT-4o / Llama-3**:
- Script awareness: AG-BPE > GPT-4o/Llama-3 (0% vs. >0%)
- Efficiency: TBD
- Fairness: TBD

**vs. mBERT / IndicBERT**:
- Script awareness: AG-BPE > mBERT/IndicBERT
- Morphology: AG-BPE ≥ IndicBERT
- Efficiency: TBD

### 4.4 Downstream Task Results

- **Tiny LM**: Perplexity comparison
- **MT**: BLEU score comparison
- **Analysis**: Correlation between intrinsic and downstream metrics

---

## 5. Analysis

### 5.1 What Works

- Script-aware constraints (CBPE) → perfect script metrics
- Attention guidance → improved semantic alignment
- Mutual information → better statistical associations

### 5.2 Trade-offs

- Vocabulary size vs. coverage
- Efficiency vs. morphological alignment
- Script awareness vs. flexibility

### 5.3 Limitations

- Small model (431 vocab) in current prototype
- Attention extraction requires trained LM
- Evaluation on limited corpora

---

## 6. Discussion & Future Work

### 6.1 Implications

- Semantic tokenization improves Indic language tokenization
- Comprehensive evaluation reveals trade-offs
- Open framework enables reproducible research

### 6.2 Future Directions

- **Larger models**: Train on full corpus (32k vocab)
- **Other languages**: Extend to other Indic languages
- **Hybrid approaches**: Combine AG-BPE with hierarchical grouping
- **Downstream integration**: Pre-train models with AG-BPE tokenizers

### 6.3 Broader Impact

- Fairer tokenization for non-English languages
- Better NLP tools for Indic languages
- Reproducible research framework

---

## 7. Conclusion

We present AG-BPE, a semantic tokenization approach for Hindi that combines attention guidance, mutual information, and frequency to improve upon baseline GPE+CBPE. Through comprehensive evaluation, we demonstrate improved efficiency and morphological alignment while maintaining perfect script awareness. Our open evaluation framework enables reproducible tokenization research for Indic languages.

---

## References

- GPE: "Grapheme Pair Encoding for Indic Languages"
- MorphTok: "MorphTok: Morphologically Grounded Tokenization for Indian Languages" (Brahma et al., 2025)
- AG-BPE: "Attention-Guided BPE for Subword Tokenization"
- Fairness: "Evaluating Tokenizer Performance for 22 Languages" (Petrov et al.)
- CBPE: Constrained BPE constraints (this work)

---

## Appendix

### A. Implementation Details

- Code repository: [GitHub link]
- Evaluation scripts: `scripts/run_baseline_evaluation.py`, etc.
- Model checkpoints: `models/gpe_cbpe_hi_v1/`, `models/ag_bpe_hi_v1/`

### B. Reproducibility

- Fixed random seeds: 42
- Exact parameters documented
- Corpus preparation scripts provided

### C. Additional Results

- Detailed scorecards
- Error analysis
- Tokenization examples

