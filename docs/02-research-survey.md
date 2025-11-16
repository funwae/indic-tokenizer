# Research Survey

This document tracks key research papers, tools, and resources relevant to Indic tokenization.

## Key Papers

### MorphTok: Morphologically Grounded Tokenization for Indian Languages
- **Authors**: Brahma et al., 2025
- **arXiv**: https://arxiv.org/abs/2504.10335
- **Key contributions**:
  - Morphology-aware pre-segmentation + Constrained BPE (CBPE)
  - EvalTok: human evaluation metric for segmentation
  - Improvements in fertility and downstream performance for Hindi/Marathi

### Optimized Indic Tokenizers for LLMs
- **arXiv**: https://arxiv.org/html/2407.12481v1
- **Focus**: Tokenizer optimization for Indic languages in LLM contexts

### Structured Encoding for Robust Multilingual Pretokenization
- **arXiv**: https://www.alphaxiv.org/overview/2505.24689v1
- **Focus**: SCRIPT-BPE and structured encoding approaches

## Tools & Libraries

### AI4Bharat
- **IndicNLP Library**: https://indicnlp.ai4bharat.org/
- **IndicBERT**: https://indicnlp.ai4bharat.org/pages/indic-bert/
- **IndicBART**: https://huggingface.co/ai4bharat/IndicBART
- **IndicTrans2**: Translation models for Indic languages

### Krutrim Tokenizer
- **Blog**: https://tech.olakrutrim.com/krutrim-tokenizer/
- **Focus**: Optimized tokenizer for Indic languages

### Saṃsādhanī (Sanskrit Tools)
- **URL**: https://sanskrit.uohyd.ac.in/scl/
- **Focus**: Sandhi splitting, morphological analysis for Sanskrit

## Datasets & Corpora

### LREC Sandhi Benchmark
- Benchmark corpus for evaluating Sanskrit sandhi tools
- Used for evaluating sandhi splitting accuracy

### AI4Bharat Corpora
- Hindi corpora and frequency lists
- Available through IndicNLP resources

## Evaluation Metrics

### EvalTok
- Human evaluation protocol for tokenization quality
- Introduced in MorphTok paper
- Rates linguistic plausibility and ease of reading

### Intrinsic Metrics
- Fertility (tokens/word)
- Characters per token
- Grapheme violation rate
- Morphology alignment

## Open Questions

- How to best handle code-mixed text (Devanagari + Latin)?
- Trade-offs between fertility and segmentation quality
- Best practices for sandhi splitting in production tokenizers
- Generalization to other Indic scripts beyond Devanagari

