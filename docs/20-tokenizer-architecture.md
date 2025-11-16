# Tokenizer Architecture

This document defines the architecture of the Indic Tokenization Lab tokenizer stack.

---

## 1. Layered design

We treat tokenization as **three stacked layers**:

1. **Script layer** (L0)

   - Unicode normalization.

   - Devanagari grapheme segmentation.

   - Script-aware regex for punctuation, URLs, numbers, etc.

   - Hard constraint: do not split grapheme clusters.

2. **Morphology layer** (L1)

   - Language-specific segmentation:

     - Hindi: morphology + light sandhi splitting.

     - Sanskrit: sandhi splitting + morphological hints.

   - Uses rule-based + ML methods.

   - May produce:

     - Single best segmentation.

     - Optional N-best segmentations (for research).

3. **Subword layer** (L2)

   - Constrained BPE (CBPE) or similar algorithm operating on L1 units.

   - Enforces:

     - No merges that violate grapheme boundaries.

     - Script-aware constraints for dependent vowels, virāma, etc.

   - Produces final subword tokens and IDs.

This aligns with MorphTok's idea: **morphology-aware pretokenization + Constrained BPE**, but extended with a stricter script layer and multi-language support.

---

## 2. Core data structures

### 2.1 Internal representation

We define a `Segment` type (conceptual):

```
Segment:
  id: string           # stable ID within a text
  text: string         # surface text
  lang: string         # 'hi', 'sa', etc.
  script: string       # 'Deva'
  layer: 'L0' | 'L1' | 'L2'
  meta:
    type: 'grapheme' | 'morpheme' | 'token'
    start_char: int
    end_char: int
    morph_features?: ...
    sandhi_info?: ...
```

The pipeline transforms:

```
raw text
  → [L0 graphemes]
  → [L1 morpheme segments]
  → [L2 subword tokens]
```

### 2.2 Tokenizer interface

High-level interface:

```python
class IndicTokenizer:
    def __init__(self, config): ...
    def tokenize(self, text: str, lang: str = "hi") -> List[str]:
        ...
    def encode(self, text: str, lang: str = "hi") -> List[int]:
        ...
    def debug(self, text: str, lang: str = "hi") -> Dict:
        # returns full L0/L1/L2 trace for visualization
```

---

## 3. Constrained BPE (CBPE)

### 3.1 Motivation

Standard BPE greedily merges frequent symbol pairs without regard for script rules, which can break Devanagari dependent vowels and grapheme clusters.

MorphTok introduces CBPE to enforce **script constraints** during merging, particularly for dependent vowels.

### 3.2 Our CBPE constraints (initial proposal)

When considering a merge of units `X Y`:

* Reject the merge if:

  * It would cause a token boundary inside a grapheme cluster.

  * `X` ends with a virāma and `Y` begins with a dependent vowel (illegal pair).

  * `X` contains certain markers we want to keep atomic (URLs, handles).

* Optionally, penalize merges that:

  * Cross morpheme boundaries flagged as "strong".

We will document the exact constraint set in `21-training-pipeline.md` once we've run empirical tests.

---

## 4. Alternate subword schemes

While BPE is the main target (for compatibility with existing LLMs), we aim to support:

* **Unigram LM (SentencePiece)** style subwords.

* **SCRIPT-BPE / structured encodings** that encode structure at the character level.

The architecture should allow:

* Plugging in different subword learners at L2.

* Keeping L0/L1 unchanged.

---

## 5. Black-box integration of external tokenizers

The lab also supports "external" tokenizers as black boxes:

* OpenAI TikToken / GPT-style tokenizers.

* AI4Bharat tokenizers for IndicBERT / IndicBART.

* Krutrim tokenizer API if accessible.

We wrap them into a common interface for:

* CLI comparisons (see `scripts/compare_tokenizers.py`).

* Playground visualization.

* Evaluation scripts in `eval/`.

---

## 6. GPU / CUDA considerations

While tokenization itself is CPU-friendly, **training models that depend on it** (e.g. small Hindi/Sanskrit LMs for downstream eval) will use your CUDA stack:

* PyTorch + HF Transformers, running on your local GPU.

* Small, fast experiments:

  * DistilBERT-style models trained on Devanagari corpora.

We keep **tokenizer training and evaluation scripts** GPU-optional, but document recommended configs for a single-GPU Ubuntu + CUDA box (your setup).

