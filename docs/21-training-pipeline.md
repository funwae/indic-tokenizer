# Training Pipeline

This document describes how we will train, validate, and iterate on the tokenizer.

---

## 1. Data flow overview

1. **Collect corpora**:

   - Hindi: AI4Bharat IndicNLP corpora, Hindi news, Wikipedia, social media (where licensing permits).

   - Sanskrit: classic texts with sandhi annotations where possible.

2. **Normalize & clean**:

   - Unicode normalization (NFC/NFKC).

   - Remove non-textual noise; handle HTML, markup.

3. **L0 segmentation**:

   - Split into graphemes and basic tokens (whitespace/punctuation).

4. **L1 segmentation**:

   - Apply morphology-aware pre-tokenization:

     - Hindi: affix lists, heuristics, optionally ML models.

     - Sanskrit: sandhi splitting tools + rules.

5. **L2 subword training**:

   - Train BPE/Unigram with CBPE constraints.

6. **Evaluation**:

   - Compute intrinsic metrics (fertility, grapheme violations).

   - Train small LMs/MT models to test downstream impact.

---

## 2. Corpora selection

### 2.1 Hindi

Sources:

- AI4Bharat IndicNLP Hindi corpus + frequency lists.

- Wikipedia dumps (open).

- Selected news corpora.

- Social media samples (carefully anonymized / license-checked).

We stratify by domain to:

- Maintain a **balanced vocabulary** (similar to Krutrim's approach), so the tokenizer doesn't overfit to one domain.

### 2.2 Sanskrit

Sources:

- Sandhi benchmark corpora.

- Classical corpora with morphology annotations when available.

---

## 3. Morphology-aware pre-tokenization

### 3.1 Hindi morphology layer

Steps:

1. **Lexicon mining**:

   - Use frequency lists to extract candidate stems and affixes.

   - Consult IndicNLP tools and Hindi morphology resources.

2. **Rule-based segmentation**:

   - Implement a set of deterministic heuristics:

     - Split at known suffixes if stem is frequent.

     - Avoid splitting inside named entities.

3. **ML refinement (optional v2)**:

   - Train a small model to predict morpheme boundaries given Devanagari sequences, similar in spirit to MorphTok's approach.

### 3.2 Sanskrit sandhi layer

Steps:

1. Integrate sandhi splitting tools (Saṃsādhanī, others) as APIs or CLI.

2. Use sandhi benchmark corpora to:

   - Tune thresholds for when to split vs keep fused.

---

## 4. Constrained BPE training

We implement CBPE as:

1. **Base algorithm**:

   - Start from standard BPE / SentencePiece training.

2. **Constraint hook**:

   - On each merge candidate, call a `is_merge_allowed(left, right)` function that checks:

     - Grapheme boundary constraints.

     - Script-specific rules.

     - Morphology boundaries (optional).

We test variants:

- **CBPE-G**: Only grapheme-aware.

- **CBPE-GM**: Grapheme + morphology constraints.

- **CBPE-GMS**: Grapheme + morphology + sandhi (Sanskrit).

---

## Phase 1.5 — Grapheme Pair Encoding (GPE) Prototype

**Status**: ✅ Implemented

We have implemented a **Grapheme Pair Encoding (GPE)** prototype that trains BPE over Unicode grapheme clusters instead of bytes/codepoints. This is a concrete implementation of the L0 grapheme layer + CBPE approach.

**Key Components**:

- `tokenizers/grapheme_segmenter.py`: Unicode grapheme clustering using `regex` with `\X` (UAX #29)
- `scripts/train_gpe_tokenizer.py`: Home-grown GPE+BPE trainer with `cbpe_merge_allowed` constraints
- `tokenizers/gpe_tokenizer.py`: GPE tokenizer adapter for loading and using trained models

**Usage**:

```bash
# Train a GPE tokenizer
python scripts/train_gpe_tokenizer.py \
  --input data/hindi/corpus.txt \
  --output-dir models/gpe_hi_v0 \
  --vocab-size 32000 \
  --min-pair-frequency 2

# Compare with other tokenizers
python scripts/compare_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert,gpe_hi_v0
```

**See**: `docs/31-gpe-prototype-plan.md` for detailed documentation on the GPE implementation, rationale, and future extensions.

---

## 5. Training environment (Ubuntu + CUDA)

Even though tokenizer training is mostly CPU, downstream LM experiments use GPU:

- Recommended stack:

  - Ubuntu 22.04, CUDA 12.x, latest PyTorch.

  - HuggingFace Transformers for LM training.

We will:

- Provide scripts in `scripts/` and `eval/`:

  - `train_lm_hindi_baseline.py`

  - `train_lm_hindi_cbpe.py`

  - `evaluate_lm_perplexity.py`

Each script will:

- Log tokenizer config used.

- Output metrics in JSON for comparison.

---

## 6. Iteration loop

For each tokenizer variant:

1. Train on same corpora.

2. Collect intrinsic metrics.

3. Train small LMs or MT models.

4. Compare:

   - Fertility vs perplexity vs human EvalTok scores.

5. Promote **Pareto-optimal** designs (good tradeoff between token count, quality, complexity).

