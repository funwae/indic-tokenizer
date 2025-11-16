# Evaluation Metrics

We evaluate tokenizers on **three axes**:

1. **Intrinsic** (segmentation quality, efficiency).

2. **Human judgment**.

3. **Downstream model impact**.

---

## Phase 1: Comprehensive Metrics (Implemented)

Phase 1 implements efficiency and script adequacy metrics that can be computed immediately without parallel corpora or annotated morphology data.

### 1. Efficiency Metrics

#### 1.1 Fertility

- **Definition:** `num_tokens / num_words` (words defined via whitespace + script-aware splitting).
- **Interpretation:** Lower is generally better, but not at the cost of mangling graphemes or morphology.
- **References:** Used in MorphTok, Dzongkha study, and Ukrainian tokenization papers.

#### 1.2 Chars per Token

- **Definition:** `num_chars / num_tokens`.
- **Interpretation:** Higher is better (more content per token).
- **Measures:** Packing efficiency.

#### 1.3 Compression Ratio (Chars)

- **Definition:** `chars / tokens`.
- **Interpretation:** Higher is better (fewer tokens per same text).
- **References:** From GPE paper.

#### 1.4 Compression Ratio (Graphemes)

- **Definition:** `graphemes / tokens`.
- **Interpretation:** Higher is better (more graphemes encoded per token).
- **Notes:** For Devanagari, grapheme-based CR is more linguistically meaningful than char-based.

#### 1.5 Normalized Sequence Length (NSL)

- **Definition:** `|t(s)| / |t0(s)|` where t0 is baseline tokenizer.
- **Interpretation:**
  - NSL < 1: more efficient than baseline
  - NSL = 1: same efficiency
  - NSL > 1: less efficient than baseline
- **Baseline:** Configurable per benchmark (default: GPT-4o once available, temporary: mBERT).
- **References:** From Dzongkha paper.

#### 1.6 Proportion of Continued Words (PCW)

- **Definition:** `(words split into ≥2 tokens) / (total words)`.
- **Interpretation:** Lower is better (fewer fragmented words).
- **References:** From Dzongkha paper.

#### 1.7 UNK Rate

- **Definition:** `(number of <unk> tokens) / (total tokens)`.
- **Interpretation:** Lower is better (fewer unknown tokens).
- **Notes:** Important for smaller vocabularies or domain-specific lexica.

### 2. Script Adequacy Metrics

#### 2.1 Grapheme Violation Rate

- **Definition:** Percent of token boundaries that fall inside a Unicode grapheme cluster.
- **Interpretation:** Lower is better (ideal = 0.0).
- **Computation:** Uses L0 grapheme segmentation (UAX #29).
- **References:** Core metric from GPE and MorphTok.

#### 2.2 Akshara Integrity Rate

- **Definition:** `1 - (split aksharas / total aksharas)`.
- **Interpretation:** Higher is better (ideal = 1.0).
- **Segmentation:** Uses v0 heuristic (consonant+virama+vowel patterns).
- **Notes:** Critical for Devanagari - a tokenizer that regularly breaks aksharas is linguistically wrong.
- **Version:** Marked as `v0_heuristic` - may be refined with proper linguistic review.

#### 2.3 Dependent Vowel Split Rate

- **Definition:** Rate of dependent vowel separations from base consonants.
- **Interpretation:** Lower is better (fewer separations).
- **Notes:** This is exactly the class of errors CBPE constrains against.

#### 2.4 Grapheme-Aligned Token Rate

- **Definition:** Fraction of tokens that align with grapheme boundaries.
- **Breakdown:**
  - Single grapheme tokens: tokens = exactly one grapheme
  - Multi grapheme tokens: tokens = multiple complete graphemes
  - Fragment tokens: tokens that split graphemes (violations)
- **Interpretation:** Higher aligned rate is better.

#### 2.5 Devanagari Token Share

- **Definition:** `% tokens that are pure Devanagari`.
- **Interpretation:** Higher is better for Hindi/Sanskrit corpora.
- **Notes:** Measures script purity.

#### 2.6 Mixed Script Token Share

- **Definition:** `% tokens mixing scripts` (e.g., Devanagari + Latin).
- **Interpretation:** Context-dependent - may be acceptable for code-mixed text.
- **Notes:** Useful for analyzing code-mixed corpora.

---

## Phase 2: Fairness & Morphology Metrics (Planned)

### 3. Fairness Metrics (Phase 2)

**Status:** Deferred - requires parallel corpora and baseline tokenizer setup.

#### 3.1 Tokenization Parity

- **Definition:** `|t(s_A)| / |t(s_B)|` for same content in languages A and B.
- **Interpretation:** TP ≈ 1 for fairness.
- **References:** From Petrov et al. and GPE paper.

#### 3.2 Tokenization Premium

- **Definition:** `E[|t(s_lang)|] / E[|t(s_en)|]` - how many more tokens a language pays vs English.
- **Interpretation:** Lower premium is better (fairer tokenization).
- **References:** From Petrov et al. - ties directly to cost, latency, and context length.

#### 3.3 Compression Ratio Disparity

- **Definition:** `ΔCR(lang1, lang2)` - difference in compression ratios.
- **Interpretation:** Lower disparity is better.
- **Goal:** Minimize disparity between Hindi and English while maintaining good Hindi morphology.

### 4. Morphology Metrics (Phase 2)

**Status:** Deferred - requires MorphTok or similar annotated morphology dataset.

#### 4.1 Morphology Alignment (Hindi)

- **Definition:** When morphology annotations are available:
  - Boundary precision/recall/F1 over morpheme boundaries
  - % of tokens that match exactly one morpheme
- **References:** Inspired by EvalTok and MorphTok.

#### 4.2 Sandhi Alignment (Sanskrit)

- **Definition:** For sandhi benchmark corpora:
  - % of sandhi splits that align with token boundaries
  - Error analysis by sandhi type (vowel vs consonant vs visarga)
- **References:** LREC sandhi benchmark tools.

---

## Legacy Metrics (Backward Compatibility)

The following metrics are still available for backward compatibility but are superseded by comprehensive metrics:

### 1.3 Grapheme violations (Legacy)

- Percent of token boundaries that fall inside a Devanagari grapheme cluster.
- Now part of script metrics (grapheme violation rate).

---

## 2. Human evaluation (EvalTok-style)

We adapt the **EvalTok** idea to our setting:

- For a subset of texts, we present human annotators with:

  - Original text.

  - Tokenization from multiple tokenizers.

- Annotators rate:

  - **Linguistic plausibility** of boundaries (1–5).

  - **Ease of reading/editing** tokens (1–5).

  - **Suitability for downstream tasks** (qualitative).

We provide:

- Annotation guidelines.

- Templates.

- Consolidation scripts.

References: MorphTok introduced EvalTok specifically for evaluating tokenization qualitatively in Hindi/Marathi.

---

## 3. Downstream metrics

We run **controlled experiments** where only the tokenizer changes:

- Tasks:

  - Hindi LM perplexity (small masked LM or causal LM).

  - Hindi MT (IndicTrans2 baseline) where feasible.

  - Optionally: classification / QA tasks (e.g. Airavata benchmarks).

Metrics:

- Perplexity / BLEU / chrF, etc.

- Parameter count and compute cost (tokens processed).

---

## 4. Benchmark Script

The `scripts/run_benchmark.py` script provides a convenient way to run comprehensive evaluation on multiple tokenizers:

```bash
python scripts/run_benchmark.py \
    --corpus data/hindi/eval_sets/news_headlines.txt \
    --tokenizers indicbert,mbert,gpe_hi_v0 \
    --lang hi \
    --baseline-tokenizer mbert \
    --output scorecards/benchmark_news.json
```

**Features:**
- Loads corpus from file (one text per line)
- Evaluates multiple tokenizers from registry
- Computes all Phase 1 metrics (efficiency + script)
- Supports baseline tokenizer for NSL computation
- Generates JSON output with aggregated metrics

**Output format:**
- Per-tokenizer metrics (efficiency, script, summary)
- Aggregated statistics across corpus
- Metadata (corpus, language, baseline, timestamp)

## 5. Scorecards

We produce **public scorecards**:

- Per tokenizer:

  - Intrinsic metrics (tables) - Phase 1 comprehensive metrics
  - Human ratings (Phase 2)
  - Downstream performance deltas vs a baseline tokenizer (Phase 2)

**Scorecard generation:**
- Use `eval/metrics.py` functions: `evaluate_tokenizer()`, `generate_scorecard()`, `export_scorecard()`
- Supports both legacy `Metrics` format and new `ComprehensiveMetrics` format
- Export formats: JSON and Markdown

**Example usage:**
```python
from eval.metrics import evaluate_tokenizer, generate_scorecard, export_scorecard

# Evaluate with comprehensive metrics (Phase 1)
metrics = evaluate_tokenizer(text, tokenizer, use_comprehensive=True)

# Generate scorecard
scorecards = generate_scorecard({"tokenizer_id": metrics}, tokenizer_names={"tokenizer_id": "Tokenizer Name"})

# Export as Markdown
markdown = export_scorecard(scorecards, format="markdown")
```

This makes it easy for an external team (OpenAI, AI4Bharat, etc.) to see:

> "If we switch our Hindi/Sanskrit tokenizer to this, we get X% fewer tokens and Y improvement on benchmarks, without breaking script rules."

