# Roadmap

## Milestone 0 — Lab bootstrap (you are here)

**Goal:** Have a usable *lab shell* with docs, basic data, and a toy playground.

### Deliverables

- Repo + docs structure (this folder).

- `data/`:

  - Small curated Hindi/Sanskrit examples (`curated_examples.jsonl`).

  - 2–3 tiny eval sets per language (news, literature/scripture, conversational).

- `scripts/compare_tokenizers.py`:

  - Can call:

    - TikToken / OpenAI encoder for a generic model.

    - HF tokenizers for `ai4bharat/indic-bert` and `ai4bharat/IndicBART`.

- `playground/`:

  - Next.js app with mocked tokenizers (word / char / naive BPE) so UI is real.

---

## Milestone 1 — Devanagari-aware baseline

**Goal:** A **script-aware tokenizer** that beats generic BPE on basic metrics, using only Unicode + regex + light heuristics.

### Work items

1. **Unicode + grapheme layer**

   - Implement robust Devanagari grapheme segmentation:

     - Handle consonant + virāma + consonant clusters.

     - Handle dependent vowels and diacritics.

   - Implement "grapheme violation" detection: any token that splits a grapheme is flagged.

2. **Rule-based pretokenizer**

   - Basic Hindi rules:

     - Split on whitespace + punctuation.

     - Preserve URLs, email, numbers, hashtags as atomic tokens.

     - Lightweight normalization (NFC/NFKC where safe).

3. **BPE training baseline**

   - Train a SentencePiece BPE model on:

     - AI4Bharat IndicNLP Hindi corpus + selected Sanskrit corpora (if licensing OK).

   - Use Unicode-aware pretokenizer as input.

4. **Evaluation**

   - Compare vs:

     - TikToken / GPT-style tokenizer.

     - IndicBERT tokenizer.

   - Metrics:

     - Fertility, chars/token, grapheme-break rate, simple downstream proxy (e.g., masked LM perplexity on small model).

---

## Milestone 2 — Morphology-aware Hindi tokenizer (MorphTok++)

**Goal:** A **morphology-aware Hindi tokenizer** that matches or exceeds MorphTok's reported fertility and downstream gains, with fully reproducible pipeline and open scripts.

### Work items

1. **Morphology + sandhi pre-segmentation**

   - Integrate or re-implement core ideas:

     - MorphTok's segmentation data/heuristics if accessible.

     - Hindi morphological analyzers / lexicons where licensed.

   - Extend to handle:

     - Social media Hindi (code-mixed, dialectal, colloquial), as identified in Indic tokenization surveys.

2. **Constrained BPE**

   - Implement Constrained BPE (CBPE) where merges cannot violate:

     - Grapheme boundaries.

     - Certain script-specific patterns (e.g. independent vs dependent vowels).

3. **EvalTok-style evaluation**

   - Implement a human evaluation protocol similar to EvalTok:

     - Annotators rate segmentation quality for sample phrases.

   - Build annotation templates + instructions in this repo.

4. **Downstream experiments**

   - Choose at least one downstream task:

     - Hindi MT (IndicTrans2) or Hindi classification / QA.

   - Fine-tune small models with different tokenizers, compare performance.

---

## Milestone 3 — Sanskrit + advanced morphology

**Goal:** Add **Sanskrit** as a first-class language, with sandhi-aware pre-segmentation and integration of existing analyzers.

### Work items

1. **Sanskrit resources integration**

   - Survey tools like:

     - Saṃsādhanī toolset (sandhi splitter, morphological analyzer).

     - LREC sandhi benchmark corpora.

   - Select resources that are:

     - High-quality.

     - Licensing compatible.

     - Script-native (avoid transliteration-only tools where possible).

2. **Sanskrit sandhi pretokenizer**

   - Implement pre-segmentation pipeline that:

     - Uses sandhi splitting for long compounds when high confidence.

     - Falls back to character/grapheme splits when ambiguous.

   - Capture multiple candidate segmentations for potential future use.

3. **Sanskrit-aware CBPE**

   - Train BPE on Sanskrit corpora with sandhi-aware segmentation as input.

4. **Evaluation**

   - Metrics similar to Hindi + specialized sandhi metrics:

     - % compounds correctly split.

     - Error distribution across sandhi types (vowel, consonant, visarga).

---

## Milestone 4 — Lexicon and ecosystem integration

**Goal:** Turn the tokenizer into a **lexicon-ready engine** and wire it into major LLM stacks.

### Work items

- Design lexicon schema (lemmas, POS, features).

- Link tokenization to lexicon entries (where available).

- Produce export formats:

  - HF tokenizer JSON.

  - SentencePiece model.

  - Compact JSON schema for use in custom LLM stacks.

- Write integration guides:

  - HuggingFace.

  - Custom PyTorch/Transformers.

  - Any OpenAI-style API gateway.

---

## Milestone 5 — Nice-to-have

- Add support for Marathi & Nepali (*still Devanagari*, but with language-specific quirks).

- Provide unified benchmarks across Devanagari languages.

- Experiment with **SCRIPT-BPE / structured encoding** approaches to further reduce bias/breakage.

