# Indic Tokenization Lab — Vision & Scope

## 0. One-liner

**Build the best open tokenizer stack for Devanagari (Hindi + Sanskrit first), with morphology-aware segmentation, script-aware constraints, and a transparent evaluation lab.**

We want something that:

- Respects **akshara / grapheme clusters** and Devanagari script rules.

- Incorporates **morphology and sandhi** instead of ignoring them.

- Is easy to **plug into LLMs** (HF, custom, cloud) and to **inspect visually**.

- Can grow into a **full lexicon + morphological layer** over time.

---

## 1. Why this lab exists (problem statement)

Modern LLM tokenizers (BPE, WordPiece, SentencePiece) work well for English and other Latin scripts, but struggle with Devanagari:

- **Misaligned splits**: BPE often cuts through dependent vowels/matras and conjuncts (क् + ष + ् etc.), creating tokens that are not meaningful units for Hindi or Sanskrit.

- **High fertility**: Many Devanagari words become 2–4× more tokens than comparable English words, inflating context length costs.

- **No morphology**: Rich inflection and compounding (especially sandhi in Sanskrit) are ignored; models learn brittle patterns over badly segmented strings.

- **Mixed-script and code-mixed reality**: Hindi and Marathi texts often mix English words, numerals, hashtags, and Romanized Hindi, which many tokenizers handle poorly.

Recent work like **MorphTok** shows that **morphology-aware pre-segmentation + Constrained BPE (CBPE)** improves fertility and downstream performance for Hindi/Marathi MT and LM tasks, and introduces a human evaluation metric, **EvalTok**, for segmentation.

But:

- It's not yet a **drop-in lab** you can use as a productized toolkit.

- Sanskrit isn't fully in-scope.

- There is no unified **playground + benchmark** that brings together MorphTok, AI4Bharat tokenizers, Krutrim, custom Devanagari tokenizers, etc., in one place.

This lab exists to fill that gap.

---

## 2. What "success" looks like

By the time v1 of this project is "done", we should have:

1. **Tokenizer stack**

   - A **morphology-aware pretokenizer** for Hindi (and later Sanskrit), including sandhi-aware splitting methods.

   - A **Constrained BPE / SCRIPT-aware BPE** variant that respects Devanagari grapheme boundaries and dependent vowels.

   - Exported vocab + merges that can be used with:

     - HuggingFace `tokenizers`

     - HF Transformers models

     - Custom LLM stacks via JSON/CLI API

2. **Evaluation lab**

   - Public **eval corpora** for Hindi & Sanskrit:

     - Domain-tagged: news, web, social, literature, scripture.

     - **Curated failure sets** demonstrating typical tokenization problems.

   - Metrics + scripts:

     - Fertility, chars/token, grapheme-break violations

     - Stability under small edits

     - Human-in-the-loop evaluation forms (EvalTok-style).

3. **Playground**

   - A **web UI** where:

     - Anyone can paste text and compare several tokenizers (OpenAI TikToken, HF models like IndicBERT/IndicBART, AI4Bharat/IndicNLP, MorphTok-style, Krutrim tokenizer if exposed).

     - Tokens are visualized as chips, with grapheme boundary hints and metrics per tokenizer.

4. **Path to lexicon**

   - Data structures and docs that make it easy to bolt on:

     - **Lexicon entries** (lemmas, POS, morphological features).

     - **Morphological analyzers and generators**, using existing resources where licensing permits.

---

## 3. Philosophy

- **Language-first, not model-first.** We start from Devanagari + Hindi/Sanskrit grammar and script constraints, then design tokenization that respects them.

- **Composable layers.**

  - Script-aware char + grapheme layer.

  - Morphology / sandhi layer.

  - Subword/BPE layer.

- **Open, inspectable, hackable.**

  - All steps are documented, versioned, and exposed via simple APIs.

- **Not reinventing the wheel.**

  - We build *on top of* AI4Bharat's IndicNLP tools, MorphTok research, sandhi corpora, and existing Sanskrit morphology/lexicon projects, where licenses allow.

---

## 4. Initial scope boundaries

In v0–v1 we will **intentionally** not:

- Solve all Indic scripts at once. Focus: **Devanagari**, with groundwork that generalizes.

- Ship our own huge Hindi LLM. We focus on tokenization + evaluation, then integrate with existing LLMs.

- Replace AI4Bharat, Krutrim, or others; instead we offer a **neutral lab** *plus* a strong tokenizer that they can optionally adopt or compare against.

---

## 5. Intended users

- **Model builders** targeting India (OpenAI, AI4Bharat, Ola Krutrim, independent teams).

- **Academics** in NLP and Indic linguistics.

- **Open-source hackers** who want better Devanagari support in their own models.

- **People who just want to see "how badly is my Hindi getting chopped up?"**

---

## 6. Relationship to other projects

- **MorphTok**

  - We treat MorphTok as a core prior: morphology-aware segmentation + Constrained BPE. We will:

    - Reproduce its core ideas.

    - Extend to Sanskrit.

    - Expose its metrics and human evaluation ideas in our lab.

- **AI4Bharat / IndicNLP, IndicBERT, IndicBART, IndicTrans**

  - We integrate their corpora, tokenizers, and evaluation benchmarks wherever licensing allows, and always benchmark against them.

- **Krutrim Tokenizer / other Indic-optimized tokenizers**

  - We treat them as fellow participants on the benchmark. For closed-source systems, we document black-box behavior if accessible via API.

---

## 7. Versioning and roadmap

See `01-roadmap.md` for milestones (v0 lab, v1 tokenizer, v2 lexicon).

