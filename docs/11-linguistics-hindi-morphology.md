# Hindi Morphology — Notes for Tokenization

This document captures Hindi-specific morphology patterns that affect tokenization.

---

## 1. Morphological richness

Hindi is morphologically rich:

- Nouns inflect for **case, number, gender**.

- Verbs inflect for **tense, aspect, mood, person, gender**.

- Derivational morphology (affixes forming new words) is common.

MorphTok and several studies on Hindi tokenizers show that tokenization quality strongly affects downstream performance, especially in MT and LM tasks.

---

## 2. Compounding & sandhi (Hindi)

While Sanskrit sandhi is more complex, Hindi also exhibits:

- Compounds (e.g., रेलगाड़ी, आत्मविश्वास).

- Phonological changes at boundaries in formal/literary Hindi.

For our purposes:

- We want to **optionally split** compounds into morphemes for some tasks.

- But we must avoid over-segmenting modern colloquial forms.

---

## 3. High-level tokenization approach

We adopt a **two-stage** view for Hindi:

1. **Morphology-aware pre-segmentation**

   - Identify potential morpheme boundaries:

     - Known prefixes/suffixes.

     - Common compounding patterns.

   - Use rule-based + ML models, guided by prior work like MorphTok.

2. **Subword segmentation (CBPE)**

   - Run Constrained BPE on morphologically segmented units:

     - So subwords align better with morphemes.

     - Dependent vowels and graphemes are respected.

---

## 4. Resources and analyzers

Indic NLP resources include:

- **Indic NLP Library** — tokenization, sentence splitting, normalization, script conversion; can act as a baseline or pre-processing layer.

- **AI4Bharat Hindi corpora and frequency lists** — useful for training & analyzing morphological productivity.

- Tokenization studies that specifically examine Hindi tokenizer variants and their effect on downstream tasks.

We plan to:

- Use these to:

  - Extract high-frequency affixes and stems.

  - Validate our heuristic rules.

- Keep licensing-sensitive artifacts in `data/` with clear notes.

---

## 5. Dialects, spelling variation, social media

Hindi in the wild:

- Contains **dialects** (Awadhi, Bhojpuri, Haryanvi…).

- Shows **non-standard spellings** and creative orthography.

- Appears in **social media** with emojis, repeated characters, etc.

Recent surveys point out these are major failure modes for existing tokenizers.

Our design:

- Treat **standard Hindi** as the core.

- Collect **dialectal / social media examples** into dedicated eval sets:

  - Tag them with features like "elongation", "emoji", "mixed script".

- Ensure our tokenizer doesn't catastrophically explode fertility in those cases.

---

## 6. Open questions for Hindi

- How aggressive should morphology-aware segmentation be by default?

- Should we provide **multiple segmentation modes**:

  - "surface-form friendly" vs "morphology-rich"?

- How do we weigh **human intuitions** vs **downstream metrics** when they disagree?

We track these questions in `99-notes-open-questions.md`.

