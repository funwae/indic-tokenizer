# Sanskrit & Sandhi — Notes for Tokenization

Sanskrit raises tokenization to "hard mode":

- No mandatory spaces.

- Heavy use of **sandhi** (phonological mergers at morpheme/word boundaries).

- Complex inflectional morphology.

---

## 1. Sandhi as a precursor to tokenization

Sandhi merges morphemes and words into long compounds. **Accurate sandhi splitting is often a prerequisite for good tokenization**, as emphasized in both classical tools and modern deep-learning sandhi splitting work.

Key ideas:

- Sandhi rules are *well defined* (Pāṇini) but:

  - Highly context-sensitive.

  - Sometimes ambiguous.

- Splitting is itself an NLP task with specialized corpora.

---

## 2. Strategy for Sanskrit in this lab

We treat Sanskrit as having **two parallel tokenization tracks**:

1. **Surface track**

   - Tokenize *without* fully resolving sandhi:

     - Use grapheme constraints + simple heuristics.

     - Appropriate for low-resource or exploratory settings.

2. **Morphology track**

   - Use sandhi splitting + morphological analysis where tools exist:

     - Saṃsādhanī sandhi splitter and morphological analyzers where licensing allows.

   - Feed split forms into CBPE, similar to Hindi.

The tokenizer must be able to:

- Export both views when requested.

- Record how sandhi decisions were made (provenance for later analysis).

---

## 3. Corpora & benchmarks

We will focus on corpora that have:

- **Gold sandhi splitting** or benchmarks:

  - LREC "Benchmark Corpus for Evaluating Sanskrit Sandhi Tools".

- **Morphology annotations** where available.

Our goal is to:

- Use these corpora to:

  - Evaluate sandhi-aware pre-segmentation quality.

  - Measure tokenization improvements when sandhi is correctly resolved.

---

## 4. Tokenization constraints specific to Sanskrit

Beyond Devanagari grapheme rules (see `10-linguistics-devanagari.md`), we add:

- **No blind splitting of compounds** if sandhi analysis is uncertain:

  - We'd rather have a long token than a confidently wrong split.

- **Optionally multi-path**:

  - For research, it may be useful to retain multiple segmentation hypotheses.

---

## 5. Future directions

- Integrate **Gemma-based sandhi splitting** style models as open-source variants appear.

- Explore bring-your-own sandhi splitter:

  - API to plug external tools into our pre-segmentation pipeline.

