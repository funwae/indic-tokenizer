# Devanagari Script — Linguistic Notes for Tokenization

This document collects **script-level facts** we treat as constraints on tokenization.

---

## 1. Script basics

- Devanagari is an **abugida**: consonants inherently carry /a/ by default; other vowels are encoded via **dependent vowel signs (matras)**.

- A visible "character" can consist of several Unicode code points:

  - Base consonant(s)

  - Virāma (्, U+094D)

  - Dependent vowels (e.g. ि, ी, ु, ू,े,ै,ो,ौ)

  - Diacritics (anusvāra, visarga, chandrabindu, etc.)

A **grapheme cluster** (what humans perceive as "one symbol") may span multiple code points and must **never** be split across tokens unless we have a strong reason.

References on Devanagari script encoding and challenges with BPE tokenization highlight how multi-byte code points and combining marks interact poorly with naive subword methods.

---

## 2. Grapheme clusters and "akshara"

### 2.1 What is an akshara?

Informally, an **akshara** is a unit of writing and pronunciation roughly corresponding to a syllabic block, often:

- One consonant + vowel (क, कि, कु, के, कौ)

- Consonant clusters with virāma (क्त, त्र, श्र)

- Plus diacritics and nasalization marks.

Operationally, we approximate akshara ≈ Unicode grapheme cluster over Devanagari.

### 2.2 Tokenization constraint

We adopt the following invariants:

1. **No Grapheme Split Rule**

   - A tokenizer must not cut inside a Devanagari grapheme cluster except in extremely controlled situations (e.g., a low-level character tokenizer).

2. **Soft boundary scoring**

   - When evaluating tokenizers, any token boundary that falls:

     - Between base character and its matra/diacritic, or

     - Within a conjunct cluster (before virāma or after it),

   - is counted as a **grapheme violation**.

These constraints align with concerns raised in MorphTok (dependent vowels) and newer Indic tokenizer work that uses Unicode-aware normalization and heuristics.

---

## 3. Unicode ranges & normalization

### 3.1 Ranges

- Devanagari: U+0900–U+097F

- Devanagari Extended: U+ A8E0–A8FF (Vedic marks, etc.)

We must be careful about:

- **Normalization** (NFC vs NFD vs NFKC):

  - Many Indic tokenizer pipelines report improvements by using **NFKC** to unify visually identical characters and reduce sparsity.

### 3.2 Normalization policy

We propose:

- **NFC** as baseline for Devanagari text (preserve canonical equivalence).

- Apply **NFKC** for:

  - Common compatibility forms (e.g., half-forms, full-width digits) if we can show a win.

- Keep all normalization steps transparent and reversible where possible.

---

## 4. Punctuation, digits, symbols

Real Hindi/Sanskrit text uses:

- Devanagari digits (०–९) and Arabic digits (0–9).

- Latin punctuation and quotes, plus Hindi-typeset quotes and dandas (।, ॥).

Tokenization rules:

- Treat **URLs, emails, @handles, hashtags** as atomic tokens (regex).

- For punctuation:

  - Split off as separate tokens but do **not** break graphemes to do so.

---

## 5. Code-mixed text

Surveys of Indic tokenization emphasize challenges with:

- Mixed scripts in the same sentence (Devanagari + Latin).

- Romanized Hindi ("namaste", "aapka") intermixed with Devanagari.

We explicitly support:

- Treating **script changes** as natural token boundaries.

- Optional special handling of **Romanized Indic** in later phases (e.g. mapping to underlying Devanagari or treating it as a separate "layer").

---

## 6. What we enforce in tokenization

The **script layer** enforces:

1. **Legal code points only** (reject or flag unusual unassigned characters).

2. **No split inside grapheme clusters.**

3. **Punctuation / whitespace boundaries only between graphemes**, never within.

These constraints are enforced before morphology or BPE and become part of our **Constrained BPE** rules later.

