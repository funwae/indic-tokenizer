# Playground UI Spec

This document describes the Indic Tokenization Playground web app.

---

## 1. Purpose

- Let users **paste Hindi/Sanskrit text** and compare tokenization from multiple tokenizers.

- Visualize:

  - Token boundaries (chips).

  - Basic metrics per tokenizer.

  - Grapheme violations (if any).

---

## 2. Feature overview

### 2.1 Core features

- Text input box with:

  - Language selector (`hi`, `sa`, `auto`).

  - Character / token count.

  - "Use example" buttons for curated Hindi/Sanskrit samples.

- Tokenizer selection:

  - Multi-select of available tokenizers:

    - `openai_gpt4o` (TikToken based).

    - `ai4bharat_indicbert` tokenizer.

    - `ai4bharat_indicbart` tokenizer.

    - `morphtok_like` (our morphology-aware prototype).

    - `krutrim` (if black-box access exists).

- Results view:

  - For each tokenizer:

    - Line of colored token chips.

    - Metrics bar: tokens, chars/token, grapheme violations.

### 2.2 Secondary features

- "Differences" mode:

  - Highlight where two tokenizers disagree on boundaries.

- Export:

  - Copy JSON of tokenization results.

- "Failure cases" panel:

  - Load curated failure examples from `data/*/curated_examples.jsonl`.

---

## 3. Data structures

### 3.1 API request

```json
POST /api/compare
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "lang": "hi",
  "tokenizerIds": ["openai_gpt4o", "indicbert", "morphtok_like"]
}
```

### 3.2 API response

```json
{
  "text": "…",
  "lang": "hi",
  "results": [
    {
      "tokenizerId": "openai_gpt4o",
      "tokenizerName": "OpenAI GPT-4.1-mini",
      "tokens": ["यहाँ", "आपका", "हिंदी", "वाक्य", "जाएगा", "।"],
      "stats": {
        "numTokens": 6,
        "chars": 26,
        "charsPerToken": 4.33,
        "graphemeViolations": 0
      }
    },
    {
      "tokenizerId": "indicbert",
      "tokenizerName": "AI4Bharat IndicBERT tokenizer",
      "tokens": ["यहाँ", "आप", "का", "हिंदी", "वाक्य", "जाएगा", "।"],
      "stats": { ... }
    }
  ]
}
```

---

## 4. UI layout (concept)

1. **Header**

   * Title: *Indic Tokenization Lab — Devanagari*

   * Short explanatory blurb.

2. **Controls row**

   * Language selector.

   * "Use example" buttons.

3. **Text input panel**

   * Large textarea.

   * Counters.

4. **Tokenizer selector**

   * Checkboxes + info tooltips.

   * "Select all".

5. **Results section**

   * For each tokenizer:

     * Name + metrics.

     * Row of token chips.

   * "Diff mode" toggle.

6. **Failure cases drawer**

   * List of curated examples.

   * Loading one replaces the text input.

---

## 5. Implementation notes

* Use Next.js (app router) + TypeScript.

* Start with **mock tokenizers** to develop UI:

  * `word_split`, `char_split`, `naive_bpe`.

* Later, wire to:

  * Local Python backend via simple HTTP/CLI.

* Keep UI design **minimal and accessible**:

  * Works on mobile + desktop.

