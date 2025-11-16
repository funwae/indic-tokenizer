# Contributing Guidelines

We want this lab to be a **collaborative hub** for better Indic tokenization.

---

## 1. How to contribute

### 1.1 Failure examples

Most impactful contributions:

- Add **real Hindi/Sanskrit snippets** where tokenizers fail.

- Include:

  - Text.

  - Context (domain, source).

  - Description of failure.

Format: append to `data/<lang>/curated_examples.jsonl`.

### 1.2 New tokenizers

If you have a tokenizer (HF, SentencePiece, custom):

- Add an adapter in `tokenizers/`.

- Register it in `tokenizers/registry.yaml`.

### 1.3 Metrics & evaluation

Help us:

- Improve metrics.

- Run downstream experiments.

- Add new tasks / datasets.

---

## 2. Style & standards

- All new code must have:

  - Type hints (Python).

  - Minimal tests.

- All new docs:

  - Use simple English.

  - Include references if based on specific research.

---

## 3. Ethics & licensing

- Respect licenses of external corpora and tools (e.g. AI4Bharat, Sanskrit morphology projects).

- No proprietary datasets dumped without explicit permission.

---

## 4. Roadmap participation

- See `01-roadmap.md`.

- Open issues tagged:

  - `good-first-issue`

  - `help-wanted`

  - `research-needed`

