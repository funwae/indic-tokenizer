# Datasets and Licensing

This document tracks datasets, corpora, and licensing considerations for the Indic Tokenization Lab.

---

## 1. Data Sources

### 1.1 Hindi Corpora

- **AI4Bharat IndicNLP Hindi Corpus**
  - Source: AI4Bharat
  - License: Check AI4Bharat licensing terms
  - Usage: Training and evaluation

- **Wikipedia Hindi**
  - Source: Wikimedia Foundation
  - License: CC BY-SA 3.0
  - Usage: Training corpus

- **News Corpora**
  - Various sources (to be documented)
  - License: Per-source (verify before use)
  - Usage: Domain-specific evaluation

- **Social Media Samples**
  - Carefully anonymized samples
  - License: Respect privacy, anonymize
  - Usage: Code-mixed text evaluation

### 1.2 Sanskrit Corpora

- **LREC Sandhi Benchmark**
  - Source: LREC conference
  - License: Check LREC dataset licensing
  - Usage: Sandhi splitting evaluation

- **Classical Texts**
  - Various sources (public domain where applicable)
  - License: Verify per-source
  - Usage: Training and evaluation

- **Saṃsādhanī Resources**
  - Source: Sanskrit University Hyderabad
  - License: Check Saṃsādhanī licensing
  - Usage: Sandhi splitting tools

---

## 2. Licensing Considerations

### 2.1 External Tools

- **IndicNLP Library**: Check AI4Bharat license
- **HuggingFace Models**: Apache 2.0 (typically)
- **Saṃsādhanī**: Verify license before integration

### 2.2 Our Contributions

- All code in this repository: [To be determined - suggest MIT or Apache 2.0]
- Documentation: CC BY 4.0 (suggested)
- Datasets we create: [To be determined]

---

## 3. Data Storage

- Large corpora: Store references, not full copies (where possible)
- Curated examples: Store in `data/` with clear attribution
- Processed data: Document preprocessing steps

---

## 4. Privacy and Ethics

- Social media data: Fully anonymize
- No personally identifiable information
- Respect source licenses and terms of service

---

## 5. Data README

Each `data/` subdirectory should contain a README.md explaining:
- Source of data
- License
- Preprocessing applied
- Usage guidelines

