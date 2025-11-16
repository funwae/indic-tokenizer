# Phase 1.5 — Grapheme Pair Encoding (GPE) Prototype for Hindi/Sanskrit

This document describes the **Grapheme Pair Encoding (GPE) prototype** implementation for the Indic Tokenization Lab. GPE extends the basic lab with grapheme-aware tokenization, following the approach proposed by Velayuthan & Sarveswaran (2024/2025).

## Overview

GPE improves BPE tokenization for complex scripts (Tamil, Sinhala, Hindi) by running BPE over **Unicode grapheme clusters** instead of bytes or raw codepoints. This approach:

- Respects script boundaries (no splitting of dependent vowels, matras, virama)
- Reduces token count (fertility) compared to byte-level tokenizers
- Improves tokenization parity with English for Indic languages

Real-world usage of GPE already exists in tools like **SLTK** (Sinhala Language Toolkit), which explicitly implements GPE-style tokenization for Sinhala NLP.

## Implementation Decisions (v0)

### Grapheme segmentation → `regex` with `\X` (UAX #29)

- ✅ Uses the Unicode-standard notion of **extended grapheme clusters** via `\X`, which is specifically designed for this job and already used in production-grade tokenization work.
- ✅ One small dependency (`regex`) and works across scripts; we can later layer Devanagari-specific tweaks on top if we find edge cases.
- ⏭ If we ever hit performance or corner-case issues, we can add a Devanagari-specialized path or swap in a dedicated grapheme library without changing the rest of the pipeline.

### BPE training → home-grown GPE+BPE trainer with `cbpe_merge_allowed` (for now)

- ✅ Full control over the merge loop, so we can **hard-wire CBPE constraints** (no illegal matra/virama starts, later morphology-aware rules) instead of fighting SentencePiece's internals.
- ✅ Easier to experiment with **GPE-specific ideas** (grapheme-aware stats, custom stopping criteria, diagnostics) and log everything for the lab.
- ⏭ Once we're happy with the behavior and constraints, we can optionally add a **SentencePiece-based path** that operates on pre-tokenized grapheme sequences for speed, using the same grapheme pretokenizer and constraints as a reference.

So for v0/v1 of the lab:

> **Graphemes:** `regex` + `\X`
> **Training:** home-grown grapheme-BPE (GPE) trainer + `cbpe_merge_allowed`

Then, if the prototype performs well, we can mirror the behavior in a SentencePiece pipeline for scale.

## Architecture

### Module 1: Grapheme Segmenter (`tokenizers/grapheme_segmenter.py`)

**Purpose**: Decompose Devanagari text into **extended grapheme clusters** (akshara-like units) that match "user-perceived characters" rather than raw codepoints.

**Implementation**:
- Uses `regex` library's `\X` pattern for UAX #29 compliant grapheme clustering
- Functions:
  - `iter_graphemes(text: str) -> Iterator[str]`: Generator for grapheme clusters
  - `segment_devanagari_graphemes(text: str, keep_non_devanagari: bool = True) -> List[str]`: Split text into grapheme list
- CLI support: `python -m tokenizers.grapheme_segmenter "किशोरी"` for debugging

**Usage**:
```python
from tokenizers.grapheme_segmenter import segment_devanagari_graphemes

graphemes = segment_devanagari_graphemes("किशोरी")
# Returns: ['कि', 'शो', 'री']
```

### Module 2: GPE BPE Trainer (`scripts/train_gpe_tokenizer.py`)

**Purpose**: Train a **Grapheme Pair Encoding–style** tokenizer for Hindi.

**Training Process**:
1. Load corpus line by line
2. Pre-tokenize each word into grapheme clusters using `segment_devanagari_graphemes()`
3. Represent each word as a sequence of grapheme symbols + `</w>` sentinel
4. Run Sennrich-style BPE training loop over grapheme symbols
5. Apply `cbpe_merge_allowed(left, right)` constraints to avoid illegal merges
6. Save vocab.json, merges.txt, and config.json

**CLI Usage**:
```bash
python scripts/train_gpe_tokenizer.py \
  --input data/hindi/corpus.txt \
  --output-dir models/gpe_hi_v0 \
  --vocab-size 32000 \
  --min-pair-frequency 2 \
  --max-lines 200000
```

**Output Format**:
- `vocab.json`: Token ID mappings (with special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`)
- `merges.txt`: BPE merge rules in HuggingFace-compatible format
- `config.json`: Metadata (type, lang, script, vocab_size, num_merges)

### Module 3: GPE Tokenizer Adapter (`tokenizers/gpe_tokenizer.py`)

**Purpose**: Load trained GPE tokenizer and implement tokenization interface.

**Implementation**:
- `GPETokenizer` class that loads vocab and merges from model directory
- `tokenize(text: str) -> List[str]`: Tokenize text using grapheme segmentation + BPE
- `encode(text: str) -> List[int]`: Encode text to token IDs
- `decode(ids: List[int]) -> str`: Decode token IDs back to text
- Compatible with `compare_tokenizers.py` interface

**Usage**:
```python
from tokenizers.gpe_tokenizer import GPETokenizer

tokenizer = GPETokenizer(
    tokenizer_id="gpe_hi_v0",
    model_path="models/gpe_hi_v0"
)

tokens = tokenizer.tokenize("यहाँ आपका हिंदी वाक्य जाएगा।")
ids = tokenizer.encode("यहाँ आपका हिंदी वाक्य जाएगा।")
```

## Integration with Lab

### Registry Entry

The GPE tokenizer is registered in `tokenizers/registry.yaml`:

```yaml
  - id: gpe_hi_v0
    type: custom_gpe
    model_path: "models/gpe_hi_v0"
    display_name: "GPE-Hi v0 (grapheme-BPE)"
```

### Comparison via CLI

Once trained, you can compare GPE-Hi with other tokenizers:

```bash
python scripts/compare_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert,gpe_hi_v0
```

Expected results:
- GPE-Hi should have **fewer tokens** than byte/char-level baselines
- Zero grapheme-break violations by design
- Better alignment with linguistic units (graphemes/akshara)

## Integration with CBPE Constraints

The GPE trainer integrates with the existing `cbpe_merge_allowed` constraint hook from `tokenizers/cbpe_constraints.py`:

- During BPE training, candidate merges are filtered through `cbpe_merge_allowed()`
- This prevents illegal merges (e.g., tokens starting with dependent vowels or virama)
- The constraints work at the grapheme level, ensuring script-aware tokenization

## Future: Sanskrit GPE + Sandhi

Once Hindi GPE is validated, extend to Sanskrit:

1. Use Sanskrit sandhi splitters + corpora to pre-segment surface forms
2. Apply `segment_devanagari_graphemes` to the **sandhi-resolved** forms
3. Train `gpe_sa_v0` with the same CBPE constraints, but over Sanskrit corpora
4. Integrate as `gpe_sa_v0` in the registry and compare against baseline tokenizers

This ties our work directly into the emerging ecosystem of "script-aware + morphology-aware tokenization" and sets us up to be a reference implementation for **Devanagari GPE** the way SLTK is becoming for Sinhala.

## References

- Velayuthan & Sarveswaran (2024/2025): "Egalitarian Language Representation in Language Models: It All Begins with Tokenizers" - GPE paper
- Unicode Text Segmentation (UAX #29): Extended grapheme clusters specification
- SLTK: Sinhala Language Toolkit with GPE implementation
- See `docs/02-research-survey.md` for additional references

