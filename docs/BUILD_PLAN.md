# Build Plan

This document outlines the concrete steps to build the Indic Tokenization Lab, organized by milestone from the roadmap.

## Current Status: Milestone 0 — Lab Bootstrap

✅ **Completed:**
- Repository structure created
- Documentation scaffold in place
- Basic tokenizer comparison script (`scripts/compare_tokenizers.py`)
- CBPE constraints module (`tokenizers/cbpe_constraints.py`)
- Tokenizer registry system

## Immediate Next Steps (Milestone 0 Completion)

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/compare_tokenizers.py --text "यहाँ आपका हिंदी वाक्य जाएगा।" --tokenizers indicbert
```

### 2. Test Tokenizer Comparison

- [ ] Test `compare_tokenizers.py` with Hindi text
- [ ] Verify IndicBERT tokenizer loads correctly
- [ ] Test JSON output format
- [ ] Add more tokenizers to registry (IndicBART, etc.)

### 3. Add Curated Examples

- [ ] Collect 10-20 real Hindi examples from various domains
- [ ] Add to `data/hindi/curated_examples.jsonl`
- [ ] Collect 5-10 Sanskrit examples
- [ ] Add to `data/sanskrit/curated_examples.jsonl`
- [ ] Document sources and licensing

### 4. Create Evaluation Sets

- [ ] Create `data/hindi/eval_sets/` directory
- [ ] Add small news headlines dataset
- [ ] Add small literature/scripture dataset
- [ ] Add conversational examples
- [ ] Repeat for Sanskrit

### 5. Playground UI (Optional for M0)

- [ ] Set up Next.js project in `playground/`
- [ ] Create basic UI with text input
- [ ] Implement mock tokenizers (word/char/naive BPE)
- [ ] Add token visualization (chips)
- [ ] Wire up to Python backend (future)

## Milestone 1 — Devanagari-Aware Baseline

### Phase 1: Unicode & Grapheme Layer

**Goal:** Implement robust Devanagari grapheme segmentation

**Tasks:**
1. Create `tokenizers/grapheme.py`:
   - Implement Unicode grapheme cluster detection
   - Handle consonant + virāma + consonant clusters
   - Handle dependent vowels and diacritics
   - Use `unicodedata` and regex

2. Create `eval/grapheme_violations.py`:
   - Detect token boundaries that split graphemes
   - Count violations per tokenizer
   - Generate violation reports

3. Test with curated examples:
   - Verify no violations for proper tokenizers
   - Document violations in existing tokenizers

**Estimated Time:** 1-2 days

### Phase 2: Rule-Based Pretokenizer

**Goal:** Basic Hindi pretokenization rules

**Tasks:**
1. Create `tokenizers/pretokenizer.py`:
   - Split on whitespace + punctuation
   - Preserve URLs, emails, numbers, hashtags (regex)
   - Apply NFC normalization
   - Script-aware splitting

2. Test with Hindi examples:
   - Verify URL/email preservation
   - Check punctuation handling
   - Validate normalization

**Estimated Time:** 1 day

### Phase 3: BPE Training Baseline

**Goal:** Train SentencePiece BPE with Unicode-aware pretokenizer

**Tasks:**
1. Data preparation:
   - Download/access AI4Bharat Hindi corpus
   - Verify licensing
   - Preprocess with L0 segmentation

2. Create `scripts/train_bpe.py`:
   - Use SentencePiece trainer
   - Apply pretokenizer before training
   - Save model + vocab

3. Train baseline model:
   - Start with vocab_size=32000
   - Train on Hindi corpus subset
   - Save to `models/hindi_baseline/`

4. Create tokenizer adapter:
   - Wrap SentencePiece model
   - Add to registry
   - Test with comparison script

**Estimated Time:** 2-3 days (including data acquisition)

### Phase 4: Evaluation

**Goal:** Compare baseline vs existing tokenizers

**Tasks:**
1. Create `eval/fertility.py`:
   - Calculate tokens/word ratio
   - Calculate chars/token
   - Generate comparison tables

2. Create `eval/metrics.py`:
   - Integrate all metrics
   - Generate scorecards
   - Export JSON results

3. Run evaluation:
   - Compare baseline vs IndicBERT vs mBERT
   - Document improvements
   - Identify failure cases

**Estimated Time:** 1-2 days

**Total M1 Estimate:** 5-8 days

## Milestone 2 — Morphology-Aware Hindi Tokenizer

### Phase 1: Morphology Pre-Segmentation

**Tasks:**
1. Research and integrate:
   - Review MorphTok segmentation approach
   - Identify Hindi morphological resources
   - Check licensing for analyzers/lexicons

2. Create `tokenizers/morphology/hindi.py`:
   - Implement affix detection
   - Rule-based morpheme boundary detection
   - Handle common compounds

3. Test with curated examples:
   - Verify compound splitting
   - Check affix handling
   - Document edge cases

**Estimated Time:** 3-5 days

### Phase 2: Constrained BPE Implementation

**Tasks:**
1. Integrate CBPE constraints:
   - Use `tokenizers/cbpe_constraints.py`
   - Extend with morphology boundaries
   - Create `tokenizers/cbpe_trainer.py`

2. Train CBPE model:
   - Apply morphology pre-segmentation
   - Train with constraints
   - Compare vs baseline BPE

**Estimated Time:** 2-3 days

### Phase 3: EvalTok-Style Evaluation

**Tasks:**
1. Create evaluation protocol:
   - Design annotation form
   - Write guidelines
   - Create templates

2. Run pilot evaluation:
   - Annotate 50-100 examples
   - Compare tokenizers
   - Analyze results

**Estimated Time:** 2-3 days

### Phase 4: Downstream Experiments

**Tasks:**
1. Choose task (e.g., Hindi classification):
   - Select dataset
   - Set up training pipeline

2. Train models:
   - Baseline tokenizer
   - CBPE tokenizer
   - Compare performance

**Estimated Time:** 3-5 days

**Total M2 Estimate:** 10-16 days

## Milestone 3 — Sanskrit + Advanced Morphology

### Phase 1: Sanskrit Resources

**Tasks:**
1. Survey tools:
   - Saṃsādhanī sandhi splitter
   - LREC sandhi benchmark
   - Check licensing

2. Integrate sandhi splitter:
   - Create API wrapper
   - Test on benchmark
   - Document usage

**Estimated Time:** 2-3 days

### Phase 2: Sanskrit Pretokenizer

**Tasks:**
1. Create `tokenizers/morphology/sanskrit.py`:
   - Sandhi splitting integration
   - Fallback to grapheme splits
   - Multi-hypothesis support (optional)

2. Test with Sanskrit examples:
   - Verify sandhi splitting
   - Check fallback behavior
   - Document accuracy

**Estimated Time:** 3-4 days

### Phase 3: Sanskrit CBPE

**Tasks:**
1. Train Sanskrit CBPE:
   - Use sandhi-aware segmentation
   - Apply constraints
   - Evaluate on benchmark

**Estimated Time:** 2-3 days

**Total M3 Estimate:** 7-10 days

## Development Workflow

### Daily Workflow

1. **Morning:** Review roadmap, pick next task
2. **Development:** Implement feature, write tests
3. **Testing:** Test with curated examples
4. **Documentation:** Update docs as needed
5. **Evening:** Commit changes, update progress

### Testing Strategy

- Unit tests for each module
- Integration tests for full pipeline
- Evaluation on curated examples
- Comparison with existing tokenizers

### Code Quality

- Type hints (Python)
- Docstrings for all functions
- Follow existing code style
- Run linters before committing

## Resource Requirements

### Hardware

- **CPU:** Standard development machine
- **GPU:** Optional for downstream LM experiments (CUDA 12.x)
- **Storage:** ~10GB for corpora and models

### Software

- Python 3.8+
- PyTorch (for downstream experiments)
- HuggingFace Transformers
- SentencePiece
- Node.js (for playground UI)

### Data

- AI4Bharat Hindi corpus (check licensing)
- Sanskrit corpora (public domain where possible)
- Evaluation benchmarks

## Risk Mitigation

### Technical Risks

1. **Data Licensing:** Verify all data sources before use
2. **Tool Integration:** Some tools may have complex dependencies
3. **Performance:** CBPE may be slower than standard BPE

### Mitigation

- Start with clearly licensed data
- Create fallback implementations
- Profile and optimize as needed

## Success Criteria

### Milestone 0 ✅
- [x] Repo structure
- [x] Documentation
- [x] Basic comparison script
- [ ] Curated examples (10+ Hindi, 5+ Sanskrit)
- [ ] Playground UI (optional)

### Milestone 1
- [ ] Grapheme violation detection working
- [ ] Baseline BPE trained and evaluated
- [ ] Metrics show improvement over generic tokenizers

### Milestone 2
- [ ] Morphology-aware segmentation implemented
- [ ] CBPE trained and evaluated
- [ ] Human evaluation completed
- [ ] Downstream task shows improvement

### Milestone 3
- [ ] Sanskrit sandhi splitting integrated
- [ ] Sanskrit CBPE trained
- [ ] Evaluation on sandhi benchmark

## Next Immediate Actions

1. **Today:**
   - Set up virtual environment
   - Test `compare_tokenizers.py`
   - Add 5-10 Hindi examples to curated_examples.jsonl

2. **This Week:**
   - Implement grapheme segmentation (Phase 1, M1)
   - Create grapheme violation detector
   - Test on existing tokenizers

3. **Next Week:**
   - Implement pretokenizer (Phase 2, M1)
   - Start BPE training setup (Phase 3, M1)

## Notes

- This is a living document; update as progress is made
- Adjust estimates based on actual experience
- Prioritize working code over perfect code
- Document decisions and trade-offs

