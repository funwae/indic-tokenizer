# Verification Checklist

This checklist helps verify that all components of the Indic Tokenization Lab are working correctly.

## Prerequisites Check

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements.txt`

## Automated Verification

Run the verification script:

```bash
python3 scripts/verify_setup.py
```

This will test:
- [ ] All imports work
- [ ] Grapheme segmentation functions
- [ ] CBPE constraints
- [ ] Registry loading
- [ ] Evaluation metrics
- [ ] Pretokenizer
- [ ] HF tokenizer loading (if transformers installed)

## Manual Verification

### 1. Grapheme Segmentation

```bash
python3 -m tokenizers.grapheme_segmenter "किशोरी"
python3 -m tokenizers.grapheme_segmenter "प्रार्थना"
python3 -m tokenizers.grapheme_segmenter "कर्मयोग"
```

Expected: Should print graphemes with Unicode code points.

### 2. Tokenizer Comparison

```bash
python3 scripts/compare_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert
```

Expected: Should print comparison table with tokens and statistics.

### 3. Evaluation

```bash
python3 scripts/evaluate_tokenizers.py \
  --text "यहाँ आपका हिंदी वाक्य जाएगा।" \
  --tokenizers indicbert,mbert \
  --output scorecards/test.json
```

Expected: Should generate JSON scorecard with metrics.

### 4. Dataset Loading

```bash
python3 -c "
from pathlib import Path
import json

# Test curated examples
with open('data/hindi/curated_examples.jsonl') as f:
    examples = [json.loads(line) for line in f if line.strip()]
    print(f'Loaded {len(examples)} Hindi examples')

# Test eval dataset
with open('data/hindi/eval_sets/news_headlines.txt') as f:
    lines = [l.strip() for l in f if l.strip()]
    print(f'Loaded {len(lines)} news headlines')
"
```

Expected: Should load examples and datasets successfully.

### 5. Unit Tests

```bash
pytest tests/ -v
```

Expected: All tests should pass (may need dependencies installed).

## Component Checklist

### Core Modules

- [x] `tokenizers/grapheme_segmenter.py` - Grapheme segmentation
- [x] `tokenizers/cbpe_constraints.py` - CBPE constraints
- [x] `tokenizers/pretokenizer.py` - Pretokenization
- [x] `tokenizers/gpe_tokenizer.py` - GPE tokenizer adapter
- [x] `tokenizers/sentencepiece_tokenizer.py` - SP tokenizer adapter

### Training Scripts

- [x] `scripts/train_gpe_tokenizer.py` - GPE training
- [x] `scripts/train_sentencepiece_baseline.py` - SP training

### Evaluation

- [x] `eval/grapheme_violations.py` - Violation detection
- [x] `eval/fertility.py` - Fertility metrics
- [x] `eval/metrics.py` - Integrated metrics

### Comparison Scripts

- [x] `scripts/compare_tokenizers.py` - Basic comparison
- [x] `scripts/evaluate_tokenizers.py` - Full evaluation
- [x] `scripts/run_full_evaluation.py` - Batch evaluation

### Data

- [x] Curated examples (20+ Hindi, 10+ Sanskrit)
- [x] Evaluation datasets (news, literature, conversational)
- [x] Dataset documentation

### Tests

- [x] `tests/test_grapheme_segmentation.py`
- [x] `tests/test_cbpe_constraints.py`
- [x] `tests/test_evaluation.py`
- [x] `tests/test_pretokenizer.py`

### Documentation

- [x] `docs/PROJECT_SPECS.md` - Complete specifications
- [x] `docs/QUICK_START.md` - Quick start guide
- [x] `README.md` - Project overview
- [x] All component documentation

### Future Components (Stubs)

- [x] `tokenizers/morphology/hindi.py` - Hindi morphology stubs
- [x] `tokenizers/morphology/sanskrit.py` - Sanskrit sandhi stubs
- [x] `playground/API_SPEC.md` - Playground API spec

## Known Limitations

1. **Dependencies**: Some features require:
   - `regex` package for grapheme segmentation
   - `transformers` for HF tokenizers
   - `sentencepiece` for SP tokenizers

2. **Training**: Requires corpus files (not included in repo)

3. **Models**: Trained models need to be generated or downloaded

## Success Criteria

The lab is ready when:

- ✅ All core modules are implemented
- ✅ Verification script runs (may show warnings for missing optional deps)
- ✅ Basic comparison works with available tokenizers
- ✅ Evaluation metrics compute correctly
- ✅ Tests pass (when dependencies installed)
- ✅ Documentation is complete

## Next Steps After Verification

1. Install missing dependencies if needed
2. Train tokenizers if corpus available
3. Run full evaluation on datasets
4. Generate scorecards
5. Start using the lab for research/development

