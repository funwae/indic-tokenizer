# Production Preview Status

## ✅ Completed

1. **CLI Entrypoints**: Added `indic-compare` and `indic-benchmark` to `pyproject.toml`
2. **Config Support**: `run_benchmark.py` now supports `--config` YAML files and `--output-dir`
3. **Demo Corpora**: Created `data/hindi/demo/news_small.txt` and `mixed_small.txt`
4. **Demo Config**: Created `configs/hi_demo.yaml` with Phase 1 metrics configuration
5. **README Updates**: Added Production Preview section with quickstart
6. **Smoke Tests**: Added `tests/test_smoke_production.py` for basic functionality checks
7. **Requirements**: Verified `requirements.txt` is minimal and correct

## 📝 Notes

### Tokenizer Availability

The demo config currently requires:
- **GPE tokenizer**: Must be trained first using `scripts/train_gpe_tokenizer.py`
- **mBERT**: Requires transformers library (may have import conflicts due to naming)

To generate golden scorecards:

1. **Option A: Train GPE tokenizer**
   ```bash
   python scripts/train_gpe_tokenizer.py \
       --input data/hindi/corpus.txt \
       --output-dir models/gpe_hi_v0 \
       --vocab-size 32000
   ```

2. **Option B: Use mBERT only** (update config to use `mbert` tokenizer)

3. **Option C: Use SentencePiece baseline** (if trained)

### Naming Conflict

There's a known naming conflict between our `tokenizers/` package and HuggingFace's `tokenizers` library. See `docs/PACKAGING_NOTE.md` for details.

## 🚀 Next Steps for Full Production Preview

1. Train GPE tokenizer on demo corpus
2. Run benchmark: `indic-benchmark --config configs/hi_demo.yaml --output-dir scorecards/hi_demo`
3. Verify scorecards in `scorecards/hi_demo/results.md`
4. Commit scorecards as "golden" reference results

## Testing

Run smoke tests:
```bash
pytest tests/test_smoke_production.py -v
```

Test CLI entrypoints:
```bash
indic-compare --text "यहाँ आपका हिंदी वाक्य जाएगा।" --tokenizers gpe_hi_v0
indic-benchmark --config configs/hi_demo.yaml --output-dir scorecards/hi_demo
```

