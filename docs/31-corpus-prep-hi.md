# Hindi Corpus Preparation

This document describes how to obtain and prepare Hindi corpus data for training the GPE+CBPE tokenizer.

## Data Source: AI4Bharat IndicNLP

The recommended corpus source is the **AI4Bharat IndicNLP Hindi corpus**.

### Obtaining the Corpus

1. **Visit AI4Bharat IndicNLP**:
   - Website: https://indicnlp.ai4bharat.org/
   - Repository: https://github.com/AI4Bharat/IndicNLP

2. **Download Hindi Corpus**:
   - Look for Hindi corpus files in the IndicNLP repository or datasets
   - Common formats: `.txt` files with one sentence per line
   - Expected size: Several million sentences

3. **License**:
   - Check the AI4Bharat IndicNLP license terms
   - Ensure compliance with usage requirements
   - Most IndicNLP resources are available under permissive licenses

### Alternative Sources

If IndicNLP corpus is not available, you can use:

- **Wikipedia Hindi**: Download from https://dumps.wikimedia.org/hiwiki/
- **OPUS**: https://opus.nlpl.eu/ (filter for Hindi)
- **Other public Hindi text corpora** (verify licensing)

## Corpus Preparation

### Step 1: Place Raw Corpus

Place the raw corpus file in the `data/hindi/raw/` directory:

```bash
# Example: if you downloaded indicnlp_hi.txt
cp /path/to/indicnlp_hi.txt data/hindi/raw/indicnlp_hi.txt
```

### Step 2: Run Preparation Script

Use the `prepare_corpus_hi.py` script to filter and normalize:

```bash
python scripts/prepare_corpus_hi.py \
  --input data/hindi/raw/indicnlp_hi.txt \
  --output data/hindi/processed/gpe_cbpe_hi_corpus.txt \
  --max-lines 500000
```

### Script Options

- `--input`: Path to raw corpus file
- `--output`: Path to output processed corpus
- `--max-lines`: Maximum number of lines to include (default: 500000)
- `--min-length`: Minimum line length in characters (default: 10)
- `--min-devanagari-ratio`: Minimum ratio of Devanagari characters (default: 0.5)
- `--random-seed`: Random seed for deterministic shuffling (default: 42)

### Processing Steps

The script performs:

1. **Unicode Normalization**: NFC normalization
2. **Filtering**:
   - Removes lines shorter than `--min-length` characters
   - Removes lines with less than `--min-devanagari-ratio` Devanagari characters
   - Removes control characters (except newlines/tabs)
3. **Normalization**:
   - Normalizes whitespace (multiple spaces → single space)
   - Preserves sentence boundaries
4. **Shuffling**: Deterministic shuffle with fixed random seed
5. **Limiting**: Keeps up to `--max-lines` lines

### Output

The processed corpus will be saved to the output path, ready for tokenizer training.

Example output:
```
Reading corpus from data/hindi/raw/indicnlp_hi.txt...
  Processed 10000 lines, kept 8234...
  ...
  Total lines read: 2000000
  Lines after filtering: 1850000
  Limited to 500000 lines
Writing to data/hindi/processed/gpe_cbpe_hi_corpus.txt...
✓ Wrote 500000 lines to data/hindi/processed/gpe_cbpe_hi_corpus.txt
  File size: 125.43 MB
```

## Corpus Requirements

For best tokenizer quality:

- **Minimum**: 50,000 lines
- **Recommended**: 300,000-500,000 lines
- **Format**: One sentence per line, UTF-8 encoding
- **Content**: Mix of domains (news, literature, conversational) preferred

## Next Steps

After preparing the corpus, proceed to:

1. Train the GPE+CBPE tokenizer (see `docs/21-training-pipeline.md`)
2. Evaluate tokenizer performance (see `docs/22-evaluation-metrics.md`)

