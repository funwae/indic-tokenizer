# Parity Datasets for Fairness Evaluation

This document describes how to obtain and prepare parallel Hindi-English corpora for tokenization parity and fairness evaluation.

## Data Source: IIT Bombay English-Hindi Parallel Corpus

The recommended corpus source is the **IIT Bombay English-Hindi Parallel Corpus**.

### Obtaining the Corpus

1. **Visit IIT Bombay Parallel Corpus**:
   - Website: http://www.cfilt.iitb.ac.in/iitb_parallel/
   - Direct download links available on the website

2. **Download Files**:
   - `train.en` - English sentences (one per line)
   - `train.hi` - Hindi sentences (one per line)
   - Expected size: ~1.5M sentence pairs

3. **License**:
   - Check IIT Bombay corpus license terms
   - Ensure compliance with usage requirements
   - Most academic/research use is permitted

### Alternative Sources

If IIT Bombay corpus is not available, you can use:

- **OPUS**: https://opus.nlpl.eu/ (filter for Hindi-English)
- **FLORES**: https://github.com/facebookresearch/flores
- **Other public parallel corpora** (verify licensing)

## Corpus Preparation

### Step 1: Place Raw Files

Place the raw corpus files in a convenient location:

```bash
# Example: if you downloaded IITB corpus
cp /path/to/train.en data/parity/iitb_train.en
cp /path/to/train.hi data/parity/iitb_train.hi
```

### Step 2: Run Preparation Script

Use the `prepare_parity_hi_en.py` script to process and sample:

```bash
python scripts/prepare_parity_hi_en.py \
  --en-file data/parity/iitb_train.en \
  --hi-file data/parity/iitb_train.hi \
  --output data/parity/hi_en_iitb_sample.jsonl \
  --max-pairs 50000
```

### Script Options

- `--en-file`: Path to English corpus file
- `--hi-file`: Path to Hindi corpus file
- `--output`: Output JSONL file path
- `--max-pairs`: Maximum number of pairs to sample (default: 50000)
- `--max-length-ratio`: Maximum length ratio for filtering (default: 3.0)
- `--random-seed`: Random seed for deterministic sampling (default: 42)

### Processing Steps

The script performs:

1. **Loading**: Reads parallel files line-by-line, pairing English and Hindi sentences
2. **Filtering**: Removes pairs with extreme length ratios (> 3:1 by default)
3. **Sampling**: Randomly samples up to `--max-pairs` pairs (deterministic with fixed seed)
4. **Saving**: Writes JSONL format with keys `{"en": "...", "hi": "..."}`

### Output Format

The output JSONL file contains one JSON object per line:

```json
{"en": "Many important events happened in India today.", "hi": "भारत में आज कई महत्वपूर्ण घटनाएं हुईं।"}
{"en": "The weather is pleasant in the capital Delhi.", "hi": "राजधानी दिल्ली में मौसम सुहावना है।"}
```

## Usage

The prepared JSONL file can be used with:

- `scripts/run_parity_benchmark.py` - Tokenization parity evaluation
- Fairness metrics computation
- Cross-language tokenization analysis

## Next Steps

After preparing the corpus, proceed to:

1. Run parity benchmark (see `scripts/run_parity_benchmark.py`)
2. Evaluate fairness metrics (see `docs/22-evaluation-metrics.md`)
3. Compare tokenization premium across tokenizers

