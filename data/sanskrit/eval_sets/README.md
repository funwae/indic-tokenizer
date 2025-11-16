# Sanskrit Evaluation Datasets

This directory contains evaluation datasets for Sanskrit tokenization.

## Files

- `classical.txt` - Classical Sanskrit texts and quotes (20 examples)
- `sandhi_examples.txt` - Sandhi-heavy examples (10 examples)

## Format

Each file contains one text per line. These are simple text files for easy processing.

## Usage

These datasets can be used with evaluation scripts:

```bash
python scripts/evaluate_tokenizers.py \
  --dataset data/sanskrit/eval_sets/classical.txt \
  --tokenizers indicbert,gpe_hi_v0
```

## Sources

- Examples are created for evaluation purposes
- All examples are in Devanagari script
- Mix of simple sentences and complex sandhi examples
- Classical Sanskrit texts and quotes

## Licensing

These are example datasets created for the Indic Tokenization Lab. They can be used freely for research and development purposes.

