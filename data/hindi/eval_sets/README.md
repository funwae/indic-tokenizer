# Hindi Evaluation Datasets

This directory contains evaluation datasets for Hindi tokenization.

## Files

- `news_headlines.txt` - News headlines (20 examples)
- `literature.txt` - Literary/descriptive text excerpts (20 examples)
- `conversational.txt` - Conversational Hindi examples (20 examples)

## Format

Each file contains one text per line. These are simple text files for easy processing.

## Usage

These datasets can be used with evaluation scripts:

```bash
python scripts/evaluate_tokenizers.py \
  --dataset data/hindi/eval_sets/news_headlines.txt \
  --tokenizers indicbert,gpe_hi_v0
```

## Sources

- Examples are created for evaluation purposes
- All examples are in Devanagari script
- Mix of simple and complex sentences
- Various domains (news, literature, conversational)

## Licensing

These are example datasets created for the Indic Tokenization Lab. They can be used freely for research and development purposes.

