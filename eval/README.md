# Evaluation Directory

This directory contains evaluation scripts, metrics, and benchmark results.

## Structure

```
eval/
├── metrics.py              # Evaluation metrics implementation
├── grapheme_violations.py  # Grapheme violation detection
├── fertility.py            # Fertility calculations
└── README.md               # This file
```

## Metrics

See `docs/22-evaluation-metrics.md` for detailed documentation on:
- Intrinsic metrics (fertility, chars/token, grapheme violations)
- Human evaluation (EvalTok-style)
- Downstream metrics (LM perplexity, MT BLEU)

## Usage

Evaluation scripts will be added as the project develops. They will:
- Compute metrics for tokenizer comparisons
- Generate scorecards
- Run downstream experiments

