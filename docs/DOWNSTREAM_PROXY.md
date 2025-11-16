# Downstream Proxy Task

## Overview

The downstream proxy task is a minimal language modeling experiment to demonstrate that better tokenization leads to better downstream performance. This is a "proxy" task - we use small models and limited data to show direction, not to achieve SOTA results.

## Architecture

- **Model**: Small transformer (2-4 layers, ~10M parameters)
- **Task**: Causal language modeling (next token prediction)
- **Training**: Small dataset (10K-50K sentences)
- **Evaluation**: Perplexity on held-out test set

## Tokenizers

We train separate models for each tokenizer:
- mBERT tokenizer
- IndicBERT tokenizer
- GPE+CBPE tokenizer

## Expected Results

Even with small models, we expect to see:
- Lower perplexity for tokenizers with better morphological alignment
- Directionally correct improvements matching MorphTok's findings
- Clear differences between tokenizers

## Note

This is a proof-of-concept implementation. For production use, you would:
- Use larger models
- Train on more data
- Use proper hyperparameter tuning
- Evaluate on multiple downstream tasks (MT, NER, etc.)

