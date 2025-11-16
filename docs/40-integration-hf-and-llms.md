# Integration with HuggingFace & LLMs

This document explains how to integrate the tokenizer with common LLM stacks.

---

## 1. HuggingFace `tokenizers` and Transformers

We will export:

- **Tokenizer JSON** for HF `PreTrainedTokenizerFast`.

- **SentencePiece models** where applicable.

Example usage (IndicBERT style):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("our-org/indic-devanagari-tokenizer-v1")

tokens = tokenizer.tokenize("यहाँ आपका हिंदी वाक्य जाएगा।")
```

We provide detailed guides for:

* Plugging into BERT-style models.

* Using with seq2seq models (IndicBART, IndicTrans2).

---

## 2. Custom LLM stacks

For other stacks:

* We provide:

  * A simple JSON spec for vocab + merges.

  * A small reference implementation in Python that:

    * Loads the tokenizer.

    * Exposes `encode/decode` functions.

---

## 3. OpenAI / TikToken comparison

We won't ship a replacement for TikToken, but:

* We provide scripts to:

  * Compare token counts between our tokenizer and OpenAI's (for Hindi/Sanskrit prompts).

* This is useful for:

  * Estimating cost savings if OpenAI (or others) adopt similar CBPE constraints for Indic languages.

---

## 4. Model training examples

We include:

* Example notebooks / scripts:

  * Training a small Hindi LLM from scratch (DistilBERT style).

  * Fine-tuning IndicBART/IndicTrans2 with our tokenizer for specific tasks.

