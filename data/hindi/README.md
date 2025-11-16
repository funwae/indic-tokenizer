# Hindi Data

This directory contains Hindi corpora, evaluation sets, and curated examples.

## Curated Examples

Add failure cases to `curated_examples.jsonl` in the format:

```json
{
  "text": "यहाँ आपका हिंदी वाक्य जाएगा।",
  "lang": "hi",
  "domain": "news",
  "source": "example",
  "description": "Description of the failure case",
  "tags": ["compound", "morphology"]
}
```

## Evaluation Sets

Place evaluation datasets in `eval_sets/` with clear documentation about:
- Source
- License
- Preprocessing applied
- Usage guidelines

