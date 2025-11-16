# Sanskrit Data

This directory contains Sanskrit corpora, evaluation sets, and curated examples.

## Curated Examples

Add failure cases to `curated_examples.jsonl` in the format:

```json
{
  "text": "संस्कृत वाक्य यहाँ होगा।",
  "lang": "sa",
  "domain": "literature",
  "source": "example",
  "description": "Description of the failure case",
  "tags": ["sandhi", "compound"]
}
```

## Evaluation Sets

Place evaluation datasets in `eval_sets/` with clear documentation about:
- Source
- License
- Preprocessing applied
- Usage guidelines

## Sandhi Resources

For sandhi-related data, see:
- LREC Sandhi Benchmark
- Saṃsādhanī resources (check licensing)

