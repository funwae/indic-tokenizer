# Data Directory

This directory contains corpora, evaluation sets, and curated examples for the Indic Tokenization Lab.

## Structure

```
data/
├── hindi/
│   ├── curated_examples.jsonl    # Curated failure examples
│   └── eval_sets/                # Evaluation datasets
├── sanskrit/
│   ├── curated_examples.jsonl    # Curated failure examples
│   └── eval_sets/                # Evaluation datasets
└── README.md                     # This file
```

## Data Format

### curated_examples.jsonl

Each line is a JSON object with:

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

## Adding Data

1. **Curated Examples**: Add failure cases to `{lang}/curated_examples.jsonl`
2. **Evaluation Sets**: Place evaluation datasets in `{lang}/eval_sets/`
3. **Documentation**: Update this README with data sources and licenses

## Licensing

See `docs/23-datasets-and-licensing.md` for licensing information. Always verify licenses before adding data.

