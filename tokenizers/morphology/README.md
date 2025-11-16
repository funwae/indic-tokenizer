# Morphology Layer (L1)

This directory contains morphology-aware segmentation modules for the Indic Tokenization Lab.

## Current Status

These modules are **stubs** that provide the interface for future morphology-aware tokenization. They currently return text unchanged (no-op) but define the API that will be implemented later.

## Modules

### `hindi.py`

Hindi morphology segmentation:
- `segment_hindi_morphology(text)` - Main segmentation function (stub)
- `split_compounds(text)` - Compound word splitting (stub)
- `identify_affixes(text)` - Affix identification (stub)

**Future Integration:**
- Hindi morphological analyzers
- Affix databases
- Compound splitting rules

### `sanskrit.py`

Sanskrit sandhi splitting:
- `split_sanskrit_sandhi(text)` - Main sandhi splitting function (stub)
- `split_sandhi_with_confidence(text, min_confidence)` - Confidence-based splitting (stub)
- `analyze_sandhi_type(text)` - Sandhi type identification (stub)
- `get_sandhi_splits(text)` - Multi-hypothesis splitting (stub)

**Future Integration:**
- Saṃsādhanī sandhi splitter
- LREC sandhi benchmark tools
- Classical sandhi rule engines

## Integration Points

These modules will be integrated into the tokenization pipeline at the L1 (morphology) layer:

```
L0: Grapheme segmentation
  ↓
L1: Morphology segmentation (this module)
  ↓
L2: BPE/CBPE tokenization
```

## Usage

```python
from tokenizers.morphology.hindi import segment_hindi_morphology
from tokenizers.morphology.sanskrit import split_sanskrit_sandhi

# Hindi (currently no-op)
segments = segment_hindi_morphology("रेलगाड़ी")
# Returns: ["रेलगाड़ी"] (will be ["रेल", "गाड़ी"] in future)

# Sanskrit (currently no-op)
segments = split_sanskrit_sandhi("तदेव")
# Returns: ["तदेव"] (will be ["तत्", "एव"] in future)
```

## Future Work

1. **Hindi Morphology**
   - Integrate morphological analyzers
   - Build affix databases
   - Implement compound splitting rules

2. **Sanskrit Sandhi**
   - Integrate Saṃsādhanī or similar tools
   - Implement sandhi rule engine
   - Add confidence scoring

3. **Evaluation**
   - Morphology alignment metrics
   - Sandhi splitting accuracy
   - Integration with evaluation pipeline

