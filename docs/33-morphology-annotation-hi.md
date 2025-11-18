# Hindi Morphology Annotation Guidelines

This document describes the schema and guidelines for annotating Hindi morphology gold set.

## Data Format

The morphology gold set is stored in TSV (tab-separated values) format with three columns:

1. **id**: Unique identifier (integer or string)
2. **text**: Full Hindi sentence
3. **morphemes**: Space-separated segmentation of the sentence, where morphemes align to character spans

## Example

```tsv
id	text	morphemes
1	कामकाजी औरतें अक्सर थक जाती हैं।	काम का जी  और तें  अक्सर  थक  जा ती  हैं ।
```

## Annotation Guidelines

### Morpheme Segmentation

1. **Word-level segmentation**: Split words into their constituent morphemes
   - Example: `कामकाजी` → `काम का जी`
   - Example: `औरतें` → `और तें`

2. **Morpheme boundaries**: Use spaces to separate morphemes
   - Each morpheme should be a meaningful morphological unit
   - Preserve word boundaries with double spaces

3. **Common patterns**:
   - **Compound words**: Split into components
     - `राजधानी` → `राज धानी`
     - `विश्वविद्यालय` → `विश्व विद्यालय`

   - **Inflectional suffixes**: Separate from stems
     - `घटनाएं` → `घटना एं` (plural marker)
     - `हुईं` → `हो ईं` (past tense + agreement)

   - **Case markers**: Separate case/postpositions
     - `में` → `में` (locative, keep as single morpheme)
     - `के लिए` → `का ए लिए` (genitive + postposition)

   - **Aspect markers**: Separate aspectual markers
     - `जा रही` → `जा रह ई` (progressive aspect)
     - `कर रहे` → `कर रह ए` (progressive aspect)

4. **Consistency**:
   - Use consistent segmentation for similar patterns
   - Document any special cases or exceptions

### Annotation Process

1. **Start with sentence**: Begin with the full Hindi sentence
2. **Identify word boundaries**: Mark word boundaries
3. **Segment each word**: Break words into morphemes
4. **Verify alignment**: Ensure morpheme sequence aligns with character spans in original text
5. **Review**: Check for consistency and correctness

### Quality Checks

- **Completeness**: All words should be segmented
- **Alignment**: Morpheme sequence should match the original text when concatenated
- **Consistency**: Similar patterns should be segmented similarly
- **Linguistic validity**: Segments should be linguistically meaningful

## Usage

The annotated gold set is used for:

- **Morphology metrics evaluation**: Boundary F1, morpheme alignment
- **Tokenizer comparison**: Compare how different tokenizers align with morphological boundaries
- **Research**: Study morphological awareness of tokenizers

## File Location

- **Gold set**: `data/hindi/morph_gold/hi_morph_gold.tsv`
- **Size**: ~150 sentences (expandable)
- **Format**: TSV with columns: id, text, morphemes

## References

- MorphTok paper: "MorphTok: Morphologically Grounded Tokenization for Indian Languages"
- IndicMorphScore: Morphology-aware evaluation metrics
- EvalTok: Evaluation framework for tokenization

