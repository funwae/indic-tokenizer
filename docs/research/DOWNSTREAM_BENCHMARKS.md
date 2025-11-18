# Downstream Benchmark Selection

**Status**: Research phase
**Goal**: Select one manageable downstream benchmark for tokenizer comparison

---

## Benchmark Options

### Option 1: Hindi Subset of IndicGenBench

**Description**: Generation benchmark for Indic languages.

**Advantages**:
- Standard benchmark in Indic NLP community
- Multiple tasks (translation, summarization, etc.)
- Well-documented

**Challenges**:
- May require significant setup
- Large dataset (need to subset)
- Multiple tasks (need to pick one)

**Resources**:
- GitHub: https://github.com/AI4Bharat/IndicGenBench
- Paper: "IndicGenBench: A Multilingual Benchmark for Indic Language Generation"

### Option 2: Small Hindi-English MT Task

**Description**: Machine translation from Hindi to English (or vice versa).

**Advantages**:
- Clear evaluation metric (BLEU)
- Manageable scope (small dataset)
- Directly shows tokenization impact

**Challenges**:
- Need parallel corpus
- Requires MT model training
- BLEU may not capture all tokenization effects

**Resources**:
- IITB parallel corpus (already have)
- FLORES dataset
- OPUS corpus

### Option 3: Hindi Text Classification

**Description**: Sentiment analysis, topic classification, etc.

**Advantages**:
- Simple evaluation (accuracy/F1)
- Fast to train small models
- Clear tokenization impact

**Challenges**:
- Need labeled dataset
- May not be as sensitive to tokenization
- Less representative of generation tasks

**Resources**:
- Hindi sentiment datasets
- Topic classification datasets

### Option 4: Hindi Named Entity Recognition (NER)

**Description**: Identify named entities in Hindi text.

**Advantages**:
- Tokenization directly affects entity boundaries
- Clear evaluation (F1 score)
- Relevant for many applications

**Challenges**:
- Need annotated dataset
- Entity boundaries may not align with tokens
- Requires sequence labeling model

**Resources**:
- Hindi NER datasets
- FIRE shared tasks

---

## Selected Benchmark: Small Hindi-English MT Task

**Rationale**:
- **Manageable**: Can use existing IITB parallel corpus
- **Clear metric**: BLEU score is standard and interpretable
- **Tokenization-sensitive**: MT directly affected by tokenization quality
- **Reusable infrastructure**: Can leverage existing tiny LM or build small seq2seq model

**Implementation Plan**:
1. Use IITB parallel corpus (or subset)
2. Train small seq2seq model (encoder-decoder transformer)
3. Train separate models with each tokenizer:
   - Baseline: `gpe_cbpe_hi_v1`
   - Semantic: `ag_bpe_hi_v1`
   - Reference: `mbert` or `indicbert`
4. Evaluate BLEU scores
5. Compare and document results

**Model Architecture**:
- Small encoder-decoder transformer
- ~2-4M parameters total
- 256-512 hidden dimensions
- 2-4 layers each

**Evaluation**:
- BLEU score on test set
- Token-level analysis
- Error analysis by tokenization type

---

## Alternative: Start with Text Classification

If MT proves too complex, fall back to text classification:
- Simpler model (just encoder)
- Faster training
- Still shows tokenization impact

---

## Next Steps

1. Prepare MT dataset (train/dev/test split from IITB corpus)
2. Implement small seq2seq model
3. Create training script for MT task
4. Train models with different tokenizers
5. Evaluate and compare BLEU scores
6. Document results

