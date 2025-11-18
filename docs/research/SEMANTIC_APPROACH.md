# Semantic/Fractal Tokenization: Research Direction

**Status**: Research phase
**Goal**: Prototype attention-guided or hierarchical tokenization for Hindi

---

## Research Options

### Option A: Attention-Guided BPE (AG-BPE)

**Approach**: Use Hindi LM attention patterns to guide BPE merges.

**Methodology**:
1. Train or load a small Hindi language model (reuse tiny LM infrastructure)
2. Extract attention patterns from LM for corpus samples
3. Compute co-attention scores for grapheme/grapheme-cluster pairs
4. Modify BPE training to weight merge candidates by attention scores
5. Train tokenizer with attention-guided merges

**Advantages**:
- Leverages semantic relationships learned by LM
- Relatively straightforward to implement
- Can reuse existing tiny LM infrastructure

**Challenges**:
- Requires trained LM (chicken-and-egg problem)
- Attention extraction can be computationally expensive
- Need to balance attention guidance with frequency

**References**:
- AG-BPE: "Attention-Guided BPE for Subword Tokenization"
- Related: Mutual information-based tokenization

### Option B: Hierarchical/Fractal Grouping

**Approach**: Multi-level tokenization preserving linguistic hierarchy.

**Methodology**:
1. Level 0: Aksharas (Devanagari base units)
2. Level 1: Morphemes (morphological segments)
3. Level 2: Sub-morpheme features (semantic components)
4. Apply BPE at each level with level-specific constraints

**Advantages**:
- Preserves linguistic structure explicitly
- Naturally handles morphology
- Fractal structure allows fine-grained control

**Challenges**:
- More complex implementation
- Requires morphological segmentation
- Need to define hierarchy clearly

**References**:
- Fractal tokenization concepts
- Hierarchical tokenization in NLP

### Option C: Hybrid Approach

**Approach**: Combine attention guidance with hierarchical structure.

**Methodology**:
- Use hierarchical grouping for structure
- Use attention guidance for merge decisions within levels
- Best of both worlds

**Advantages**:
- Combines benefits of both approaches
- More flexible

**Challenges**:
- Most complex to implement
- More hyperparameters to tune

---

## Selected Approach: Attention-Guided BPE (AG-BPE)

**Rationale**:
- More straightforward to implement (start modest)
- Can reuse existing tiny LM infrastructure
- Well-aligned with emerging research (AG-BPE papers)
- Easier to iterate and refine

**Alignment with Recent AG-BPE Work**:
Recent AG-BPE research (HuggingFace blog + papers) demonstrates:
- Start from vanilla BPE
- Add semantic score from Transformer ("ContextAnalyzer") on top of frequency
- Merge pairs by hybrid score (e.g., `freq + λ × attention_score`)
- Results show more morphologically meaningful vocabularies and better compression vs GPT-4-style tokenizers

Our implementation follows this approach, applied to Hindi with:
- Mutual information weighting (statistical association)
- Attention patterns (if LM available)
- Frequency-based scoring
- Combined hybrid score for merge decisions

**Important Note: Pre-Training Pitfall**:
AG-BPE literature warns about the "pre-training pitfall": if the context/attention model is **pretrained and frozen**, semantics may misalign relative to the LM that will use the tokenizer. They argue for **joint training** (tokenizer guidance + LM).

**Current Implementation (v1)**:
- **Offline/pretrained**: Uses frozen LM for attention extraction (if available)
- **Future work (v2)**: Joint training of tokenizer guidance + LM for better alignment

For exploratory research on Hindi tokenization, the offline approach is acceptable. For production deployment, joint training would be preferred.

**Implementation Plan**:
1. Train small Hindi LM (or use existing tiny LM)
2. Extract attention patterns for corpus samples
3. Compute co-attention/mutual information for pairs
4. Modify BPE training to weight merges by attention
5. Train AG-BPE tokenizer
6. Evaluate against baseline

---

## Success Criteria

For AG-BPE tokenizer to be considered successful:

1. **Script & Morphology**: Maintains or improves vs baseline
   - Grapheme violation ≤ baseline (0%)
   - Akshara integrity ≥ baseline (100%)
   - Morphology F1 ≥ baseline (0.469)

2. **Efficiency**: Reduces fertility/NSL/token tax
   - Fertility < baseline
   - NSL < baseline
   - Token tax < baseline

3. **Downstream**: Improves or matches tiny LM perplexity
   - Perplexity ≤ baseline

---

## Next Steps

1. Implement AG-BPE trainer (`tokenizers/ag_bpe_trainer.py`)
2. Create AG-BPE tokenizer adapter (`tokenizers/ag_bpe_tokenizer.py`)
3. Register in tokenizer registry
4. Train AG-BPE tokenizer
5. Run comprehensive evaluation
6. Compare with baseline and document results

