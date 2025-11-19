#!/bin/bash
# Quick-start script for training all tokenizers on desktop machine
# Usage: ./scripts/train_all_tokenizers.sh

set -e  # Exit on error

echo "=========================================="
echo "Training All Tokenizers"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if corpus exists
CORPUS="data/hindi/processed/gpe_cbpe_hi_corpus.txt"
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: Corpus not found at $CORPUS"
    echo "Please ensure the corpus file exists."
    exit 1
fi

echo ""
echo "Corpus: $CORPUS"
echo "Size: $(du -h $CORPUS | cut -f1)"
echo ""

# Train GPE+CBPE
echo "=========================================="
echo "1. Training GPE+CBPE Tokenizer"
echo "=========================================="
python3 scripts/train_gpe_tokenizer.py \
    --input "$CORPUS" \
    --output-dir models/gpe_cbpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2

echo ""
echo "✓ GPE+CBPE training complete"
echo ""

# Verify GPE+CBPE
if [ -f "models/gpe_cbpe_hi_v1/vocab.json" ]; then
    VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('models/gpe_cbpe_hi_v1/vocab.json'))))")
    echo "  Vocab size: $VOCAB_SIZE"
fi

# Train AG-BPE
echo "=========================================="
echo "2. Training AG-BPE Tokenizer"
echo "=========================================="
python3 scripts/train_ag_bpe_tokenizer.py \
    --input "$CORPUS" \
    --output-dir models/ag_bpe_hi_v1 \
    --vocab-size 32000 \
    --min-pair-frequency 2 \
    --attention-weight 0.5 \
    --mi-weight 0.3 \
    --frequency-weight 0.2

echo ""
echo "✓ AG-BPE training complete"
echo ""

# Verify AG-BPE
if [ -f "models/ag_bpe_hi_v1/vocab.json" ]; then
    VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('models/ag_bpe_hi_v1/vocab.json'))))")
    echo "  Vocab size: $VOCAB_SIZE"
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run evaluation suite:"
echo "   python3 scripts/run_baseline_evaluation.py --tokenizer-id gpe_cbpe_hi_v1"
echo "   python3 scripts/run_semantic_evaluation.py --tokenizer-id ag_bpe_hi_v1"
echo ""
echo "2. Train tiny LMs for perplexity comparison"
echo "3. Fill in results documentation"
echo ""

