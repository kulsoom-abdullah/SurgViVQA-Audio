#!/bin/bash
# Evaluate the 50-sample checkpoint to see how it did

set -e

echo "🔍 Evaluating 50-sample checkpoint..."
echo ""

# Check if checkpoint exists
CHECKPOINT="./checkpoints/surgical_vqa_50"
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found at: $CHECKPOINT"
    echo "Trying to find it..."
    CHECKPOINT=$(find ./checkpoints -name "surgical_vqa*" -type d | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "❌ No checkpoint found!"
        exit 1
    fi
    echo "✓ Found: $CHECKPOINT"
fi

# Run evaluation
python src/evaluate_checkpoint.py \
    --checkpoint_path "$CHECKPOINT" \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/out_002-001 \
    --output_file results/eval_50sample_results.jsonl \
    --batch_size 1 \
    --max_image_size 384

echo ""
echo "✅ Evaluation complete!"
echo ""
echo "Results saved to: results/eval_50sample_results.jsonl"
echo ""
echo "To view per-question-type breakdown:"
echo "  cat results/eval_50sample_results.jsonl | python3 -m json.tool | less"
