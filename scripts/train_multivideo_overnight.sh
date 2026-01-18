#!/bin/bash
# Overnight training on full multi-video dataset
# Videos: 002-001, 002-002, 002-003 (video 002-004 held out for final test)
# Estimated time: 5-8 hours on 2x RTX 4090

set -e

echo "🌙 Overnight Multi-Video Training"
echo "=================================="
echo ""
echo "Dataset:"
echo "  - Training: Videos 002-001, 002-002, 002-003 (~2,300 samples)"
echo "  - Eval: Same videos, stratified split (~400 samples)"
echo "  - Test: Video 002-004 held out (~1,000 samples, for final evaluation)"
echo ""
echo "Training config:"
echo "  - Max epochs: 8"
echo "  - Early stopping: patience=3"
echo "  - Expected time: 5-8 hours"
echo "  - Will likely stop around epoch 5-6"
echo ""

# Set W&B project
export WANDB_PROJECT="surgical-vqa"
export WANDB_LOG_MODEL="false"

# Multi-GPU settings
NUM_GPUS=${NUM_GPUS:-2}

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create multi-video stratified split
echo "📊 Creating multi-video stratified split..."
python3 scripts/create_multivideo_split.py

# Check audio directories exist
echo ""
echo "🔍 Checking audio directories..."
for vid in 002-001 002-002 002-003; do
    if [ ! -d "data/audio/out_${vid}" ]; then
        echo "⚠️  Warning: data/audio/out_${vid} not found"
        echo "   Training will use silent audio for this video"
    else
        echo "✓ data/audio/out_${vid} found"
    fi
done

echo ""
echo "🚀 Starting overnight training..."
echo "Monitor at: https://wandb.ai/AI_Healthcare/surgical-vqa"
echo ""

# Build command
CMD="python"
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=12345"
fi

$CMD src/train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_multivideo \
    --run_name "surgical-vqa-multivideo-overnight-$(date +%Y%m%d-%H%M%S)" \
    --train_data_path data/train_multivideo.jsonl \
    --eval_data_path data/eval_multivideo.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio \
    --eval_audio_dir data/audio \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 8 \
    --warmup_ratio 0.05 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --early_stopping_patience 3 \
    --bf16 True \
    --logging_steps 20 \
    --report_to wandb

echo ""
echo "✅ Training complete!"
echo ""
echo "Results:"
echo "  - Best checkpoint: ./checkpoints/surgical_vqa_multivideo"
echo "  - Training log: ./checkpoints/surgical_vqa_multivideo/trainer_state.json"
echo ""
echo "Next steps:"
echo "1. Evaluate on eval set (from training):"
echo "   python src/evaluate_checkpoint.py --checkpoint_path ./checkpoints/surgical_vqa_multivideo ..."
echo ""
echo "2. Final evaluation on held-out video 002-004:"
echo "   python src/evaluate_checkpoint.py --checkpoint_path ./checkpoints/surgical_vqa_multivideo \\"
echo "       --eval_data_path data/test_multivideo.jsonl \\"
echo "       --frames_dir data/frames --audio_dir data/audio \\"
echo "       --output_file results/final_test_002004.jsonl"
