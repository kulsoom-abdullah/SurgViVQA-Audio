#!/bin/bash
# Train on full 002-001 with stratified split and early stopping
# This runs while other videos (002-002, 003, 004) are uploading

set -e

echo "🚀 Training on Full Video 002-001 with Stratified Split"
echo "========================================================"
echo ""
echo "Strategy:"
echo "  - Stratified train/eval split (85%/15% by question type)"
echo "  - Natural language questions (OUT dataset)"
echo "  - Early stopping enabled (patience=3, monitors eval_loss)"
echo "  - 10 max epochs (will likely stop early around epoch 5-7)"
echo "  - Estimated time: 2-3 hours on 2x RTX 4090"
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

# Create stratified split
echo "📊 Creating stratified train/eval split..."
python3 create_stratified_split.py

echo ""
echo "🚀 Starting training with early stopping..."
echo ""

# Build command
CMD="python"
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=12345"
fi

$CMD src/train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_002001_stratified \
    --run_name "surgical-vqa-002001-stratified-$(date +%Y%m%d-%H%M%S)" \
    --train_data_path train_002001_stratified.jsonl \
    --eval_data_path eval_002001_stratified.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/out_002-001 \
    --eval_audio_dir data/audio/out_002-001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 10 \
    --warmup_ratio 0.05 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 25 \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --early_stopping_patience 3 \
    --bf16 True \
    --logging_steps 10 \
    --report_to wandb

echo ""
echo "✅ Training complete!"
echo ""
echo "Results:"
echo "  - Best checkpoint: ./checkpoints/surgical_vqa_002001_stratified"
echo "  - Training stopped early based on eval_loss plateau"
echo "  - W&B logs: https://wandb.ai/AI_Healthcare/surgical-vqa"
echo ""
echo "Next steps:"
echo "1. Run evaluation: ./eval_50sample_checkpoint.sh (update checkpoint path)"
echo "2. Upload videos 002-002, 002-003, 002-004"
echo "3. Launch multi-video training (2700 samples)"
