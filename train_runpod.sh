#!/bin/bash
# Quick training script for RunPod
# Run this on RunPod after uploading your data

set -e

echo "🚀 Starting Audio-Grafted Surgical VQA Training"
echo "================================================"
echo ""
echo "Configuration:"
echo "  - Model: kulsoom-abdullah/Qwen2-Audio-7B-Transcription (4-bit quantized)"
echo "  - Method: QLoRA (4-bit base + bf16 LoRA adapters)"
echo "  - Attention: SDPA (compatible with quantized weights)"
echo "  - Frames: 8 frames @ 384px (train AND eval)"
echo "  - Batch size: 1 (effective=4 with gradient accumulation)"
echo ""


# Set W&B project (change if needed)
export WANDB_PROJECT="surgical-vqa"
export WANDB_LOG_MODEL="false"  # Don't upload full checkpoints to W&B
# export PYTHONPATH=/workspace/SurgViVQA-Audio/transformers_fork/src:$PYTHONPATH


NUM_GPUS=${NUM_GPUS:-1}          # export NUM_GPUS=2 to use both 4090s

if [ -d "venv" ]; then
    source venv/bin/activate
fi
export WANDB_PROJECT="surgical-vqa"
export WANDB_LOG_MODEL="false"

# Build the command: python by default, torchrun when >1 GPU
CMD="python"
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=12345"
fi

$CMD src/train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_50 \
    --run_name "surgical-vqa-50-samples-$(date +%Y%m%d-%H%M%S)" \
    --train_data_path test_set/in_002-001.jsonl \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/in_002-001 \
    --eval_audio_dir data/audio/out_002-001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --warmup_ratio 0.05 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 5 \
    --report_to wandb

echo ""
echo "✅ Training complete!"
echo "Checkpoint saved to: ./checkpoints/surgical_vqa_50"
