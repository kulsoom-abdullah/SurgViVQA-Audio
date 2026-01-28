#!/bin/bash
# Fine-tune Qwen 3.0-VL (text+image only, NO audio) for fair comparison
# Uses same train/eval/test split as original Qwen 2.0 + audio run
# Expected time: 5-8 hours on 2x RTX 4090

set -e

echo "🎯 Qwen 3.0-VL Fine-Tuning (Text+Image Only)"
echo "=============================================="
echo ""
echo "Experimental goal:"
echo "  Test if Qwen 3.0's superior vision encoder can LEARN temporal reasoning"
echo "  from training data (without audio)."
echo ""
echo "Dataset:"
echo "  - Training: Videos 002-001, 002-002, 002-003 (~2,300 samples)"
echo "  - Eval: Same videos, stratified split (~400 samples)"
echo "  - Test: Video 002-004 held out (~1,000 samples)"
echo ""
echo "Comparison:"
echo "  - Qwen 3.0 zero-shot: 54.10% overall, 16% motion"
echo "  - Qwen 2.0 + audio fine-tuned: 63.40% overall, 20% motion"
echo ""
echo "Training config:"
echo "  - Base model: Qwen/Qwen3-VL-8B-Instruct"
echo "  - Method: QLoRA (4-bit base + bf16 LoRA)"
echo "  - Max epochs: 8"
echo "  - Early stopping: patience=3"
echo "  - Expected time: 5-8 hours"
echo ""

# Set W&B project
export WANDB_PROJECT="surgical-vqa"
export WANDB_LOG_MODEL="false"

# Multi-GPU settings
NUM_GPUS=${NUM_GPUS:-2}

# Check data files exist
echo "🔍 Checking data files..."
if [ ! -f "data/train_multivideo.jsonl" ]; then
    echo "❌ Error: data/train_multivideo.jsonl not found"
    echo "   Run: python scripts/create_multivideo_split.py"
    exit 1
fi

if [ ! -f "data/eval_multivideo.jsonl" ]; then
    echo "❌ Error: data/eval_multivideo.jsonl not found"
    exit 1
fi

echo "✓ Training data found"
echo "✓ Eval data found"
echo ""

echo "🚀 Starting Qwen 3.0 fine-tuning..."
echo "Monitor at: https://wandb.ai/AI_Healthcare/surgical-vqa"
echo ""

# Build command
CMD="python"
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=12346"
fi

$CMD src/train_qwen3_text_image.py \
    --output_dir ./checkpoints/qwen3_text_image \
    --run_name "qwen3-text-image-$(date +%Y%m%d-%H%M%S)" \
    --model_name_or_path "Qwen/Qwen3-VL-8B-Instruct" \
    --train_data_path data/train_multivideo.jsonl \
    --eval_data_path data/eval_multivideo.jsonl \
    --frames_dir data/frames \
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

# Auto-evaluate on test set
echo "🧪 Auto-evaluating on test set (1,000 held-out samples)..."
echo ""

python src/evaluate_qwen3.py \
    --checkpoint_path ./checkpoints/qwen3_text_image \
    --test_file data/test_multivideo.jsonl \
    --frames_dir data/frames \
    --output results/qwen3_finetuned_test.jsonl

echo ""
echo "Compare to:"
echo "  - Qwen 3.0 zero-shot: 54.10% (results/vision_comparison/qwen3.0_full.jsonl)"
echo "  - Qwen 2.0 + audio: 63.40% (from your original run)"
echo ""

# Optional: Auto-terminate RunPod instance when done
if [ -n "$RUNPOD_API_KEY" ] && [ -n "$RUNPOD_POD_ID" ]; then
    echo "🛑 Auto-terminating RunPod instance..."
    curl -s -X POST "https://api.runpod.io/v2/${RUNPOD_POD_ID}/terminate" \
         -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
         -H "Content-Type: application/json"
    echo "✅ Pod termination requested"
else
    echo "💡 Tip: Set RUNPOD_API_KEY and RUNPOD_POD_ID to auto-terminate when done"
fi
