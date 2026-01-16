#!/bin/bash
# Train on FULL 002-001 dataset (700 samples) while other videos download
# This keeps GPUs busy and validates the pipeline at scale

set -e

echo "🚀 Training on FULL Video 002-001 (700 samples)"
echo "================================================"
echo ""
echo "Strategy:"
echo "  - Training: First 600 samples from 002-001 (OUT natural questions)"
echo "  - Eval: Last 100 samples from 002-001 (OUT natural questions)"
echo "  - Epochs: 5 (optimal for 600 samples based on 50-sample results)"
echo "  - This will take ~2.5 hours on 2x RTX 4090"
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

# Create train/val split (600 train, 100 eval)
echo "📊 Creating train/eval split..."
python3 << 'EOF'
import json

# Load full 002-001 OUT data (700 samples, natural language)
with open('out_template.jsonl') as f:
    all_data = [json.loads(line) for line in f]

# Filter only 002-001
video_002_001 = [s for s in all_data if s['video_id'] == '002-001']
print(f"Total 002-001 samples: {len(video_002_001)}")

# Split: 600 train, 100 eval
train_data = video_002_001[:600]
eval_data = video_002_001[600:]

# Save splits
with open('train_002001_600.jsonl', 'w') as f:
    for sample in train_data:
        f.write(json.dumps(sample) + '\n')

with open('eval_002001_100.jsonl', 'w') as f:
    for sample in eval_data:
        f.write(json.dumps(sample) + '\n')

print(f"✓ Created train_002001_600.jsonl ({len(train_data)} samples)")
print(f"✓ Created eval_002001_100.jsonl ({len(eval_data)} samples)")
EOF

echo ""
echo "🚀 Starting training..."
echo ""

# Build command
CMD="python"
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=12345"
fi

$CMD src/train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_002001_600 \
    --run_name "surgical-vqa-002001-600samples-$(date +%Y%m%d-%H%M%S)" \
    --train_data_path train_002001_600.jsonl \
    --eval_data_path eval_002001_100.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/out_002-001 \
    --eval_audio_dir data/audio/out_002-001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 5 \
    --warmup_ratio 0.05 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 25 \
    --save_strategy steps \
    --save_steps 150 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --bf16 True \
    --logging_steps 10 \
    --report_to wandb

echo ""
echo "✅ Training complete!"
echo "Checkpoint saved to: ./checkpoints/surgical_vqa_002001_600"
echo ""
echo "Next steps:"
echo "1. Run evaluation: python src/evaluate_checkpoint.py ..."
echo "2. Upload other videos (002-002, 002-003, 002-004)"
echo "3. Launch full multi-video training"
