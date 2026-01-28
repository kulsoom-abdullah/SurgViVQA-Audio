#!/bin/bash
# Training with evaluation enabled (memory-optimized with fixes)
#
# FIXES APPLIED:
# 1. PAD token separated from EOS (stops ForCanBeConverted spam)
# 2. Eval uses 6 frames instead of 8 (reduces memory)
# 3. Labels only train on answer (not vision tokens)
# 4. Eval batch size = 1 with accumulation

cd ~/audiograft/SurgViVQA-Audio
source ~/venvs/surg-audio/bin/activate

python train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_with_eval \
    --run_name "surgical-vqa-eval-v1" \
    --train_data_path test_set/in_002-001.jsonl \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/in_002-001 \
    --eval_audio_dir audio/out_002-001 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --warmup_ratio 0.0 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --bf16 True \
    --report_to wandb \
    --logging_steps 5

# Key settings:
# - Eval uses 6 frames (train uses 8) - set in train_vqa.py
# - PAD token automatically added if missing
# - Eval batch size 1 + accumulation 4 (mirrors training)
# - SDPA attention (memory efficient)
