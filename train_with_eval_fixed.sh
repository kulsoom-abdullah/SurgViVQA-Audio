#!/bin/bash
# Training with eval fix (batch size 1 + eval accumulation)
# Use this after merging to proper bf16 checkpoint

python train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_bf16_with_eval \
    --run_name "surgical-vqa-bf16-eval-001" \
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
    --gradient_checkpointing True \
    --report_to wandb \
    --logging_steps 5

# Key additions for eval fix:
# --per_device_eval_batch_size 1
# --eval_accumulation_steps 4
# --gradient_checkpointing True (forces it during eval too)
