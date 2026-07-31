# BF16 FlashAttention Training Setup

## Problem Summary

- **Issue**: QLoRA (4-bit quantization) creates `uint8` tensors that FlashAttention can't process
- **Solution**: Merge Stage-2 adapters into full BF16 checkpoint, train WITHOUT quantization

## Step 1: Merge Stage-2 to BF16

First, you need to provide the paths to your checkpoints:
- **Stage-1**: Your audio-grafted Qwen2-VL base model (bf16)
- **Stage-2**: Your LoRA adapters from audio-only training

Run the merge script:

```bash
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio

python merge_stage2_to_bf16.py \
    --stage1_model_path /path/to/your/stage1/checkpoint \
    --stage2_adapter_path /path/to/your/stage2/checkpoint \
    --output_path ./qwen2_audio_vl_merged_bf16
```

**Example if your checkpoints are:**
- Stage-1: `/Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft/checkpoints/stage1_audio_graft`
- Stage-2: `/Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft/checkpoints/stage2_lora_audio`

```bash
python merge_stage2_to_bf16.py \
    --stage1_model_path /Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft/checkpoints/stage1_audio_graft \
    --stage2_adapter_path /Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft/checkpoints/stage2_lora_audio \
    --output_path ./qwen2_audio_vl_merged_bf16
```

This will create a merged checkpoint at `./qwen2_audio_vl_merged_bf16/`

## Step 2: Update train_vqa.py

Edit line 64 in `train_vqa.py`:

```python
# BEFORE
MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

# AFTER (use your local merged checkpoint)
MODEL_ID = "./qwen2_audio_vl_merged_bf16"
```

## Step 3: Upload to Lambda Labs

```bash
# Upload both files
scp merge_stage2_to_bf16.py train_vqa.py ubuntu@132-145-138-210:~/audiograft/SurgViVQA-Audio/

# If you already have the merged checkpoint, upload it too:
scp -r qwen2_audio_vl_merged_bf16 ubuntu@132-145-138-210:~/audiograft/SurgViVQA-Audio/
```

## Step 4: Run Training on A100

```bash
# SSH to Lambda
ssh ubuntu@132-145-138-210
cd ~/audiograft/SurgViVQA-Audio
source ~/venvs/surg-audio/bin/activate

python train_vqa.py \
    --model_name_or_path ./qwen2_audio_vl_merged_bf16 \
    --output_dir ./checkpoints/surgical_vqa_bf16 \
    --run_name "surgical-vqa-bf16-001" \
    --train_data_path test_set/in_002-001.jsonl \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/in_002-001 \
    --eval_audio_dir audio/out_002-001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --warmup_ratio 0.0 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --bf16 True \
    --report_to wandb
```

## Memory Estimate (40GB A100)

With full BF16 + FlashAttention2:

```
Model (bf16):            ~14 GB
LoRA adapters (bf16):    ~400 MB
8 frames @ 512px:        ~4 GB
Gradient buffers:        ~3 GB
Optimizer (8-bit):       ~4 GB
FlashAttention overhead: ~1 GB
Working memory:          ~2 GB
-------------------------
Total:                   ~28.5 GB ✓
```

## Key Changes Made

1. **Removed quantization**: No more `BitsAndBytesConfig` or `prepare_model_for_kbit_training`
2. **Fixed collator**: Now explicitly casts `pixel_values` and `input_features` to bf16
3. **Removed aggressive casting**: Trust PEFT to handle LoRA dtypes correctly
4. **Kept FlashAttention2**: Memory efficient for long sequences (8 frames + 1500 audio tokens)
5. **Reduced frame size**: 768px → 512px for extra memory headroom

## Troubleshooting

### If you still get OOM:
- Reduce frames to 6: modify dataset `__getitem__` to sample fewer frames
- Reduce resolution to 384px: change line 126 threshold
- Add gradient checkpointing (already enabled by default)

### If you still get dtype errors:
```bash
# Add debug to see what dtype is causing issues
python -c "
import torch
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained('./qwen2_audio_vl_merged_bf16', torch_dtype=torch.bfloat16, device_map='cpu')
for name, param in model.named_parameters():
    if param.dtype not in [torch.bfloat16, torch.float16]:
        print(f'{name}: {param.dtype}')
"
```

All parameters should be bf16. If you see float32, the merge didn't work correctly.
