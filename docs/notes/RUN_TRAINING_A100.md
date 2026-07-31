# Running Surgical VQA Training on A100 (BF16 + FlashAttention)

## What Changed

**Problem**: Loading your checkpoint with 4-bit quantization created `uint8` tensors incompatible with FlashAttention

**Solution**: Load the same checkpoint in full BF16 (no quantization). Your checkpoint is already merged Stage 1 + Stage 2!

## Upload Training Script

```bash
# On your Mac
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio

scp train_vqa.py ubuntu@132-145-138-210:~/audiograft/SurgViVQA-Audio/
```

## Run Training on A100

```bash
# SSH to Lambda
ssh ubuntu@132-145-138-210
cd ~/audiograft/SurgViVQA-Audio
source ~/venvs/surg-audio/bin/activate

# Run training with BF16 (no quantization)
python train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_bf16_test \
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

## Key Changes in train_vqa.py

1. ✅ **Removed quantization**: No `BitsAndBytesConfig`, loads full BF16
2. ✅ **FlashAttention2**: Memory-efficient attention for 8 frames
3. ✅ **Fixed collator**: Explicitly casts `pixel_values` and `input_features` to bf16
4. ✅ **8 frames @ 512px**: Optimal for temporal questions in your dataset
5. ✅ **LoRA on decoder + full training on audio_projector**: Same as your working Stage-2 setup

## Memory Estimate (40GB A100)

```
Model (bf16):                 ~14 GB
LoRA adapters (bf16):         ~400 MB
8 frames @ 512px activations: ~4 GB
Gradient buffers:             ~3 GB
Optimizer (8-bit AdamW):      ~4 GB
FlashAttention overhead:      ~1 GB
Working memory:               ~2 GB
--------------------------------------------
Total:                        ~28.5 GB ✓ (fits comfortably in 40GB)
```

## What to Expect

### During Load:
```
⏳ Loading merged BF16 model with FlashAttention2...
🔧 Ensuring multimodal components are bf16...
✓ Model loaded in full BF16
🧠 Adding LoRA Adapters...
trainable params: XXX M || all params: 7.6B || trainable%: ~2-3%
```

### During Training:
- First epoch should take ~5-10 minutes (50 samples, batch=1, grad_accum=4)
- Eval every 10 steps
- Watch W&B for loss curves

## Troubleshooting

### If you get OOM:
```bash
# Reduce to 6 frames or 384px resolution
# Edit line 126 in train_vqa.py:
if max(img.size) > 384:  # Was 512
    img.thumbnail((384, 384))
```

### If you get dtype errors:
This means the checkpoint wasn't loaded correctly. Run this debug:
```bash
python -c "
from transformers import Qwen2VLForConditionalGeneration
import torch
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'kulsoom-abdullah/Qwen2-Audio-7B-Transcription',
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    trust_remote_code=True
)
print('Model dtype check:')
for name, param in model.named_parameters():
    if param.dtype not in [torch.bfloat16, torch.float16]:
        print(f'  {name}: {param.dtype}')
"
```

All params should be bf16. If you see uint8/float32, the HuggingFace checkpoint has issues.

### Check GPU memory during training:
```python
# Add to train() function after first batch
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Success Criteria

- ✅ Training starts without dtype errors
- ✅ Loss decreases over epochs
- ✅ Eval loss improves (goal: beat 46% baseline accuracy)
- ✅ Memory usage stays under 35GB

Let me know when you start training and I'll help monitor progress!
python -c "
import torch
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    './qwen2_audio_vl_merged_bf16',
    torch_dtype=torch.bfloat16,
    device_map='cpu'
)
print('vision qkv:', model.visual.blocks[0].attn.qkv.weight.dtype)
print('vision proj:', model.visual.blocks[0].attn.proj.weight.dtype)
print('vision block dtype:', model.visual.get_dtype())
"