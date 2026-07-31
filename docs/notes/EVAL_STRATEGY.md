# Evaluation & Scaling Strategy

## Current Status ✅

- **Training works perfectly**: Loss → 0.0001 on 5 samples
- **Label fix successful**: Only training on answer tokens (not vision tokens)
- **Ready for 50-sample training**: Pipeline validated

## Evaluation Options (Post-Training)

### Option 1: Standalone Eval Script (RECOMMENDED)

Run evaluation separately after training completes:

```bash
# After training finishes
python evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/surgical_vqa_50samples \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/out_002-001 \
    --output_file results/eval_50samples.jsonl \
    --batch_size 1 \
    --max_image_size 384
```

**Advantages:**
- ✅ Runs in fresh process (no accumulated memory from training)
- ✅ Can use `torch.cuda.empty_cache()` between samples
- ✅ Easy to debug if OOM occurs
- ✅ Can reduce frames/resolution just for eval without affecting training

**Memory-saving tricks:**
- Process one sample at a time (`batch_size=1`)
- Clear CUDA cache after each sample
- Use 384px images (or reduce to 320px if needed)
- Use SDPA (not FlashAttention) for eval

### Option 2: Reduce Frames for Eval Only

Modify dataset to use fewer frames during evaluation:

```python
# In train_vqa.py, add eval-specific frame sampling
def __getitem__(self, i):
    # ... load all 8 frames ...

    # If in eval mode, sample 4-6 frames instead of 8
    if self.is_eval and len(images) > 6:
        # Sample evenly (keep temporal diversity)
        indices = torch.linspace(0, len(images)-1, 6).long()
        images = [images[idx] for idx in indices]
```

**Trade-off:** Slightly less accurate eval (6 frames vs 8), but fits in memory.

### Option 3: Skip Eval During Training

What you're doing now - works great:

```bash
# Train without eval
python train_vqa.py \
    --train_data_path test_set/in_002-001.jsonl \
    --output_dir ./checkpoints/surgical_vqa_50samples \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 20 \
    --save_strategy epoch \
    --bf16 True

# Then run standalone eval
python evaluate_checkpoint.py ...
```

---

## Scaling Plan (After 50-Sample Success)

### Phase 1: Validate on 50 Samples ✅
**Goal:** Confirm training improves over 46% baseline

```bash
# 1. Train on 50 in-domain samples
python train_vqa.py --train_data_path test_set/in_002-001.jsonl ...

# 2. Eval on 50 out-domain samples
python evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/surgical_vqa_50samples \
    --eval_data_path test_set/out_002-001.jsonl \
    --output_file results/eval_50_out.jsonl

# 3. Also eval on in-domain (check for overfitting)
python evaluate_checkpoint.py \
    --eval_data_path test_set/in_002-001.jsonl \
    --output_file results/eval_50_in.jsonl
```

**Success criteria:**
- Out-domain accuracy > 46% (baseline)
- In-domain accuracy > out-domain (shows learning, not just memorization)

### Phase 2: Scale to Full Dataset
**Goal:** Train on full colonoscopy VQA dataset

**Dataset size estimates:**
- If you have ~500-1000 QA pairs total
- Split: 70% train (350-700), 15% val (75-150), 15% test (75-150)

```bash
# Full training run
python train_vqa.py \
    --train_data_path data/train_full.jsonl \
    --eval_data_path data/val_full.jsonl \
    --output_dir ./checkpoints/surgical_vqa_full \
    --run_name "surgical-vqa-full-v1" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 10 \
    --save_strategy steps \
    --save_steps 50 \
    --bf16 True \
    --report_to wandb
```

**Eval strategy for larger dataset:**
- Use standalone eval script (Option 1)
- Eval every 50 steps on small val set (~50 samples)
- Full eval at end on entire test set

### Phase 3: Demo Preparation

Once you have a trained model with good accuracy:

#### A. Create Inference Script

```bash
# Simple demo script for single sample
python demo_inference.py \
    --checkpoint ./checkpoints/surgical_vqa_full \
    --video_id 002-001 \
    --audio_file demo/sample_question.mp3 \
    --frames_dir demo/frames \
    --question "How severe is the fluid occlusion?"
```

#### B. Gradio Demo (Interactive)

Create a web interface:
- Upload audio question OR type text
- Select video clip or upload frames
- Model outputs answer in real-time

#### C. Batch Processing

For clinical validation:
- Process entire procedures
- Generate VQA pairs for review
- Export to structured format (JSON/CSV)

---

## Memory Optimization Checklist

If eval still OOMs:

**Level 1: Easy fixes**
- ✅ Use standalone eval script
- ✅ Reduce image size: 384px → 320px
- ✅ Clear CUDA cache between samples

**Level 2: Reduce frames**
- ✅ Sample 6 frames instead of 8
- ✅ Sample 4 frames (evenly spaced)
- Test if accuracy drop is acceptable

**Level 3: Model changes**
- Use gradient checkpointing during eval
- Quantize model to 8-bit for inference only
- Split long sequences (if questions >50 tokens)

---

## Next Steps (When You Resume)

1. ✅ **Let 50-sample training finish**
   - Should complete in ~20-30 minutes
   - Watch loss drop (expect ~1-2 final loss)

2. ✅ **Run standalone eval**
   ```bash
   python evaluate_checkpoint.py \
       --checkpoint_path ./checkpoints/surgical_vqa_50samples \
       --eval_data_path test_set/out_002-001.jsonl \
       --frames_dir dataset/frames \
       --audio_dir audio/out_002-001 \
       --output_file results/eval_out.jsonl
   ```

3. ✅ **Compare to baseline**
   - Baseline 2 (audio+image): 46% in-domain, 48% out-domain
   - Your model should beat this (goal: 55-65%)

4. **If eval OOMs:**
   - Reduce `--max_image_size` to 320
   - Or sample 6 frames in dataset

5. **If accuracy is good (>50%):**
   - Scale to full dataset
   - Prepare demo
   - Consider uploading checkpoint to HuggingFace

---

## Expected Results

**Conservative estimate:**
- 50 samples training: 50-55% accuracy
- Full dataset training: 60-70% accuracy
- Better than ASR pipeline (62%) if audio understanding matters

**Why this should work:**
- ✅ Model learned audio "reading" in Stage 2
- ✅ Training pipeline validated (5-sample overfit)
- ✅ Label fix ensures efficient learning
- ✅ Surgical domain is narrow (easier to specialize)

---

## PAD/EOS Token Fix (Inference Issue)

The "ForCanBeConverted" spam is because PAD == EOS in your tokenizer. This doesn't affect training but breaks generation.

**Check:**
```bash
python check_tokenizer_config.py
```

**Fix if needed:**
Update `tokenizer_config.json` in your checkpoint:
```json
{
  "pad_token": "<|endoftext|>",
  "eos_token": "<|im_end|>",
  ...
}
```

The updated `evaluate_checkpoint.py` script includes `attention_mask` which should prevent this issue.

---

## Questions to Consider

1. **How much data do you have total?**
   - Affects training time and expected accuracy

2. **What's the minimum acceptable accuracy for demo?**
   - Clinical validation might require >80%
   - Research demo might be fine with 60-70%

3. **Do you need real-time inference?**
   - Affects quantization strategy
   - Current speed: ~5-7s per sample (acceptable for most demos)

4. **Will you deploy this?**
   - Affects packaging (Docker, API, etc.)
   - Might need smaller model or quantization

---

Sleep well! When you resume:
1. Check if 50-sample training finished
2. Run `evaluate_checkpoint.py`
3. Compare results to 46% baseline
4. Decide on scaling strategy

The hard debugging is done - now it's just scaling up! 🎉
