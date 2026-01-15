# SurgViVQA-Audio: Audio-Grafted VLM for Surgical Video Question Answering

Audio-enhanced vision-language model for colonoscopy video understanding. Grafts Whisper audio encoder into Qwen2-VL for direct audio+vision processing.

## Overview

**Problem:** Existing surgical VQA models rely on ASR transcription, adding latency and error propagation.

**Solution:** Direct audio embedding via grafted Whisper encoder, enabling the model to "hear" and "see" surgical procedures simultaneously.

**Results:**
- Baseline (Audio+Image, zero-shot): 46% accuracy
- This model (fine-tuned): Training in progress
- Inference speed: 2.5× faster than ASR pipeline

## Architecture

```
Audio (MP3) ──→ Whisper Encoder ──→ Audio Projector ──┐
                                                        ├──→ Qwen2-VL-7B ──→ Answer
Frames (8×) ──→ Vision Encoder ──────────────────────→┘
Question ──→ Text Embeddings ─────────────────────────→┘
```

**Key Innovation:** Audio tokens (1500) are injected directly into the multimodal input sequence, bypassing ASR.

## Project Structure

```
SurgViVQA-Audio/
├── src/
│   ├── train_vqa.py          # Training script (QLoRA + audio grafting)
│   ├── evaluate_checkpoint.py # Standalone evaluation
│   └── dataset.py             # SurgicalVQADataset with audio injection
├── baselines/
│   ├── baseline1_text_image.py   # Text + Image (no audio)
│   ├── baseline2_audio_image.py  # Audio + Image (direct, ours)
│   ├── baseline3_asr_pipeline.py # ASR → Text + Image
│   └── utils.py                   # Shared utilities
├── transformers_fork/         # Modified transformers (audio support)
├── scripts/
│   ├── bootstrap.sh           # Environment setup
│   └── verify_tokenizer.sh   # Tokenizer validation
├── test_set/                  # Small JSONL test sets (kept in Git)
├── docs/                      # Documentation and guides
│   ├── EVAL_STRATEGY.md
│   ├── FIXES_APPLIED.md
│   └── RUN_TRAINING_A100.md
├── requirements.txt
└── README.md
```

## Quick Start

### Setup (RunPod Network Volume)

```bash
# 1. Clone repo
git clone https://github.com/your-username/SurgViVQA-Audio
cd SurgViVQA-Audio

# 2. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download data (one-time, on network volume)
# Sync from your data source (HuggingFace/R2/upload)
```

### Training

```bash
# Train on 50 samples (test run)
python src/train_vqa.py \
    --train_data_path test_set/in_002-001.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/in_002-001 \
    --output_dir checkpoints/surgical_vqa_50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --bf16 True
```

### Evaluation

```bash
# Standalone evaluation (after training)
python src/evaluate_checkpoint.py \
    --checkpoint_path checkpoints/surgical_vqa_50 \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/out_002-001 \
    --output_file results/eval_out.jsonl
```

## Technical Details

### Audio Grafting

- **Base Model:** Qwen2-VL-7B-Instruct (vision + language)
- **Audio Encoder:** Whisper Large v3 Turbo (frozen)
- **Audio Projector:** 1280 → 3584 (trainable)
- **Stage 1:** Train audio projector on audio→text task
- **Stage 2:** Fine-tune on surgical VQA with LoRA

### Training Configuration

- **Method:** QLoRA (4-bit base + bf16 LoRA adapters)
- **LoRA rank:** 64, alpha 16
- **Target modules:** Q/K/V/O projections + MLP (decoder only)
- **Modules to save:** Audio projector (full precision)
- **Label masking:** Train only on assistant's answer (not vision tokens)

### Dataset Format

```jsonl
{
  "id": "qa_000357",
  "question": "How severe is the fluid occlusion?",
  "answer": "Fluid occlusion is absent and clean mucosa stays visible.",
  "short_answer": "absent",
  "question_type": "fluid_occlusion_level",
  "frames": ["002-001_18743", "002-001_18788", ...],  // 8 frames
}
```

### Memory Optimization

- **Image resize:** 1350×1080 → 384×384 (preserves detail, reduces tokens)
- **Attention:** SDPA (handles mixed precision, ~25% slower than FlashAttention)
- **Eval frames:** 6 frames (vs 8 for training) to prevent OOM
- **Gradient checkpointing:** Enabled (reduces activation memory)

## Key Fixes Applied

### 1. Label Construction (Critical)
**Problem:** Training on vision tokens (unpredictable) wasted model capacity.

**Fix:** Mask everything except assistant's answer.
```python
# Find assistant response start, mask everything before it
for i in range(len(final_labels) - assistant_start_len):
    if torch.equal(final_input_ids[i:i+assistant_start_len], assistant_start_ids):
        final_labels[:i+assistant_start_len] = -100  # Ignore
        break
```

**Result:** Trainable tokens: 46% → 10%, loss: 5.x → 0.0001 (5-sample overfit)

### 2. PAD/EOS Token Separation
**Status:** Already correct in tokenizer (PAD: 151643, EOS: 151645)

**Safety:** Training script adds dedicated PAD token if missing.

### 3. Eval Memory Management
**Solution:** Use 6 evenly-sampled frames for eval (vs 8 for training)
```python
stride = len(frames) / max_eval_frames
selected = [frames[int(i * stride)] for i in range(max_eval_frames)]
```

## Baselines

| Method | In-Domain | Out-Domain | Speed (samples/sec) |
|--------|-----------|------------|---------------------|
| Text + Image (no audio) | 58% | 50% | 0.11 |
| **Audio + Image (ours, zero-shot)** | **46%** | **48%** | **1.07** |
| ASR → Text + Image | 62% | 60% | 0.43 |

**Goal:** Fine-tuning should improve 46% → 55-65% while maintaining 2.5× speed advantage.

## Citation

```bibtex
@article{abdullah2024surgvivqa,
  title={SurgViVQA-Audio: Audio-Grafted Vision-Language Models for Surgical Video QA},
  author={Abdullah, Kulsoom},
  journal={In preparation},
  year={2024}
}
```

## Acknowledgments

- Base model: [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- Audio encoder: [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- Surgical VQA dataset: EndoVis Challenge

## License

[Your License - MIT/Apache 2.0/etc.]
