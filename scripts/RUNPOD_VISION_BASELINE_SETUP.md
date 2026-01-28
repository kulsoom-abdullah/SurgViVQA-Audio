# RunPod Setup for Vision Baseline Comparison

## 🖥️ Hardware Requirements

**For Apples-to-Apples Comparison:**
Use **A100 80GB** (matches your original baseline experiments from January 14)

**Why A100?**
- Your original baselines ran on Lambda Labs A100 80GB
- Inference latency comparisons will be fair
- Faster inference (~2-3 hours for 3,000 samples)

**Alternative (cheaper but slower):**
- RTX 4090 (24GB) - ~4-6 hours for full experiment
- Still valid if all 3 baselines run on same GPU

---

## 🚀 RunPod Setup Steps

### 1. Launch Pod

```
GPU: A100 80GB (PCIE or SXM)
Template: PyTorch 2.x + CUDA 12.1
Volume: At least 100GB
```

### 2. Clone Repository

```bash
cd /workspace
git clone https://github.com/kulsoom-abdullah/SurgViVQA-Audio.git
cd SurgViVQA-Audio
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Install transformers fork (if needed for audio grafting)
# Not needed for text+image baselines
```

### 4. Upload Test Data

**Required files:**
- `test_multivideo.jsonl` (1,000 samples)
- `data/frames/` (frame images for all videos)

**Upload via RunPod web interface or rsync:**

```bash
# From local machine:
rsync -avz --progress \
    test_multivideo.jsonl \
    user@runpod-instance:/workspace/SurgViVQA-Audio/

rsync -avz --progress \
    data/frames/ \
    user@runpod-instance:/workspace/SurgViVQA-Audio/data/frames/
```

**Or download from cloud storage if you have it there**

### 5. Run Quick Test (Sanity Check)

```bash
# Run unit tests
python scripts/test_vision_baselines.py

# Run quick pipeline test (10 samples)
bash scripts/quick_baseline_test.sh
```

**Expected output:**
```
✅ ALL TESTS PASSED
✅ QUICK PIPELINE TEST PASSED!
```

### 6. Run Full Experiment

```bash
# This will take 2-3 hours on A100
bash scripts/run_all_vision_baselines.sh
```

**What it does:**
1. Runs Qwen 2.0 text+image on 1,000 samples
2. Runs Qwen 2.5 text+image on 1,000 samples
3. Runs Qwen 3.0 text+image on 1,000 samples
4. Generates comparison report

### 7. Download Results

```bash
# From local machine:
rsync -avz --progress \
    user@runpod-instance:/workspace/SurgViVQA-Audio/results/vision_comparison/ \
    ./results/vision_comparison/
```

---

## 📊 Expected Results Location

After completion, you'll have:

```
results/vision_comparison/
├── baseline_qwen2.0_full_test.jsonl    # 1,000 results
├── baseline_qwen2.5_full_test.jsonl    # 1,000 results
├── baseline_qwen3.0_full_test.jsonl    # 1,000 results
└── vision_comparison_report.txt        # Detailed comparison
```

---

## 🎯 Key Questions the Experiment Answers

**Question 1:** Does Qwen 3.0's vision encoder improve temporal reasoning?
- Check `lesion_motion_direction` (5-way): Did it go from ~20% to 40%+?
- Check `lesion_screen_position` (4-way): Did spatial reasoning improve?

**Question 2:** Should you audio-graft Qwen 3.0 next?
- **If yes:** Qwen 3.0 shows >10 point improvement on temporal questions
- **If no:** Need video-native model (VideoLLaMA, Video-ChatGPT)

---

## 💰 Cost Estimate

**A100 80GB on RunPod:**
- ~$2.50/hour
- 2-3 hours runtime
- **Total: ~$5-7.50**

**RTX 4090 (cheaper alternative):**
- ~$0.50/hour
- 4-6 hours runtime
- **Total: ~$2-3**

---

## ⚠️ Troubleshooting

**"Out of memory":**
- Close other processes
- Reduce batch size in baseline script (already set to 1)
- Use gradient checkpointing (already enabled)

**"Model download timeout":**
```bash
# Pre-download models
huggingface-cli login  # If needed
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')"
```

**"Frames not found":**
- Verify `data/frames/002-004/` exists and has .jpg files
- Check paths in test_multivideo.jsonl match actual frame names

---

## 📝 After Experiment

1. Download all results
2. Review `vision_comparison_report.txt`
3. Update your README with findings
4. Decide: Audio-graft Qwen 3.0 or switch to video-native model?
5. Shut down RunPod instance to stop charges
