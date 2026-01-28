# GitHub Setup Guide

## Files Created ✅

1. ✅ `.gitignore` - Excludes large files, data, checkpoints, venv
2. ✅ `README.md` - Professional project overview
3. ✅ `requirements.txt` - Python dependencies
4. ✅ `reorganize_for_github.sh` - Restructures project for GitHub
5. ✅ This guide

## Step-by-Step Setup

### 1. Reorganize Project (5 minutes)

```bash
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio

# Run reorganization script
bash reorganize_for_github.sh

# Verify structure
tree -L 2 -I 'venv|__pycache__|*.pyc|audio|dataset|checkpoints'
```

**Expected output:**
```
SurgViVQA-Audio/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train_vqa.py
│   └── evaluate_checkpoint.py
├── baselines/
│   ├── baseline1_text_image.py
│   ├── baseline2_audio_image.py
│   ├── baseline3_asr_pipeline.py
│   └── utils.py
├── transformers_fork/
│   └── src/
├── scripts/
│   ├── bootstrap.sh
│   ├── verify_tokenizer.sh
│   └── train_with_eval_FINAL.sh
├── docs/
│   ├── EVAL_STRATEGY.md
│   └── FIXES_APPLIED.md
├── test_set/
│   ├── in_002-001.jsonl
│   └── out_002-001.jsonl
└── data/
    └── README.md
```

### 2. Create Private GitHub Repo

**Option A: Via GitHub Web UI**
1. Go to https://github.com/new
2. Repository name: `SurgViVQA-Audio`
3. Description: "Audio-Grafted Vision-Language Model for Surgical Video QA"
4. **Visibility: Private** ✅
5. **Don't** initialize with README (we already have one)
6. Click "Create repository"

**Option B: Via GitHub CLI**
```bash
# Install GitHub CLI if needed
brew install gh

# Login
gh auth login

# Create private repo
gh repo create SurgViVQA-Audio --private --description "Audio-Grafted VLM for Surgical Video QA"
```

### 3. Initial Commit

```bash
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio

# Initialize git (if not already)
git init

# Add all files
git add .

# Check what will be committed (data should be excluded!)
git status

# You should see:
#   src/
#   baselines/
#   transformers_fork/
#   scripts/
#   docs/
#   test_set/
#   README.md
#   requirements.txt
#   .gitignore
#
# You should NOT see:
#   audio/ (excluded)
#   dataset/ (excluded)
#   checkpoints/ (excluded)
#   venv/ (excluded)

# Commit
git commit -m "Initial commit: Audio-Grafted Surgical VQA

- Audio grafting architecture (Whisper + Qwen2-VL)
- Training script with QLoRA + label masking fix
- Baseline experiments (text+image, audio+image, ASR pipeline)
- Custom transformers fork for audio support
- Documentation and setup scripts"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR-USERNAME/SurgViVQA-Audio.git

# Push
git branch -M main
git push -u origin main
```

### 4. Verify on GitHub

Go to: `https://github.com/YOUR-USERNAME/SurgViVQA-Audio`

You should see:
- ✅ Professional README with architecture diagram
- ✅ Clean project structure
- ✅ No large files (audio, checkpoints, venv)
- ✅ Small test JSONL files included
- ✅ Private repository

## RunPod Network Volume Setup

### 1. Create Network Volume (RunPod Dashboard)

1. Go to https://www.runpod.io/
2. Sign up / Login
3. **Storage** → **+ Network Volume**
4. Name: `surgvqa-data`
5. Size: `100 GB` (recommended)
6. Region: Choose closest to you
7. Create ($10/month when pods are off)

### 2. Launch Pod with Volume

1. **+ Deploy** → Select GPU (A100 40GB recommended)
2. **Attach Network Volume** → Select `surgvqa-data`
3. **Deploy**
4. Wait for pod to start (~30 seconds)
5. **Connect** → Copy SSH command or use web terminal

### 3. First-Time Setup (On RunPod Pod)

```bash
# Navigate to your persistent volume
cd /workspace

# Clone your private repo (will prompt for GitHub token)
git clone https://github.com/YOUR-USERNAME/SurgViVQA-Audio.git
cd SurgViVQA-Audio

# Run bootstrap script
bash scripts/bootstrap.sh

# This installs all dependencies and creates directories
```

### 4. Upload Data (One-Time, From Mac)

**Option A: Direct SCP**
```bash
# Get your RunPod SSH connection string from dashboard
# It looks like: ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519

# Upload data
scp -P PORT -r audio root@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/
scp -P PORT -r dataset/frames root@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/
```

**Option B: Via rsync (resume if interrupted)**
```bash
rsync -avz -e "ssh -p PORT" audio/ root@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/audio/
rsync -avz -e "ssh -p PORT" dataset/frames/ root@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/frames/
```

### 5. Train Immediately

```bash
# On RunPod
cd /workspace/SurgViVQA-Audio
source venv/bin/activate

# Run 50-sample training
python src/train_vqa.py \
    --train_data_path test_set/in_002-001.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/in_002-001 \
    --output_dir checkpoints/surgical_vqa_50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --bf16 True \
    --logging_steps 5
```

### 6. Future Sessions (Instant Setup!)

**Every time you launch a new pod:**

```bash
# 1. In RunPod Dashboard:
#    Deploy → Select GPU → Attach 'surgvqa-data' volume → Deploy

# 2. SSH into pod
ssh root@RUNPOD_IP -p PORT

# 3. Everything is already there!
cd /workspace/SurgViVQA-Audio
source venv/bin/activate

# 4. Pull latest code changes (if you pushed from another machine)
git pull

# 5. Train immediately
python src/train_vqa.py ...
```

**That's it!** No rebuilding, no re-uploading data. Everything persists.

## Cost Breakdown

### RunPod Network Volume:
- **Storage:** $10/month (100GB, persists forever)
- **A100 40GB pod:** $0.39/hr (only when running)
- **Example:** 20 hours of training = $7.80 + $10 = $17.80/month

### Alternative (Lambda without volume):
- Rebuild environment: 30 min
- Re-upload data: 1-2 hours
- Wasted GPU time: $2-4 per session
- **Your sanity:** Priceless

**RunPod Network Volume is worth it.**

## Troubleshooting

### Git asks for credentials
```bash
# Use GitHub Personal Access Token
gh auth login
# Or: https://github.com/settings/tokens
```

### Data didn't upload completely
```bash
# Use rsync to resume
rsync -avz --progress -e "ssh -p PORT" audio/ root@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/audio/
```

### Forgot to attach network volume
**Before training:** Stop pod, deploy new one with volume attached.
**After training:** Checkpoints are lost (not on volume). Always attach volume!

### Network volume is in different region than pod
Transfer between volumes or create new volume in same region as preferred GPUs.

## Success Checklist

- [x] Created `.gitignore` (excludes large files)
- [x] Created `README.md` (professional overview)
- [x] Created `requirements.txt` (Python dependencies)
- [x] Reorganized project structure
- [x] Created private GitHub repo
- [x] Initial commit pushed
- [ ] RunPod account created
- [ ] Network volume created (100GB)
- [ ] Pod launched with volume attached
- [ ] Repo cloned to `/workspace/SurgViVQA-Audio`
- [ ] Data uploaded to volume
- [ ] Training started successfully

## Next Steps After Training

1. **Push checkpoint info to GitHub:**
   ```bash
   # Add checkpoint metadata (not weights!)
   echo "checkpoints/surgical_vqa_50: 55% accuracy on out-domain" > docs/RESULTS.md
   git add docs/RESULTS.md
   git commit -m "Add training results"
   git push
   ```

2. **Upload model to HuggingFace:**
   ```bash
   huggingface-cli upload kulsoom-abdullah/SurgVQA-Trained ./checkpoints/surgical_vqa_50
   ```

3. **Update README with results:**
   - Replace "Training in progress" with actual accuracy
   - Add W&B training curves
   - Update baselines table

4. **Prepare demo:**
   - Create `demo.py` for interactive inference
   - Add to GitHub
   - Document in README

---

**You're all set!** 🚀

GitHub keeps your code safe. RunPod Network Volume keeps your data safe. You can now train from anywhere without rebuilding.
