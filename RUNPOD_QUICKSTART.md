# RunPod Quick Start - Copy & Paste Commands

## Step 1: Upload Files from Local Machine

### A. Get RunPod Connection Info
1. Go to your RunPod dashboard
2. Click "Connect" on your pod
3. Copy the SSH command (looks like: `ssh root@X.X.X.X -p XXXXX`)
4. Note the PORT number (the XXXXX after -p)

### B. Create Directory on RunPod
```bash
# Replace PORT and HOST with your values
ssh root@HOST -p PORT

# Create directory structure
mkdir -p /workspace/SurgViVQA-Audio/{baselines,test_set,audio,dataset/frames,results}
exit
```

### C. Upload from Local Machine
```bash
# Open new terminal on your Mac
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio

# Replace PORT and HOST in ALL commands below
PORT=XXXXX  # Your RunPod SSH port
HOST=X.X.X.X  # Your RunPod IP

# Upload code files (1 minute)
scp -P $PORT -r baselines test_set *.py *.sh *.md root@$HOST:/workspace/SurgViVQA-Audio/

# Upload audio files (2-5 minutes)
scp -P $PORT -r audio/in_002-001 root@$HOST:/workspace/SurgViVQA-Audio/audio/
scp -P $PORT -r audio/out_002-001 root@$HOST:/workspace/SurgViVQA-Audio/audio/

# Upload frames - LARGEST (10-30 minutes)
# Use rsync for resumable upload
rsync -avz --progress -e "ssh -p $PORT" \
    dataset/frames/002-001/ \
    root@$HOST:/workspace/SurgViVQA-Audio/dataset/frames/002-001/
```

## Step 2: Setup Environment on RunPod

```bash
# SSH into RunPod
ssh root@HOST -p PORT

# Navigate to project
cd /workspace/SurgViVQA-Audio

# Run setup script (will install conda, PyTorch, dependencies)
bash setup_runpod.sh
```

**Expected output:**
- Creates conda environment 'surg-audio'
- Installs PyTorch with CUDA
- Installs all dependencies
- Verifies GPU access
- Takes ~5-10 minutes

## Step 3: Verify Upload

```bash
# Still on RunPod, in /workspace/SurgViVQA-Audio

# Check file counts
echo "Frame count (should be 25998):"
ls dataset/frames/002-001/ | wc -l

echo "Audio in-domain (should be 50):"
ls audio/in_002-001/ | wc -l

echo "Audio out-domain (should be 50):"
ls audio/out_002-001/ | wc -l

# Run verification script
python verify_setup.py \
    --test_file test_set/in_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/in_002-001
```

**Expected output:**
```
✓ SETUP VERIFIED - Ready to run experiments!
```

## Step 4: Run Experiments

```bash
# Run all 3 baselines on both test sets
# This will take ~1-2 hours
bash run_all_baselines.sh
```

**What happens:**
1. Baseline 1: Text + Image VQA (~10 min per test set)
2. Baseline 2: Audio + Image VQA (~15 min per test set)
3. Baseline 3: Audio → ASR → Text (~20 min per test set)
4. Results saved to `results/` directory

## Step 5: Download Results

```bash
# From your LOCAL Mac terminal (new window)
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio

# Download all results
scp -P PORT -r root@HOST:/workspace/SurgViVQA-Audio/results/ ./results/

# Check what you got
ls -lh results/
```

**Expected files:**
- `baseline1_in.jsonl` - Text+Image on in-domain
- `baseline1_out.jsonl` - Text+Image on out-domain
- `baseline2_in.jsonl` - Audio+Image on in-domain
- `baseline2_out.jsonl` - Audio+Image on out-domain
- `baseline3_in_medium.jsonl` - ASR pipeline on in-domain
- `baseline3_out_medium.jsonl` - ASR pipeline on out-domain

## Troubleshooting

### "conda: command not found"
```bash
# Initialize conda for current shell
source /root/miniconda3/etc/profile.d/conda.sh
conda activate surg-audio
```

### "Out of Memory" during Baseline 3
```bash
# Edit run_all_baselines.sh and change line 19:
WHISPER_MODEL="base"  # Instead of "medium"
```

### Upload interrupted/failed
```bash
# Resume frame upload with rsync
rsync -avz --progress -e "ssh -p PORT" \
    dataset/frames/002-001/ \
    root@HOST:/workspace/SurgViVQA-Audio/dataset/frames/002-001/
```

### Model download fails
```bash
# Pre-download the model
python -c "from transformers import Qwen2VLForConditionalGeneration; \
Qwen2VLForConditionalGeneration.from_pretrained('kulsoom-abdullah/Qwen2-Audio-7B-Transcription')"
```

## Cost Tracking

**RTX A6000 @ $0.79/hour**

- Upload + Setup: ~20 min = $0.26
- Experiments: ~2 hours = $1.58
- **Total: ~$1.84**

Stop pod immediately after downloading results to save money!

```bash
# On RunPod dashboard: Click "Stop" button
```

## What You Get

After downloading results, you'll have:
1. **6 JSONL files** with per-sample predictions and timing
2. **Accuracy metrics** for each baseline
3. **Inference time** comparison (ASR overhead analysis)
4. **System latency** data (typing vs speaking)

Ready for analysis and visualization!
