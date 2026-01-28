# Setup Baseline 2 (Audio-Grafted Model)

Baseline 2 uses your custom audio-grafted Qwen2-VL which requires modified transformers code.

## Step 1: Copy Modified Transformers to Lambda Labs

On your **Mac**, run:

```bash
cd /Users/kulsoom/workspace/learning/SurgViVQA-Audio/baselines

# Create the directory on Lambda Labs
ssh ubuntu@132-145-135-107 "mkdir -p ~/audiograft/SurgViVQA-Audio/baselines/transformers_local/transformers/models"

# Copy the modified qwen2_vl module
scp -r /Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft/transformers_fork/src/transformers/models/qwen2_vl \
    ubuntu@132-145-135-107:~/audiograft/SurgViVQA-Audio/baselines/transformers_local/transformers/models/

# Copy the updated baseline2 script
scp baseline2_audio_image.py ubuntu@132-145-135-107:~/audiograft/SurgViVQA-Audio/baselines/
```

## Step 2: Verify Setup on Lambda Labs

```bash
# SSH to Lambda Labs
ssh ubuntu@132-145-135-107

# Check the fork is in place
ls -la ~/audiograft/SurgViVQA-Audio/baselines/transformers_local/transformers/models/qwen2_vl/
# Should show: modeling_qwen2_vl.py, configuration_qwen2_vl.py, etc.
```

## Step 3: Run Baseline 2

```bash
cd ~/audiograft/SurgViVQA-Audio
source ~/venvs/surg-audio/bin/activate

# Test on in-domain data
python baselines/baseline2_audio_image.py \
    --test_file test_set/in_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/in_002-001 \
    --output results/baseline2_in.jsonl

# Test on out-domain data
python baselines/baseline2_audio_image.py \
    --test_file test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/out_002-001 \
    --output results/baseline2_out.jsonl
```

## What This Does

- Baseline 2 tests your **novel audio-grafted architecture**
- Direct audio embedding (no ASR transcription step)
- Uses Whisper encoder features directly in Qwen2-VL
- Measures accuracy and inference time for audio+image VQA
