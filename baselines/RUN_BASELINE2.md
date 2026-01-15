# Quick Start: Run Baseline 2 NOW

## On Lambda Labs (Run These Commands)

```bash
# 1. Navigate to project
cd ~/audiograft/SurgViVQA-Audio
source ~/venvs/surg-audio/bin/activate

# 2. Download transformers fork from HuggingFace (no Mac needed!)
cd baselines
git clone https://huggingface.co/kulsoom-abdullah/Qwen2-Audio-7B-Transcription temp_hf
mv temp_hf/transformers_fork/src transformers_local
rm -rf temp_hf
cd ..

# 3. Run Baseline 2 - In-Domain
python baselines/baseline2_audio_image.py \
    --test_file test_set/in_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/in_002-001 \
    --output results/baseline2_in.jsonl

# 4. Run Baseline 2 - Out-Domain
python baselines/baseline2_audio_image.py \
    --test_file test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/out_002-001 \
    --output results/baseline2_out.jsonl
```

## Expected Output

```
✓ Using local transformers fork from: /home/ubuntu/audiograft/SurgViVQA-Audio/baselines/transformers_local
⏳ Loading model from: kulsoom-abdullah/Qwen2-Audio-7B-Transcription...
✓ Model loaded on device: cuda:0
✓ Loaded 50 test samples from test_set/in_002-001.jsonl

================================================================================
Running inference...
================================================================================
Processing samples: 100%|██████████| 50/50 [XX:XX<00:00, X.XXs/it]

✓ Results saved to results/baseline2_in.jsonl

================================================================================
BASELINE 2 RESULTS SUMMARY
================================================================================
Total samples: 50
Correct answers: XX
Accuracy: XX.XX%
...
```

## What This Tests

- Your **novel audio-grafted architecture**
- Direct audio embedding (Whisper → Qwen2-VL)
- No ASR transcription step
- Compare to Baseline 1 (text) and Baseline 3 (ASR pipeline)

---

**Note:** This downloads the fork from HuggingFace, so no `scp` from your Mac needed!
