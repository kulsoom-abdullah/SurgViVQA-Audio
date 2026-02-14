# Vertical Slice Setup Guide - Video 002-001

This guide walks through testing the complete pipeline using ONLY video `002-001` (700 questions, ~26K frames).

## Quick Start

```bash
# 1. Activate your conda environment
conda activate surg-audio

# 2. Generate test sets (already done!)
# ✓ test_set/in_002-001.jsonl (50 samples)
# ✓ test_set/out_002-001.jsonl (50 samples)

# 3. Generate audio for test sets
python generate_audio_subset.py --input test_set/in_002-001.jsonl --output audio/in_002-001
python generate_audio_subset.py --input test_set/out_002-001.jsonl --output audio/out_002-001

# 4. Verify frames are accessible
python verify_setup.py --test_file test_set/in_002-001.jsonl --frames_dir dataset/frames

# 5. Run baseline experiments (see baselines/ directory)
```

## What's Been Set Up

### ✓ Data Organization
- **Frames**: `dataset/frames/002-001/` (25,998 frames)
- **Annotations**: `in_template.jsonl` and `out_template.jsonl` (700 questions for video 002-001)
- **Test Sets**: 50 stratified samples each for in/out templates

### ✓ Scripts Updated
- `create_test_set.py`: Now supports `--video_id` filtering
- `generate_audio_subset.py`: New script for targeted audio generation

## Directory Structure

```
SurgViVQA-Audio/
├── dataset/
│   └── frames/
│       └── 002-001/           # 25,998 frames
│           ├── 002-001_0.jpg
│           ├── 002-001_1.jpg
│           └── ...
├── test_set/
│   ├── in_002-001.jsonl       # 50 test questions (in-domain)
│   └── out_002-001.jsonl      # 50 test questions (out-domain)
├── audio/                      # Generated next
│   ├── in_002-001/
│   └── out_002-001/
└── baselines/                  # Experiment scripts
```

## Next Steps

1. **Generate Audio** (~1-2 minutes for 100 files)
2. **Verify Setup** (check frames/audio are accessible)
3. **Run Baseline 1** (Text input VQA)
4. **Run Baseline 2** (Audio input - Whisper ASR → Text)
5. **Run Baseline 3** (Direct Audio Embedding)

## Scaling Up

Once vertical slice works:
- Increase test set size: `--size 200` or `--size 700` (all questions)
- Generate audio for full dataset: Use `generate_all_audio.py` with video filter
- Download more videos from the 002-series

## Video Stats

Video 002-001 has:
- 700 questions total
- 14 question types (all represented)
- 25,998 frames
- Smallest dataset in the 002-series
