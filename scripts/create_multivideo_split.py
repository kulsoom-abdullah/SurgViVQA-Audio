#!/usr/bin/env python3
"""
Create stratified train/eval/test split for multi-video dataset
Videos 002-001, 002-002, 002-003 for training
Video 002-004 held out as test set (never seen during training)
"""

import json
from collections import defaultdict
import random

# Set random seed for reproducibility
random.seed(42)

# Load full OUT data (natural language questions)
print("Loading data...")
with open('data/out_template.jsonl') as f:
    all_data = [json.loads(line) for line in f]

# Split by video
train_videos = ['002-001', '002-002', '002-003']
test_video = '002-004'

train_pool = [s for s in all_data if s['video_id'] in train_videos]
test_data = [s for s in all_data if s['video_id'] == test_video]

print(f"\nData distribution:")
print(f"  Train pool (videos {', '.join(train_videos)}): {len(train_pool)} samples")
print(f"  Held-out test (video {test_video}): {len(test_data)} samples")

# Group training pool by question type for stratified split
by_qtype = defaultdict(list)
for sample in train_pool:
    by_qtype[sample['question_type']].append(sample)

print(f"\nQuestion types: {len(by_qtype)}")
for qtype, samples in sorted(by_qtype.items()):
    print(f"  {qtype}: {len(samples)} samples")

# Stratified split: 85% train, 15% eval
train_data = []
eval_data = []

print("\nCreating stratified train/eval split (85%/15%)...")
for qtype, samples in by_qtype.items():
    # Shuffle samples within each type
    random.shuffle(samples)

    # Calculate split point
    n_eval = max(1, int(len(samples) * 0.15))
    n_train = len(samples) - n_eval

    # Split
    train_data.extend(samples[:n_train])
    eval_data.extend(samples[n_train:])

    print(f"  {qtype}: {n_train} train, {n_eval} eval")

# Shuffle final datasets
random.shuffle(train_data)
random.shuffle(eval_data)
random.shuffle(test_data)

print(f"\n✓ Train set: {len(train_data)} samples")
print(f"✓ Eval set: {len(eval_data)} samples")
print(f"✓ Test set: {len(test_data)} samples (held out)")

# Verify all question types present
train_qtypes = set(s['question_type'] for s in train_data)
eval_qtypes = set(s['question_type'] for s in eval_data)
test_qtypes = set(s['question_type'] for s in test_data)

print(f"\nQuestion type coverage:")
print(f"  Train: {len(train_qtypes)} types")
print(f"  Eval: {len(eval_qtypes)} types")
print(f"  Test: {len(test_qtypes)} types")

# Save splits
print("\nSaving splits...")
with open('data/train_multivideo.jsonl', 'w') as f:
    for sample in train_data:
        f.write(json.dumps(sample) + '\n')

with open('data/eval_multivideo.jsonl', 'w') as f:
    for sample in eval_data:
        f.write(json.dumps(sample) + '\n')

with open('data/test_multivideo.jsonl', 'w') as f:
    for sample in test_data:
        f.write(json.dumps(sample) + '\n')

print(f"✅ Saved data/train_multivideo.jsonl ({len(train_data)} samples)")
print(f"✅ Saved data/eval_multivideo.jsonl ({len(eval_data)} samples)")
print(f"✅ Saved data/test_multivideo.jsonl ({len(test_data)} samples)")
print("\nReady for overnight training!")
