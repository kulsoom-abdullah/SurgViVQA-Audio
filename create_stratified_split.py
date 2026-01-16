#!/usr/bin/env python3
"""
Create stratified train/eval split for video 002-001
Ensures all question types are represented in both train and eval sets
"""

import json
from collections import defaultdict
import random

# Set random seed for reproducibility
random.seed(42)

# Load full 002-001 OUT data (natural language questions)
print("Loading data...")
with open('out_template.jsonl') as f:
    all_data = [json.loads(line) for line in f]

# Filter only 002-001
video_002_001 = [s for s in all_data if s['video_id'] == '002-001']
print(f"Total 002-001 samples: {len(video_002_001)}")

# Group by question type
by_qtype = defaultdict(list)
for sample in video_002_001:
    by_qtype[sample['question_type']].append(sample)

print(f"\nQuestion types found: {len(by_qtype)}")
for qtype, samples in sorted(by_qtype.items()):
    print(f"  {qtype}: {len(samples)} samples")

# Stratified split: 85% train, 15% eval
train_data = []
eval_data = []

print("\nCreating stratified split (85% train, 15% eval)...")
for qtype, samples in by_qtype.items():
    # Shuffle samples within each type
    random.shuffle(samples)

    # Calculate split point
    n_eval = max(1, int(len(samples) * 0.15))  # At least 1 sample for eval
    n_train = len(samples) - n_eval

    # Split
    train_data.extend(samples[:n_train])
    eval_data.extend(samples[n_train:])

    print(f"  {qtype}: {n_train} train, {n_eval} eval")

# Shuffle the final datasets
random.shuffle(train_data)
random.shuffle(eval_data)

print(f"\n✓ Train set: {len(train_data)} samples")
print(f"✓ Eval set: {len(eval_data)} samples")

# Verify stratification
print("\nVerifying stratification...")
train_qtypes = defaultdict(int)
eval_qtypes = defaultdict(int)

for s in train_data:
    train_qtypes[s['question_type']] += 1
for s in eval_data:
    eval_qtypes[s['question_type']] += 1

all_qtypes = set(train_qtypes.keys()) | set(eval_qtypes.keys())
print(f"Question types in train: {len(train_qtypes)}/{len(all_qtypes)}")
print(f"Question types in eval: {len(eval_qtypes)}/{len(all_qtypes)}")

# Save splits
print("\nSaving splits...")
with open('train_002001_stratified.jsonl', 'w') as f:
    for sample in train_data:
        f.write(json.dumps(sample) + '\n')

with open('eval_002001_stratified.jsonl', 'w') as f:
    for sample in eval_data:
        f.write(json.dumps(sample) + '\n')

print(f"✅ Saved train_002001_stratified.jsonl ({len(train_data)} samples)")
print(f"✅ Saved eval_002001_stratified.jsonl ({len(eval_data)} samples)")
print("\nReady for training!")
