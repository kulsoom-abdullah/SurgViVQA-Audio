#!/usr/bin/env python3
"""
Quick test to verify frame loading works correctly
"""
import json
from baselines.utils import load_frames

# Load one test sample
with open('test_set/in_002-001.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print(f"Test sample: {sample['id']}")
print(f"Video: {sample['video_id']}")
print(f"Frame IDs: {sample['frames'][:3]}...")  # Show first 3

# Try loading frames
frames_dir = "dataset/frames"
images = load_frames(sample['frames'], frames_dir)

print(f"\n✓ Loaded {len(images)} images")
print(f"First image size: {images[0].size}")
print(f"First image mode: {images[0].mode}")

# Check if any are black placeholders (size 224x224)
# Real frames are (1350, 1080), placeholders are (224, 224)
placeholder_count = sum(1 for img in images if img.size == (224, 224))
if placeholder_count > 0:
    print(f"⚠️ Warning: {placeholder_count}/{len(images)} images are black placeholders")
else:
    print(f"✓ All {len(images)} images loaded successfully (no placeholders)")
