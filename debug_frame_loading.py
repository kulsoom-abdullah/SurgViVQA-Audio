#!/usr/bin/env python3
"""
Debug script to see exactly what's happening in frame loading
"""
from pathlib import Path

# Test with one frame name
frame_name = "002-001_18743"
frames_dir = "dataset/frames"

print(f"Testing frame: {frame_name}")
print(f"Frames dir: {frames_dir}")
print()

# Test the rsplit logic
if '_' in frame_name:
    video_id = frame_name.rsplit('_', 1)[0]
else:
    video_id = "unknown"

print(f"Extracted video_id: '{video_id}'")
print()

# Build path
path_a = Path(frames_dir) / video_id / f"{frame_name}.jpg"
print(f"Constructed path: {path_a}")
print(f"Path exists: {path_a.exists()}")
print()

# Check if file actually exists
import os
if path_a.exists():
    print(f"✓ File found!")
    print(f"  Absolute path: {path_a.absolute()}")
    print(f"  File size: {os.path.getsize(path_a)} bytes")
else:
    print(f"❌ File NOT found")
    print(f"  Checking parent directory:")
    parent = path_a.parent
    print(f"  Parent exists: {parent.exists()}")
    if parent.exists():
        print(f"  Files in parent:")
        for f in sorted(parent.iterdir())[:10]:
            print(f"    - {f.name}")
