#!/usr/bin/env python3
"""
Verify that the vertical slice setup is correct:
- Frames are accessible
- Audio files match questions
- Data format is correct
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def verify_setup(test_file, frames_dir, audio_dir=None):
    """Verify setup for a test file"""

    print("=" * 80)
    print("VERTICAL SLICE SETUP VERIFICATION")
    print("=" * 80)

    # Load test data
    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"\n📊 Test File: {test_file}")
    print(f"   Total samples: {len(data)}")

    # Check video IDs
    video_ids = set(item['video_id'] for item in data)
    print(f"   Video IDs: {', '.join(sorted(video_ids))}")

    # Check question types
    q_types = defaultdict(int)
    for item in data:
        q_types[item['question_type']] += 1

    print(f"\n📋 Question Type Distribution:")
    for qtype, count in sorted(q_types.items()):
        print(f"   {qtype}: {count}")

    # Verify frames
    print(f"\n🖼️  Verifying Frames...")
    frames_path = Path(frames_dir)
    missing_frames = []
    checked_frames = 0

    for item in data:
        for frame_id in item['frames']:
            checked_frames += 1
            video_id = frame_id.split('_')[0]
            frame_path = frames_path / video_id / f"{frame_id}.jpg"

            if not frame_path.exists():
                missing_frames.append(str(frame_path))

    if missing_frames:
        print(f"   ❌ FAILED: {len(missing_frames)}/{checked_frames} frames missing")
        print(f"   First missing: {missing_frames[0]}")
    else:
        print(f"   ✓ All {checked_frames} frame references verified")

    # Verify audio if directory provided
    missing_audio = []  # Initialize here to avoid UnboundLocalError
    if audio_dir:
        print(f"\n🔊 Verifying Audio...")
        audio_path = Path(audio_dir)

        if not audio_path.exists():
            print(f"   ⚠️  Audio directory doesn't exist: {audio_dir}")
            print(f"   Run: python generate_audio_subset.py --input {test_file} --output {audio_dir}")
            missing_audio = data  # Mark all as missing if directory doesn't exist
        else:
            for item in data:
                audio_file = audio_path / f"{item['id']}.mp3"
                if not audio_file.exists():
                    missing_audio.append(item['id'])

            if missing_audio:
                print(f"   ❌ {len(missing_audio)}/{len(data)} audio files missing")
                print(f"   Run: python generate_audio_subset.py --input {test_file} --output {audio_dir}")
            else:
                print(f"   ✓ All {len(data)} audio files present")

    # Summary
    print("\n" + "=" * 80)
    if not missing_frames and (not audio_dir or not missing_audio):
        print("✓ SETUP VERIFIED - Ready to run experiments!")
    else:
        print("⚠️  SETUP INCOMPLETE - See issues above")
    print("=" * 80)

    return len(missing_frames) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify vertical slice setup")
    parser.add_argument("--test_file", required=True, help="Test JSONL file")
    parser.add_argument("--frames_dir", default="dataset/frames", help="Frames directory")
    parser.add_argument("--audio_dir", default=None, help="Audio directory (optional)")

    args = parser.parse_args()

    verify_setup(args.test_file, args.frames_dir, args.audio_dir)
