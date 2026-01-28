#!/usr/bin/env python3
"""
Generate TTS audio files for a SPECIFIC subset of questions (e.g., filtered by video_id).
This is optimized for vertical slice testing with a single video.

Usage:
    python generate_audio_subset.py --input test_set/in_002-001.jsonl --output audio/in_002-001
    python generate_audio_subset.py --input test_set/out_002-001.jsonl --output audio/out_002-001
"""

import json
import edge_tts
import asyncio
from pathlib import Path
from tqdm import tqdm
import time
import argparse

# Configuration
VOICE = "en-US-AriaNeural"  # Professional female voice

async def generate_audio(text, output_file):
    """Generate audio for a given text"""
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(str(output_file))
        return True
    except Exception as e:
        print(f"\n❌ Error generating {output_file}: {e}")
        return False

async def main(input_file, output_dir, skip_existing=True):
    print("=" * 80)
    print("TTS Audio Generation for SurgViVQA Subset")
    print("=" * 80)
    print(f"Voice: {VOICE}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Skip existing: {skip_existing}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"\nLoaded {len(data)} samples from {input_file}")

    total_generated = 0
    total_skipped = 0
    total_failed = 0
    start_time = time.time()

    # Generate audio for each sample
    for item in tqdm(data, desc=f"Generating audio"):
        qa_id = item['id']
        question = item['question']
        audio_file = output_path / f"{qa_id}.mp3"

        # Skip if already exists
        if skip_existing and audio_file.exists():
            total_skipped += 1
            continue

        # Generate audio
        success = await generate_audio(question, audio_file)

        if success:
            total_generated += 1
        else:
            total_failed += 1

        # Small delay to avoid rate limiting (unofficial API)
        await asyncio.sleep(0.1)  # 100ms delay between requests

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✓ Audio Generation Complete!")
    print("=" * 80)
    print(f"Total generated: {total_generated}")
    print(f"Total skipped (existing): {total_skipped}")
    print(f"Total failed: {total_failed}")
    print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    if total_generated > 0:
        print(f"Average per file: {elapsed_time/total_generated:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio for a subset of SurgViVQA questions")
    parser.add_argument("--input", required=True, help="Input JSONL file (e.g., test_set/in_002-001.jsonl)")
    parser.add_argument("--output", required=True, help="Output directory for audio files")
    parser.add_argument("--no-skip", action="store_true", help="Regenerate existing audio files")

    args = parser.parse_args()

    asyncio.run(main(
        input_file=args.input,
        output_dir=args.output,
        skip_existing=not args.no_skip
    ))
