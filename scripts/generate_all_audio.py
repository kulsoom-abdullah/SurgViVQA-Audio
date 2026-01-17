#!/usr/bin/env python3
"""
Generate TTS audio files for ALL questions in the SurgViVQA dataset.
This creates the audio input needed for Baseline 2 and Baseline 3.

Strategy: Process files sequentially (not parallel) to avoid rate limits.
Expected time: ~10,400 questions * ~0.5s/question = ~90 minutes
"""

import json
import edge_tts
import asyncio
from pathlib import Path
from tqdm import tqdm
import time

# Configuration
VOICE = "en-US-AriaNeural"  # Professional female voice
FILES = {
    "in_template.jsonl": "audio/in_template",
    "out_template.jsonl": "audio/out_template"
}

async def generate_audio(text, output_file):
    """Generate audio for a given text"""
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(str(output_file))
        return True
    except Exception as e:
        print(f"\n❌ Error generating {output_file}: {e}")
        return False

async def main():
    print("=" * 80)
    print("TTS Audio Generation for SurgViVQA Dataset")
    print("=" * 80)
    print(f"Voice: {VOICE}")
    print(f"Files to process: {len(FILES)}")
    print("=" * 80)

    total_generated = 0
    total_failed = 0
    start_time = time.time()

    for jsonl_file, output_dir in FILES.items():
        print(f"\n📂 Processing: {jsonl_file}")
        print("-" * 80)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        with open(jsonl_file, 'r') as f:
            data = [json.loads(line) for line in f]

        print(f"Loaded {len(data)} samples")

        # Generate audio for each sample
        for item in tqdm(data, desc=f"Generating audio ({jsonl_file})"):
            qa_id = item['id']
            question = item['question']
            audio_file = output_path / f"{qa_id}.mp3"

            # Skip if already exists
            if audio_file.exists():
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
    print(f"Total failed: {total_failed}")
    print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Average per file: {elapsed_time/total_generated:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
