#!/usr/bin/env python3
"""Quick command-line test of edge-tts before using the notebook"""

import json
import random
import edge_tts
import asyncio

async def test_tts():
    # Load a random sample
    with open("out_template.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]

    # Pick 3 samples: random, NBI question, and mucosa question
    random_sample = random.choice(data)
    nbi_samples = [s for s in data if 'NBI' in s['question']]
    mucosa_samples = [s for s in data if 'mucosa' in s['question'].lower()]

    test_samples = [
        ("Random", random_sample),
        ("NBI Test", random.choice(nbi_samples) if nbi_samples else random_sample),
        ("Mucosa Test", random.choice(mucosa_samples) if mucosa_samples else random_sample)
    ]

    VOICE = "en-US-AriaNeural"

    print("=" * 70)
    print("Edge-TTS Quick Test")
    print("=" * 70)

    for label, sample in test_samples:
        text = sample['question']
        output_file = f"test_{label.replace(' ', '_').lower()}.mp3"

        print(f"\n[{label}]")
        print(f"Question Type: {sample['question_type']}")
        print(f"Text: {text}")
        print(f"Generating audio... ", end='', flush=True)

        try:
            communicate = edge_tts.Communicate(text, VOICE, rate="-15%"))
            await communicate.save(output_file)
            print(f"✓ Saved to {output_file}")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 70)
    print("Test complete! Check the generated MP3 files.")
    print("If these work, the notebook will work too.")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_tts())
