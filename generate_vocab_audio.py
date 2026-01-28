#!/usr/bin/env python3
"""
Generate audio files for all unique medical vocabulary words.
This creates a pronunciation reference for doctor verification.
"""

import json
import collections
import re
import edge_tts
import asyncio
from pathlib import Path

# Configuration
VOICE = "en-US-AriaNeural"
OUTPUT_DIR = Path("vocab_audio")
FILES = ["in_template.jsonl", "out_template.jsonl"]

async def generate_audio(text, output_file):
    """Generate audio for a given text"""
    communicate = edge_tts.Communicate(text, VOICE, rate="-15%")
    await communicate.save(str(output_file))

async def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Collect all unique words from both files
    all_words = set()
    word_to_sample_sentences = collections.defaultdict(list)

    print("=" * 70)
    print("Extracting unique medical vocabulary...")
    print("=" * 70)

    for filename in FILES:
        with open(filename, 'r') as f:
            data = [json.loads(line) for line in f]

        for item in data:
            question = item['question']
            words = re.findall(r'\b[a-z]+\b', question.lower())

            # Filter for words > 6 letters
            for word in words:
                if len(word) > 6:
                    all_words.add(word)
                    # Store sample sentence for context
                    if len(word_to_sample_sentences[word]) < 2:  # Keep 2 examples
                        word_to_sample_sentences[word].append(question)

    unique_words = sorted(all_words)
    print(f"\nFound {len(unique_words)} unique medical vocabulary words (>6 letters)\n")

    # Generate audio for each word
    print("Generating audio files...")
    print("-" * 70)

    # 1. Generate isolated word pronunciations
    isolated_dir = OUTPUT_DIR / "isolated"
    isolated_dir.mkdir(exist_ok=True)

    # 2. Generate in-context pronunciations
    context_dir = OUTPUT_DIR / "in_context"
    context_dir.mkdir(exist_ok=True)

    for i, word in enumerate(unique_words, 1):
        print(f"[{i}/{len(unique_words)}] {word}")

        # Generate isolated pronunciation
        isolated_file = isolated_dir / f"{word}.mp3"
        await generate_audio(word, isolated_file)
        print(f"  ✓ Isolated: {isolated_file}")

        # Generate in-context pronunciation (first example sentence)
        if word_to_sample_sentences[word]:
            sample_sentence = word_to_sample_sentences[word][0]
            context_file = context_dir / f"{word}_context.mp3"
            await generate_audio(sample_sentence, context_file)
            print(f"  ✓ Context: {context_file}")
            print(f"    Sentence: {sample_sentence}")

        print()

    # Create a summary text file for the doctor
    summary_file = OUTPUT_DIR / "vocabulary_list.txt"
    with open(summary_file, 'w') as f:
        f.write("Medical Vocabulary - Pronunciation Verification\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total unique words: {len(unique_words)}\n")
        f.write(f"Voice used: {VOICE}\n\n")
        f.write("Words and sample contexts:\n")
        f.write("-" * 70 + "\n\n")

        for word in unique_words:
            f.write(f"Word: {word}\n")
            if word_to_sample_sentences[word]:
                f.write(f"  Example: {word_to_sample_sentences[word][0]}\n")
                if len(word_to_sample_sentences[word]) > 1:
                    f.write(f"  Example: {word_to_sample_sentences[word][1]}\n")
            f.write(f"  Audio files:\n")
            f.write(f"    - isolated/{word}.mp3\n")
            f.write(f"    - in_context/{word}_context.mp3\n")
            f.write("\n")

    print("=" * 70)
    print(f"✓ Complete! Generated audio for {len(unique_words)} words")
    print(f"✓ Summary saved to: {summary_file}")
    print(f"✓ Isolated pronunciations: {isolated_dir}/")
    print(f"✓ In-context pronunciations: {context_dir}/")
    print("=" * 70)
    print("\nNext step: Share vocab_audio/ folder with doctor for verification")

if __name__ == "__main__":
    asyncio.run(main())
