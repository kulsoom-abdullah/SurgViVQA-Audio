#!/usr/bin/env python3
"""
Create a small test set for debugging and initial pipeline verification.
Selects diverse samples across all question types.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

def create_test_set(
    input_file="out_template.jsonl",
    output_file="test_set/test_samples.jsonl",
    test_size=100,
    seed=42,
    video_id=None
):
    """
    Create a stratified test set ensuring representation from all question types

    Args:
        input_file: Source JSONL file
        output_file: Output JSONL file for test set
        test_size: Number of samples in test set
        seed: Random seed for reproducibility
        video_id: Optional video_id to filter (e.g., "002-001" for vertical slice)
    """
    random.seed(seed)

    print("=" * 70)
    print("Creating Test Set")
    print("=" * 70)

    # Load data
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} total samples from {input_file}")

    # Filter by video_id if specified
    if video_id:
        data = [item for item in data if item.get('video_id') == video_id]
        print(f"Filtered to {len(data)} samples for video_id='{video_id}'")

    # Group by question type
    by_type = defaultdict(list)
    for item in data:
        by_type[item['question_type']].append(item)

    print(f"\nFound {len(by_type)} unique question types:")
    for qtype, items in sorted(by_type.items()):
        print(f"  {qtype}: {len(items)} samples")

    # Stratified sampling: take roughly equal number from each type
    samples_per_type = test_size // len(by_type)
    remainder = test_size % len(by_type)

    test_samples = []
    for i, (qtype, items) in enumerate(sorted(by_type.items())):
        # Add 1 extra sample for first 'remainder' types
        n_samples = samples_per_type + (1 if i < remainder else 0)
        n_samples = min(n_samples, len(items))  # Don't exceed available samples

        sampled = random.sample(items, n_samples)
        test_samples.extend(sampled)

    # Shuffle the final test set
    random.shuffle(test_samples)

    print(f"\n✓ Created test set with {len(test_samples)} samples")

    # Verify distribution
    test_types = defaultdict(int)
    for item in test_samples:
        test_types[item['question_type']] += 1

    print("\nTest set distribution:")
    for qtype in sorted(test_types.keys()):
        print(f"  {qtype}: {test_types[qtype]} samples")

    # Save test set
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in test_samples:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ Test set saved to: {output_file}")
    print("=" * 70)

    return test_samples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test set for SurgViVQA evaluation")
    parser.add_argument("--input", default="out_template.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="test_set/test_samples.jsonl", help="Output JSONL file")
    parser.add_argument("--size", type=int, default=100, help="Test set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--video_id", default=None, help="Filter to specific video (e.g., '002-001' for vertical slice)")

    args = parser.parse_args()

    create_test_set(
        input_file=args.input,
        output_file=args.output,
        test_size=args.size,
        seed=args.seed,
        video_id=args.video_id
    )
