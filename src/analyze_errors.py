#!/usr/bin/env python3
"""
Analyze model predictions vs ground truth for categories where model struggled.
Shows confusion matrices, mode collapse detection, and specific error examples.

Usage:
    python src/analyze_errors.py --results_file results/qwen3_finetuned_test.jsonl
    python src/analyze_errors.py --results_file results/qwen3_finetuned_test.jsonl --question_types lesion_motion_direction lesion_screen_position
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_results(results_file: str) -> List[Dict]:
    """Load JSONL results file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_answer(text: str) -> str:
    """Extract clean answer from prediction text."""
    # Remove common prefixes
    text = text.lower().strip()
    prefixes = ['answer:', 'the answer is:', 'the answer is', ':']
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Clean up punctuation
    text = text.rstrip('.,!?;')
    return text


def get_ground_truth(sample: Dict) -> str:
    """Get ground truth answer from sample."""
    # Try short_answer first (preferred), fall back to ground_truth
    if 'short_answer' in sample:
        return sample['short_answer'].lower().strip()
    return sample.get('ground_truth', '').lower().strip()


def get_predicted(sample: Dict) -> str:
    """Get predicted answer from sample."""
    # Handle both 'predicted_answer' and 'predicted' fields
    pred = sample.get('predicted_answer') or sample.get('predicted', '')
    return extract_answer(pred)


def is_correct(sample: Dict) -> bool:
    """Check if prediction is correct."""
    # Handle both 'exact_match' and 'correct' fields
    if 'exact_match' in sample:
        return bool(sample['exact_match'])
    return bool(sample.get('correct', False))


def compute_confusion_matrix(samples: List[Dict]) -> Tuple[Dict, List[str]]:
    """
    Compute confusion matrix for a question type.
    Returns (confusion_matrix, label_order)
    """
    # Get all unique labels
    all_labels = set()
    for sample in samples:
        gt = get_ground_truth(sample)
        pred = get_predicted(sample)
        all_labels.add(gt)
        all_labels.add(pred)

    label_order = sorted(all_labels)

    # Build confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        gt = get_ground_truth(sample)
        pred = get_predicted(sample)
        confusion[gt][pred] += 1

    return dict(confusion), label_order


def print_confusion_matrix(confusion: Dict, label_order: List[str], question_type: str):
    """Pretty print confusion matrix."""
    print(f"\n{'='*80}")
    print(f"CONFUSION MATRIX: {question_type}")
    print(f"{'='*80}")
    print(f"Rows = Ground Truth | Columns = Predicted\n")

    # Header
    max_label_len = max(len(label) for label in label_order)
    print(f"{'GT \\ Pred':<{max_label_len + 2}}", end="")
    for pred_label in label_order:
        print(f"{pred_label[:10]:>12}", end="")
    print("   TOTAL")
    print("-" * 80)

    # Rows
    total_correct = 0
    total_samples = 0
    for gt_label in label_order:
        print(f"{gt_label:<{max_label_len + 2}}", end="")
        row_total = 0
        for pred_label in label_order:
            count = confusion.get(gt_label, {}).get(pred_label, 0)
            if gt_label == pred_label:
                total_correct += count
            row_total += count
            total_samples += count
            print(f"{count:>12}", end="")
        print(f"{row_total:>8}")

    print("-" * 80)
    print(f"Accuracy: {total_correct}/{total_samples} = {100*total_correct/total_samples:.1f}%\n")


def analyze_mode_collapse(samples: List[Dict], question_type: str):
    """Detect if model has collapsed to predicting one or few answers."""
    print(f"\n{'='*80}")
    print(f"MODE COLLAPSE ANALYSIS: {question_type}")
    print(f"{'='*80}\n")

    # Count ground truth distribution
    gt_counts = Counter()
    pred_counts = Counter()

    for sample in samples:
        gt = get_ground_truth(sample)
        pred = get_predicted(sample)
        gt_counts[gt] += 1
        pred_counts[pred] += 1

    total = len(samples)

    print(f"Ground Truth Distribution ({total} samples):")
    for answer, count in gt_counts.most_common():
        pct = 100 * count / total
        print(f"  {answer:<20} {count:>3} ({pct:>5.1f}%)")

    print(f"\nPredicted Distribution ({total} samples):")
    for answer, count in pred_counts.most_common():
        pct = 100 * count / total
        print(f"  {answer:<20} {count:>3} ({pct:>5.1f}%)")

    # Mode collapse detection
    print(f"\n{'─'*80}")
    most_common_pred, max_count = pred_counts.most_common(1)[0]
    max_pct = 100 * max_count / total

    if max_pct > 60:
        print(f"⚠️  MODE COLLAPSE DETECTED!")
        print(f"    Model predicts '{most_common_pred}' {max_pct:.1f}% of the time")
        print(f"    This suggests the model has collapsed to a 'safe' default answer.")
    elif max_pct > 40:
        print(f"⚠️  STRONG BIAS DETECTED!")
        print(f"    Model heavily favors '{most_common_pred}' ({max_pct:.1f}%)")
    else:
        print(f"✓  No strong mode collapse detected")
        print(f"    Most common prediction: '{most_common_pred}' ({max_pct:.1f}%)")

    # Calculate entropy to measure randomness
    import math
    pred_probs = [count/total for count in pred_counts.values()]
    entropy = -sum(p * math.log2(p) for p in pred_probs if p > 0)
    max_entropy = math.log2(len(pred_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    print(f"\n    Prediction entropy: {entropy:.2f} bits")
    print(f"    Normalized entropy: {normalized_entropy:.2f} (1.0 = uniform, 0.0 = deterministic)")

    if normalized_entropy < 0.5:
        print(f"    → Low entropy suggests model is NOT exploring answer space")
    else:
        print(f"    → Moderate/high entropy suggests diverse predictions")


def show_error_examples(samples: List[Dict], question_type: str, num_examples: int = 10):
    """Show specific examples of incorrect predictions."""
    print(f"\n{'='*80}")
    print(f"ERROR EXAMPLES: {question_type}")
    print(f"{'='*80}\n")

    wrong_samples = [s for s in samples if not is_correct(s)]

    if not wrong_samples:
        print("No errors found! Model is 100% accurate on this category.")
        return

    print(f"Showing {min(num_examples, len(wrong_samples))} of {len(wrong_samples)} errors:\n")

    for i, sample in enumerate(wrong_samples[:num_examples], 1):
        gt = get_ground_truth(sample)
        pred = get_predicted(sample)
        pred_full = sample.get('predicted_answer') or sample.get('predicted', '')

        print(f"Example {i}:")
        print(f"  Question:      {sample['question']}")
        print(f"  Ground Truth:  {gt}")
        print(f"  Predicted:     {pred}")
        print(f"  Full Response: {pred_full[:100]}...")
        print()


def analyze_question_type(results: List[Dict], question_type: str, show_examples: bool = True):
    """Full analysis for a single question type."""
    # Filter samples for this question type
    samples = [r for r in results if r.get('question_type') == question_type]

    if not samples:
        print(f"\n⚠️  No samples found for question type: {question_type}")
        return

    print(f"\n\n{'#'*80}")
    print(f"# ANALYSIS: {question_type}")
    print(f"# Total samples: {len(samples)}")
    print(f"{'#'*80}")

    # Compute accuracy
    correct = sum(1 for s in samples if is_correct(s))
    accuracy = 100 * correct / len(samples)
    print(f"\nOverall Accuracy: {correct}/{len(samples)} = {accuracy:.1f}%")

    # Confusion matrix
    confusion, label_order = compute_confusion_matrix(samples)
    print_confusion_matrix(confusion, label_order, question_type)

    # Mode collapse analysis
    analyze_mode_collapse(samples, question_type)

    # Error examples
    if show_examples and correct < len(samples):
        show_error_examples(samples, question_type)


def get_struggling_categories(results: List[Dict], threshold: float = 0.4) -> List[str]:
    """Identify question types with accuracy below threshold."""
    accuracy_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in results:
        qt = r.get('question_type')
        if qt:
            accuracy_by_type[qt]['total'] += 1
            if is_correct(r):
                accuracy_by_type[qt]['correct'] += 1

    struggling = []
    for qt, stats in accuracy_by_type.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        if acc < threshold:
            struggling.append((qt, acc, stats['total']))

    # Sort by accuracy (worst first)
    struggling.sort(key=lambda x: x[1])
    return struggling


def main():
    parser = argparse.ArgumentParser(description="Analyze model prediction errors")
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to JSONL results file')
    parser.add_argument('--question_types', type=str, nargs='+',
                       help='Specific question types to analyze (default: auto-detect struggling categories)')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='Accuracy threshold for auto-detecting struggling categories (default: 0.4)')
    parser.add_argument('--no_examples', action='store_true',
                       help='Skip showing error examples')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} predictions\n")

    # Determine which question types to analyze
    if args.question_types:
        question_types = args.question_types
        print(f"Analyzing specified question types: {', '.join(question_types)}")
    else:
        # Auto-detect struggling categories
        struggling = get_struggling_categories(results, args.threshold)
        print(f"\n{'='*80}")
        print(f"AUTO-DETECTED STRUGGLING CATEGORIES (accuracy < {args.threshold*100:.0f}%):")
        print(f"{'='*80}\n")

        if not struggling:
            print(f"No categories found with accuracy below {args.threshold*100:.0f}%!")
            return

        print(f"{'Question Type':<35} {'Accuracy':>10} {'Samples':>10}")
        print("-" * 80)
        for qt, acc, total in struggling:
            print(f"{qt:<35} {acc*100:>9.1f}% {total:>10}")

        question_types = [qt for qt, _, _ in struggling]
        print(f"\nAnalyzing {len(question_types)} struggling categories...\n")

    # Analyze each question type
    for qt in question_types:
        analyze_question_type(results, qt, show_examples=not args.no_examples)

    print(f"\n\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
