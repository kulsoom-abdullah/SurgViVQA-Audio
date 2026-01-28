#!/usr/bin/env python3
"""
Compare vision encoder performance across Qwen 2.0, 2.5, and 3.0
Generates a detailed comparison report by question type
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

def load_results(filepath):
    """Load results from JSONL file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def calculate_stats(results):
    """Calculate overall and per-type accuracy"""
    overall = {'correct': 0, 'total': 0}
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in results:
        # Overall
        overall['total'] += 1
        overall['correct'] += result['exact_match']

        # By type
        qtype = result['question_type']
        by_type[qtype]['total'] += 1
        by_type[qtype]['correct'] += result['exact_match']

    return overall, by_type

def format_report(qwen2_stats, qwen25_stats, qwen3_stats, finetuned_stats):
    """Generate formatted comparison report"""
    report = []

    report.append("="*100)
    report.append("VISION ENCODER COMPARISON: Qwen 2.0 vs 2.5 vs 3.0 (All Zero-Shot Text+Image)")
    report.append("="*100)
    report.append("")

    # Overall accuracy
    report.append("OVERALL ACCURACY:")
    report.append("-"*100)

    qwen2_overall, qwen2_by_type = qwen2_stats
    qwen25_overall, qwen25_by_type = qwen25_stats
    qwen3_overall, qwen3_by_type = qwen3_stats
    finetuned_overall, finetuned_by_type = finetuned_stats

    qwen2_acc = (qwen2_overall['correct'] / qwen2_overall['total'] * 100) if qwen2_overall['total'] > 0 else 0
    qwen25_acc = (qwen25_overall['correct'] / qwen25_overall['total'] * 100) if qwen25_overall['total'] > 0 else 0
    qwen3_acc = (qwen3_overall['correct'] / qwen3_overall['total'] * 100) if qwen3_overall['total'] > 0 else 0
    finetuned_acc = (finetuned_overall['correct'] / finetuned_overall['total'] * 100) if finetuned_overall['total'] > 0 else 0

    report.append(f"Qwen 2.0 VL (zero-shot):      {qwen2_acc:.2f}% ({qwen2_overall['correct']}/{qwen2_overall['total']})")
    report.append(f"Qwen 2.5 VL (zero-shot):      {qwen25_acc:.2f}% ({qwen25_overall['correct']}/{qwen25_overall['total']})")
    report.append(f"Qwen 3.0 VL (zero-shot):      {qwen3_acc:.2f}% ({qwen3_overall['correct']}/{qwen3_overall['total']})")
    report.append(f"Qwen 2.0 Audio (fine-tuned):  {finetuned_acc:.2f}% ({finetuned_overall['correct']}/{finetuned_overall['total']}) ✅ Your Model")
    report.append("")

    # Per-type comparison
    report.append("ACCURACY BY QUESTION TYPE:")
    report.append("-"*100)
    report.append(f"{'Question Type':<30} | {'Qwen 2.0':>12} | {'Qwen 2.5':>12} | {'Qwen 3.0':>12} | {'Fine-tuned':>12} | Insight")
    report.append("-"*100)

    # Get all question types
    all_types = set(qwen2_by_type.keys()) | set(qwen25_by_type.keys()) | set(qwen3_by_type.keys()) | set(finetuned_by_type.keys())

    for qtype in sorted(all_types):
        qwen2_type_acc = (qwen2_by_type[qtype]['correct'] / qwen2_by_type[qtype]['total'] * 100) if qtype in qwen2_by_type and qwen2_by_type[qtype]['total'] > 0 else 0
        qwen25_type_acc = (qwen25_by_type[qtype]['correct'] / qwen25_by_type[qtype]['total'] * 100) if qtype in qwen25_by_type and qwen25_by_type[qtype]['total'] > 0 else 0
        qwen3_type_acc = (qwen3_by_type[qtype]['correct'] / qwen3_by_type[qtype]['total'] * 100) if qtype in qwen3_by_type and qwen3_by_type[qtype]['total'] > 0 else 0
        finetuned_type_acc = (finetuned_by_type[qtype]['correct'] / finetuned_by_type[qtype]['total'] * 100) if qtype in finetuned_by_type and finetuned_by_type[qtype]['total'] > 0 else 0

        # Determine insight
        vision_improvement = qwen3_type_acc - qwen2_type_acc
        finetuning_improvement = finetuned_type_acc - qwen2_type_acc

        if 'motion' in qtype.lower() or 'position' in qtype.lower():
            insight = "🎯 TEMPORAL TEST"
        elif abs(vision_improvement) > 10:
            insight = f"Vision encoder: {'+' if vision_improvement > 0 else ''}{vision_improvement:.1f}%"
        elif finetuning_improvement > 20:
            insight = "Fine-tuning helps"
        else:
            insight = ""

        report.append(f"{qtype:<30} | {qwen2_type_acc:>11.1f}% | {qwen25_type_acc:>11.1f}% | {qwen3_type_acc:>11.1f}% | {finetuned_type_acc:>11.1f}% | {insight}")

    report.append("-"*100)
    report.append("")

    # Key findings
    report.append("KEY FINDINGS:")
    report.append("-"*100)
    report.append("")

    # Check temporal questions specifically
    temporal_types = ['lesion_motion_direction', 'lesion_screen_position', 'scope_motion', 'scope_forward_motion', 'scope_backward_motion']
    temporal_found = [t for t in temporal_types if t in all_types]

    if temporal_found:
        report.append("🎯 TEMPORAL REASONING (Motion/Position Questions):")
        for qtype in temporal_found:
            qwen2_type_acc = (qwen2_by_type[qtype]['correct'] / qwen2_by_type[qtype]['total'] * 100) if qtype in qwen2_by_type and qwen2_by_type[qtype]['total'] > 0 else 0
            qwen3_type_acc = (qwen3_by_type[qtype]['correct'] / qwen3_by_type[qtype]['total'] * 100) if qtype in qwen3_by_type and qwen3_by_type[qtype]['total'] > 0 else 0
            finetuned_type_acc = (finetuned_by_type[qtype]['correct'] / finetuned_by_type[qtype]['total'] * 100) if qtype in finetuned_by_type and finetuned_by_type[qtype]['total'] > 0 else 0

            improvement = qwen3_type_acc - qwen2_type_acc
            report.append(f"  {qtype}: Qwen 2.0 {qwen2_type_acc:.1f}% → Qwen 3.0 {qwen3_type_acc:.1f}% ({'+' if improvement >= 0 else ''}{improvement:.1f}%) | Fine-tuned: {finetuned_type_acc:.1f}%")

        avg_qwen2_temporal = sum(qwen2_by_type[t]['correct'] for t in temporal_found) / sum(qwen2_by_type[t]['total'] for t in temporal_found if t in qwen2_by_type) * 100
        avg_qwen3_temporal = sum(qwen3_by_type[t]['correct'] for t in temporal_found) / sum(qwen3_by_type[t]['total'] for t in temporal_found if t in qwen3_by_type) * 100

        report.append("")
        if avg_qwen3_temporal - avg_qwen2_temporal > 5:
            report.append(f"✅ CONCLUSION: Qwen 3.0 vision encoder improves temporal reasoning by {avg_qwen3_temporal - avg_qwen2_temporal:.1f} points")
            report.append("   → Recommendation: Audio-graft Qwen 3.0 for next iteration")
        else:
            report.append(f"❌ CONCLUSION: Vision encoder version doesn't significantly help temporal reasoning ({avg_qwen3_temporal - avg_qwen2_temporal:.1f} point change)")
            report.append("   → Recommendation: Need video-native architecture (VideoLLaMA, Video-ChatGPT)")

    report.append("")
    report.append("="*100)

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Compare vision encoder baselines")
    parser.add_argument("--qwen2", required=True, help="Qwen 2.0 results JSONL")
    parser.add_argument("--qwen25", required=True, help="Qwen 2.5 results JSONL")
    parser.add_argument("--qwen3", required=True, help="Qwen 3.0 results JSONL")
    parser.add_argument("--finetuned", required=True, help="Fine-tuned model results JSONL")
    parser.add_argument("--output", required=True, help="Output report file")

    args = parser.parse_args()

    # Load results
    print("Loading results...")
    qwen2_results = load_results(args.qwen2)
    qwen25_results = load_results(args.qwen25)
    qwen3_results = load_results(args.qwen3)
    finetuned_results = load_results(args.finetuned)

    # Calculate stats
    print("Calculating statistics...")
    qwen2_stats = calculate_stats(qwen2_results)
    qwen25_stats = calculate_stats(qwen25_results)
    qwen3_stats = calculate_stats(qwen3_results)
    finetuned_stats = calculate_stats(finetuned_results)

    # Generate report
    print("Generating report...")
    report = format_report(qwen2_stats, qwen25_stats, qwen3_stats, finetuned_stats)

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\n✅ Comparison report saved to: {args.output}\n")
    print(report)

if __name__ == "__main__":
    main()
