#!/usr/bin/env python3
"""
Fix scoring by removing prompt leakage from predictions
"""
import json
import sys

def clean_prediction(text):
    """Extract only the assistant's response, removing leaked prompt"""
    # Split by the "assistant" token
    if "assistant\n" in text:
        # Take everything after the last "assistant\n"
        return text.split("assistant\n")[-1].strip()
    elif "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant")[-1].strip()
    return text.strip()

def check_match(pred, gt, short):
    """Check if prediction matches ground truth"""
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    short = short.lower().strip()
    return (short in pred) or (gt in pred) or (pred in gt)

def fix_results_file(input_file, output_file):
    """Process results file and recalculate accuracy"""
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    total = 0
    correct = 0
    fixed_results = []

    print(f"Processing {len(lines)} samples from {input_file}...")
    print()

    for i, line in enumerate(lines, 1):
        data = json.loads(line)
        total += 1

        # Clean the prediction (remove prompt)
        raw_pred = data["predicted_answer"]
        clean_pred = clean_prediction(raw_pred)

        # Show first 3 examples
        if i <= 3:
            print(f"Example {i}:")
            print(f"  Raw prediction (first 100 chars): {raw_pred[:100]}...")
            print(f"  Cleaned prediction: {clean_pred}")
            print(f"  Ground truth: {data['short_answer']}")
            print()

        # Re-evaluate
        is_correct = check_match(clean_pred, data["ground_truth"], data["short_answer"])
        if is_correct:
            correct += 1

        # Save fixed data
        data["predicted_answer"] = clean_pred
        data["exact_match"] = 1 if is_correct else 0
        fixed_results.append(data)

    # Write fixed results
    with open(output_file, 'w') as f_out:
        for item in fixed_results:
            f_out.write(json.dumps(item) + "\n")

    print("="*60)
    print(f"TRUE Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    print(f"Previous (inflated) accuracy was likely ~62%")
    print("="*60)
    print(f"Cleaned results saved to: {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "results/baseline1_in.jsonl"
    output_file = input_file.replace(".jsonl", "_clean.jsonl")

    fix_results_file(input_file, output_file)
