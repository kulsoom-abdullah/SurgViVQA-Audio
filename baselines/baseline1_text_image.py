"""
Baseline 1: Text + Image → VQA Model (Control)
Traditional VQA using text questions and image frames
"""

import sys
import os
import json
import time
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Import utilities
from utils import (
    load_model,
    load_frames,
    build_text_vqa_messages,
    process_text_vqa_inputs,
    check_answer_match,
    load_jsonl,
    save_jsonl
)

def run_baseline1(test_file, frames_dir, output_file, model_path=None):
    """
    Run Baseline 1: Text + Image VQA

    Args:
        test_file: Path to test JSONL file
        frames_dir: Path to frames directory
        output_file: Path to save results
        model_path: Optional model checkpoint path
    """
    print("="*80)
    print("BASELINE 1: TEXT + IMAGE VQA")
    print("="*80)

    # Load model
    if model_path:
        model, tokenizer, processor, _ = load_model(model_path)
    else:
        model, tokenizer, processor, _ = load_model()

    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device}")

    # Load test data
    test_data = load_jsonl(test_file)
    print(f"✓ Loaded {len(test_data)} test samples from {test_file}")

    # Results storage
    results = []
    correct = 0
    total = 0
    total_inference_time = 0

    # Run inference
    print("\n" + "="*80)
    print("Running inference...")
    print("="*80)

    for sample in tqdm(test_data, desc="Processing samples"):
        try:
            # Load frames
            images = load_frames(sample['frames'], frames_dir)

            # Build messages structure (standard Qwen2-VL format)
            question = sample['question']
            messages, images = build_text_vqa_messages(question, images)

            # Process using Qwen2-VL processor (handles vision token injection)
            inputs = process_text_vqa_inputs(messages, images, processor)

            # Move to device and convert to bfloat16
            inputs = {k: v.to(device).to(torch.bfloat16) if v.dtype == torch.float else v.to(device)
                     for k, v in inputs.items()}

            # Run inference with timing
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1
                )

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Decode only the generated tokens (skip input)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Check answer
            is_correct = check_answer_match(
                predicted_answer,
                sample['answer'],
                sample['short_answer']
            )

            # Store result
            result = {
                'question_id': sample['id'],  # Fixed: key is 'id' not 'question_id'
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'short_answer': sample['short_answer'],
                'predicted_answer': predicted_answer,
                'exact_match': int(is_correct),
                'inference_time_ms': inference_time
            }
            results.append(result)

            # Update counters
            total += 1
            if is_correct:
                correct += 1
            total_inference_time += inference_time

        except Exception as e:
            print(f"\n❌ Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue

    # Save results
    save_jsonl(results, output_file)
    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("BASELINE 1 RESULTS SUMMARY")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {(correct/total)*100:.2f}%")
    print(f"Average inference time: {total_inference_time/total:.2f} ms")
    print(f"Median inference time: {sorted([r['inference_time_ms'] for r in results])[len(results)//2]:.2f} ms")
    print(f"Throughput: {1000/(total_inference_time/total):.2f} samples/second")
    print("="*80)

    return results

def main():
    parser = argparse.ArgumentParser(description="Baseline 1: Text + Image VQA")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_path", type=str, default=None, help="Optional model checkpoint path")

    args = parser.parse_args()

    run_baseline1(args.test_file, args.frames_dir, args.output, args.model_path)

if __name__ == "__main__":
    main()
