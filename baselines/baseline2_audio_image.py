"""
Baseline 2: Audio + Image → VQA Model (Novel Approach)
Direct audio processing using Whisper encoder grafted into Qwen2-VL
"""

import sys
import os
import json
import time
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# CRITICAL: Use local transformers fork with audio grafting logic
# This must be done BEFORE any transformers imports
script_dir = os.path.dirname(os.path.abspath(__file__))
fork_path = os.path.join(script_dir, "transformers_local")
if os.path.exists(fork_path):
    sys.path.insert(0, fork_path)
    print(f"✓ Using local transformers fork from: {fork_path}")
else:
    print(f"⚠️  WARNING: Local fork not found at {fork_path}")
    print("   Baseline 2 requires modified transformers with audio support!")
    print("   Run: bash baselines/setup_fork_from_hf.sh")
    sys.exit(1)

# Import utilities
from utils import (
    load_model,
    load_frames,
    process_audio,
    build_audio_vision_vqa_inputs,  # NEW: Handles audio + vision tokens
    check_answer_match,
    load_jsonl,
    save_jsonl
)

def run_baseline2(test_file, frames_dir, audio_dir, output_file, model_path=None):
    """
    Run Baseline 2: Audio + Image VQA (Direct Audio Embedding)

    Args:
        test_file: Path to test JSONL file
        frames_dir: Path to frames directory
        audio_dir: Path to audio files directory
        output_file: Path to save results
        model_path: Optional model checkpoint path
    """
    print("="*80)
    print("BASELINE 2: AUDIO + IMAGE VQA (DIRECT AUDIO EMBEDDING)")
    print("="*80)

    # Load model
    if model_path:
        model, tokenizer, processor, feature_extractor = load_model(model_path)
    else:
        model, tokenizer, processor, feature_extractor = load_model()

    # Processor now loads correctly from HF checkpoint (Qwen2VLProcessor with audio+vision support)

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

            # Process images using processor - CRITICAL: Must capture image_grid_thw!
            image_inputs = processor.image_processor(images, return_tensors="pt")
            pixel_values = image_inputs.pixel_values.to(device).to(torch.bfloat16)
            image_grid_thw = image_inputs.image_grid_thw.to(device)  # Required for Qwen2-VL

            # Load and process audio
            audio_filename = f"{sample['id']}.mp3"
            audio_path = Path(audio_dir) / audio_filename

            if not audio_path.exists():
                print(f"\n⚠️  Audio file not found: {audio_path}")
                continue

            input_features = process_audio(audio_path, feature_extractor, device)

            # Build input_ids with BOTH audio and vision tokens
            # This properly injects vision placeholder tokens so model doesn't get "tokens: 0" error
            input_ids, attention_mask = build_audio_vision_vqa_inputs(tokenizer, processor, images, device)

            # Run inference with timing
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    input_features=input_features,  # Whisper mel-spectrogram features
                    pixel_values=pixel_values,       # Vision features
                    image_grid_thw=image_grid_thw,   # Grid dimensions (REQUIRED!)
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Decode output (skip the input tokens, only decode generated tokens)
            predicted_answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Check answer
            is_correct = check_answer_match(
                predicted_answer,
                sample['answer'],
                sample['short_answer']
            )

            # Store result
            result = {
                'question_id': sample['id'],
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
            import traceback
            traceback.print_exc()
            continue

    # Save results
    save_jsonl(results, output_file)
    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("BASELINE 2 RESULTS SUMMARY")
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
    parser = argparse.ArgumentParser(description="Baseline 2: Audio + Image VQA (Direct Audio)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio files directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_path", type=str, default=None, help="Optional model checkpoint path")

    args = parser.parse_args()

    run_baseline2(args.test_file, args.frames_dir, args.audio_dir, args.output, args.model_path)

if __name__ == "__main__":
    main()
