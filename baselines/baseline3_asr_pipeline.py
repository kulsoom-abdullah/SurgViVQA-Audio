"""
Baseline 3: Audio → ASR → Text + Image VQA (Traditional Pipeline)
Two-stage pipeline: Whisper ASR transcription followed by text VQA
"""

import sys
import os
import json
import time
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import whisper

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

def run_baseline3(test_file, frames_dir, audio_dir, output_file, model_path=None, whisper_model="large-v3"):
    """
    Run Baseline 3: Audio → ASR → Text + Image VQA (Two-stage pipeline)

    Args:
        test_file: Path to test JSONL file
        frames_dir: Path to frames directory
        audio_dir: Path to audio files directory
        output_file: Path to save results
        model_path: Optional VQA model checkpoint path
        whisper_model: Whisper model size for ASR
    """
    print("="*80)
    print("BASELINE 3: AUDIO → ASR → TEXT + IMAGE VQA (TWO-STAGE PIPELINE)")
    print("="*80)

    # Load VQA model
    if model_path:
        model, tokenizer, processor, _ = load_model(model_path)
    else:
        model, tokenizer, processor, _ = load_model()

    device = next(model.parameters()).device
    print(f"✓ VQA Model loaded on device: {device}")

    # Load Whisper ASR model
    print(f"⏳ Loading Whisper {whisper_model} for ASR...")
    whisper_asr = whisper.load_model(whisper_model, device=device)
    print("✓ Whisper ASR model loaded")

    # Load test data
    test_data = load_jsonl(test_file)
    print(f"✓ Loaded {len(test_data)} test samples from {test_file}")

    # Results storage
    results = []
    correct = 0
    total = 0
    total_asr_time = 0
    total_vqa_time = 0
    total_inference_time = 0

    # Run inference
    print("\n" + "="*80)
    print("Running two-stage inference...")
    print("="*80)

    for sample in tqdm(test_data, desc="Processing samples"):
        try:
            # Load audio file
            audio_filename = f"{sample['id']}.mp3"
            audio_path = Path(audio_dir) / audio_filename

            if not audio_path.exists():
                print(f"\n⚠️  Audio file not found: {audio_path}")
                continue

            # STAGE 1: ASR (Whisper transcription)
            asr_start = time.time()

            with torch.no_grad():
                asr_result = whisper_asr.transcribe(str(audio_path))
                transcribed_text = asr_result['text']

            asr_time = (time.time() - asr_start) * 1000  # Convert to ms

            # STAGE 2: VQA (Text + Image)
            # Load frames
            images = load_frames(sample['frames'], frames_dir)

            # Build messages structure with transcribed text
            messages, images = build_text_vqa_messages(transcribed_text, images)

            # Process using Qwen2-VL processor
            inputs = process_text_vqa_inputs(messages, images, processor)

            # Move to device and convert to bfloat16
            inputs = {k: v.to(device).to(torch.bfloat16) if v.dtype == torch.float else v.to(device)
                     for k, v in inputs.items()}

            # Run VQA inference with timing
            vqa_start = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1
                )

            vqa_time = (time.time() - vqa_start) * 1000  # Convert to ms

            # Total inference time
            inference_time = asr_time + vqa_time

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
                'question_id': sample['id'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'transcribed_question': transcribed_text,
                'ground_truth': sample['answer'],
                'short_answer': sample['short_answer'],
                'predicted_answer': predicted_answer,
                'exact_match': int(is_correct),
                'asr_time_ms': asr_time,
                'vqa_time_ms': vqa_time,
                'inference_time_ms': inference_time
            }
            results.append(result)

            # Update counters
            total += 1
            if is_correct:
                correct += 1
            total_asr_time += asr_time
            total_vqa_time += vqa_time
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
    print("BASELINE 3 RESULTS SUMMARY")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {(correct/total)*100:.2f}%")
    print(f"\nTiming Breakdown:")
    print(f"  Average ASR time: {total_asr_time/total:.2f} ms")
    print(f"  Average VQA time: {total_vqa_time/total:.2f} ms")
    print(f"  Average total time: {total_inference_time/total:.2f} ms")
    print(f"  Median total time: {sorted([r['inference_time_ms'] for r in results])[len(results)//2]:.2f} ms")
    print(f"\nThroughput: {1000/(total_inference_time/total):.2f} samples/second")
    print(f"ASR overhead: {(total_asr_time/total_inference_time)*100:.1f}% of total time")
    print("="*80)

    return results

def main():
    parser = argparse.ArgumentParser(description="Baseline 3: Audio → ASR → Text + Image VQA")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio files directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_path", type=str, default=None, help="Optional VQA model checkpoint path")
    parser.add_argument("--whisper_model", type=str, default="large-v3",
                       help="Whisper model size (tiny, base, small, medium, large, large-v3)")

    args = parser.parse_args()

    run_baseline3(args.test_file, args.frames_dir, args.audio_dir, args.output,
                  args.model_path, args.whisper_model)

if __name__ == "__main__":
    main()
