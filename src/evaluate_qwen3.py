#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen 3.0 checkpoint on test set
"""

import torch
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from peft import PeftModel

# Dynamic import of correct Qwen model class
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration

from transformers import AutoTokenizer, AutoProcessor

def load_frames(frame_names, frames_dir, max_size=384):
    """Load and resize frames"""
    images = []
    for frame_name in frame_names:
        vid_id = frame_name.rsplit('_', 1)[0] if '_' in frame_name else "unknown"
        path = Path(frames_dir) / vid_id / f"{frame_name}.jpg"

        if not path.exists():
            frame_num = frame_name.rsplit('_', 1)[1] if '_' in frame_name else frame_name
            path = Path(frames_dir) / vid_id / f"{frame_num}.jpg"

        if path.exists():
            img = Image.open(path).convert("RGB")
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))
            images.append(img)
        else:
            print(f"⚠️  Frame not found: {path}")
            images.append(Image.new('RGB', (224, 224), color='black'))

    return images

def evaluate(checkpoint_path, test_file, frames_dir, output_file):
    """Evaluate fine-tuned checkpoint on test set"""

    print("="*80)
    print("EVALUATING FINE-TUNED QWEN 3.0")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test file: {test_file}")
    print("")

    # Load base model
    print("⏳ Loading base model...")
    base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # Load LoRA adapters
    print("⏳ Loading LoRA adapters from checkpoint...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    processor.tokenizer = tokenizer

    print(f"✅ Model loaded on device: {model.device}")

    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(test_data)} test samples\n")

    # Run inference
    results = []
    correct = 0
    total = 0
    total_time = 0

    print("Running inference...")
    for sample in tqdm(test_data, desc="Processing"):
        try:
            # Load frames
            images = load_frames(sample['frames'], frames_dir)

            # Build prompt
            content = [{"type": "image"} for _ in images]
            content.append({
                "type": "text",
                "text": f"Question: {sample['question']}\nProvide a brief answer."
            })

            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process inputs
            inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            inference_time = (time.time() - start_time) * 1000
            total_time += inference_time

            # Decode
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            predicted = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Check answer
            is_correct = sample['short_answer'].lower() in predicted.lower()

            # Store result
            result = {
                'question_id': sample['id'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'short_answer': sample['short_answer'],
                'predicted_answer': predicted,
                'exact_match': int(is_correct),
                'inference_time_ms': inference_time
            }
            results.append(result)

            total += 1
            if is_correct:
                correct += 1

            # Free memory
            del outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error on sample {sample['id']}: {e}")
            continue

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print summary
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg inference time: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.2f} samples/sec")
    print("="*80)

    # Per-type breakdown
    print("\n📊 Accuracy by Question Type:")
    type_stats = {}
    for result in results:
        qtype = result['question_type']
        if qtype not in type_stats:
            type_stats[qtype] = {'correct': 0, 'total': 0}
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['correct'] += result['exact_match']

    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {qtype}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    print("="*80)

    return results, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--test_file", required=True, help="Test data JSONL file")
    parser.add_argument("--frames_dir", required=True, help="Directory containing video frames")
    parser.add_argument("--output", required=True, help="Output JSONL file for results")

    args = parser.parse_args()
    evaluate(args.checkpoint_path, args.test_file, args.frames_dir, args.output)
