#!/usr/bin/env python3
"""
Zero-shot evaluation for any Qwen VLM (no LoRA, no fine-tuning).
Outputs per-sample JSONL compatible with analyze_errors.py.

Usage:
    python src/evaluate_zeroshot.py --model Qwen/Qwen2-VL-7B-Instruct \
        --test_file data/test_multivideo.jsonl --frames_dir data/frames \
        --output results/qwen2_zeroshot_test.jsonl

    python src/evaluate_zeroshot.py --model Qwen/Qwen3-VL-8B-Instruct \
        --test_file data/test_multivideo.jsonl --frames_dir data/frames \
        --output results/qwen3_zeroshot_test.jsonl
"""

import torch
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

# Import all available Qwen VL model classes
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_QWEN25 = True
except ImportError:
    HAS_QWEN25 = False

from transformers import Qwen2VLForConditionalGeneration


def load_frames(frame_names, frames_dir, max_size=384):
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
            images.append(Image.new('RGB', (224, 224), color='black'))
    return images


def evaluate(model_name, test_file, frames_dir, output_file):
    print("=" * 80)
    print(f"ZERO-SHOT EVALUATION: {model_name}")
    print("=" * 80)

    # Auto-detect model class based on model name
    model_name_lower = model_name.lower()
    if ("qwen3" in model_name_lower or "Qwen3" in model_name) and HAS_QWEN3:
        print("Using Qwen3VLForConditionalGeneration")
        model_cls = Qwen3VLForConditionalGeneration
    elif ("qwen2.5" in model_name_lower or "qwen2_5" in model_name_lower or "Qwen2.5" in model_name) and HAS_QWEN25:
        print("Using Qwen2_5_VLForConditionalGeneration")
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        print("Using Qwen2VLForConditionalGeneration")
        model_cls = Qwen2VLForConditionalGeneration

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model loaded on: {model.device}")

    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]
    print(f"Loaded {len(test_data)} test samples\n")

    results = []
    correct = 0
    total = 0
    total_time = 0

    for sample in tqdm(test_data, desc="Processing"):
        try:
            images = load_frames(sample['frames'], frames_dir)

            content = [{"type": "image"} for _ in images]
            content.append({
                "type": "text",
                "text": f"Question: {sample['question']}\nProvide a brief answer."
            })
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            inference_time = (time.time() - start_time) * 1000
            total_time += inference_time

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            predicted = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            is_correct = sample['short_answer'].lower() in predicted.lower()

            result = {
                'question_id': sample['id'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'short_answer': sample['short_answer'],
                'predicted_answer': predicted,
                'exact_match': int(is_correct),
                'correct': int(is_correct),
                'inference_time_ms': inference_time
            }
            results.append(result)

            total += 1
            if is_correct:
                correct += 1

            del outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError on {sample['id']}: {e}")
            continue

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 80}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Avg inference: {avg_time:.1f} ms")

    # Per-type breakdown
    type_stats = {}
    for r in results:
        qt = r['question_type']
        if qt not in type_stats:
            type_stats[qt] = {'correct': 0, 'total': 0}
        type_stats[qt]['total'] += 1
        type_stats[qt]['correct'] += r['exact_match']

    print(f"\nBy question type:")
    for qt in sorted(type_stats):
        s = type_stats[qt]
        print(f"  {qt}: {s['correct']/s['total']*100:.1f}% ({s['correct']}/{s['total']})")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.model, args.test_file, args.frames_dir, args.output)
