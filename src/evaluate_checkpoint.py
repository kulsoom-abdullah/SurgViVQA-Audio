"""
Standalone evaluation script for surgical VQA model
Runs after training completes, with memory-efficient settings

Usage:
python evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/surgical_vqa_50samples \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/out_002-001 \
    --output_file results/eval_results.jsonl \
    --batch_size 1
"""

import warnings
# Suppress "copying from non-meta parameter" warnings during checkpoint loading
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*meta parameter.*", category=UserWarning)

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, WhisperFeatureExtractor
from PIL import Image
import librosa

# Using your merged Stage 1 + Stage 2 checkpoint from HuggingFace
AUDIO_ADAPTED_MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

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
            images.append(Image.new('RGB', (224, 224), color='black'))

    return images

def evaluate(args):
    print("="*80)
    print("SURGICAL VQA EVALUATION")
    print("="*80)

    # Load base model + LoRA adapters
    print(f"\n⏳ Loading base model and adapters from: {args.checkpoint_path}")

    # Load tokenizer/processor from base model (not from PEFT checkpoint)
    print(f"Loading tokenizer from base: {AUDIO_ADAPTED_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(AUDIO_ADAPTED_MODEL_ID, trust_remote_code=True, use_fast=False)

    # Load base model (4-bit quantized)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        AUDIO_ADAPTED_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    # Load LoRA adapters
    from peft import PeftModel
    print(f"Loading LoRA adapters from: {args.checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    model.eval()

    # Load processor from Qwen2-VL base
    print("Loading processor from Qwen2-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, use_fast=False)
    processor.tokenizer = tokenizer

    # Fix PAD/EOS if needed
    if tokenizer.pad_token_id == tokenizer.eos_token_id or tokenizer.pad_token_id is None:
        print("🔧 Setting dedicated pad token for generation")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        # Note: We don't resize embeddings here since checkpoint already has it
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # Load eval data
    print(f"📁 Loading eval data: {args.eval_data_path}")
    eval_data = [json.loads(line) for line in open(args.eval_data_path)]
    print(f"✓ Loaded {len(eval_data)} samples")

    # Run evaluation
    results = []
    correct = 0
    total = 0

    print(f"\n🚀 Running evaluation (batch_size={args.batch_size})...")

    for sample in tqdm(eval_data, desc="Evaluating"):
        try:
            # Load audio
            audio_path = Path(args.audio_dir) / f"{sample['id']}.mp3"
            if audio_path.exists():
                y, _ = librosa.load(audio_path, sr=16000, mono=True)
            else:
                y = torch.zeros(16000 * 2)

            audio_inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
            input_features = audio_inputs.input_features.to(model.device).to(torch.bfloat16)

            # Load frames
            images = load_frames(sample['frames'], args.frames_dir, max_size=args.max_image_size)

            # Build prompt
            content = [{"type": "image"} for _ in images]
            content.append({
                "type": "text",
                "text": f"User Question: {sample['question']}\nAnswer the question concisely based on the visual and audio evidence."
            })
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = processor(text=[text], images=images, return_tensors="pt")

            # Inject audio tokens
            AUDIO_TOKEN_ID = 151657
            NUM_AUDIO_TOKENS = 1500

            audio_tokens = torch.tensor([[AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS], device=model.device)
            audio_header = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
            audio_footer = tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False, return_tensors="pt").to(model.device)

            user_prefix_len = len(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
            vision_content = batch.input_ids[:, user_prefix_len:].to(model.device)

            input_ids = torch.cat([audio_header, audio_tokens, audio_footer, vision_content], dim=1)
            attention_mask = torch.ones_like(input_ids)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    pixel_values=batch.pixel_values.to(model.device).to(torch.bfloat16),
                    image_grid_thw=batch.image_grid_thw.to(model.device),
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            predicted = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Check answer
            is_correct = sample['short_answer'].lower() in predicted.lower()

            result = {
                'question_id': sample['id'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'short_answer': sample['short_answer'],
                'predicted_answer': predicted,
                'correct': int(is_correct)
            }
            results.append(result)

            total += 1
            if is_correct:
                correct += 1

            # Free memory
            del input_features, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error on sample {sample['id']}: {e}")
            continue

    # Save results
    print(f"\n💾 Saving results to: {args.output_file}")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print summary
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Goal: Beat 46% baseline (audio+image zero-shot)")
    print("="*80)

    # Question type breakdown
    if 'question_type' in eval_data[0]:
        print("\nAccuracy by Question Type:")
        type_stats = {}
        for result, sample in zip(results, eval_data):
            qtype = sample['question_type']
            if qtype not in type_stats:
                type_stats[qtype] = {'correct': 0, 'total': 0}
            type_stats[qtype]['total'] += 1
            type_stats[qtype]['correct'] += result['correct']

        for qtype, stats in sorted(type_stats.items()):
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {qtype}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    return results, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate surgical VQA checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained checkpoint")
    parser.add_argument("--eval_data_path", type=str, required=True,
                       help="Path to eval JSONL file")
    parser.add_argument("--frames_dir", type=str, required=True,
                       help="Path to frames directory")
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Path to audio directory")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (keep at 1 for memory)")
    parser.add_argument("--max_image_size", type=int, default=384,
                       help="Max image dimension")

    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
