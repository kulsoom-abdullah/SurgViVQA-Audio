import sys
import os
import torch
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor, 
    WhisperFeatureExtractor
)
from peft import PeftModel

# --- CRITICAL: Import the exact dataset class from your training script ---
sys.path.append(os.getcwd())
from src.train_vqa import SurgicalVQADataset, DataCollatorForSurgicalVQA

# Using your merged Stage 1 + Stage 2 checkpoint from HuggingFace
AUDIO_ADAPTED_MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

def calculate_metrics(results):
    """Break down accuracy by 'question_type'."""
    by_type = {}
    total_correct = 0
    total_samples = 0
    
    for r in results:
        q_type = r['type']
        if q_type not in by_type:
            by_type[q_type] = {'correct': 0, 'total': 0}
            
        pred = r['pred'].strip().lower()
        truth = r['true'].strip().lower()
        
        # Simple inclusion check (e.g. "absent" in "absent.")
        is_correct = pred == truth or truth in pred
        
        by_type[q_type]['total'] += 1
        if is_correct:
            by_type[q_type]['correct'] += 1
            total_correct += 1

    print("\n" + "="*60)
    print(f"{'Question Type':<30} | {'Acc':<8} | {'Count':<8}")
    print("-" * 60)
    
    sorted_types = sorted(by_type.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for q_type, stats in sorted_types:
        acc = (stats['correct'] / stats['total']) * 100
        print(f"{q_type:<30} | {acc:5.1f}%   | {stats['total']:<8}")
        
    print("-" * 60)
    overall_acc = (total_correct / len(results)) * 100 if len(results) > 0 else 0
    print(f"OVERALL ACCURACY: {overall_acc:.2f}% ({total_correct}/{len(results)})")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoint-200)")
    parser.add_argument("--eval_file", type=str, default="eval_002001_stratified.jsonl")
    parser.add_argument("--frames_dir", type=str, default="data/frames")
    parser.add_argument("--audio_dir", type=str, default="data/audio/out_002-001")
    parser.add_argument("--device", type=str, default="cuda:0", help="Force specific GPU (cuda:0 or cuda:1)")
    args = parser.parse_args()

    print(f"🔄 Loading model from {args.checkpoint} on {args.device}...")
    
    # 1. Load Model with EXPLICIT device mapping
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        AUDIO_ADAPTED_MODEL_ID,
        quantization_config=bnb_config,
        device_map=args.device,  # <--- FORCE THE DEVICE HERE
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    
    # Load Adapters
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    # 2. Setup Processors
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # 3. Load Dataset
    print(f"📂 Loading eval data: {args.eval_file}")
    dataset = SurgicalVQADataset(
        args.eval_file, args.frames_dir, args.audio_dir,
        processor, tokenizer, feature_extractor,
        max_eval_frames=6, 
        is_eval=True 
    )
    
    # Batch size must be 1 to handle variable sequence lengths safely during simple inference
    collator = DataCollatorForSurgicalVQA(tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    # 4. Inference
    results = []
    print(f"🚀 Running inference on {len(dataset)} samples...")
    
    with torch.inference_mode():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move batch to target device manually
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)
            
            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                input_features=batch['input_features'],
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Slice output to remove input prompt
            input_len = batch['input_ids'].shape[1]
            new_tokens = generated_ids[:, input_len:]
            pred_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            sample = dataset.data[i]
            results.append({
                "type": sample['question_type'],
                "true": sample['answer'],
                "pred": pred_text
            })

    # 5. Metrics
    calculate_metrics(results)
    
    out_file = "results_stratified.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {out_file}")

if __name__ == "__main__":
    main()