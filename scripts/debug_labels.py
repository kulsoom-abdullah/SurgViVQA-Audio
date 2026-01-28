"""
Debug script: Check if labels are properly constructed
Verifies that we're not masking out all the answer tokens
"""
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, WhisperFeatureExtractor
from train_vqa import SurgicalVQADataset

tokenizer = AutoTokenizer.from_pretrained("kulsoom-abdullah/Qwen2-Audio-7B-Transcription", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
processor.tokenizer = tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

# Create dataset
dataset = SurgicalVQADataset(
    "test_set/tiny_5samples.jsonl",
    "dataset/frames",
    "audio/in_002-001",
    processor,
    tokenizer,
    feature_extractor
)

print("="*80)
print("LABEL CONSTRUCTION CHECK")
print("="*80)

for i in range(min(3, len(dataset))):
    print(f"\n{'='*80}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*80}")

    sample_data = dataset.data[i]
    print(f"Question: {sample_data['question']}")
    print(f"Answer: {sample_data['answer']}")

    # Get dataset item
    item = dataset[i]
    input_ids = item['input_ids']
    labels = item['labels']

    print(f"\nSequence lengths:")
    print(f"  Input IDs: {len(input_ids)}")
    print(f"  Labels: {len(labels)}")

    # Count non-ignored labels
    non_ignored = (labels != -100).sum().item()
    total = len(labels)

    print(f"\nLabel stats:")
    print(f"  Total tokens: {total}")
    print(f"  Ignored (-100): {total - non_ignored}")
    print(f"  Trainable: {non_ignored}")
    print(f"  Trainable %: {100 * non_ignored / total:.1f}%")

    # Decode the trainable portion
    trainable_tokens = input_ids[labels != -100]
    if len(trainable_tokens) > 0:
        decoded_trainable = tokenizer.decode(trainable_tokens, skip_special_tokens=False)
        print(f"\nTrainable text (what model learns to predict):")
        print(f"  {repr(decoded_trainable)}")
    else:
        print(f"\n❌ NO TRAINABLE TOKENS! All labels are -100!")

    # Check if answer is in trainable portion
    if sample_data['answer'].lower() in tokenizer.decode(trainable_tokens, skip_special_tokens=True).lower():
        print(f"  ✅ Answer IS in trainable portion")
    else:
        print(f"  ❌ Answer NOT in trainable portion!")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("If trainable % is 0% → BUG: All labels masked, model can't learn")
print("If trainable % is 5-15% → GOOD: Learning answer only (teacher forcing)")
print("If answer NOT in trainable portion → BUG: Labels misaligned")
