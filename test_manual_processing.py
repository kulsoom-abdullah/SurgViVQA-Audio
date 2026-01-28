#!/usr/bin/env python3
"""
Test manual processing - bypass the processor
"""
import json
import torch
from baselines.utils import load_model, load_frames, build_text_vqa_messages

print("Loading model...")
model, tokenizer, processor, _ = load_model()

print("\nLoading test sample...")
with open('test_set/in_002-001.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print("\nLoading frames...")
images = load_frames(sample['frames'], 'dataset/frames')

print("\nBuilding messages...")
messages, images = build_text_vqa_messages(sample['question'], images)

print("\nApplying chat template...")
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"Text template (first 300 chars):\n{text[:300]}...")

# Try manual processing - separate tokenizer and image processor
print("\n=== Manual Processing ===")

# 1. Tokenize text manually
print("Tokenizing text...")
text_inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=False  # No padding for single sequence
)
print(f"✓ Text tokenized - input_ids shape: {text_inputs['input_ids'].shape}")

# 2. Process images manually
print("\nProcessing images...")
image_inputs = processor.image_processor(
    images,
    return_tensors="pt"
)
print(f"✓ Images processed - pixel_values shape: {image_inputs['pixel_values'].shape}")

# 3. Combine manually
print("\nCombining inputs...")
combined_inputs = {
    'input_ids': text_inputs['input_ids'],
    'attention_mask': text_inputs['attention_mask'],
    'pixel_values': image_inputs['pixel_values'],
    'image_grid_thw': image_inputs['image_grid_thw']
}

print("\n✓ Manual processing succeeded!")
print(f"Combined inputs keys: {combined_inputs.keys()}")
for key, value in combined_inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
