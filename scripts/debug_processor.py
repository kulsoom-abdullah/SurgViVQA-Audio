#!/usr/bin/env python3
"""
Debug what the processor is actually receiving and returning
"""
import json
import torch
from baselines.utils import load_model, load_frames, build_text_vqa_messages

print("Loading model...")
model, tokenizer, processor, _ = load_model()

print("\nLoading test sample...")
with open('test_set/in_002-001.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print(f"Sample: {sample['id']}")
print(f"Question: {sample['question']}")
print(f"Frames: {sample['frames']}")

print("\nLoading frames...")
images = load_frames(sample['frames'], 'dataset/frames')
print(f"Loaded {len(images)} images")
print(f"Image types: {[type(img) for img in images[:3]]}")
print(f"Image sizes: {[img.size for img in images[:3]]}")

print("\nBuilding messages...")
messages, images = build_text_vqa_messages(sample['question'], images)
print(f"Messages structure: {messages}")
print(f"Number of images: {len(images)}")

print("\nApplying chat template...")
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Template output (first 200 chars): {text[:200]}...")

print("\nCalling processor...")
print(f"Input - text type: {type(text)}")
print(f"Input - text content: {text[:100]}...")
print(f"Input - images type: {type(images)}")
print(f"Input - images length: {len(images)}")
print(f"Input - images[0] type: {type(images[0])}")

try:
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt"
    )

    print("\n✓ Processor succeeded!")
    print(f"Output keys: {inputs.keys()}")
    for key, value in inputs.items():
        if value is None:
            print(f"  ⚠️ {key}: None")
        elif isinstance(value, torch.Tensor):
            print(f"  ✓ {key}: tensor shape {value.shape}")
        else:
            print(f"  ? {key}: {type(value)}")

except Exception as e:
    print(f"\n❌ Processor failed: {e}")
    import traceback
    traceback.print_exc()
