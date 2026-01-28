#!/usr/bin/env python3
"""
Debug the tokenizer directly
"""
from transformers import AutoTokenizer

print("=== Testing Your Model's Tokenizer ===")
model_path = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)

print(f"✓ Tokenizer loaded from {model_path}")
print(f"Tokenizer type: {type(tokenizer)}")

# Simple test
test_text = "Hello, how are you?"
print(f"\nTest 1: Simple text: '{test_text}'")

# Try encoding without return_tensors first
encoded = tokenizer.encode(test_text, add_special_tokens=True)
print(f"✓ Encoded (list): {encoded[:10]}...")

# Try with return_tensors
try:
    encoded_pt = tokenizer(test_text, return_tensors="pt")
    print(f"✓ Encoded (tensors): {encoded_pt.keys()}")
    print(f"  input_ids shape: {encoded_pt['input_ids'].shape}")
except Exception as e:
    print(f"❌ Failed with return_tensors='pt': {e}")

# Try with the actual template text
print("\n=== Testing With Vision Template ===")
template_text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>How are you?<|im_end|>
<|im_start|>assistant
"""

print(f"Template length: {len(template_text)} chars")

# Try encoding
try:
    print("\nAttempt 1: Direct encode...")
    result = tokenizer.encode(template_text)
    print(f"✓ Encoded result type: {type(result)}")
    print(f"  Length: {len(result)}")
    print(f"  First 10: {result[:10]}")
except Exception as e:
    print(f"❌ Encode failed: {e}")
    import traceback
    traceback.print_exc()

# Try with return_tensors
try:
    print("\nAttempt 2: With return_tensors='pt'...")
    result = tokenizer(template_text, return_tensors="pt", padding=False)
    print(f"✓ Success! Keys: {result.keys()}")
    for k, v in result.items():
        if v is None:
            print(f"  ⚠️ {k}: None")
        else:
            print(f"  ✓ {k}: {type(v)} shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Check tokenizer attributes
print("\n=== Tokenizer Attributes ===")
print(f"pad_token: {tokenizer.pad_token}")
print(f"pad_token_id: {tokenizer.pad_token_id}")
print(f"eos_token: {tokenizer.eos_token}")
print(f"eos_token_id: {tokenizer.eos_token_id}")
