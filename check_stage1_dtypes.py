"""
Check if Stage-1 checkpoint is pure bf16
"""
import torch
from transformers import Qwen2VLForConditionalGeneration

print("Checking Stage-1 checkpoint dtypes...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "kulsoom-abdullah/Qwen2-Audio-Stage1",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

print("\n" + "="*80)
print("DTYPE SUMMARY")
print("="*80)

dtype_counts = {}
for name, param in model.named_parameters():
    dtype_str = str(param.dtype)
    if dtype_str not in dtype_counts:
        dtype_counts[dtype_str] = 0
    dtype_counts[dtype_str] += 1

for dtype, count in sorted(dtype_counts.items()):
    print(f"  {dtype}: {count} parameters")

if 'torch.uint8' in dtype_counts:
    print("\n❌ Stage-1 is also quantized (uint8)")
    print("   You'll need to merge from unquantized checkpoints")
elif 'torch.bfloat16' in dtype_counts and len(dtype_counts) == 1:
    print("\n✅ Stage-1 is pure BF16!")
    print("   Use this as your base for training")
else:
    print(f"\n⚠️ Mixed dtypes found: {list(dtype_counts.keys())}")
