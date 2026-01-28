"""
Debug script to check all parameter dtypes in the model
"""
import torch
from transformers import Qwen2VLForConditionalGeneration

print("Loading model to check dtypes...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "kulsoom-abdullah/Qwen2-Audio-7B-Transcription",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

print("\n" + "="*80)
print("CHECKING ALL PARAMETER DTYPES")
print("="*80)

# Check visual encoder
print("\n🔍 VISUAL ENCODER:")
if hasattr(model, 'visual'):
    non_bf16_count = 0
    for name, param in model.visual.named_parameters():
        if param.dtype not in [torch.bfloat16, torch.float16]:
            print(f"  ❌ {name}: {param.dtype}")
            non_bf16_count += 1

    if non_bf16_count == 0:
        print("  ✅ All visual encoder params are bf16/fp16")
    else:
        print(f"  ⚠️ Found {non_bf16_count} non-bf16 params in visual encoder")

# Check audio encoder
print("\n🔍 AUDIO ENCODER:")
if hasattr(model, 'audio_encoder'):
    non_bf16_count = 0
    for name, param in model.audio_encoder.named_parameters():
        if param.dtype not in [torch.bfloat16, torch.float16]:
            print(f"  ❌ {name}: {param.dtype}")
            non_bf16_count += 1

    if non_bf16_count == 0:
        print("  ✅ All audio encoder params are bf16/fp16")
    else:
        print(f"  ⚠️ Found {non_bf16_count} non-bf16 params in audio encoder")

# Check decoder
print("\n🔍 DECODER (LLM):")
non_bf16_count = 0
for name, param in model.model.named_parameters():
    if param.dtype not in [torch.bfloat16, torch.float16]:
        print(f"  ❌ {name}: {param.dtype}")
        non_bf16_count += 1
        if non_bf16_count > 10:
            print(f"  ... (truncated, found {non_bf16_count} total)")
            break

if non_bf16_count == 0:
    print("  ✅ All decoder params are bf16/fp16")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Count all params by dtype
dtype_counts = {}
for name, param in model.named_parameters():
    dtype_str = str(param.dtype)
    if dtype_str not in dtype_counts:
        dtype_counts[dtype_str] = 0
    dtype_counts[dtype_str] += 1

for dtype, count in sorted(dtype_counts.items()):
    print(f"  {dtype}: {count} parameters")

print("\n✅ If all params are torch.bfloat16, checkpoint is good")
print("❌ If you see torch.float32, the checkpoint has mixed dtypes")
