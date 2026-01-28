#!/usr/bin/env python3
"""
Inspect Qwen 3.0 architecture to find correct LoRA target modules
"""

import torch
from transformers import Qwen3VLForConditionalGeneration

print("Loading Qwen 3.0 model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load on CPU for inspection
    trust_remote_code=True
)

print("\n" + "="*80)
print("MODEL ARCHITECTURE INSPECTION")
print("="*80)

# Get all named modules
all_modules = dict(model.named_modules())

print("\n1. TOP LEVEL MODULES:")
print("-" * 80)
for name, module in all_modules.items():
    if '.' not in name and name:  # Top level only
        print(f"  {name}: {type(module).__name__}")

print("\n2. LANGUAGE MODEL LAYERS (first 3):")
print("-" * 80)
layer_count = 0
for name, module in all_modules.items():
    if 'language_model.layers' in name and layer_count < 3:
        if name.count('.') <= 3:  # Don't go too deep
            print(f"  {name}: {type(module).__name__}")
        if '.layers.' in name and name.endswith(('.0', '.1', '.2')):
            layer_count += 1

print("\n3. ATTENTION PROJECTION LAYERS (sample from layer 0):")
print("-" * 80)
for name in sorted(all_modules.keys()):
    if 'language_model.layers.0.self_attn' in name and 'proj' in name:
        print(f"  {name}")

print("\n4. MLP LAYERS (sample from layer 0):")
print("-" * 80)
for name in sorted(all_modules.keys()):
    if 'language_model.layers.0.mlp' in name and 'proj' in name:
        print(f"  {name}")

print("\n5. VISUAL ENCODER LAYERS (sample):")
print("-" * 80)
for name in sorted(all_modules.keys()):
    if 'visual' in name and ('proj' in name or 'attn' in name):
        if name.count('.') <= 3:  # Keep it readable
            print(f"  {name}")

# Count total layers
language_layers = [name for name in all_modules.keys() if 'language_model.layers.' in name and name.count('.') == 3]
num_layers = len(set([name.split('.')[2] for name in language_layers]))

print("\n" + "="*80)
print(f"SUMMARY")
print("="*80)
print(f"Total language model layers: {num_layers}")
print("")
print("RECOMMENDED LORA TARGET REGEX:")
print("-" * 80)
print('r"model\\.language_model\\.layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)"')
print("")
print("This targets:")
print("  - All attention projections (q_proj, k_proj, v_proj, o_proj)")
print("  - All MLP projections (gate_proj, up_proj, down_proj)")
print("  - Across all language model layers")
print("="*80)
