"""
Merge Stage-2 LoRA adapters into full BF16 checkpoint for training with FlashAttention

This loads:
- Stage-1: Audio-grafted Qwen2-VL base model (bf16)
- Stage-2: LoRA adapters from audio-only training
Then merges them into a single bf16 checkpoint for surgical VQA fine-tuning.

Usage:
python merge_stage2_to_bf16.py \
    --stage1_model_path /path/to/stage1/checkpoint \
    --stage2_adapter_path /path/to/stage2/checkpoint \
    --output_path ./qwen2_audio_vl_merged_bf16
"""

import torch
import argparse
from transformers import Qwen2VLForConditionalGeneration
from peft import PeftModel

def merge_adapters(stage1_path, stage2_path, output_path):
    print("="*80)
    print("MERGING STAGE-2 LORA ADAPTERS INTO BF16 CHECKPOINT")
    print("="*80)

    # 1. Load Stage-1 base model (bf16)
    print(f"\n⏳ Loading Stage-1 base model from: {stage1_path}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        stage1_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ Stage-1 base model loaded")

    # 2. Load Stage-2 LoRA adapters
    print(f"\n⏳ Loading Stage-2 LoRA adapters from: {stage2_path}")
    model = PeftModel.from_pretrained(
        base_model,
        stage2_path,
        torch_dtype=torch.bfloat16
    )
    print("✓ Stage-2 adapters loaded")

    # 3. Merge adapters into base model
    print("\n🔧 Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    print("✓ Adapters merged")

    # 4. Ensure everything is bf16
    print("\n🔧 Casting to bfloat16...")
    merged_model = merged_model.to(torch.bfloat16)

    # 5. Save merged checkpoint
    print(f"\n💾 Saving merged bf16 checkpoint to: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    # 6. Copy tokenizer and processor config
    print("\n📋 Copying tokenizer and config files...")
    from transformers import AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained(stage1_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Copy processor config if it exists
    try:
        import shutil
        from pathlib import Path
        stage1_path_obj = Path(stage1_path)
        output_path_obj = Path(output_path)

        processor_files = [
            "preprocessor_config.json",
            "chat_template.json"
        ]

        for filename in processor_files:
            src = stage1_path_obj / filename
            if src.exists():
                shutil.copy(src, output_path_obj / filename)
                print(f"  ✓ Copied {filename}")
    except Exception as e:
        print(f"  ⚠️ Could not copy processor configs: {e}")

    print("\n" + "="*80)
    print("✅ MERGE COMPLETE!")
    print("="*80)
    print(f"\nMerged checkpoint saved to: {output_path}")
    print("\nUpdate your train_vqa.py:")
    print(f'  MODEL_ID = "{output_path}"')
    print("\nThen remove BitsAndBytesConfig and prepare_model_for_kbit_training")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Merge Stage-2 LoRA to BF16")
    parser.add_argument("--stage1_model_path", type=str,
                       default="kulsoom-abdullah/Qwen2-Audio-Stage1",
                       help="Path to Stage-1 base model checkpoint (local or HF)")
    parser.add_argument("--stage2_adapter_path", type=str, required=True,
                       help="Path to Stage-2 LoRA adapter checkpoint (must be adapters, not merged)")
    parser.add_argument("--output_path", type=str,
                       default="./qwen2_audio_vl_merged_bf16",
                       help="Path to save merged bf16 checkpoint")

    args = parser.parse_args()

    merge_adapters(args.stage1_model_path, args.stage2_adapter_path, args.output_path)

if __name__ == "__main__":
    main()
