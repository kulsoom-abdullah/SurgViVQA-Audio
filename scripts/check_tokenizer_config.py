"""
Check tokenizer pad/eos token configuration
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "kulsoom-abdullah/Qwen2-Audio-7B-Transcription",
    trust_remote_code=True
)

print("="*80)
print("TOKENIZER CONFIGURATION")
print("="*80)

print(f"\nPAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if tokenizer.pad_token_id == tokenizer.eos_token_id:
    print("❌ PAD and EOS are the SAME token!")
    print(f"   Both are ID: {tokenizer.pad_token_id}")
    print("\n   FIX: Set pad_token to a different token (usually <|endoftext|>)")
else:
    print("✅ PAD and EOS are different tokens")
    print(f"   PAD: {tokenizer.pad_token_id}, EOS: {tokenizer.eos_token_id}")

print("\nTo fix in your checkpoint, update tokenizer_config.json:")
print('  "pad_token": "<|endoftext|>",  # Different from EOS')
print('  "eos_token": "<|im_end|>",')
