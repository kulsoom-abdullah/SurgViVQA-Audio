"""
Check what PAD and EOS tokens ACTUALLY are (by ID, not just name)
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "kulsoom-abdullah/Qwen2-Audio-7B-Transcription",
    trust_remote_code=True
)

print("="*80)
print("ACTUAL TOKEN ANALYSIS")
print("="*80)

print("\nToken Strings:")
print(f"  pad_token:  {repr(tokenizer.pad_token)}")
print(f"  eos_token:  {repr(tokenizer.eos_token)}")
print(f"  bos_token:  {repr(tokenizer.bos_token)}")

print("\nToken IDs:")
print(f"  pad_token_id:  {tokenizer.pad_token_id}")
print(f"  eos_token_id:  {tokenizer.eos_token_id}")
print(f"  bos_token_id:  {tokenizer.bos_token_id}")

print("\nActual vocabulary lookup:")
print(f"  vocab['<|endoftext|>']:  {tokenizer.convert_tokens_to_ids('<|endoftext|>')}")
print(f"  vocab['<|im_end|>']:     {tokenizer.convert_tokens_to_ids('<|im_end|>')}")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if tokenizer.pad_token_id == tokenizer.eos_token_id:
    print("❌ PAD and EOS have the SAME ID!")
    print(f"   Both are ID {tokenizer.pad_token_id}")
    print(f"   PAD token '{tokenizer.pad_token}' → ID {tokenizer.pad_token_id}")
    print(f"   EOS token '{tokenizer.eos_token}' → ID {tokenizer.eos_token_id}")
else:
    print("✅ PAD and EOS have DIFFERENT IDs!")
    print(f"   PAD: '{tokenizer.pad_token}' (ID {tokenizer.pad_token_id})")
    print(f"   EOS: '{tokenizer.eos_token}' (ID {tokenizer.eos_token_id})")
    print("\n   This is CORRECT! Training should work fine.")

print("\n" + "="*80)
print("WHY THIS MATTERS")
print("="*80)
print("The TOKEN STRING doesn't matter - only the ID matters!")
print("Qwen2 uses:")
print("  - '<|endoftext|>' for padding")
print("  - '<|im_end|>' for end-of-sequence (chat format)")
print("These map to DIFFERENT token IDs, so generation works correctly.")
