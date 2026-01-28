#!/bin/bash
# Quick verification script for tokenizer PAD/EOS configuration

cd ~/audiograft/SurgViVQA-Audio
source ~/venvs/surg-audio/bin/activate

python -c "
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('kulsoom-abdullah/Qwen2-Audio-7B-Transcription', trust_remote_code=True)

print('='*80)
print('TOKENIZER VERIFICATION')
print('='*80)
print()
print(f'pad_token: {repr(tok.pad_token)} (ID: {tok.pad_token_id})')
print(f'eos_token: {repr(tok.eos_token)} (ID: {tok.eos_token_id})')
print(f'bos_token: {repr(tok.bos_token)} (ID: {tok.bos_token_id})')
print()

if tok.pad_token_id == tok.eos_token_id:
    print('❌ ISSUE: PAD and EOS tokens are THE SAME')
    print(f'   Both have ID: {tok.pad_token_id}')
    print()
    print('This causes generation to loop (ForCanBeConverted spam)')
    print()
    print('✅ FIX: Updated train_vqa.py adds dedicated <|pad|> token')
    print('   This will be applied automatically during training')
else:
    print('✅ GOOD: PAD and EOS are different tokens')
    print(f'   PAD: {tok.pad_token_id}, EOS: {tok.eos_token_id}')

print('='*80)
"
