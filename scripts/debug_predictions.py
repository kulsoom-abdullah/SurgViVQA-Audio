"""
Debug script: Check what model actually predicts for 5 training samples
Run this AFTER training to see if model learned anything
"""
import torch
import json
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, WhisperFeatureExtractor
from PIL import Image
import librosa

# Load trained model
MODEL_PATH = "./checkpoints/overfit_test_5samples"  # Your checkpoint path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("kulsoom-abdullah/Qwen2-Audio-7B-Transcription", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
processor.tokenizer = tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

# Load 5 training samples
data = [json.loads(line) for line in open("test_set/tiny_5samples.jsonl")]

print("="*80)
print("PREDICTION CHECK - MODEL vs GROUND TRUTH")
print("="*80)

for i, sample in enumerate(data[:5]):
    print(f"\n{'='*80}")
    print(f"SAMPLE {i+1}/{5}")
    print(f"{'='*80}")
    print(f"Question: {sample['question']}")
    print(f"Ground Truth: {sample['answer']}")
    print(f"Short Answer: {sample['short_answer']}")

    # Load audio
    audio_path = Path("audio/in_002-001") / f"{sample['id']}.mp3"
    if audio_path.exists():
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
    else:
        y = torch.zeros(16000 * 2)

    audio_inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    input_features = audio_inputs.input_features.to(model.device).to(torch.bfloat16)

    # Load images
    images = []
    for frame_name in sample['frames']:
        vid_id = frame_name.rsplit('_', 1)[0] if '_' in frame_name else "unknown"
        path = Path("dataset/frames") / vid_id / f"{frame_name}.jpg"

        if not path.exists():
            frame_num = frame_name.rsplit('_', 1)[1] if '_' in frame_name else frame_name
            path = Path("dataset/frames") / vid_id / f"{frame_num}.jpg"

        if path.exists():
            img = Image.open(path).convert("RGB")
            if max(img.size) > 384:
                img.thumbnail((384, 384))
            images.append(img)

    # Build prompt
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": f"User Question: {sample['question']}\nAnswer the question concisely based on the visual and audio evidence."})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=images, return_tensors="pt")

    # Inject audio tokens
    AUDIO_TOKEN_ID = 151657
    NUM_AUDIO_TOKENS = 1500

    audio_tokens = torch.tensor([[AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS], device=model.device)
    audio_header = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
    audio_footer = tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False, return_tensors="pt").to(model.device)

    user_prefix_len = len(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
    vision_content = batch.input_ids[:, user_prefix_len:].to(model.device)

    input_ids = torch.cat([audio_header, audio_tokens, audio_footer, vision_content], dim=1)

    # Create attention mask (all 1s, no padding in this input)
    attention_mask = torch.ones_like(input_ids)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            pixel_values=batch.pixel_values.to(model.device).to(torch.bfloat16),
            image_grid_thw=batch.image_grid_thw.to(model.device),
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    predicted = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\n{'─'*80}")
    print(f"MODEL PREDICTION: {predicted}")
    print(f"{'─'*80}")

    # Check if prediction matches
    if sample['short_answer'].lower() in predicted.lower():
        print("✅ MATCH!")
    else:
        print("❌ NO MATCH")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("If predictions are random/nonsense → model didn't learn")
print("If predictions are same for all samples → model collapsed to single output")
print("If predictions match ground truth → training worked!")
