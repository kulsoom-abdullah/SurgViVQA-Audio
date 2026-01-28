"""
Utility functions - Load everything from your model (no hybrid)
"""
import sys
import os
import torch
import json
import librosa
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    WhisperFeatureExtractor
)

def load_model(model_path="kulsoom-abdullah/Qwen2-Audio-7B-Transcription"):
    """Load everything directly from your model"""
    print(f"⏳ Loading model from: {model_path}...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load tokenizer from base Qwen2-VL (your model has wrong tokenizer!)
    print("⏳ Loading tokenizer from base Qwen2-VL...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        use_fast=False
    )

    # Load processor from base Qwen2-VL
    print("⏳ Loading processor from base Qwen2-VL...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        use_fast=False
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-large-v3-turbo"
    )

    print("✓ Model, tokenizer, and processor loaded successfully")
    return model, tokenizer, processor, feature_extractor

def load_frames(frame_names, frames_dir):
    """Load PIL images with robust path checking"""
    images = []

    if getattr(load_frames, 'first_run', True):
        print(f"📁 Loading frames from {frames_dir}")
        load_frames.first_run = False

    for frame_name in frame_names:
        try:
            video_id = frame_name.rsplit('_', 1)[0]
        except IndexError:
            video_id = "unknown"

        path_a = Path(frames_dir) / video_id / f"{frame_name}.jpg"

        if '_' in frame_name:
            frame_num = frame_name.rsplit('_', 1)[-1]
        else:
            frame_num = frame_name
        path_b = Path(frames_dir) / video_id / f"{frame_num}.jpg"

        image = None
        try:
            if path_a.exists():
                image = Image.open(path_a).convert("RGB")
            elif path_b.exists():
                image = Image.open(path_b).convert("RGB")
            else:
                for ext in ['.png', '.jpeg']:
                    if path_a.with_suffix(ext).exists():
                        image = Image.open(path_a.with_suffix(ext)).convert("RGB")
                        break
        except Exception:
            pass

        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        images.append(image)

    return images

def build_prompt_for_vqa(question, num_images):
    """Create messages with image placeholders"""
    content = []
    for _ in range(num_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})

    return [{"role": "user", "content": content}]

def process_images(images, messages, processor):
    """Process using native API - simple version"""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Simplest possible call - let processor handle everything
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt"
    )
    return inputs

def build_text_vqa_messages(question, images):
    messages = build_prompt_for_vqa(question, num_images=len(images))
    return messages, images

def process_text_vqa_inputs(messages, images, processor):
    return process_images(images, messages, processor)

def process_audio(audio_path, feature_extractor, device):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device).to(torch.bfloat16)
    return input_features

def build_audio_vqa_inputs(tokenizer, instruction, device):
    AUDIO_TOKEN_ID = 151657
    NUM_AUDIO_TOKENS = 1500
    audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
    input_ids_audio = torch.tensor([audio_tokens], device=device, dtype=torch.long)

    p1 = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(device)
    p2 = tokenizer.encode(f"<|audio_eos|>\n{instruction}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt").to(device)

    return torch.cat([p1, input_ids_audio, p2], dim=1), torch.ones_like(torch.cat([p1, input_ids_audio, p2], dim=1))

def check_answer_match(predicted, ground_truth, short_answer):
    pred, gt, short = predicted.lower().strip(), ground_truth.lower().strip(), short_answer.lower().strip()
    return (short in pred) or (gt in pred) or (pred in gt)

def load_jsonl(path):
    with open(path, 'r') as f: return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data: f.write(json.dumps(item) + "\n")
