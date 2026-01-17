"""
Utility functions for baseline experiments
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

# Using your merged Stage 1 + Stage 2 checkpoint from HuggingFace
AUDIO_ADAPTED_MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

def load_model(model_path=AUDIO_ADAPTED_MODEL_ID):
    """
    Load Qwen2-Audio model with all components from your HuggingFace repo.
    Now that preprocessor_config.json is uploaded, everything loads cleanly!
    """
    print(f"⏳ Loading model from: {model_path}...")

    # Load model weights with Flash Attention 2 to prevent OOM
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # Linear memory scaling
        )
        print("✓ Using Flash Attention 2 (linear memory)")
    except Exception as e:
        print(f"⚠️ Flash Attention 2 not available, using standard attention: {e}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Load processor - hybrid approach needed
    # Your model has preprocessor_config.json but missing chat_template.json
    # So we load from base Qwen and override with your tokenizer
    print("⏳ Loading processor from base Qwen2-VL (for chat template)...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        use_fast=False
    )
    # Use your model's tokenizer (has audio tokens)
    processor.tokenizer = tokenizer

    # Load Whisper feature extractor for audio (Baseline 2)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-large-v3-turbo"
    )

    print("✓ Model, tokenizer, and processor loaded successfully")
    return model, tokenizer, processor, feature_extractor

def load_frames(frame_names, frames_dir):
    """Load PIL images with robust path checking"""
    images = []
    print(f"📁 Loading {len(frame_names)} frames from {frames_dir}")

    for frame_name in frame_names:
        # Frame format: "002-001_18743" - video ID before LAST underscore
        if '_' in frame_name:
            video_id = frame_name.rsplit('_', 1)[0]  # "002-001"
        else:
            video_id = "unknown"

        # Primary path: dataset/frames/002-001/002-001_18743.jpg
        path_a = Path(frames_dir) / video_id / f"{frame_name}.jpg"

        # Fallback: dataset/frames/002-001/18743.jpg (frame num only)
        frame_num = frame_name.rsplit('_', 1)[1] if '_' in frame_name else frame_name
        path_b = Path(frames_dir) / video_id / f"{frame_num}.jpg"

        image = None
        try:
            if path_a.exists():
                image = Image.open(path_a).convert("RGB")
            elif path_b.exists():
                image = Image.open(path_b).convert("RGB")
            else:
                # Try other extensions
                for ext in ['.png', '.jpeg']:
                    if path_a.with_suffix(ext).exists():
                        image = Image.open(path_a.with_suffix(ext)).convert("RGB")
                        break

                if image is None:
                    print(f"⚠️ Frame not found: {frame_name} (tried {path_a}, {path_b})")
                    raise FileNotFoundError(f"Frame {frame_name} not found")

        except Exception as e:
            print(f"⚠️ Error loading {frame_name}: {e}")
            # Black placeholder to prevent crash
            image = Image.new('RGB', (224, 224), color='black')

        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        # CRITICAL: Resize large images to prevent OOM
        # Surgical frames are 1350x1080 which creates too many vision tokens
        max_size = 768  # Reasonable size for video frames
        if image.width > max_size or image.height > max_size:
            # Maintain aspect ratio
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        images.append(image)

    return images

# =============================================================================
# Text + Image VQA Functions (Baselines 1 & 3)
# =============================================================================

def build_prompt_for_vqa(question, num_images):
    """Create the Qwen chat template for VQA with multiple images"""
    # Build content list: use placeholders (not actual images)
    # Images are passed separately to processor
    content = []
    for _ in range(num_images):
        content.append({"type": "image"})  # Placeholder only
    content.append({"type": "text", "text": question})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def process_images(images, messages, processor):
    """Process images and text into model inputs"""
    # 1. Prepare text with chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 2. Process with images directly (simpler, more compatible)
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt"
    )
    return inputs

# Legacy function names for backwards compatibility with baseline scripts
def build_text_vqa_messages(question, images):
    """Build messages structure for text + vision VQA (legacy wrapper)"""
    messages = build_prompt_for_vqa(question, num_images=len(images))
    return messages, images

def process_text_vqa_inputs(messages, images, processor):
    """Process messages and images using processor (legacy wrapper)"""
    return process_images(images, messages, processor)

# =============================================================================
# Audio + Image VQA Functions (Baseline 2)
# =============================================================================

def process_audio(audio_path, feature_extractor, device):
    """
    Process audio file and return Whisper features
    Args:
        audio_path: Path to audio file
        feature_extractor: WhisperFeatureExtractor
        device: torch device
    Returns:
        input_features: Tensor of shape [1, 128, 3000] (Whisper mel spectrogram)
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)  # 16kHz for Whisper
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device).to(torch.bfloat16)
    return input_features

def build_audio_vqa_inputs(tokenizer, instruction, device):
    """
    Build input_ids for audio + vision VQA
    Manually constructs input with audio token injection

    Args:
        tokenizer: Qwen2-VL tokenizer
        instruction: Text instruction (e.g., "Answer this question based on the audio and images.")
        device: torch device
    Returns:
        input_ids: Tensor with audio tokens injected
        attention_mask: Attention mask
    """
    # Audio token configuration
    AUDIO_TOKEN_ID = 151657
    NUM_AUDIO_TOKENS = 1500

    # Create audio token sequence
    audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
    input_ids_audio = torch.tensor([audio_tokens], device=device, dtype=torch.long)

    # Build prompt in parts
    p1 = tokenizer.encode(
        "<|im_start|>user\n<|audio_bos|>",
        add_special_tokens=False,
        return_tensors="pt"
    ).to(device)

    p2 = tokenizer.encode(
        f"<|audio_eos|>\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        return_tensors="pt"
    ).to(device)

    # Concatenate: prefix + audio_tokens + suffix
    input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask

def build_audio_vision_vqa_inputs(tokenizer, processor, images, device):
    """
    Build input_ids for Audio + Vision VQA
    Properly combines audio tokens with vision tokens
    CRITICAL: Uses processor to generate vision tokens (not tokenizer!)

    Args:
        tokenizer: Qwen2-VL tokenizer
        processor: Qwen2-VL processor (for vision token generation)
        images: List of PIL images
        device: torch device
    Returns:
        input_ids: Tensor with audio AND vision tokens injected
        attention_mask: Attention mask
    """
    # Audio token configuration
    AUDIO_TOKEN_ID = 151657
    NUM_AUDIO_TOKENS = 1500

    # 1. Build messages with image placeholders
    content = []
    for _ in range(len(images)):
        content.append({"type": "image"})
    content.append({"type": "text", "text": "Answer the question based on the audio and images."})
    messages = [{"role": "user", "content": content}]

    # 2. Use processor to create FULL input_ids with actual vision token counts
    # This is CRITICAL - processor looks at real image sizes to calculate token counts
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_inputs = processor(text=[text], images=images, return_tensors="pt")
    vision_input_ids = vision_inputs['input_ids'].to(device)  # Has correct vision placeholder tokens!

    # 3. Now inject audio tokens BEFORE the vision content
    # Format: <|im_start|>user\n<|audio_bos|>[1500 AUDIO]<|audio_eos|>\n[VISION_TOKENS]...
    audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
    input_ids_audio = torch.tensor([audio_tokens], device=device, dtype=torch.long)

    # Build audio prefix
    audio_prefix = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(device)
    audio_suffix = tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False, return_tensors="pt").to(device)

    # Find where user content starts (after "<|im_start|>user\n")
    user_prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False, return_tensors="pt")
    user_prefix_len = user_prefix.shape[1]

    # Get vision content (everything after user prefix)
    vision_content = vision_input_ids[:, user_prefix_len:]

    # Combine: audio_prefix + audio_tokens + audio_suffix + vision_content
    final_input_ids = torch.cat([audio_prefix, input_ids_audio, audio_suffix, vision_content], dim=1)
    attention_mask = torch.ones_like(final_input_ids)

    return final_input_ids, attention_mask

# =============================================================================
# Shared Utility Functions
# =============================================================================

def check_answer_match(predicted, ground_truth, short_answer):
    """Fuzzy match prediction against ground truth"""
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    short = short_answer.lower().strip()

    return (short in pred) or (gt in pred) or (pred in gt)

def load_jsonl(path):
    """Load JSONL file"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    """Save to JSONL file"""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
