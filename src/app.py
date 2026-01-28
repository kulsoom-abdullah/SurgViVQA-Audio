"""
SurgViVQA-Audio: Interactive Demo with Flipbook Animation
Natural language questions (OUT template) + Question type filtering
"""

import streamlit as st
import torch
import librosa
import numpy as np
import tempfile
import os
import json
import random
import time
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    WhisperFeatureExtractor,
    BitsAndBytesConfig
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CHECKPOINT_PATH = "./checkpoints/surgical_vqa_multivideo"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
AUDIO_ADAPTED_MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

st.set_page_config(page_title="SurgViVQA Assistant", layout="wide", page_icon="🩺")

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model_system():
    """Loads model once and keeps it in memory"""
    with st.spinner("⏳ Loading Audio-Grafted Model + LoRA Adapters..."):
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load base model
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            AUDIO_ADAPTED_MODEL_ID,
            quantization_config=bnb_config,
            device_map=DEVICE,
            attn_implementation="sdpa",
            trust_remote_code=True
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        model.eval()

        # Processors
        tokenizer = AutoTokenizer.from_pretrained(
            AUDIO_ADAPTED_MODEL_ID,
            trust_remote_code=True,
            use_fast=False
        )
        if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            trust_remote_code=True,
            use_fast=False
        )
        processor.tokenizer = tokenizer

        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

        return model, processor, tokenizer, feature_extractor

# --- 2. DATA LOADING ---
@st.cache_data
def load_samples():
    """Loads test/eval samples - prioritizes OUT (natural language) data"""
    # Priority: Test Multivideo -> Eval Multivideo -> Out test set
    paths = [
        "data/test_multivideo.jsonl",
        "data/eval_multivideo.jsonl",
        "data/eval_002001_stratified.jsonl"
    ]
    data = []
    loaded_from = None
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                data = [json.loads(line) for line in f]
            loaded_from = p
            break

    return data, loaded_from

def get_frames(sample, frames_dir="data/frames"):
    """Loads frames for a sample"""
    images = []
    for frame_name in sample['frames']:
        # Handle file naming (e.g., 002-001_12345 or just 12345)
        if "_" in frame_name:
            vid_id = frame_name.rsplit('_', 1)[0]
        else:
            vid_id = sample.get('video_id', 'unknown')

        path = Path(frames_dir) / vid_id / f"{frame_name}.jpg"

        if path.exists():
            img = Image.open(path).convert("RGB")
            # Resize to match training (384px)
            if max(img.size) > 384:
                img.thumbnail((384, 384))
            images.append(img)
        else:
            # Gray placeholder if missing
            images.append(Image.new('RGB', (384, 384), (100, 100, 100)))
    return images

# --- INFERENCE FUNCTION ---
def run_inference(model, processor, tokenizer, feature_extractor, frames, audio_array, question_text):
    """Run model inference with audio adapter"""

    # 1. Process audio
    audio_inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = audio_inputs.input_features.to(DEVICE).to(torch.bfloat16)

    # 2. Build prompt
    content = [{"type": "image"} for _ in frames]
    content.append({
        "type": "text",
        "text": f"User Question: {question_text}\nAnswer the question concisely based on the visual and audio."
    })
    msgs = [{"role": "user", "content": content}]

    text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # 3. Process vision inputs
    batch = processor(text=[text_prompt], images=frames, return_tensors="pt")

    input_ids = batch.input_ids.to(DEVICE)
    pixel_values = batch.pixel_values.to(DEVICE).to(torch.bfloat16)
    image_grid_thw = batch.image_grid_thw.to(DEVICE)

    # 4. Audio grafting (inject audio tokens)
    AUDIO_TOKEN_ID = 151657
    NUM_AUDIO_TOKENS = 1500

    audio_tokens = torch.tensor([[AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS], device=DEVICE, dtype=torch.long)
    audio_header = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(DEVICE)
    audio_footer = tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False, return_tensors="pt").to(DEVICE)

    # Strip user prefix to avoid duplication
    user_prefix_len = len(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
    vision_content = input_ids[:, user_prefix_len:]

    final_input_ids = torch.cat([audio_header, audio_tokens, audio_footer, vision_content], dim=1)
    attention_mask = torch.ones_like(final_input_ids)

    # 5. Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=final_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_features=input_features,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 6. Decode
    new_tokens = generated_ids[0, final_input_ids.shape[1]:]
    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return pred_text

# --- MAIN APP ---
st.title("🩺 SurgViVQA-Audio: Interactive Demo")
st.markdown(f"**Qwen2-VL-Audio-Adapter** | Device: `{DEVICE}`")

# Load model (happens once, cached)
model, processor, tokenizer, feature_extractor = load_model_system()

# Initialize session state
if "data" not in st.session_state:
    data, loaded_from = load_samples()
    if not data:
        st.error("❌ No eval data found!")
        st.stop()
    st.session_state.data = data
    st.session_state.loaded_from = loaded_from
    st.session_state.filtered_data = data
    st.session_state.sample = random.choice(data)
    st.session_state.prediction = None

def next_sample():
    if st.session_state.filtered_data:
        st.session_state.sample = random.choice(st.session_state.filtered_data)
        st.session_state.prediction = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🩻 Case Selection")

    # Show data source
    st.success(f"✅ Loaded {len(st.session_state.data)} samples")
    st.caption(f"From: {st.session_state.loaded_from}")

    st.divider()

    # 1. Filter by Question Type
    all_types = sorted(list(set([d['question_type'] for d in st.session_state.data])))
    selected_type = st.selectbox(
        "Filter by Question Type",
        ["All"] + all_types,
        help="Select specific question type or 'All' for random sampling"
    )

    # Update filtered data
    if selected_type == "All":
        st.session_state.filtered_data = st.session_state.data
    else:
        st.session_state.filtered_data = [d for d in st.session_state.data if d['question_type'] == selected_type]

    st.caption(f"🔍 {len(st.session_state.filtered_data)} cases match filter")

    # 2. Random button
    if st.button("🎲 Random New Case", type="primary", use_container_width=True):
        next_sample()
        st.rerun()

    st.divider()

    # 3. Case info
    sample = st.session_state.sample
    st.markdown("### 📊 Current Case")
    st.write(f"**Type:** {sample.get('question_type', 'N/A')}")
    st.write(f"**Video:** {sample.get('video_id', 'N/A')}")
    st.write(f"**ID:** {sample.get('id', 'N/A')}")

    st.divider()

    # 4. Ground truth
    st.markdown("### 🎯 Ground Truth")
    # Prefer full answer over short_answer for natural language
    truth_text = sample.get('answer', sample.get('short_answer', 'N/A'))
    st.success(f"**{truth_text}**")

    st.divider()

# 5. Model Strengths (The Behavioral Profile)
    st.markdown("### 🧠 Model Capability Profile")
    st.info("""
    **✅ 100% Reliable (Static):**
    * Blue Dye Presence
    * Lighting Mode (NBI/White)
    * Endoscope Visibility
    
    **⚠️ 84% Reliable (Safety):**
    * Occlusion Check (Blocked View?)
    
    **📉 20-55% Reliable (Dynamic):**
    * Motion (Is scope advancing?)
    * Precise Localization (Which quadrant?)
    """)
    
    st.caption("Based on held-out test video 002-004")

# --- MAIN CONTENT ---
sample = st.session_state.sample

st.subheader(f"❓ {sample['question']}")

# Get frames
frames = get_frames(sample)

# 1. VISUALIZATION (Tabs for Grid vs Animation)
tab_grid, tab_anim = st.tabs(["🎞️ Frame Grid", "▶️ Play Sequence"])

with tab_grid:
    if frames:
        st.image(frames, width=150, caption=[f"Frame {i+1}" for i in range(len(frames))])
    else:
        st.error("No frames found for this sample")

with tab_anim:
    col_play, col_desc = st.columns([1, 2])

    with col_play:
        if st.button("▶️ Play Sequence (2 FPS)", use_container_width=True):
            placeholder = st.empty()
            for loop in range(2):  # Play twice
                for i, frame in enumerate(frames):
                    placeholder.image(frame, caption=f"Frame {i+1}/{len(frames)}", width=400)
                    time.sleep(0.5)  # 2 FPS for better visibility

    with col_desc:
        st.markdown("""
        **Video Frames Animation:**

        Motion questions like *"Is the scope advancing?"* require understanding temporal changes.

        The model processes all 8 frames simultaneously to detect patterns like:
        - 🔄 Scope rotation
        - ⬆️ Forward/backward movement
        - 🎯 Tool insertion/removal

        Click "Play Sequence" to see the motion yourself!
        """)

st.divider()

# 2. AUDIO INPUT + INFERENCE
col_audio, col_result = st.columns([1, 1])

with col_audio:
    st.markdown("### 🎤 Record Your Question")
    st.info(f'**Say:** "{sample["question"]}"')

    audio_value = st.audio_input("Click microphone to record")

    if audio_value:
        st.audio(audio_value)

        # Run inference button
        if st.button("🚀 Run Inference", type="primary", use_container_width=True):
            with st.spinner("Processing audio + vision..."):
                # Load audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_value.getvalue())
                    tmp_path = tmp.name

                audio_array, _ = librosa.load(tmp_path, sr=16000, mono=True)
                os.remove(tmp_path)

                # Run inference
                prediction = run_inference(
                    model, processor, tokenizer, feature_extractor,
                    frames, audio_array, sample['question']
                )

                st.session_state.prediction = prediction
                st.rerun()

with col_result:
    st.markdown("### 🤖 Model Prediction")

    if st.session_state.prediction:
        pred = st.session_state.prediction
        ground_truth = sample.get('answer', sample.get('short_answer', ''))

        # Check correctness (fuzzy matching)
        # Both "yes" in "yes" and "the scope is advancing" in "yes, the scope is advancing"
        is_correct = (
            ground_truth.lower() in pred.lower() or
            pred.lower() in ground_truth.lower()
        )

        if is_correct:
            st.success(f"✅ **{pred}**")
            st.balloons()
        else:
            st.warning(f"❌ **{pred}**")

        st.caption(f"Ground Truth: {ground_truth}")

        # Analysis
        st.divider()
        if is_correct:
            st.success("🎉 **Correct!** Model prediction matches ground truth.")
        else:
            st.info(f"💡 **Analysis:** This is a `{sample['question_type']}` question. These can be challenging.")
            if sample['question_type'] in ['scope_motion', 'lesion_motion_direction']:
                st.caption("Motion questions have lower accuracy (45-54%) - temporal modeling is hard!")
    else:
        st.info("👆 Record audio and click 'Run Inference' to see prediction")

# Footer
st.divider()
st.caption("Built with Streamlit | Audio-Grafted Qwen2-VL + Whisper-v3-Turbo | QLoRA Fine-tuning")
