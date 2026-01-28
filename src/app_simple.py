"""
Simple Streamlit Test - Start Here
Tests: Frame loading, audio recording, basic UI
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import numpy as np
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Surgical VQA Demo", layout="wide")

st.title("🏥 Surgical VQA Demo - Step by Step Test")

# Step 1: Load eval samples
st.header("Step 1: Load Sample Data")

EVAL_DATA = "eval_multivideo.jsonl"
if Path(EVAL_DATA).exists():
    with open(EVAL_DATA) as f:
        samples = [json.loads(line) for line in f]
    st.success(f"✅ Loaded {len(samples)} eval samples")

    # Pick one sample to test
    sample = samples[0]
    st.json({
        "question": sample['question'],
        "question_type": sample['question_type'],
        "video_id": sample['video_id'],
        "answer": sample['short_answer']
    })
else:
    st.error(f"❌ {EVAL_DATA} not found")
    st.stop()

# Step 2: Load frame
st.header("Step 2: Load Surgical Frame")

frame_name = sample['frames'][0]  # First frame
vid_id = frame_name.rsplit('_', 1)[0] if '_' in frame_name else "unknown"
frame_path = Path("data/frames") / vid_id / f"{frame_name}.jpg"

if frame_path.exists():
    image = Image.open(frame_path).convert("RGB")
    st.image(image, caption=f"Frame: {frame_name}", width=400)
    st.success(f"✅ Loaded frame from {frame_path}")
else:
    st.error(f"❌ Frame not found: {frame_path}")
    st.stop()

# Step 3: Display question
st.header("Step 3: Question")
st.info(f"**{sample['question']}**")
st.write(f"Ground truth answer: **{sample['short_answer']}**")

# Step 4: Audio recording
st.header("Step 4: Record Your Voice")
st.write("Click the microphone to record yourself asking the question")

audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#3498db",
    icon_size="2x"
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.success("✅ Audio recorded!")

    # Save for inspection
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    st.write(f"Audio shape: {audio_array.shape}")
    st.write(f"Sample rate: 16000 Hz (assumed)")
else:
    st.info("No audio recorded yet")

# Step 5: Next steps
st.header("Step 5: Next - Add Model Inference")
st.write("Once steps 1-4 work, we'll add the model!")

st.sidebar.markdown("### Progress")
st.sidebar.markdown("""
- ✅ Step 1: Load data
- ✅ Step 2: Load frame
- ✅ Step 3: Show question
- 🎤 Step 4: Record audio
- ⏳ Step 5: Model inference (next)
""")
